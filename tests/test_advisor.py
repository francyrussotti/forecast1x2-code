import pandas as pd
import pytest

from forecast1x2 import advisor as ad


def vt():
    """value_table-like rows: 3 matches."""
    rows = [
        # match A: most probable Under 4.5 (1.20), cheapest 1X (smaller margin), a long shot and a suspicious quote
        ("A", "Under/Over", "Under 4.5", 0.80, 1.20), ("A", "Doppia chance", "1X", 0.70, 1.40),
        ("A", "1X2", "2", 0.20, 4.50), ("A", "1X2", "X", 0.25, 9.00),
        ("A", "Draw no bet", "1 (rimborso se X)", 0.90, 1.21),          # excluded group
        # match B: best odds too low (1.10), then Over 1.5
        ("B", "Under/Over", "Under 5.5", 0.95, 1.10), ("B", "Under/Over", "Over 1.5", 0.75, 1.30),
        # match C: Goal 1.55 and a very expensive Over 2.5
        ("C", "Goal/No goal", "Goal", 0.62, 1.55), ("C", "Under/Over", "Over 2.5", 0.55, 1.50),
    ]
    df = pd.DataFrame(rows, columns=["home", "group", "market", "prob", "odds"])
    df["league"], df["away"], df["kickoff_utc"] = "I1", df.home + "x", "2026-10-24T14:00:00Z"
    df["fair_odds"] = 1 / df.prob
    df["value"] = df.odds * df.prob - 1
    df["verdict"] = ["quota sospetta" if o > 2 * f else "nessun valore" for o, f in zip(df.odds, df.fair_odds)]
    return df


def test_candidates_apply_odds_cost_and_group_rules():
    c = ad.candidates(vt())
    # out: 1.10 (too low), 4.50 and 9.00 (too high / suspicious), DNB (excluded group), Over 2.5 (cost -17.5%)
    assert set(c.market) == {"Under 4.5", "1X", "Over 1.5", "Goal"}


def test_best_and_cheapest_picks_one_per_match():
    bp = ad.best_picks(vt()).set_index("home")
    assert bp.loc["A", "market"] == "Under 4.5" and bp.loc["B", "market"] == "Over 1.5" and bp.loc["C", "market"] == "Goal"
    assert list(bp.index) == ["A", "B", "C"]   # sorted by probability
    cp = ad.cheapest_picks(vt()).set_index("home")
    assert cp.loc["A", "market"] == "1X"       # -2% beats Under 4.5 at -4%
    assert cp.loc["C", "market"] == "Goal"


def test_best_multiple_uses_different_matches_and_reaches_the_target():
    m = ad.best_multiple(vt(), 2, 1.5)
    assert m["odds"] >= 1.5 and m["legs"].home.is_unique
    assert set(m["legs"].home) == {"A", "B"} and m["prob"] == pytest.approx(0.8 * 0.75)
    assert m["value"] == pytest.approx(0.8 * 0.75 * 1.20 * 1.30 - 1)
    # total >= 2.00: Over 1.5 (1.30) x Goal (1.55) = 2.015 is the most probable valid double (0.75 x 0.62)
    m2 = ad.best_multiple(vt(), 2, 2.0)
    assert list(m2["legs"].market) == ["Over 1.5", "Goal"] and m2["prob"] == pytest.approx(0.75 * 0.62)
    assert ad.best_multiple(vt(), 3, 50.0) is None


def test_alternative_slips_share_no_match_and_value_objective():
    slips = ad.alternative_slips(vt(), 1, 1.2, n=5)
    homes = [m["legs"].home.iloc[0] for m in slips]
    assert homes == ["A", "B", "C"]                      # each match used once, most probable first
    best_value = ad.best_multiple(vt(), 1, 1.2, objective="value")
    assert best_value["legs"].market.iloc[0] == "1X"     # 0.70 x 1.40 = 0.98: the smallest loss
    with pytest.raises(ValueError):
        ad.best_multiple(vt(), 1, 1.2, objective="boh")


def test_pareto_options_drop_dominated_choices():
    opts = {d.home.iloc[0]: list(d.market) for d in ad.pareto_options(ad.candidates(vt()))}
    assert opts["A"] == ["Under 4.5", "1X"]   # 1X is less likely but pays more: kept


def test_exotic_suggestions_and_money():
    fair = pd.DataFrame({"group": ["Multigoal", "Multigoal", "Combo DC + U/O", "1X2"],
                         "market": ["Multigoal 1-3", "Multigoal 2-4", "1X + Under 3.5", "1"],
                         "prob": [0.55, 0.65, 0.60, 0.70], "fair_odds": [1 / 0.55, 1 / 0.65, 1 / 0.60, 1 / 0.70]})
    ex = ad.exotic_suggestions(fair)
    assert list(ex.market) == ["Multigoal 2-4", "1X + Under 3.5"]   # one per group, most probable, no 1X2
    o = ad.expected_outcome(0.6, 1.5, 10)
    assert o["payout"] == 15 and o["avg_result"] == pytest.approx(-1.0) and o["win_pct"] == pytest.approx(60)
