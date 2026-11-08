import json

import numpy as np
import pandas as pd
import pytest

from forecast1x2 import tracking as tr

NOW = pd.Timestamp("2026-10-23 12:00", tz="UTC")
PARAMS = {"rho": -0.05, "nu": 1.1, "first_half_share": 0.44}


def vt():
    """value_table-like rows for 3 matches (A and B on Saturday, C starts in 30 minutes)."""
    rows = [("A", "Under/Over", "Under 4.5", 0.80, 1.20, "2026-10-24T14:00:00Z"),
            ("A", "Doppia chance", "1X", 0.70, 1.40, "2026-10-24T14:00:00Z"),
            ("B", "Under/Over", "Over 1.5", 0.75, 1.30, "2026-10-24T16:00:00Z"),
            ("C", "Goal/No goal", "Goal", 0.62, 1.55, "2026-10-23T12:30:00Z")]
    df = pd.DataFrame(rows, columns=["home", "group", "market", "prob", "odds", "kickoff_utc"])
    df["league"], df["away"], df["bookmaker"] = "I1", df.home + "x", "bet365.it"
    df["fair_odds"], df["value"], df["verdict"] = 1 / df.prob, df.odds * df.prob - 1, "nessun valore"
    return df


def empty_picks():
    return pd.DataFrame(columns=tr.PICK_COLS)


def test_record_picks_once_per_match_and_kind_before_kickoff():
    reg = tr.record_picks(vt(), empty_picks(), NOW)
    got = {(k, h): m for k, h, m in zip(reg.kind, reg.home, reg.market)}
    assert got == {("più probabile", "A"): "Under 4.5", ("più probabile", "B"): "Over 1.5",
                   ("più conveniente", "A"): "1X", ("più conveniente", "B"): "Over 1.5"}   # C starts too soon
    assert (reg.status == "aperta").all()
    later = vt().assign(odds=2.0)                      # odds move: the logged pick must not change
    reg2 = tr.record_picks(later, reg, NOW + pd.Timedelta(hours=2))
    assert len(reg2) == len(reg) and reg2.odds.tolist() == reg.odds.tolist()


def test_record_slips_once_a_day():
    reg = tr.record_slips(vt(), pd.DataFrame(columns=tr.SLIP_COLS), NOW)
    assert "Singola (quota ≥ 1,50)" not in set(reg.preset)          # no single pick with odds >= 1.50 before kickoff
    assert "Doppia prudente (≥ 1,50)" in set(reg.preset)
    legs = json.loads(reg.set_index("preset").loc["Doppia prudente (≥ 1,50)", "legs"])
    assert {l["home"] for l in legs} == {"A", "B"}
    again = tr.record_slips(vt(), reg, NOW + pd.Timedelta(hours=3))  # same day: nothing new
    assert len(again) == len(reg)


def results(rows):
    """results_table-like frame: (home, date, hg, ag, h1, a1) with closing probabilities."""
    df = pd.DataFrame(rows, columns=["home", "date", "hg", "ag", "h1", "a1"])
    return df.assign(league="I1", away=df.home + "x", c1=0.45, c2=0.28, cover=0.52)


def test_settle_picks_outcomes_refunds_voids_and_clv():
    reg = pd.DataFrame({"logged_at": NOW.isoformat(), "kind": "più probabile", "league": "I1",
                        "kickoff_utc": ["2026-10-24T14:00:00Z"] * 3 + ["2026-10-01T14:00:00Z"],
                        "home": ["A", "B", "D", "E"], "away": ["Ax", "Bx", "Dx", "Ex"],
                        "group": ["Under/Over", "1X2", "Draw no bet", "1X2"],
                        "market": ["Over 1.5", "1", "1 (rimborso se X)", "1"], "bookmaker": "x",
                        "odds": [1.3, 2.4, 1.6, 2.0], "prob": 0.5, "value": 0.0, "status": "aperta",
                        "profit": np.nan, "close_prob": np.nan, "clv": np.nan})
    res = results([("A", "2026-10-24", 2, 1, 1, 0), ("B", "2026-10-24", 0, 0, 0, 0), ("D", "2026-10-24", 1, 1, 0, 1)])
    out = tr.settle_picks(reg, res, PARAMS, pd.Timestamp("2026-10-25 12:00", tz="UTC")).set_index("home")
    assert out.loc["A", "status"] == "vinta" and out.loc["A", "profit"] == pytest.approx(0.3)
    assert out.loc["B", "status"] == "persa" and out.loc["B", "profit"] == -1
    assert out.loc["D", "status"] == "rimborsata" and out.loc["D", "profit"] == 0
    assert out.loc["E", "status"] == "annullata"                      # never played, more than 7 days ago
    assert 0.4 < out.loc["B", "close_prob"] < 0.5                     # closing «1»: close to the 0.45 given
    assert out.loc["B", "clv"] == pytest.approx(2.4 * out.loc["B", "close_prob"] - 1, abs=1e-4)


def test_settle_slips_win_refund_at_one_and_loss():
    def leg(home, group, market, odds):
        return {"league": "I1", "kickoff_utc": "2026-10-24T14:00:00Z", "home": home, "away": home + "x",
                "group": group, "market": market, "bookmaker": "x", "odds": odds, "prob": 0.5}
    over = leg("A", "Under/Over", "Over 1.5", 1.3)
    reg = pd.DataFrame({"logged_at": NOW.isoformat(), "preset": ["win", "refund", "loss", "pending"],
                        "legs": [json.dumps([over, leg("B", "Under/Over", "Under 2.5", 1.8)]),
                                 json.dumps([over, leg("D", "Draw no bet", "1 (rimborso se X)", 1.6)]),
                                 json.dumps([over, leg("B", "1X2", "1", 2.4)]),
                                 json.dumps([over, leg("Z", "1X2", "1", 2.0)])],
                        "odds": 2.0, "prob": 0.5, "value": 0.0, "status": "aperta", "profit": np.nan, "clv": np.nan})
    res = results([("A", "2026-10-24", 2, 1, 1, 0), ("B", "2026-10-24", 0, 0, 0, 0), ("D", "2026-10-24", 1, 1, 0, 1)])
    out = tr.settle_slips(reg, res, PARAMS, pd.Timestamp("2026-10-25 12:00", tz="UTC")).set_index("preset")
    assert out.loc["win", "status"] == "vinta" and out.loc["win", "profit"] == pytest.approx(1.3 * 1.8 - 1)
    assert out.loc["refund", "status"] == "vinta" and out.loc["refund", "profit"] == pytest.approx(0.3)   # x 1.00
    assert out.loc["loss", "status"] == "persa"
    assert out.loc["pending", "status"] == "aperta"                    # Z not played yet, less than 7 days
    assert not np.isnan(out.loc["win", "clv"])


def test_summary():
    reg = pd.DataFrame({"kickoff_utc": ["2026-10-24T14:00:00Z"] * 3, "status": ["vinta", "persa", "aperta"],
                        "profit": [0.5, -1.0, np.nan], "clv": [0.02, -0.04, np.nan], "kind": ["a", "b", "a"]})
    s = tr.summary(reg, "kind", n_boot=50).set_index("group")
    assert s.loc["tutte", "bets"] == 2 and s.loc["tutte", "roi"] == pytest.approx(-0.25)
    assert s.loc["tutte", "clv"] == pytest.approx(-0.01) and s.loc["a", "wins"] == 1
