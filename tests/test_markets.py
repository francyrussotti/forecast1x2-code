import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from forecast1x2 import markets as mk

PARAMS = {"rho": -0.05, "nu": 1.1, "first_half_share": 0.44}


def probs(lh=1.6, la=1.1):
    return mk.market_probs([lh], [la], PARAMS["rho"], PARAMS["first_half_share"], PARAMS["nu"]).iloc[0]


def test_exhaustive_markets_sum_to_one():
    p = probs()
    for group, names in [("1X2", ["1", "X", "2"]), ("Goal/No goal", ["Goal", "No goal"]),
                         ("Pari/Dispari", ["Pari", "Dispari"]), ("Tempo con più gol", ["1° tempo", "2° tempo", "Uguale"]),
                         ("Primo tempo", ["1X2 1T: 1", "1X2 1T: X", "1X2 1T: 2"]),
                         ("Handicap 1X2", ["casa -1: 1", "casa -1: X", "casa -1: 2"])]:
        assert sum(p[(group, n)] for n in names) == pytest.approx(1, abs=1e-9), group
    assert sum(p[("Risultato esatto", n)] for (g, n) in p.index if g == "Risultato esatto") == pytest.approx(1)
    assert sum(p[("Tempo/finale", n)] for (g, n) in p.index if g == "Tempo/finale") == pytest.approx(1, abs=1e-6)
    for line in [0.5, 1.5, 2.5, 3.5]:
        assert p[("Under/Over", f"Under {line}")] + p[("Under/Over", f"Over {line}")] == pytest.approx(1)
    assert p[("Doppia chance", "1X")] == pytest.approx(p[("1X2", "1")] + p[("1X2", "X")])
    assert p[("Combo 1X2 + U/O", "1 + Over 2.5")] < min(p[("1X2", "1")], p[("Under/Over", "Over 2.5")])


def test_halves_add_up_to_the_match():
    p = probs()
    # no goal in the match <=> no goal in either half: the halves model must roughly agree with the full-time one
    no_goal_halves = p[("Primo tempo", "Under 0.5 1T")] * p[("Secondo tempo", "Under 0.5 2T")]
    assert no_goal_halves == pytest.approx(p[("Under/Over", "Under 0.5")], abs=0.02)


def test_fit_rates_reproduces_market_probabilities():
    M = mk.score_matrices([1.7, 0.9], [0.8, 1.5], -0.05, nu=1.1)
    target = mk._core_probs(M)
    lh, la, resid = mk.fit_rates(target[:, 0], target[:, 1], target[:, 2], -0.05, 1.1)
    assert np.allclose(lh, [1.7, 0.9], atol=1e-3) and np.allclose(la, [0.8, 1.5], atol=1e-3)
    assert resid.max() < 1e-6


def test_goal_distribution():
    k = np.arange(mk.G)
    assert np.allclose(mk.goal_pmf([1.3], nu=1.0)[0], poisson.pmf(k, 1.3))
    p1, p2 = mk.goal_pmf([1.3], nu=1.0)[0], mk.goal_pmf([1.3], nu=1.2)[0]
    var = lambda p: (p * k ** 2).sum() - (p * k).sum() ** 2
    assert var(p2) / (p2 * k).sum() < var(p1) / (p1 * k).sum()   # nu > 1: less dispersed than Poisson
    m = mk.rate_for_mean([0.6, 1.2], mk.GH, 1.1)
    assert np.allclose((mk.goal_pmf(m, mk.GH, 1.1) * np.arange(mk.GH)).sum(axis=1), [0.6, 1.2], atol=1e-3)


def test_outcomes_of_a_real_score():
    # 2-1 at full time, 0-1 at half time: second half 2-0
    y = mk.market_outcomes([2, 1], [1, 1], [0, 0], [1, 0]).iloc[0]
    won = {("1X2", "1"), ("Doppia chance", "1X"), ("Doppia chance", "12"), ("Under/Over", "Over 2.5"),
           ("Goal/No goal", "Goal"), ("Multigoal", "Multigoal 2-4"), ("Combo 1X2 + GG/NG", "1 + Goal"),
           ("Pari/Dispari", "Dispari"), ("Risultato esatto", "2-1"), ("Handicap 1X2", "casa -1: X"),
           ("Primo tempo", "1X2 1T: 2"), ("Secondo tempo", "1X2 2T: 1"), ("Tempo/finale", "2/1"),
           ("Tempo con più gol", "2° tempo"), ("Gol nei tempi", "Gol in entrambi i tempi"),
           ("Draw no bet", "1 (rimborso se X)")}
    for key in won:
        assert y[key] == 1, key
    for key in [("1X2", "X"), ("Under/Over", "Under 2.5"), ("Goal/No goal", "No goal"), ("Tempo/finale", "1/1")]:
        assert y[key] == 0, key
    draw = mk.market_outcomes([1], [1], [0], [0]).iloc[0]   # 1-1: draw-no-bet refunded
    assert np.isnan(draw[("Draw no bet", "1 (rimborso se X)")])


def test_1x2_calibration_recovers_known_parameters():
    rng = np.random.default_rng(1)
    q = rng.dirichlet([4, 2.5, 3], 20000)                     # "market" probabilities
    true = {"a_X": 0.06, "a_2": -0.02, "b": 1.12}
    p1, p2 = mk.calibrate_1x2(q[:, 0], q[:, 2], {"calib_1x2": true})
    pt = np.stack([p1, 1 - p1 - p2, p2], 1)
    result = (rng.random(len(q))[:, None] > np.cumsum(pt, 1)).sum(1)   # outcomes drawn from the true model
    est = mk.fit_1x2_calibration(q[:, 0], q[:, 2], result)
    assert est["b"] == pytest.approx(1.12, abs=0.05) and est["a_X"] == pytest.approx(0.06, abs=0.04)
    same = mk.calibrate_1x2(q[:5, 0], q[:5, 2], {"calib_1x2": {"a_X": 0, "a_2": 0, "b": 1}})
    assert np.allclose(same[0], q[:5, 0]) and np.allclose(same[1], q[:5, 2])   # a = 0, b = 1: unchanged


def test_override_touches_only_1x2_markets():
    P = mk.market_probs([1.5], [1.1], -0.05, 0.44, 1.1)
    params = {"calib_1x2": {"a_X": 0.06, "a_2": 0.0, "b": 1.11}}
    O = mk.override_1x2(P, [0.45], [0.28], params).iloc[0]
    assert O[("1X2", "1")] + O[("1X2", "X")] + O[("1X2", "2")] == pytest.approx(1)
    assert O[("Doppia chance", "1X")] == pytest.approx(O[("1X2", "1")] + O[("1X2", "X")])
    assert O[("1X2", "1")] > 0.45                               # b > 1: the favourite gets a bit more
    assert O[("Goal/No goal", "Goal")] == P.iloc[0][("Goal/No goal", "Goal")]   # other markets untouched
    assert mk.override_1x2(P, [0.45], [0.28], {}).equals(P)    # no calibration: nothing changes


def test_upcoming_rates_skip_started_matches_and_stale_odds():
    fd = pd.DataFrame({"Div": ["I1", "I1", "B1"], "Date": ["24/10/2026", "23/10/2026", "24/10/2026"],
                       "Time": ["15:00", "10:00", "15:00"], "HomeTeam": ["Roma", "Lazio", "Genk"],
                       "AwayTeam": ["Inter", "Milan", "Gent"],
                       "AvgH": ["2.5"] * 3, "AvgD": ["3.3"] * 3, "AvgA": ["2.9"] * 3,
                       "Avg>2.5": ["1.9"] * 3, "Avg<2.5": ["1.95"] * 3})
    now = pd.Timestamp("2026-10-23 12:00", tz="UTC")
    r = mk.upcoming_rates(fd, now, now, PARAMS)
    assert r.home.tolist() == ["Roma"]                       # already started and other leagues are dropped
    fo = mk.fair_odds_for(r.iloc[0], PARAMS).set_index(["group", "market"])
    p1 = fo.loc[("1X2", "1"), "prob"]
    inv = np.array([1 / 2.5, 1 / 3.3, 1 / 2.9])
    assert p1 == pytest.approx(inv[0] / inv.sum(), abs=0.01)  # close to the margin-free market price
    assert mk.upcoming_rates(fd, now, now - pd.Timedelta(days=8), PARAMS).empty
