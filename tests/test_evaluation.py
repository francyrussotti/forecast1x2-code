import numpy as np
import pandas as pd

from forecast1x2.evaluation import (
    bootstrap_diff, brier, combine, devig_proportional, devig_shin, log_loss, rps, walk_forward,
)

P = np.array([[0.5, 0.3, 0.2]] * 3)
Y = np.array([0, 1, 2])  # home win, draw, away win


def test_log_loss_by_hand():
    assert np.allclose(log_loss(P, Y), [-np.log(0.5), -np.log(0.3), -np.log(0.2)])


def test_brier_by_hand():
    # home win: (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.38 ; draw: 0.25+0.49+0.04 ; away: 0.25+0.09+0.64
    assert np.allclose(brier(P, Y), [0.38, 0.78, 0.98])


def test_rps_by_hand():
    # cumulative forecast [0.5, 0.8]; observed home [1,1], draw [0,1], away [0,0]
    # home: ((-.5)^2 + (-.2)^2)/2 = 0.145 ; draw: (.5^2 + (-.2)^2)/2 = 0.145 ; away: (.25 + .64)/2 = 0.445
    assert np.allclose(rps(P, Y), [0.145, 0.145, 0.445])


def test_rps_rewards_nearby_outcome():
    # same probability on the observed home win; the rest closer to it (draw) is better for RPS
    assert rps(np.array([[0.6, 0.3, 0.1]]), np.array([0])) < rps(np.array([[0.6, 0.1, 0.3]]), np.array([0]))


def test_devig_proportional_by_hand():
    p = devig_proportional([[2.0, 3.5, 4.0]])
    q = np.array([0.5, 1 / 3.5, 0.25])
    assert np.allclose(p, q / q.sum())
    assert np.isclose(p.sum(), 1)


def test_shin_sums_to_one_and_favours_favourite():
    odds = np.array([[1.5, 4.2, 7.0], [2.6, 3.2, 2.9]])
    shin, prop = devig_shin(odds), devig_proportional(odds)
    assert np.allclose(shin.sum(axis=1), 1)
    # Shin attributes the margin mostly to longshots: favourite gets more than proportional
    assert shin[0, 0] > prop[0, 0] and shin[0, 2] < prop[0, 2]


def test_devig_without_margin_is_identity():
    odds = np.array([[2.0, 4.0, 4.0]])
    assert np.allclose(devig_shin(odds), [[0.5, 0.25, 0.25]])
    assert np.allclose(devig_proportional(odds), [[0.5, 0.25, 0.25]])


def test_combine_log_pool():
    pm, pk = np.array([[0.6, 0.25, 0.15]]), np.array([[0.4, 0.3, 0.3]])
    assert np.allclose(combine(pm, pk, 1.0), pm) and np.allclose(combine(pm, pk, 0.0), pk)
    mid = combine(pm, pk, 0.5)
    assert np.isclose(mid.sum(), 1) and pk[0, 0] < mid[0, 0] < pm[0, 0]
    expected = np.sqrt(pm * pk) / np.sqrt(pm * pk).sum()
    assert np.allclose(mid, expected)


def _two_seasons(seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(6)]
    rows, day = [], pd.Timestamp("2023-08-05")
    for season in ["2324", "2425"]:
        for r in range(10):
            for i in range(0, 6, 2):
                h, a = rng.permutation(teams)[:2]
                hg, ag = rng.poisson(1.5), rng.poisson(1.1)
                d = day + pd.Timedelta(days=int(rng.integers(0, 4)))  # several days within each week
                rows.append({"league": "I1", "tier": 1, "season": season, "date": d, "home": h, "away": a, "hg": hg,
                             "ag": ag, "result": "H" if hg > ag else "D" if hg == ag else "A",
                             "odds_h": 2.2, "odds_d": 3.3, "odds_a": 3.4})
            # second-division matches on the same days, including later in the same week
            for k in range(2):
                rows.append({"league": "I2", "tier": 2, "season": season,
                             "date": day + pd.Timedelta(days=int(rng.integers(0, 7))),
                             "home": f"B{k}", "away": f"B{k + 2}", "hg": rng.poisson(1.2), "ag": rng.poisson(1.0),
                             "result": "H", "odds_h": np.nan, "odds_d": np.nan, "odds_a": np.nan})
            day += pd.Timedelta(days=7)
        day += pd.Timedelta(days=60)
    df = pd.DataFrame(rows)
    df.loc[df.tier == 2, "result"] = np.where(df.hg > df.ag, "H", np.where(df.hg == df.ag, "D", "A"))[df.tier == 2]
    return df


def test_backtest_has_no_leakage():
    m = _two_seasons()
    out = walk_forward(m, "2425", xi=0.002, l2=1.0)
    assert len(out) == ((m.season == "2425") & (m.tier == 1)).sum()   # only top-flight matches are tested
    assert (out.train_max_date < out.date).all()
    assert (out.train_max_date < out.week_start).all()   # nothing from the same calendar week either
    for r in out.itertuples():
        assert r.n_train == (m.date < r.week_start).sum()  # both divisions, only matches before the week


def test_bootstrap_ci_contains_mean():
    m = _two_seasons()
    out = walk_forward(m, "2425", xi=0.002, l2=1.0)
    for k, o in enumerate("1X2"):
        out[f"market_{o}"] = devig_proportional(out[["odds_h", "odds_d", "odds_a"]].to_numpy())[:, k]
    b = bootstrap_diff(out, sources=("model",), n_boot=500)
    assert ((b.ci_low <= b.mean_diff) & (b.mean_diff <= b.ci_high)).all()
