import numpy as np
import pandas as pd

from forecast1x2.elo import (BASE, HFA, elo_walk_forward, fit_ols, fit_ordlogit, mov_multiplier, ols_probs,
                             ordlogit_probs, run_elo)
from forecast1x2.evaluation import reliability_table


def _m(rows):
    df = pd.DataFrame(rows, columns=["season", "date", "home", "away", "hg", "ag"])
    df["date"] = pd.to_datetime(df.date)
    df["league"] = "I1"
    df["result"] = np.where(df.hg > df.ag, "H", np.where(df.hg == df.ag, "D", "A"))
    return df


def test_mov_multiplier():
    assert [mov_multiplier(g) for g in [0, 1, -1, 2, 3, 5]] == [1.0, 1.0, 1.0, 1.5, 14 / 8, 16 / 8]


def test_elo_update_by_hand():
    df = _m([("2425", "2024-08-17", "A", "B", 2, 0), ("2425", "2024-08-24", "B", "A", 1, 1)])
    diff, r = run_elo(df, k=20, regress=0.0)
    assert diff.tolist()[0] == 0.0                       # both start at BASE
    e = 1 / (1 + 10 ** (-HFA / 400))
    d1 = 20 * 1.5 * (1 - e)                              # home win by 2 goals
    assert np.isclose(diff.tolist()[1], (BASE - d1) - (BASE + d1))   # second week uses updated ratings
    e2 = 1 / (1 + 10 ** (-((BASE - d1) + HFA - (BASE + d1)) / 400))
    d2 = 20 * 1.0 * (0.5 - e2)
    assert np.isclose(r["B"], BASE - d1 + d2) and np.isclose(r["A"], BASE + d1 - d2)


def test_newcomer_starts_from_leavers_mean():
    df = _m([("2324", "2024-01-06", "A", "B", 3, 0), ("2324", "2024-01-13", "C", "A", 0, 0),
             ("2425", "2024-08-17", "A", "N", 1, 0)])    # B and C leave, N arrives
    _, r_before = run_elo(df[df.season == "2324"], 20, 0.0)
    diff, _ = run_elo(df, 20, 0.0)
    expected_n = np.mean([r_before["B"], r_before["C"]])
    assert np.isclose(diff.tolist()[2], r_before["A"] - expected_n)


def test_ordlogit_and_ols_are_valid_probabilities():
    rng = np.random.default_rng(0)
    d = rng.normal(0, 150, 2000)
    y = np.where(rng.random(2000) < 1 / (1 + np.exp(-(d / 100 + 0.3))), 0, np.where(rng.random(2000) < 0.5, 1, 2))
    for P in (ordlogit_probs(fit_ordlogit(d, y), d), ols_probs(fit_ols(d, y), d)):
        assert np.allclose(P.sum(axis=1), 1) and (P > 0).all()
    P = ordlogit_probs(fit_ordlogit(d, y), np.array([-300.0, 0.0, 300.0]))
    assert P[0, 0] < P[1, 0] < P[2, 0] and P[0, 2] > P[2, 2]   # stronger home team -> more home wins


def test_elo_walk_forward_has_no_leakage():
    rng = np.random.default_rng(1)
    teams = list("ABCDEF")
    rows, day = [], pd.Timestamp("2023-08-05")
    for season in ["2324", "2425"]:
        for r in range(80):   # > MIN_TRAIN (200) matches before the test season
            for i in range(3):
                h, a = rng.permutation(teams)[:2]
                rows.append((season, day + pd.Timedelta(days=int(rng.integers(0, 3))), h, a,
                             rng.poisson(1.5), rng.poisson(1.1)))
            day += pd.Timedelta(days=7)
    df = _m(rows)
    out = elo_walk_forward(df, "2425", k=20, regress=0.2)
    wk = out.date.dt.normalize() - pd.to_timedelta(out.date.dt.weekday, unit="D")
    assert len(out) == (df.season == "2425").sum()
    assert (out.elo_train_max_week < wk).all()


def test_reliability_by_hand():
    df = pd.DataFrame({"result": ["H", "A", "D", "H"],
                       "model_1": [0.6, 0.5, 0.3, 0.2], "model_X": [0.2, 0.2, 0.4, 0.3], "model_2": [0.2, 0.3, 0.3, 0.5]})
    for s in ["baseline", "market"]:
        df[f"{s}_1"], df[f"{s}_X"], df[f"{s}_2"] = (0.45, 0.27, 0.28) if s == "baseline" else (0.5, 0.25, 0.25)
    r = reliability_table(df, ["model"]).iloc[0]
    assert r.accuracy_pct == 50.0                       # favourite right in matches 1 and 3
    ll = lambda p: -np.mean(np.log(p))
    lm, lb, lk = ll([0.6, 0.3, 0.4, 0.2]), ll([0.45, 0.28, 0.27, 0.45]), ll([0.5, 0.25, 0.25, 0.5])
    assert np.isclose(r.market_share_pct, 100 * (lb - lm) / (lb - lk))
    assert np.isclose(r.skill_vs_baseline_pct, 100 * (1 - lm / lb))
