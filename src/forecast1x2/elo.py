"""Elo ratings + ordered logit: a second, structurally different 1X2 model for the ensemble.

Ratings are updated match by match from results (with a goal-difference multiplier). Predictions for a
calendar week use the ratings at the start of that week, and the ordered logit mapping rating
difference -> P(1), P(X), P(2) is fitted only on earlier weeks. An OLS linear-probability model on the
same feature is kept as a comparison.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

BASE = 1500.0
HFA = 60.0          # home advantage inside the Elo expected score (rating points)
MIN_TRAIN = 200     # matches needed before the ordered logit is fitted


def week_start(dates: pd.Series) -> pd.Series:
    return dates.dt.normalize() - pd.to_timedelta(dates.dt.weekday, unit="D")


def mov_multiplier(gd: int) -> float:
    """World Football Elo goal-difference multiplier."""
    gd = abs(int(gd))
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8


def run_elo(top: pd.DataFrame, k: float, regress: float):
    """Sequential Elo over one league's top-flight matches.

    Returns (diff, ratings): diff[i] = home rating - away rating at the start of the match's calendar
    week; ratings = dict after the last match. At each new season ratings are pulled towards the mean
    by `regress`, and newcomers start from the mean rating of the teams that left the league.
    """
    top = top.sort_values("date")
    weeks = week_start(top.date)
    ratings, season, diff = {}, None, pd.Series(np.nan, index=top.index)
    for wk in sorted(weeks.unique()):
        g = top[weeks == wk]
        s = g.season.iloc[0]
        if s != season:
            teams = set(top.loc[top.season == s, "home"]) | set(top.loc[top.season == s, "away"])
            if ratings:
                mean = np.mean(list(ratings.values()))
                ratings = {t: mean + (r - mean) * (1 - regress) for t, r in ratings.items()}
                leavers = [r for t, r in ratings.items() if t not in teams]
                start = np.mean(leavers) if leavers else mean - 50
            else:
                start = BASE
            for t in teams:
                ratings.setdefault(t, start)
            season = s
        for i, r in g.iterrows():
            diff[i] = ratings[r.home] - ratings[r.away]
        for r in g.itertuples():
            exp_home = 1 / (1 + 10 ** (-(ratings[r.home] + HFA - ratings[r.away]) / 400))
            score = {"H": 1.0, "D": 0.5, "A": 0.0}[r.result]
            delta = k * mov_multiplier(r.hg - r.ag) * (score - exp_home)
            ratings[r.home] += delta
            ratings[r.away] -= delta
    return diff, ratings


# ------------------------------------------------------------- ordered logit

def ordlogit_probs(theta, d) -> np.ndarray:
    """theta = (slope, c1, log(c2 - c1)); z = slope * diff/100; columns P(1), P(X), P(2)."""
    b, c1, ld = theta
    z = b * np.asarray(d, dtype=float) / 100
    pa = expit(c1 - z)
    pad = expit(c1 + np.exp(ld) - z)
    P = np.column_stack([1 - pad, pad - pa, pa])
    return np.clip(P, 1e-12, 1)


def fit_ordlogit(d, y) -> np.ndarray:
    """Maximum likelihood; y in {0: home, 1: draw, 2: away}."""
    d, y = np.asarray(d, dtype=float), np.asarray(y)

    def nll(theta):
        return -np.log(ordlogit_probs(theta, d)[np.arange(len(y)), y]).mean()

    res = minimize(nll, np.array([0.5, -0.6, 0.0]), method="BFGS")
    return res.x


def fit_ols(d, y) -> np.ndarray:
    """Linear probability model: one-hot outcome on [1, diff/100], least squares."""
    X = np.column_stack([np.ones(len(d)), np.asarray(d, dtype=float) / 100])
    beta, *_ = np.linalg.lstsq(X, np.eye(3)[np.asarray(y)], rcond=None)
    return beta


def ols_probs(beta, d) -> np.ndarray:
    X = np.column_stack([np.ones(len(d)), np.asarray(d, dtype=float) / 100])
    P = np.clip(X @ beta, 0.01, None)   # linear model can go below 0: clip and renormalise
    return P / P.sum(axis=1, keepdims=True)


RESULT_INDEX = {"H": 0, "D": 1, "A": 2}


def elo_walk_forward(top: pd.DataFrame, season: str, k: float, regress: float) -> pd.DataFrame:
    """Weekly walk-forward predictions (Elo + ordered logit, and OLS) for the matches of `season`."""
    top = top.sort_values("date").copy()
    top["diff"], _ = run_elo(top, k, regress)
    top["week_start"] = week_start(top.date)
    top["y"] = top.result.map(RESULT_INDEX)
    test = top[top.season == season]
    rows = []
    for wk, g in test.groupby("week_start"):
        train = top[top.week_start < wk]
        if len(train) < MIN_TRAIN:
            raise ValueError(f"troppe poche partite prima di {wk.date()} per stimare il modello Elo")
        theta, beta = fit_ordlogit(train["diff"], train.y), fit_ols(train["diff"], train.y)
        pe, po = ordlogit_probs(theta, g["diff"]), ols_probs(beta, g["diff"])
        for j, r in enumerate(g.itertuples()):
            rows.append({"league": r.league, "date": r.date, "home": r.home, "away": r.away,
                         "elo_1": pe[j, 0], "elo_X": pe[j, 1], "elo_2": pe[j, 2],
                         "ols_1": po[j, 0], "ols_X": po[j, 1], "ols_2": po[j, 2],
                         "elo_train_max_week": train.week_start.max()})
    return pd.DataFrame(rows)


def elo_current(top: pd.DataFrame, k: float, regress: float):
    """Ratings after all played matches and the ordered logit fitted on all of them (for predictions)."""
    top = top.sort_values("date").copy()
    diff, ratings = run_elo(top, k, regress)
    theta = fit_ordlogit(diff, top.result.map(RESULT_INDEX))
    return ratings, theta
