"""Dixon-Coles model with time decay, fitted by weighted maximum likelihood."""
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from .data import (CURRENT_SEASON, LEAGUES, PROCESSED, ROOT, DataError, league_history, load_fixtures,
                   load_matches, load_upcoming_odds, promoted_teams, upcoming_round, usable_odds,
                   write_atomic, write_csv_atomic)

OUTPUTS = ROOT / "outputs"
MAX_GOALS = 10
L2 = 1.0          # default ridge on attack/defence (calibrated per league in hyperparams.json)
RHO_BOUNDS = (-0.2, 0.2)
MIN_WEIGHT = 1e-3
TAU_FLOOR = 1e-2


@dataclass
class Params:
    teams: list
    attack: np.ndarray
    defence: np.ndarray   # higher = concedes more
    mu: float
    home: float
    rho: float
    delta: float = 0.0    # log-rate offset of second-division matches (not used for top-flight predictions)
    kappa: float = 0.0    # strength gap of a team in its first top-flight season after promotion

    def rates(self, home: str, away: str, home_promoted: bool = False, away_promoted: bool = False):
        idx = {t: i for i, t in enumerate(self.teams)}
        a = lambda t: self.attack[idx[t]] if t in idx else 0.0   # unseen team: league average
        d = lambda t: self.defence[idx[t]] if t in idx else 0.0
        k = self.kappa * (int(away_promoted) - int(home_promoted))
        lam = np.exp(self.mu + self.home + a(home) + d(away) + k)
        mu = np.exp(self.mu + a(away) + d(home) - k)
        return lam, mu


def tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction (vectorised)."""
    t = np.ones(np.broadcast(x, y, lam, mu).shape)
    x, y, lam, mu = (np.broadcast_to(v, t.shape) for v in (x, y, lam, mu))
    m00, m01, m10, m11 = (x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)
    t[m00] = 1 - lam[m00] * mu[m00] * rho
    t[m01] = 1 + lam[m01] * rho
    t[m10] = 1 + mu[m10] * rho
    t[m11] = 1 - rho
    return t


N_GLOBAL = 5   # mu, home, rho, delta, kappa


def _unpack(theta, n):
    mu, home, rho, delta, kappa = theta[:N_GLOBAL]
    a = np.append(theta[N_GLOBAL:N_GLOBAL + n - 1], 0.0)
    d = np.append(theta[N_GLOBAL + n - 1:], 0.0)
    a[-1], d[-1] = -a[:-1].sum(), -d[:-1].sum()   # identifiability: sum(attack) = sum(defence) = 0
    return mu, home, rho, delta, kappa, a, d


def neg_loglik(theta, hi, ai, x, y, w, n, l2=L2, tier2=None, promo=None, xe=None, ye=None):
    """Weighted negative log-likelihood + L2 penalty, divided by total weight; returns (value, gradient).

    x, y are the actual goals (used by the tau correction); xe, ye are the "effective goals" of the
    Poisson part, a blend of goals and xG (default: the goals themselves).

    tier2 marks second-division matches, whose goal rates get the extra offset delta.
    promo = away_promoted - home_promoted (in {-1, 0, 1}); a promoted team scores exp(-kappa)
    times less and concedes exp(kappa) times more than its rating implies.
    """
    mu, home, rho, delta, kappa, a, d = _unpack(theta, n)
    xe = x if xe is None else xe
    ye = y if ye is None else ye
    t2 = np.zeros(len(x)) if tier2 is None else tier2.astype(float)
    pr = np.zeros(len(x)) if promo is None else promo.astype(float)
    eta_h = mu + delta * t2 + home + a[hi] + d[ai] + kappa * pr
    eta_a = mu + delta * t2 + a[ai] + d[hi] - kappa * pr
    lam, m = np.exp(eta_h), np.exp(eta_a)

    t = np.ones_like(lam)
    dt_dlam, dt_dm, dt_drho = np.zeros_like(lam), np.zeros_like(lam), np.zeros_like(lam)
    m00, m01, m10, m11 = (x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)
    t[m00] = 1 - lam[m00] * m[m00] * rho
    dt_dlam[m00], dt_dm[m00], dt_drho[m00] = -m[m00] * rho, -lam[m00] * rho, -lam[m00] * m[m00]
    t[m01] = 1 + lam[m01] * rho
    dt_dlam[m01], dt_drho[m01] = rho, lam[m01]
    t[m10] = 1 + m[m10] * rho
    dt_dm[m10], dt_drho[m10] = rho, m[m10]
    t[m11] = 1 - rho
    dt_drho[m11] = -1.0
    # log(tau) continues linearly below TAU_FLOOR: tau can reach 0 or below for lopsided matches with
    # a low score (e.g. lam*mu*rho > 1); value and gradient stay finite and consistent
    low = t < TAU_FLOOR
    log_t = np.where(low, np.log(TAU_FLOOR) + (t - TAU_FLOOR) / TAU_FLOOR, np.log(np.where(low, 1.0, t)))
    dlog_t = np.where(low, 1 / TAU_FLOOR, 1 / np.where(low, 1.0, t))

    W = w.sum()
    ll = xe * eta_h - lam + ye * eta_a - m + log_t
    pen = l2 * (a @ a + d @ d)
    value = (-(w @ ll) + pen) / W

    gh = w * (xe - lam + lam * dt_dlam * dlog_t)    # d ll / d eta_h
    ga = w * (ye - m + m * dt_dm * dlog_t)          # d ll / d eta_a
    g_a = -(np.bincount(hi, gh, n) + np.bincount(ai, ga, n)) + 2 * l2 * a
    g_d = -(np.bincount(ai, gh, n) + np.bincount(hi, ga, n)) + 2 * l2 * d
    grad = np.concatenate([
        [-(gh.sum() + ga.sum()), -gh.sum(), -(w * dt_drho * dlog_t).sum(), -((gh + ga) * t2).sum(), -((gh - ga) * pr).sum()],
        g_a[:-1] - g_a[-1],
        g_d[:-1] - g_d[-1],
    ]) / W
    return value, grad


def fit(train: pd.DataFrame, ref_date, xi: float, l2: float = L2, init: Params = None,
        xg_weight: float = 0.0) -> Params:
    """Fit on matches in `train` (all must be before ref_date) with weights exp(-xi * days).

    `train` may include second-division matches (column tier == 2): teams moving between
    divisions link the two scales, so promoted teams start from a data-based rating.
    With xg_weight > 0 and columns hxg/axg, team strengths are fitted on
    (1 - xg_weight) * goals + xg_weight * xG; matches without xG keep their goals.
    """
    days = (pd.Timestamp(ref_date) - train.date).dt.days.to_numpy()
    if (days < 0).any():
        raise ValueError("partite di training con data successiva al riferimento")
    w = np.exp(-xi * days)
    keep = w >= MIN_WEIGHT   # matches weighing less than 1/1000 change nothing but cost time
    train, w = train[keep], w[keep]
    teams = sorted(set(train.home) | set(train.away))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    hi = train.home.map(idx).to_numpy()
    ai = train.away.map(idx).to_numpy()

    tier2 = (train.tier == 2).to_numpy() if "tier" in train else np.zeros(len(train), bool)
    promo = (train.away_promoted.astype(int) - train.home_promoted.astype(int)).to_numpy() \
        if "home_promoted" in train else None

    theta0 = np.zeros(N_GLOBAL + 2 * (n - 1))
    theta0[:3] = np.log(max(train.ag.mean(), 0.1)), 0.2, -0.05
    if init is not None:   # warm start with previous fit, mapping teams by name
        theta0[:N_GLOBAL] = init.mu, init.home, init.rho, init.delta, init.kappa
        prev = dict(zip(init.teams, zip(init.attack, init.defence)))
        for t, i in idx.items():
            if t in prev and i < n - 1:
                theta0[N_GLOBAL + i], theta0[N_GLOBAL + n - 1 + i] = prev[t]

    x, y = train.hg.to_numpy(dtype=float), train.ag.to_numpy(dtype=float)
    xe, ye = x, y
    if xg_weight > 0 and "hxg" in train:
        hxg, axg = train.hxg.to_numpy(dtype=float), train.axg.to_numpy(dtype=float)
        has = ~(np.isnan(hxg) | np.isnan(axg))
        xe = np.where(has, (1 - xg_weight) * x + xg_weight * np.nan_to_num(hxg), x)
        ye = np.where(has, (1 - xg_weight) * y + xg_weight * np.nan_to_num(axg), y)

    bounds = [(None, None), (None, None), RHO_BOUNDS, (None, None), (None, None)] + [(None, None)] * (2 * (n - 1))
    res = minimize(neg_loglik, theta0,
                   args=(hi, ai, x, y, w, n, l2, tier2, promo, xe, ye),
                   jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8})
    if not res.success:
        raise RuntimeError(f"ottimizzazione non convergente: {res.message}")
    mu, home, rho, delta, kappa, a, d = _unpack(res.x, n)
    return Params(teams, a, d, mu, home, rho, delta, kappa)


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """P(home=i, away=j) for i, j in 0..max_goals, renormalised after truncation."""
    g = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    M[:2, :2] *= tau(g[:2, None], g[None, :2], lam, mu, rho)
    M = np.clip(M, 0, None)
    return M / M.sum()


def outcome_probs(M: np.ndarray):
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()


def predict_match(params: Params, home: str, away: str, home_promoted=False, away_promoted=False) -> dict:
    lam, mu = params.rates(home, away, home_promoted, away_promoted)
    M = score_matrix(lam, mu, params.rho)
    p1, px, p2 = outcome_probs(M)
    g = np.arange(M.shape[0])
    i, j = np.unravel_index(M.argmax(), M.shape)
    return {"p1": p1, "px": px, "p2": p2,
            "xg_home": M.sum(axis=1) @ g, "xg_away": M.sum(axis=0) @ g,
            "top_score": f"{i}-{j}", "top_score_prob": M[i, j]}


# ------------------------------------------------------------------- predict

def load_hyperparams() -> dict:
    from .evaluation import HYPER_PATH, calibrate
    if not HYPER_PATH.exists() or "ensemble_a" not in json.loads(HYPER_PATH.read_text()):
        print("iperparametri mancanti o di una versione precedente: lancio la calibrazione (richiede vari minuti)...")
        calibrate(load_matches())
    return json.loads(HYPER_PATH.read_text())


def predict_all() -> None:
    from .elo import elo_current, ordlogit_probs
    from .evaluation import combine, devig_proportional, risk_level

    matches = load_matches()
    fixtures = load_fixtures()
    hyper = load_hyperparams()
    now = pd.Timestamp.now(tz="UTC")
    odds = usable_odds(load_upcoming_odds(), fixtures, now).set_index(["league", "home", "away"])
    today = now.tz_localize(None).normalize()
    # without Sportmonks (no token, as on GitHub) there are no xG: goals only, and meta.json says so
    has_xg = "hxg" in matches and matches.hxg.notna().any()
    xg_weight = hyper.get("xg_weight", 0.0) if has_xg else 0.0
    if not has_xg:
        print("xG non disponibili: previsioni sui soli gol (ensemble senza xG)")
    OUTPUTS.mkdir(exist_ok=True)

    rows, ratings, model_meta = [], [], {}
    for div, (name, *_) in LEAGUES.items():
        h = league_history(matches, div)
        train = h[h.date < today]
        params = fit(train, today, hyper["xi"], l2=hyper["l2"], xg_weight=xg_weight)
        played = h[(h.season == CURRENT_SEASON) & (h.tier == 1)]
        promoted = promoted_teams(h, CURRENT_SEASON)
        up = upcoming_round(fixtures[fixtures.league == div], played, now)
        top = h[(h.tier == 1) & (h.date < today)]
        elo_ratings, theta = elo_current(top, hyper["elo_k"], hyper["elo_regress"])
        unknown = sorted((set(up.home) | set(up.away)) - (set(params.teams) & set(elo_ratings)))
        if unknown:   # usually a name-mapping slip: never forecast an unknown team as "average"
            raise DataError(f"{name}: squadre senza dati storici nel modello: {unknown}")
        for r in up.itertuples():
            pred = predict_match(params, r.home, r.away, r.home in promoted, r.away in promoted)
            dc = np.array([pred["p1"], pred["px"], pred["p2"]])
            elo = ordlogit_probs(theta, [elo_ratings[r.home] - elo_ratings[r.away]])[0]
            p3 = hyper["ensemble_a"] * dc + (1 - hyper["ensemble_a"]) * elo   # final forecast
            pred.update({"p1": p3[0], "px": p3[1], "p2": p3[2]})
            row = {"league": div, "round": r.round, "kickoff_utc": r.kickoff_utc, "home": r.home,
                   "away": r.away, "status": r.status, "fixture_source": r.source, **pred,
                   "dc_1": dc[0], "dc_X": dc[1], "dc_2": dc[2], "elo_1": elo[0], "elo_X": elo[1], "elo_2": elo[2],
                   "fav": "1X2"[p3.argmax()], "risk_level": int(risk_level(p3.max()))}
            key = (div, r.home, r.away)
            if key in odds.index and odds.loc[key, ["open_h", "open_d", "open_a"]].notna().all():
                o = odds.loc[key, ["open_h", "open_d", "open_a"]].to_numpy(dtype=float)[None, :]
                mk = devig_proportional(o)
                cb = combine(np.array([[pred["p1"], pred["px"], pred["p2"]]]), mk, hyper["w"])[0]
                row.update({"open_1": o[0, 0], "open_X": o[0, 1], "open_2": o[0, 2],
                            "market_1": mk[0, 0], "market_X": mk[0, 1], "market_2": mk[0, 2],
                            "comb_1": cb[0], "comb_X": cb[1], "comb_2": cb[2]})
            rows.append(row)
        top_now = set(played.home) | set(played.away)
        for t, a, d in zip(params.teams, params.attack, params.defence):
            if t in top_now:
                ratings.append({"league": div, "team": t, "attack": a, "defence": d, "promoted": t in promoted})
        model_meta[div] = {"home": params.home, "rho": params.rho, "mu": params.mu, "delta": params.delta,
                           "kappa": params.kappa, "train_matches": len(train),
                           "last_train_date": str(train.date.max().date())}
        with_odds = sum("comb_1" in r_ for r_ in rows if r_["league"] == div)
        print(f"{name}: {len(up)} partite in previsione (giornata {up['round'].max() if len(up) else '-'}), "
              f"{with_odds} con quote pre-partita")

    cols = ["league", "round", "kickoff_utc", "home", "away", "status", "fixture_source", "p1", "px", "p2",
            "xg_home", "xg_away", "top_score", "top_score_prob", "fav", "risk_level",
            "dc_1", "dc_X", "dc_2", "elo_1", "elo_X", "elo_2",
            "open_1", "open_X", "open_2", "market_1", "market_X", "market_2", "comb_1", "comb_X", "comb_2"]
    write_csv_atomic(pd.DataFrame(rows).reindex(columns=cols), OUTPUTS / "predictions.csv")
    write_csv_atomic(pd.DataFrame(ratings), OUTPUTS / "ratings.csv")
    update_meta = json.loads((PROCESSED / "update_meta.json").read_text())
    meta = {**update_meta, "predictions_generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "xi": hyper["xi"], "l2": hyper["l2"], "w": hyper["w"], "xg_weight": xg_weight,
            "ensemble_a": hyper["ensemble_a"], "elo_k": hyper["elo_k"], "elo_regress": hyper["elo_regress"],
            "model": model_meta}
    write_atomic(OUTPUTS / "meta.json", json.dumps(meta, indent=2))
