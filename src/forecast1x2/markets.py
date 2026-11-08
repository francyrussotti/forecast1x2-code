"""Quote giuste per i mercati sul risultato (regolamento scommesse calcio), ricavate dalle quote di mercato.

Idea: le quote 1X2 e Over/Under 2.5 sono i mercati più liquidi e con meno margine. Dalle loro probabilità
senza margine si ricavano i gol attesi delle due squadre (Dixon-Coles con correzione dei punteggi bassi ρ),
e da questi la probabilità di ogni altro mercato: Doppia chance, Multigoal, Combo, GG/NG, primo tempo, ecc.
Il primo tempo usa una quota fissa dei gol attesi (FIRST_HALF_SHARE), con gol indipendenti tra i due tempi.

ρ e la quota del primo tempo si stimano sulle stagioni 2015/16-2023/24; 2024/25-2025/26 servono da verifica
(`python -m forecast1x2 quote-giuste-verifica`). Ogni mercato è una funzione dei gol, usata sia per la
probabilità sia per l'esito reale, così la verifica controlla esattamente quello che si mostra.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import poisson

from .data import (LEAGUES, ODDS_MAX_AGE, RAW, ROOT, SEASONS, SECOND_DIV, fetched_at_of, parse_dates,
                   parse_fd_fixtures, parse_results_csv, results_path, write_atomic, write_csv_atomic)

G = 11          # final score grid 0..10
GH = 8          # half grid 0..7
DIVS = list(LEAGUES) + list(SECOND_DIV.values())
CAL_SEASONS = [s for s in SEASONS if s <= "2324"]
TEST_SEASONS = ["2425", "2526"]
RHO_GRID = [-0.15, -0.1, -0.075, -0.05, -0.025, 0.0, 0.025]
NU_GRID = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
PARAMS_PATH = ROOT / "outputs" / "markets_params.json"
CALIBRATION_OUT = ROOT / "outputs" / "markets_calibration.csv"
FAIR_ODDS_OUT = ROOT / "outputs" / "fair_odds.csv"
DEFAULT_PARAMS = {"rho": -0.05, "nu": 1.0, "first_half_share": 0.44}


# ------------------------------------------------------------------ markets

def _sign(h, a):
    return np.where(h > a, "1", np.where(h == a, "X", "2"))


def _mg(lo, hi):
    return lambda h, a: (h + a >= lo) & (h + a <= hi)


def _build_markets():
    """(group, market, scope, predicate). scope: ft (final score), 1t, 2t, joint (h1, a1, h2, a2)."""
    m = []
    add = lambda g, name, scope, f: m.append((g, name, scope, f))
    for o in ["1", "X", "2"]:
        add("1X2", o, "ft", lambda h, a, o=o: _sign(h, a) == o)
    add("Doppia chance", "1X", "ft", lambda h, a: h >= a)
    add("Doppia chance", "X2", "ft", lambda h, a: h <= a)
    add("Doppia chance", "12", "ft", lambda h, a: h != a)
    for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        add("Under/Over", f"Under {line}", "ft", lambda h, a, l=line: h + a < l)
        add("Under/Over", f"Over {line}", "ft", lambda h, a, l=line: h + a > l)
    add("Goal/No goal", "Goal", "ft", lambda h, a: (h > 0) & (a > 0))
    add("Goal/No goal", "No goal", "ft", lambda h, a: (h == 0) | (a == 0))
    for lo, hi in [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6)]:
        add("Multigoal", f"Multigoal {lo}-{hi}", "ft", _mg(lo, hi))
    add("Multigoal", "Multigoal 7+", "ft", lambda h, a: h + a >= 7)
    for side, goals in [("casa", lambda h, a: h), ("ospite", lambda h, a: a)]:
        for lo, hi in [(1, 2), (1, 3), (2, 3)]:
            add("Multigoal squadra", f"Multigoal {side} {lo}-{hi}", "ft",
                lambda h, a, g=goals, lo=lo, hi=hi: (g(h, a) >= lo) & (g(h, a) <= hi))
        for line in [0.5, 1.5, 2.5]:
            add("Under/Over squadra", f"Over {line} {side}", "ft", lambda h, a, g=goals, l=line: g(h, a) > l)
            add("Under/Over squadra", f"Under {line} {side}", "ft", lambda h, a, g=goals, l=line: g(h, a) < l)
    for o in ["1", "X", "2"]:
        for line in [1.5, 2.5, 3.5]:
            add("Combo 1X2 + U/O", f"{o} + Under {line}", "ft", lambda h, a, o=o, l=line: (_sign(h, a) == o) & (h + a < l))
            add("Combo 1X2 + U/O", f"{o} + Over {line}", "ft", lambda h, a, o=o, l=line: (_sign(h, a) == o) & (h + a > l))
        add("Combo 1X2 + GG/NG", f"{o} + Goal", "ft", lambda h, a, o=o: (_sign(h, a) == o) & (h > 0) & (a > 0))
        add("Combo 1X2 + GG/NG", f"{o} + No goal", "ft", lambda h, a, o=o: (_sign(h, a) == o) & ((h == 0) | (a == 0)))
    dc = {"1X": lambda h, a: h >= a, "X2": lambda h, a: h <= a, "12": lambda h, a: h != a}
    for name, f in dc.items():
        for line in [1.5, 2.5, 3.5]:
            add("Combo DC + U/O", f"{name} + Under {line}", "ft", lambda h, a, f=f, l=line: f(h, a) & (h + a < l))
            add("Combo DC + U/O", f"{name} + Over {line}", "ft", lambda h, a, f=f, l=line: f(h, a) & (h + a > l))
        add("Combo DC + GG/NG", f"{name} + Goal", "ft", lambda h, a, f=f: f(h, a) & (h > 0) & (a > 0))
        add("Combo DC + GG/NG", f"{name} + No goal", "ft", lambda h, a, f=f: f(h, a) & ((h == 0) | (a == 0)))
        for lo, hi in [(1, 3), (2, 4), (2, 5)]:
            add("Combo DC + Multigoal", f"{name} + Multigoal {lo}-{hi}", "ft",
                lambda h, a, f=f, lo=lo, hi=hi: f(h, a) & (h + a >= lo) & (h + a <= hi))
    for o in ["1", "2"]:
        for lo, hi in [(1, 3), (2, 4), (2, 5)]:
            add("Combo 1X2 + Multigoal", f"{o} + Multigoal {lo}-{hi}", "ft",
                lambda h, a, o=o, lo=lo, hi=hi: (_sign(h, a) == o) & (h + a >= lo) & (h + a <= hi))
    for gg in ["Goal", "No goal"]:
        for ou in ["Under", "Over"]:
            add("Combo GG/NG + U/O", f"{gg} + {ou} 2.5", "ft",
                lambda h, a, gg=gg, ou=ou: (((h > 0) & (a > 0)) == (gg == "Goal")) & ((h + a > 2.5) == (ou == "Over")))
    add("Pari/Dispari", "Pari", "ft", lambda h, a: (h + a) % 2 == 0)
    add("Pari/Dispari", "Dispari", "ft", lambda h, a: (h + a) % 2 == 1)
    for hc, lab in [(-1, "casa -1"), (1, "casa +1"), (-2, "casa -2")]:
        for o in ["1", "X", "2"]:
            add("Handicap 1X2", f"{lab}: {o}", "ft", lambda h, a, hc=hc, o=o: _sign(h + hc, a) == o)
    for o in ["1", "2"]:
        add("Draw no bet", f"{o} (rimborso se X)", "dnb", o)
    for i in range(4):
        for j in range(4):
            add("Risultato esatto", f"{i}-{j}", "ft", lambda h, a, i=i, j=j: (h == i) & (a == j))
    add("Risultato esatto", "Altro", "ft", lambda h, a: (h > 3) | (a > 3))
    for o in ["1", "X", "2"]:
        add("Primo tempo", f"1X2 1T: {o}", "1t", lambda h, a, o=o: _sign(h, a) == o)
    for line in [0.5, 1.5, 2.5]:
        add("Primo tempo", f"Over {line} 1T", "1t", lambda h, a, l=line: h + a > l)
        add("Primo tempo", f"Under {line} 1T", "1t", lambda h, a, l=line: h + a < l)
    add("Primo tempo", "Goal 1T", "1t", lambda h, a: (h > 0) & (a > 0))
    for o in ["1", "X", "2"]:
        add("Secondo tempo", f"1X2 2T: {o}", "2t", lambda h, a, o=o: _sign(h, a) == o)
    for line in [0.5, 1.5, 2.5]:
        add("Secondo tempo", f"Over {line} 2T", "2t", lambda h, a, l=line: h + a > l)
        add("Secondo tempo", f"Under {line} 2T", "2t", lambda h, a, l=line: h + a < l)
    for o1 in ["1", "X", "2"]:
        for o2 in ["1", "X", "2"]:
            add("Tempo/finale", f"{o1}/{o2}", "joint",
                lambda h1, a1, h2, a2, o1=o1, o2=o2: (_sign(h1, a1) == o1) & (_sign(h1 + h2, a1 + a2) == o2))
    add("Tempo con più gol", "1° tempo", "joint", lambda h1, a1, h2, a2: h1 + a1 > h2 + a2)
    add("Tempo con più gol", "2° tempo", "joint", lambda h1, a1, h2, a2: h1 + a1 < h2 + a2)
    add("Tempo con più gol", "Uguale", "joint", lambda h1, a1, h2, a2: h1 + a1 == h2 + a2)
    add("Gol nei tempi", "Gol in entrambi i tempi", "joint", lambda h1, a1, h2, a2: (h1 + a1 > 0) & (h2 + a2 > 0))
    return m


MARKETS = _build_markets()


# ------------------------------------------------------------- distributions

def goal_pmf(lam, n=G, nu=1.0):
    """Goals of one team: P(k) ∝ λ^k / (k!)^ν (Conway-Maxwell-Poisson). ν = 1 is Poisson;
    ν > 1 is less dispersed (fewer 0s and fewer big scores), as real football goals are."""
    lam = np.atleast_1d(lam).astype(float)
    k = np.arange(n)
    if nu == 1.0:
        return poisson.pmf(k[None, :], lam[:, None])
    logp = k[None, :] * np.log(lam[:, None]) - nu * gammaln(k + 1)[None, :]
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def score_matrices(lh, la, rho, n=G, nu=1.0, kappa=0.0):
    """Dixon-Coles score matrices (N, n, n): home goals on axis 1, away on axis 2.

    kappa: dependence between the two teams' goals; every score where both teams scored is multiplied by
    exp(kappa) before normalising (kappa > 0: «both teams score» more often than with independent goals)."""
    lh, la = np.atleast_1d(lh).astype(float), np.atleast_1d(la).astype(float)
    M = goal_pmf(lh, n, nu)[:, :, None] * goal_pmf(la, n, nu)[:, None, :]
    M[:, 0, 0] *= 1 - lh * la * rho
    M[:, 0, 1] *= 1 + lh * rho
    M[:, 1, 0] *= 1 + la * rho
    M[:, 1, 1] *= 1 - rho
    if kappa:
        M[:, 1:, 1:] *= np.exp(kappa)
    M = np.clip(M, 0, None)
    return M / M.sum(axis=(1, 2), keepdims=True)


_I, _J = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")


def _core_probs(M):
    """P(1), P(2), P(Over 2.5) for each matrix."""
    p1 = (M * (_I > _J)).sum(axis=(1, 2))
    p2 = (M * (_I < _J)).sum(axis=(1, 2))
    po = (M * (_I + _J > 2.5)).sum(axis=(1, 2))
    return np.stack([p1, p2, po], axis=1)


def fit_rates(p1, p2, p_over, rho, nu=1.0, iters: int = 25, kappa=0.0):
    """Rates (home, away) whose Dixon-Coles probabilities match P(1), P(2), P(Over 2.5).

    Vectorised damped Gauss-Newton on log-rates (2 unknowns, 3 targets: least squares).
    """
    target = np.stack([p1, p2, p_over], axis=1).astype(float)
    n = len(target)
    x = np.log(np.full((n, 2), [1.4, 1.1]))
    eps = 1e-4
    sm = lambda a, b: score_matrices(np.exp(a), np.exp(b), rho, nu=nu, kappa=kappa)
    for _ in range(iters):
        f = _core_probs(sm(x[:, 0], x[:, 1]))
        r = f - target
        J = np.empty((n, 3, 2))
        for k in range(2):
            xe = x.copy()
            xe[:, k] += eps
            J[:, :, k] = (_core_probs(sm(xe[:, 0], xe[:, 1])) - f) / eps
        A = np.einsum("nik,nil->nkl", J, J) + 1e-6 * np.eye(2)
        b = -np.einsum("nik,ni->nk", J, r)
        step = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        x += np.clip(step, -0.5, 0.5)
        if np.abs(step).max() < 1e-7:
            break
    resid = np.abs(_core_probs(sm(x[:, 0], x[:, 1])) - target).max(axis=1)
    return np.exp(x[:, 0]), np.exp(x[:, 1]), resid


def expected_goals(lh, la, rho, nu=1.0, kappa=0.0):
    """Mean goals (home, away) of the final-score matrix."""
    M = score_matrices(lh, la, rho, nu=nu, kappa=kappa)
    return (M * _I).sum(axis=(1, 2)), (M * _J).sum(axis=(1, 2))


_LAM_TABLE = np.exp(np.linspace(np.log(0.01), np.log(8.0), 400))


def rate_for_mean(mean, n=GH, nu=1.0):
    """Rate λ whose (truncated) Conway-Maxwell-Poisson distribution has the given mean (by interpolation)."""
    k = np.arange(n)
    means = (goal_pmf(_LAM_TABLE, n, nu) * k).sum(axis=1)
    return np.exp(np.interp(np.asarray(mean, dtype=float), means, np.log(_LAM_TABLE)))


def market_probs(lh, la, rho, share, nu=1.0, kappa=0.0):
    """Probability of every market for each match: DataFrame (N rows x markets).

    Halves: same dispersion ν, with share / (1 - share) of the expected goals, independent of each other."""
    lh, la = np.atleast_1d(lh), np.atleast_1d(la)
    M = score_matrices(lh, la, rho, nu=nu, kappa=kappa)
    mh, ma = (M * _I).sum(axis=(1, 2)), (M * _J).sum(axis=(1, 2))
    half = lambda m: rate_for_mean(m, GH, nu)
    M1 = score_matrices(half(share * mh), half(share * ma), 0.0, GH, nu)
    M2 = score_matrices(half((1 - share) * mh), half((1 - share) * ma), 0.0, GH, nu)
    h, a = np.meshgrid(np.arange(GH), np.arange(GH), indexing="ij")
    j = np.meshgrid(*[np.arange(GH)] * 4, indexing="ij")
    out = {}
    for group, name, scope, f in MARKETS:
        if scope == "ft":
            p = (M * f(_I, _J)).sum(axis=(1, 2))
        elif scope == "1t":
            p = (M1 * f(h, a)).sum(axis=(1, 2))
        elif scope == "2t":
            p = (M2 * f(h, a)).sum(axis=(1, 2))
        elif scope == "dnb":
            p1, p2 = (M * (_I > _J)).sum(axis=(1, 2)), (M * (_I < _J)).sum(axis=(1, 2))
            p = (p1 if f == "1" else p2) / (p1 + p2)
        else:
            mask = f(*j).astype(float)
            p = np.einsum("nab,ncd,abcd->n", M1, M2, mask)
        out[(group, name)] = p
    return pd.DataFrame(out)


def market_outcomes(hg, ag, h1, a1) -> pd.DataFrame:
    """Actual outcome (1 won, 0 lost; DNB: NaN when refunded) of every market."""
    hg, ag, h1, a1 = (np.asarray(v, dtype=int) for v in (hg, ag, h1, a1))
    h2, a2 = hg - h1, ag - a1
    out = {}
    for group, name, scope, f in MARKETS:
        if scope == "ft":
            y = f(hg, ag)
        elif scope == "1t":
            y = f(h1, a1)
        elif scope == "2t":
            y = f(h2, a2)
        elif scope == "dnb":
            y = np.where(hg == ag, np.nan, (hg > ag) if f == "1" else (hg < ag))
        else:
            y = f(h1, a1, h2, a2)
        out[(group, name)] = np.asarray(y, dtype=float)
    return pd.DataFrame(out)


# ------------------------------------------------------------------- odds

def devig(*odds):
    inv = [1 / np.asarray(o, dtype=float) for o in odds]
    s = sum(inv)
    return [i / s for i in inv]


def _num(df, *names):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def fit_1x2_calibration(p1, p2, result) -> dict:
    """Conditional logit P(j) ∝ exp(a_j + b·log q_j) on margin-free 1X2 probabilities (Hill et al. 2018, 16.4).

    result: 0 home win, 1 draw, 2 away win. With a = 0, b = 1 the market would be perfectly calibrated;
    b > 1 means favourites are a bit more likely than the odds say (favourite-longshot bias)."""
    from scipy import optimize
    Q = np.log(np.clip(np.stack([p1, 1 - np.asarray(p1) - np.asarray(p2), p2], 1), 1e-9, 1))
    Y = np.eye(3)[np.asarray(result, dtype=int)]

    def nll(t):
        u = Q * t[2] + np.array([0.0, t[0], t[1]])
        u -= u.max(1, keepdims=True)
        return -(Y * (u - np.log(np.exp(u).sum(1, keepdims=True)))).sum()

    t = optimize.minimize(nll, np.array([0.0, 0.0, 1.0]), method="BFGS").x
    return {"a_X": round(float(t[0]), 5), "a_2": round(float(t[1]), 5), "b": round(float(t[2]), 5)}


def calibrate_1x2(p1, p2, params: dict):
    """Apply the 1X2 calibration in params['calib_1x2'] (if any) to margin-free probabilities."""
    c = params.get("calib_1x2")
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    if not c:
        return p1, p2
    Q = np.log(np.clip(np.stack([p1, 1 - p1 - p2, p2], 1), 1e-9, 1))
    u = Q * c["b"] + np.array([0.0, c["a_X"], c["a_2"]])
    P = np.exp(u - u.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    return P[:, 0], P[:, 2]


def override_1x2(P: pd.DataFrame, p1, p2, params: dict) -> pd.DataFrame:
    """Replace the markets that depend only on the 1X2 (1X2, Doppia chance, Draw no bet) with the calibrated
    1X2 probabilities. The other markets keep the goals engine: carrying the correction through the expected
    goals made them worse (verification 2024/25-2025/26), while applied here it improves the 1X2 itself."""
    if not params.get("calib_1x2"):
        return P
    c1, c2 = calibrate_1x2(p1, p2, params)
    cx = 1 - c1 - c2
    P = P.copy()
    for col, v in {("1X2", "1"): c1, ("1X2", "X"): cx, ("1X2", "2"): c2, ("Doppia chance", "1X"): c1 + cx,
                   ("Doppia chance", "X2"): cx + c2, ("Doppia chance", "12"): c1 + c2,
                   ("Draw no bet", "1 (rimborso se X)"): c1 / (c1 + c2),
                   ("Draw no bet", "2 (rimborso se X)"): c2 / (c1 + c2)}.items():
        if col in P.columns:
            P[col] = v
    return P


def market_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Margin-free P(1), P(2), P(Over 2.5) from football-data average odds (Bet365 as fallback)."""
    h, d, a = (_num(df, f"Avg{c}", f"BbAv{c}").fillna(_num(df, f"B365{c}")) for c in "HDA")
    o = _num(df, "Avg>2.5", "BbAv>2.5").fillna(_num(df, "B365>2.5"))
    u = _num(df, "Avg<2.5", "BbAv<2.5").fillna(_num(df, "B365<2.5"))
    p1, _, p2 = devig(h, d, a)
    po, _ = devig(o, u)
    return pd.DataFrame({"p1": p1, "p2": p2, "p_over": po}, index=df.index)


# ------------------------------------------------------------- calibration

def load_history() -> pd.DataFrame:
    rows = []
    for div in DIVS:
        for season in SEASONS:
            raw = parse_results_csv(results_path(season, div).read_bytes())
            raw = raw.dropna(subset=["FTHG", "FTAG", "HTHG", "HTAG"])
            inp = market_inputs(raw)
            rows.append(pd.DataFrame({
                "league": div, "season": season, "date": parse_dates(raw.Date),
                "hg": pd.to_numeric(raw.FTHG), "ag": pd.to_numeric(raw.FTAG),
                "h1": pd.to_numeric(raw.HTHG), "a1": pd.to_numeric(raw.HTAG),
            }).join(inp))
    hist = pd.concat(rows, ignore_index=True).dropna()
    ok = (hist.h1 <= hist.hg) & (hist.a1 <= hist.ag)   # rare data slips
    return hist[ok].reset_index(drop=True)


def choose_params(cal: pd.DataFrame) -> dict:
    """ρ and ν maximising the exact-score likelihood, and the first-half share of goals, on calibration seasons."""
    i, j = cal.hg.clip(upper=G - 1).to_numpy(int), cal.ag.clip(upper=G - 1).to_numpy(int)
    grid = []
    for nu in NU_GRID:
        for rho in RHO_GRID:
            lh, la, _ = fit_rates(cal.p1.to_numpy(), cal.p2.to_numpy(), cal.p_over.to_numpy(), rho, nu)
            M = score_matrices(lh, la, rho, nu=nu)
            grid.append({"nu": nu, "rho": rho, "loglik": float(np.log(M[np.arange(len(cal)), i, j]).mean())})
    best = max(grid, key=lambda r: r["loglik"])
    share = (cal.h1 + cal.a1).sum() / (cal.hg + cal.ag).sum()
    result = np.select([cal.hg > cal.ag, cal.hg == cal.ag], [0, 1], 2)
    calib = fit_1x2_calibration(cal.p1.to_numpy(), cal.p2.to_numpy(), result)
    return {"rho": best["rho"], "nu": best["nu"], "first_half_share": round(float(share), 4), "calib_1x2": calib,
            "exact_score_loglik": round(best["loglik"], 5), "grid": grid, "calibration_seasons": CAL_SEASONS,
            "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def calibration_table(hist: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Per market and period: predicted vs observed frequency, z-score and binned calibration error."""
    rho, nu = params["rho"], params.get("nu", 1.0)
    lh, la, _ = fit_rates(hist.p1.to_numpy(), hist.p2.to_numpy(), hist.p_over.to_numpy(), rho, nu)
    P = override_1x2(market_probs(lh, la, rho, params["first_half_share"], nu), hist.p1, hist.p2, params)
    Y = market_outcomes(hist.hg, hist.ag, hist.h1, hist.a1)
    rows = []
    for label, seasons in [("calibrazione 2015/16-2023/24", CAL_SEASONS), ("verifica 2024/25-2025/26", TEST_SEASONS)]:
        sel = hist.season.isin(seasons).to_numpy()
        for (group, name) in P.columns:
            p, y = P[(group, name)].to_numpy()[sel], Y[(group, name)].to_numpy()[sel]
            ok = ~np.isnan(y)
            p, y = p[ok], y[ok]
            se = np.sqrt((p * (1 - p)).sum()) / len(p)
            bins = np.clip((p * 10).astype(int), 0, 9)
            ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).sum() for b in np.unique(bins)) / len(p)
            rows.append({"period": label, "group": group, "market": name, "n": len(p),
                         "pred_pct": 100 * p.mean(), "obs_pct": 100 * y.mean(),
                         "diff_pp": 100 * (y.mean() - p.mean()), "z": (y.mean() - p.mean()) / se,
                         "ece_pp": 100 * ece})
    return pd.DataFrame(rows)


def run_calibration() -> None:
    hist = load_history()
    cal = hist[hist.season.isin(CAL_SEASONS)]
    print(f"partite con quote 1X2, over/under e primo tempo: {len(hist)} ({len(cal)} di calibrazione)")
    params = choose_params(cal)
    print(f"ρ = {params['rho']}, ν = {params['nu']}, quota dei gol nel primo tempo = {params['first_half_share']}, "
          f"calibrazione 1X2 (logit condizionale) = {params['calib_1x2']}")
    PARAMS_PATH.parent.mkdir(exist_ok=True)
    write_atomic(PARAMS_PATH, json.dumps(params, indent=2))
    table = calibration_table(hist, params)
    write_csv_atomic(table, CALIBRATION_OUT)
    test = table[table.period.str.startswith("verifica")]
    worst = test.reindex(test.diff_pp.abs().sort_values(ascending=False).index).head(10)
    pd.set_option("display.width", 160)
    print("verifica 2024/25-2025/26: mercati con lo scarto maggiore tra previsto e osservato")
    print(worst[["group", "market", "n", "pred_pct", "obs_pct", "diff_pp", "z", "ece_pp"]].round(2).to_string(index=False))
    print(f"scarto medio assoluto su {len(test)} mercati: {test.diff_pp.abs().mean():.2f} punti %, "
          f"|z| > 2 in {(test.z.abs() > 2).sum()} mercati")


# ---------------------------------------------------------- upcoming matches

def load_params() -> dict:
    return json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else DEFAULT_PARAMS


UPCOMING_COLS = ["league", "kickoff_utc", "home", "away", "p1", "p2", "p_over",
                 "rate_home", "rate_away", "exp_home", "exp_away", "odds_fetched_at"]


def upcoming_rates(fd: pd.DataFrame, now: pd.Timestamp, fetched_at: pd.Timestamp, params: dict) -> pd.DataFrame:
    """One row per upcoming match (our 10 leagues): market inputs and fitted rates. Markets are computed
    from the rates with market_probs(), on the fly, so the file stays small."""
    from .tips import kickoff_utc
    fd = fd[fd["Div"].str.strip().isin(DIVS)].copy()
    if fd.empty or now - fetched_at > ODDS_MAX_AGE:
        return pd.DataFrame(columns=UPCOMING_COLS)
    inp = market_inputs(fd)
    ko = kickoff_utc(fd)
    keep = inp.notna().all(axis=1) & ko.notna() & (ko > now)
    fd, inp, ko = fd[keep], inp[keep], ko[keep]
    if fd.empty:
        return pd.DataFrame(columns=UPCOMING_COLS)
    rho, nu = params["rho"], params.get("nu", 1.0)
    lh, la, _ = fit_rates(inp.p1.to_numpy(), inp.p2.to_numpy(), inp.p_over.to_numpy(), rho, nu)
    mh, ma = expected_goals(lh, la, rho, nu)
    return pd.DataFrame({
        "league": fd["Div"].str.strip().to_numpy(), "kickoff_utc": ko.dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
        "home": fd["HomeTeam"].str.strip().to_numpy(), "away": fd["AwayTeam"].str.strip().to_numpy(),
        "p1": inp.p1.round(5).to_numpy(), "p2": inp.p2.round(5).to_numpy(), "p_over": inp.p_over.round(5).to_numpy(),
        "rate_home": lh.round(6), "rate_away": la.round(6), "exp_home": mh.round(3), "exp_away": ma.round(3),
        "odds_fetched_at": fetched_at.isoformat(timespec="seconds"),
    })[UPCOMING_COLS]


def fair_odds_for(row, params: dict) -> pd.DataFrame:
    """Probability and fair odds of every market for one upcoming match (a row of upcoming_rates)."""
    P = market_probs([row["rate_home"]], [row["rate_away"]], params["rho"], params["first_half_share"],
                     params.get("nu", 1.0))
    if "p1" in row and "p2" in row:
        P = override_1x2(P, [row["p1"]], [row["p2"]], params)
    p = P.iloc[0].to_numpy()
    return pd.DataFrame({"group": [g for g, _ in P.columns], "market": [m for _, m in P.columns],
                         "prob": p, "fair_odds": np.where(p > 0, 1 / np.maximum(p, 1e-9), np.nan)})


def update_fair_odds(now: pd.Timestamp = None, fd_path=RAW / "fd_fixtures.csv", out=FAIR_ODDS_OUT) -> list:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    if not fd_path.exists():
        return ["quote giuste: fixtures.csv non disponibile"]
    table = upcoming_rates(parse_fd_fixtures(fd_path.read_bytes()), now, fetched_at_of(fd_path), load_params())
    out.parent.mkdir(exist_ok=True)
    write_csv_atomic(table, out)
    print(f"quote giuste: {len(table)} partite, {len(MARKETS)} mercati")
    return []
