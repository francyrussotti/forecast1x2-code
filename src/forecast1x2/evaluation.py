"""Walk-forward backtest, scoring rules, margin removal and comparison with the market."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .data import LEAGUES, league_history, load_matches, write_csv_atomic
from .elo import elo_walk_forward
from .model import OUTPUTS, fit, predict_match

HYPER_PATH = OUTPUTS / "hyperparams.json"
XI_GRID = [0.001, 0.0015, 0.002, 0.003, 0.004, 0.006]
L2_GRID = [0.01, 0.03, 0.1, 0.25, 0.5, 1.0]
W_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
XG_WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
MIN_XG_COVERAGE = 0.9   # share of calibration matches with xG needed to calibrate the xG weight
# xG exist only from 2024/25 (Sportmonks). Cross-season test: weight chosen on one season, tested on the
# other, both ways; predictions use the weight chosen on both seasons together.
XG_SEASONS = ["2425", "2526"]
# Hyperparameters are shared by all leagues: choosing them per league overfits (validated by picking
# on 2022/23 and scoring 2023/24: per-league 0.9857 vs shared 0.9751 log-loss).
CALIBRATION_SEASONS = ["2122", "2223", "2324"]
BACKTEST_SEASONS = ["2425", "2526"]
SOURCES = {"ensemble_xg": "Ensemble + xG (in uso)", "ensemble": "Ensemble (Dixon-Coles + Elo)",
           "model": "Dixon-Coles", "model_xg": "Dixon-Coles + xG", "elo": "Elo", "ols": "OLS (regressione lineare)",
           "combined": "Modello + mercato apertura", "market_open": "Mercato apertura",
           "market": "Mercato chiusura (prop.)", "market_shin": "Mercato chiusura (Shin)", "baseline": "Baseline"}
ELO_K_GRID = [10, 15, 20, 25, 30, 40]
ELO_REGRESS_GRID = [0.0, 0.1, 0.2, 0.33, 0.5]
ENSEMBLE_GRID = np.round(np.arange(0, 1.0001, 0.1), 2)   # weight of Dixon-Coles in the linear average
OUTCOMES = ["1", "X", "2"]
RESULT_INDEX = {"H": 0, "D": 1, "A": 2}


# ------------------------------------------------------------------- metrics
# P: (n, 3) probabilities in order 1, X, 2; y: (n,) index of the observed outcome

def log_loss(P, y):
    return -np.log(P[np.arange(len(y)), y])


def brier(P, y):
    return ((P - np.eye(3)[y]) ** 2).sum(axis=1)


def rps(P, y):
    cum = np.cumsum(P, axis=1)[:, :2] - np.cumsum(np.eye(3)[y], axis=1)[:, :2]
    return (cum ** 2).sum(axis=1) / 2


METRICS = {"log_loss": log_loss, "brier": brier, "rps": rps}


# ------------------------------------------------------------- margin removal

def devig_proportional(odds):
    q = 1 / np.asarray(odds, dtype=float)
    return q / q.sum(axis=1, keepdims=True)


def _shin_row(q):
    B = q.sum()
    if B <= 1:
        return q / B

    def probs(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / B) - z) / (2 * (1 - z))

    z = brentq(lambda z: probs(z).sum() - 1, 0.0, 0.5)
    return probs(z)


def devig_shin(odds):
    q = 1 / np.asarray(odds, dtype=float)
    return np.vstack([_shin_row(r) for r in q])


def combine(P_model, P_market, w):
    """Logarithmic pool: P ∝ P_model^w · P_market^(1-w)."""
    P = np.asarray(P_model) ** w * np.asarray(P_market) ** (1 - w)
    return P / P.sum(axis=1, keepdims=True)


# --------------------------------------------------------------- walk-forward

def week_start(dates: pd.Series) -> pd.Series:
    return dates.dt.normalize() - pd.to_timedelta(dates.dt.weekday, unit="D")


def walk_forward(history: pd.DataFrame, season: str, xi: float, l2: float, xg_weight: float = 0.0) -> pd.DataFrame:
    """Predict each calendar week of the top-flight `season` with a model fitted only on matches before that week.

    `history` is one league's data as returned by league_history (top flight + second division).
    """
    h = history.sort_values("date")
    tier = h.tier if "tier" in h else pd.Series(1, index=h.index)
    test = h[(h.season == season) & (tier == 1)].copy()
    test["week_start"] = week_start(test.date)
    rows, params = [], None
    for wk, games in test.groupby("week_start"):
        train = h[h.date < wk]
        params = fit(train, wk, xi, l2=l2, init=params, xg_weight=xg_weight)
        top = train[tier[train.index] == 1]
        freq = top.result.value_counts(normalize=True).reindex(["H", "D", "A"]).fillna(0).to_numpy()
        known = set(params.teams)
        for g in games.itertuples():
            hp, ap = getattr(g, "home_promoted", False), getattr(g, "away_promoted", False)
            pred = predict_match(params, g.home, g.away, hp, ap)
            rows.append({
                "league": g.league, "season": season, "date": g.date, "week_start": wk,
                "home": g.home, "away": g.away, "result": g.result,
                "model_1": pred["p1"], "model_X": pred["px"], "model_2": pred["p2"],
                "baseline_1": freq[0], "baseline_X": freq[1], "baseline_2": freq[2],
                "odds_h": g.odds_h, "odds_d": g.odds_d, "odds_a": g.odds_a,
                "open_h": getattr(g, "open_h", np.nan), "open_d": getattr(g, "open_d", np.nan),
                "open_a": getattr(g, "open_a", np.nan),
                "train_max_date": train.date.max(), "n_train": len(train),
                "unseen_team": g.home not in known or g.away not in known,
                "promoted_match": bool(hp or ap),
            })
    return pd.DataFrame(rows)


def calibrate(matches: pd.DataFrame) -> dict:
    """Choose shared xi and l2, then the model/market pool weight, on 2022/23 + 2023/24 (before the backtest)."""
    hist = matches[matches.season <= CALIBRATION_SEASONS[-1]]
    histories = {div: league_history(hist, div) for div in LEAGUES}
    grid = []
    for xi in XI_GRID:
        for l2 in L2_GRID:
            p = pd.concat([walk_forward(h, s, xi, l2) for h in histories.values() for s in CALIBRATION_SEASONS],
                          ignore_index=True)
            ll = log_loss(probs(p, "model"), outcome_index(p)).mean()
            grid.append({"xi": xi, "l2": l2, "log_loss": ll, "preds": p})
            print(f"  xi={xi:<6} l2={l2:<5} log-loss {ll:.5f}")
    best = min(grid, key=lambda r: r["log_loss"])
    print(f"scelti xi={best['xi']} (emivita {np.log(2) / best['xi']:.0f} giorni), l2={best['l2']}")

    df = add_market(best["preds"].dropna(subset=["open_h", "open_d", "open_a"]).reset_index(drop=True))
    y = outcome_index(df)
    w_grid = [{"w": float(w), "log_loss": log_loss(combine(probs(df, "model"), probs(df, "market_open"), w), y).mean()}
              for w in W_GRID]
    w = min(w_grid, key=lambda r: r["log_loss"])
    print(f"peso del modello nel pool con le quote di apertura: w={w['w']} (log-loss {w['log_loss']:.5f})")

    elo = calibrate_elo_ensemble(hist, best["preds"])
    xg = calibrate_xg(matches, best["xi"], best["l2"])
    hyper = {"calibration_seasons": CALIBRATION_SEASONS, "xi": best["xi"], "l2": best["l2"], "w": w["w"], **xg, **elo,
             "grid": [{k: r[k] for k in ("xi", "l2", "log_loss")} for r in grid], "w_grid": w_grid}
    OUTPUTS.mkdir(exist_ok=True)
    HYPER_PATH.write_text(json.dumps(hyper, indent=2))
    return hyper


def elo_predictions(matches: pd.DataFrame, seasons, k: float, regress: float) -> pd.DataFrame:
    top = matches[matches.tier == 1]
    return pd.concat([elo_walk_forward(top[top.league == div], s, k, regress) for div in LEAGUES for s in seasons],
                     ignore_index=True)


def add_ensemble(df: pd.DataFrame, a: float, dc: str = "model", name: str = "ensemble") -> pd.DataFrame:
    """Linear average a * Dixon-Coles + (1 - a) * Elo."""
    _set(df, name, a * probs(df, dc) + (1 - a) * probs(df, "elo"))
    return df


KEYS = ["league", "date", "home", "away"]


def calibrate_elo_ensemble(hist: pd.DataFrame, dc_preds: pd.DataFrame) -> dict:
    """Elo K and between-season regression, then the Dixon-Coles/Elo ensemble weight, on the calibration seasons."""
    grid = []
    for k in ELO_K_GRID:
        for reg in ELO_REGRESS_GRID:
            p = elo_predictions(hist, CALIBRATION_SEASONS, k, reg)
            p = p.merge(hist[KEYS + ["result"]], on=KEYS)
            grid.append({"k": k, "regress": reg, "log_loss": log_loss(probs(p, "elo"), outcome_index(p)).mean(),
                         "preds": p})
    best = min(grid, key=lambda r: r["log_loss"])
    print(f"Elo: K={best['k']}, regressione verso la media {best['regress']} (log-loss {best['log_loss']:.5f})")
    both = dc_preds.merge(best["preds"][KEYS + ["elo_1", "elo_X", "elo_2"]], on=KEYS, how="inner")
    assert len(both) == len(dc_preds)
    y = outcome_index(both)
    ens = [{"a": float(a), "log_loss": log_loss(a * probs(both, "model") + (1 - a) * probs(both, "elo"), y).mean()}
           for a in ENSEMBLE_GRID]
    a = min(ens, key=lambda r: r["log_loss"])
    print(f"ensemble: peso Dixon-Coles {a['a']}, Elo {1 - a['a']:.1f} (log-loss {a['log_loss']:.5f})")
    return {"elo_k": best["k"], "elo_regress": best["regress"], "ensemble_a": a["a"],
            "elo_grid": [{x: r[x] for x in ("k", "regress", "log_loss")} for r in grid], "ensemble_grid": ens}


def calibrate_xg(matches: pd.DataFrame, xi: float, l2: float) -> dict:
    """xG weight by walk-forward log-loss on 2024/25 and 2025/26 (xi, l2 already fixed).

    Returns the production weight (best on both seasons together) and, for the cross-season test,
    the weight each season would choose on its own. Weight 0 if xG are missing.
    """
    top = matches[(matches.tier == 1) & matches.season.isin(XG_SEASONS)]
    coverage = float(top.hxg.notna().mean()) if "hxg" in top and len(top) else 0.0
    if coverage < MIN_XG_COVERAGE:
        print(f"xG non usati: copertura {'+'.join(XG_SEASONS)} {coverage:.0%} (< {MIN_XG_COVERAGE:.0%})")
        return {"xg_weight": 0.0, "xg_coverage": coverage, "xg_grid": [], "xg_folds": {}}
    histories = [league_history(matches, div) for div in LEAGUES]
    grid = []
    for xw in XG_WEIGHT_GRID:
        row = {"xg_weight": xw}
        for season in XG_SEASONS:
            p = pd.concat([walk_forward(h, season, xi, l2, xw) for h in histories], ignore_index=True)
            ll = log_loss(probs(p, "model"), outcome_index(p))
            row[f"log_loss_{season}"], row[f"n_{season}"] = ll.mean(), len(ll)
        n = sum(row[f"n_{s}"] for s in XG_SEASONS)
        row["log_loss"] = sum(row[f"log_loss_{s}"] * row[f"n_{s}"] for s in XG_SEASONS) / n
        grid.append(row)
        print("  peso xG {}: ".format(xw) + ", ".join(f"{s} {row[f'log_loss_{s}']:.5f}" for s in XG_SEASONS))
    best = min(grid, key=lambda r: r["log_loss"])["xg_weight"]
    # test season -> weight chosen on the other season
    folds = {test: min(grid, key=lambda r: r[f"log_loss_{cal}"])["xg_weight"]
             for test, cal in zip(XG_SEASONS, XG_SEASONS[::-1])}
    print(f"peso xG per le previsioni (scelto su {'+'.join(XG_SEASONS)}): {best}; "
          + ", ".join(f"per testare {t}: {w}" for t, w in folds.items()))
    return {"xg_weight": best, "xg_coverage": coverage, "xg_seasons": XG_SEASONS, "xg_grid": grid, "xg_folds": folds}


# ----------------------------------------------------------- reliability

def reliability_table(df: pd.DataFrame, sources) -> pd.DataFrame:
    """Reliability in plain percentages, per source, on the same matches.

    accuracy: how often the most likely outcome happened. calib_error_pp: average gap (percentage
    points) between forecast and observed frequency over 10 probability bins, weighted by matches.
    skill_vs_baseline: log-loss improvement over historical 1/X/2 frequencies. market_share: share of
    the market's improvement over the baseline that the source achieves (100% = as good as closing odds).
    """
    y = outcome_index(df)
    ll = {s: log_loss(probs(df, s), y).mean() for s in set(sources) | {"baseline", "market"}}
    gap = ll["baseline"] - ll["market"]
    rows = []
    for s in sources:
        P = probs(df, s)
        cal = calibration_table(df, sources=[s])
        rows.append({"source": s, "n": len(df),
                     "accuracy_pct": 100 * (P.argmax(axis=1) == y).mean(),
                     "calib_error_pp": 100 * np.average((cal.mean_pred - cal.obs_freq).abs(), weights=cal.n),
                     "skill_vs_baseline_pct": 100 * (1 - ll[s] / ll["baseline"]),
                     "market_share_pct": 100 * (ll["baseline"] - ll[s]) / gap,
                     "log_loss": ll[s]})
    return pd.DataFrame(rows)


# --------------------------------------------------------- risk and betting

# (level, minimum probability of the most likely outcome); checked on the 2024-26 backtest:
# ~25% / 41% / 34% of matches, favourite win rate close to the forecast in each level
RISK_TIERS = [(1, 0.60), (2, 0.45), (3, 0.0)]
VALUE_THRESHOLDS = [0.0, 0.05, 0.10]


def risk_level(p_max) -> np.ndarray:
    p = np.asarray(p_max, dtype=float)
    return np.select([p >= t for _, t in RISK_TIERS], [lvl for lvl, _ in RISK_TIERS])


def risk_table(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Per risk level: share of matches, mean forecast of the favourite, how often the favourite won."""
    P, y = probs(df, source), outcome_index(df)
    lvl, fav, pmax = risk_level(P.max(axis=1)), P.argmax(axis=1), P.max(axis=1)
    rows = []
    for level, _ in RISK_TIERS:
        m = lvl == level
        rows.append({"source": source, "level": level, "n": int(m.sum()), "share": m.mean(),
                     "mean_pred": pmax[m].mean() if m.any() else np.nan,
                     "fav_win_rate": (fav[m] == y[m]).mean() if m.any() else np.nan})
    return pd.DataFrame(rows)


def _roi_ci(profit: np.ndarray, weeks: np.ndarray, n_boot: int, rng) -> tuple:
    if len(profit) == 0:
        return np.nan, np.nan
    blocks = pd.DataFrame({"wk": weeks, "p": profit}).groupby("wk").p.agg(["sum", "count"])
    s, c = blocks["sum"].to_numpy(), blocks["count"].to_numpy()
    idx = rng.integers(0, len(blocks), size=(n_boot, len(blocks)))
    boot = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return np.quantile(boot, 0.025), np.quantile(boot, 0.975)


def betting_simulation(df: pd.DataFrame, source: str, thresholds=VALUE_THRESHOLDS,
                       n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Flat 1-unit stakes at the early (pre-match) market-average odds, at most one bet per match.

    "valore": bet the outcome with the highest expected value p*odds-1 when it exceeds the threshold.
    "favorito": always bet the market favourite (reference: shows the bookmaker margin).
    ROI = profit per unit staked; 95% CI by resampling calendar weeks.
    """
    rng = np.random.default_rng(seed)
    O = df[["open_h", "open_d", "open_a"]].to_numpy(dtype=float)
    y, weeks = outcome_index(df), df.week_start.to_numpy()
    rows = []

    def record(strategy, threshold, bet, pick):
        win = pick[bet] == y[bet]
        profit = np.where(win, O[bet, pick[bet]] - 1, -1.0)
        lo, hi = _roi_ci(profit, weeks[bet], n_boot, rng)
        rows.append({"source": source, "strategy": strategy, "threshold": threshold, "matches": len(df),
                     "bets": int(bet.sum()), "wins": int(win.sum()), "profit": profit.sum(),
                     "roi": profit.mean() if bet.any() else np.nan, "ci_low": lo, "ci_high": hi})

    ev = probs(df, source) * O - 1
    pick, best = ev.argmax(axis=1), ev.max(axis=1)
    for t in thresholds:
        record("valore", t, best > t, pick)
    record("favorito", np.nan, np.ones(len(df), bool), O.argmin(axis=1))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- summaries

def probs(df: pd.DataFrame, source: str) -> np.ndarray:
    return df[[f"{source}_{o}" for o in OUTCOMES]].to_numpy()


def outcome_index(df: pd.DataFrame) -> np.ndarray:
    return df.result.map(RESULT_INDEX).to_numpy()


def _set(df, source, P):
    for k, o in enumerate(OUTCOMES):
        df[f"{source}_{o}"] = P[:, k]


def add_market(df: pd.DataFrame, w: float = None) -> pd.DataFrame:
    """Margin-free market probabilities (closing: proportional and Shin; early odds: proportional) and the pool."""
    closing = df[["odds_h", "odds_d", "odds_a"]].to_numpy()
    _set(df, "market", devig_proportional(closing))
    _set(df, "market_shin", devig_shin(closing))
    _set(df, "market_open", devig_proportional(df[["open_h", "open_d", "open_a"]].to_numpy()))
    if w is not None:
        _set(df, "combined", combine(probs(df, "model"), probs(df, "market_open"), w))
    return df


def _groups(df):
    """(league, season) labels for every aggregation level."""
    yield from (((lg, ss), g) for (lg, ss), g in df.groupby(["league", "season"]))
    yield from (((lg, "tutte"), g) for lg, g in df.groupby("league"))
    yield from ((("tutti", ss), g) for ss, g in df.groupby("season"))
    yield ("tutti", "tutte"), df


def present_sources(df: pd.DataFrame) -> list:
    return [s for s in SOURCES if f"{s}_1" in df]


def metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    y_all = outcome_index(df)
    sources = present_sources(df)
    scores = {(s, m): f(probs(df, s), y_all) for s in sources for m, f in METRICS.items()}
    rows = []
    for (lg, ss), g in _groups(df):
        pos = df.index.get_indexer(g.index)
        for s in sources:
            rows.append({"league": lg, "season": ss, "source": s, "n": len(g),
                         **{m: scores[(s, m)][pos].mean() for m in METRICS}})
    return pd.DataFrame(rows)


def bootstrap_diff(df: pd.DataFrame, sources=("model",), n_boot: int = 2000, seed: int = 0,
                   against: str = "market") -> pd.DataFrame:
    """95% CI of mean(source - against), resampling whole calendar weeks (matches in a week are not independent)."""
    rng = np.random.default_rng(seed)
    y_all = outcome_index(df)
    rows = []
    for source in sources:
        for metric in ["log_loss", "rps"]:
            f = METRICS[metric]
            diff_all = f(probs(df, source), y_all) - f(probs(df, against), y_all)
            for (lg, ss), g in _groups(df):
                pos = df.index.get_indexer(g.index)
                blocks = pd.DataFrame({"wk": g.week_start.to_numpy(), "d": diff_all[pos]}).groupby("wk").d.agg(["sum", "count"])
                s, c = blocks["sum"].to_numpy(), blocks["count"].to_numpy()
                idx = rng.integers(0, len(blocks), size=(n_boot, len(blocks)))
                boot = s[idx].sum(axis=1) / c[idx].sum(axis=1)
                rows.append({"source": source, "against": against, "league": lg, "season": ss, "metric": metric, "n": len(g),
                             "weeks": len(blocks), "mean_diff": diff_all[pos].mean(),
                             "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975)})
    return pd.DataFrame(rows)


def calibration_table(df: pd.DataFrame, bins: int = 10, sources=None) -> pd.DataFrame:
    """Mean forecast vs observed frequency per probability bin, for each source and outcome."""
    y = np.eye(3)[outcome_index(df)]
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for s in (sources or present_sources(df)):
        P = probs(df, s)
        for k, o in enumerate(OUTCOMES):
            b = np.clip(np.digitize(P[:, k], edges) - 1, 0, bins - 1)
            for i in np.unique(b):
                m = b == i
                rows.append({"source": s, "outcome": o, "bin_low": edges[i], "bin_high": edges[i + 1],
                             "mean_pred": P[m, k].mean(), "obs_freq": y[m, k].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def run_backtest() -> None:
    matches = load_matches()
    print(f"calibrazione su {' + '.join(CALIBRATION_SEASONS)} (xi e l2 comuni, Elo, ensemble, xG)...")
    hyper = calibrate(matches)

    parts = []
    for div, (name, *_) in LEAGUES.items():
        h = league_history(matches, div)
        for season in BACKTEST_SEASONS:
            p = walk_forward(h, season, hyper["xi"], hyper["l2"])   # goals only: comparable with earlier runs
            print(f"{name} {season}: {len(p)} partite, {p.week_start.nunique()} settimane")
            parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    odds_cols = ["odds_h", "odds_d", "odds_a", "open_h", "open_d", "open_a"]
    missing = df[odds_cols].isna().any(axis=1)
    if missing.any():
        print(f"escluse {missing.sum()} partite senza quote (confronto sulle stesse partite)")
        df = df[~missing].reset_index(drop=True)
    elo = elo_predictions(matches, BACKTEST_SEASONS, hyper["elo_k"], hyper["elo_regress"])
    df = df.merge(elo, on=KEYS, how="left")
    assert df.elo_1.notna().all() and (df.elo_train_max_week < df.week_start).all()
    df = add_ensemble(add_market(df, hyper["w"]), hyper["ensemble_a"])
    assert (df.train_max_date < df.week_start).all() and (df.week_start <= df.date).all()

    df.to_csv(OUTPUTS / "backtest_matches.csv", index=False)
    metrics = metrics_table(df)
    metrics.to_csv(OUTPUTS / "backtest_metrics.csv", index=False)
    boot = bootstrap_diff(df, sources=("ensemble", "model", "elo", "ols", "market_open"))
    boot.to_csv(OUTPUTS / "backtest_bootstrap.csv", index=False)
    calibration_table(df).to_csv(OUTPUTS / "calibration.csv", index=False)

    tot = metrics[(metrics.league == "tutti") & (metrics.season == "tutte")].set_index("source")
    print(tot[["n", "log_loss", "brier", "rps"]].round(4).to_string())
    b = boot[(boot.league == "tutti") & (boot.season == "tutte")]
    print(b[["source", "metric", "mean_diff", "ci_low", "ci_high"]].round(5).to_string(index=False))

    xg_test = evaluate_xg(matches, df, hyper)

    # risk levels, betting simulation and reliability: goals-only sources on the backtest, xG sources on
    # the cross-season xG test (same 3504 matches)
    frames = [(df, ["ensemble", "model", "elo"])]
    if xg_test is not None:
        frames.append((xg_test, ["ensemble_xg", "model_xg"]))
    risk = pd.concat([risk_table(d, s) for d, ss in frames for s in ss], ignore_index=True)
    sims = pd.concat([betting_simulation(d, s) for d, ss in frames for s in ss], ignore_index=True)
    rel = pd.concat([reliability_table(df, present_sources(df))]
                    + ([reliability_table(xg_test, ["ensemble_xg", "model_xg"])] if xg_test is not None else []),
                    ignore_index=True)
    write_csv_atomic(risk, OUTPUTS / "risk_tiers.csv")
    write_csv_atomic(sims, OUTPUTS / "betting_sim.csv")
    write_csv_atomic(rel, OUTPUTS / "reliability.csv")
    print("\nAffidabilità:")
    print(rel.round(2).to_string(index=False))
    print("\nLivelli di rischio:")
    print(risk.round(3).to_string(index=False))
    print("\nSimulazione scommesse (1 unità a partita, quote pre-partita medie):")
    print(sims[["source", "strategy", "threshold", "bets", "wins", "profit", "roi", "ci_low", "ci_high"]]
          .round(3).to_string(index=False))


def evaluate_xg(matches: pd.DataFrame, backtest: pd.DataFrame, hyper: dict):
    """Cross-season test of the xG model against the goals-only model and the market, on the same matches.

    Each season is predicted (walk-forward) with the xG weight chosen on the other season; the two
    test seasons are then pooled. The weight is the only quantity chosen across seasons: model fits
    always use past matches only.
    """
    paths = [OUTPUTS / "xg_metrics.csv", OUTPUTS / "xg_bootstrap.csv"]
    folds = hyper.get("xg_folds") or {}
    if not folds:
        print("xG: non disponibili, nessun test")
        for p in paths:
            p.unlink(missing_ok=True)
        return None
    parts = []
    for season, w in folds.items():
        xgp = pd.concat([walk_forward(league_history(matches, div), season, hyper["xi"], hyper["l2"], w)
                         for div in LEAGUES], ignore_index=True)
        xgp = xgp[["league", "date", "home", "away", "model_1", "model_X", "model_2"]].rename(
            columns={f"model_{o}": f"model_xg_{o}" for o in OUTCOMES})
        part = backtest[backtest.season == season].merge(xgp, on=["league", "date", "home", "away"], how="inner")
        assert len(part) == (backtest.season == season).sum()
        parts.append(part.assign(xg_weight_used=w))
    test = add_ensemble(pd.concat(parts, ignore_index=True), hyper["ensemble_a"], dc="model_xg", name="ensemble_xg")
    metrics = metrics_table(test)
    write_csv_atomic(metrics, paths[0])
    boot = pd.concat([bootstrap_diff(test, sources=("model_xg", "ensemble_xg"), against="model"),
                      bootstrap_diff(test, sources=("model_xg", "ensemble_xg"), against="market")], ignore_index=True)
    write_csv_atomic(boot, paths[1])
    tot = metrics[(metrics.league == "tutti") & (metrics.season == "tutte")].set_index("source")
    print("\nTest incrociato degli xG (" + ", ".join(f"{s} con peso {w}" for s, w in folds.items()) + "):")
    print(tot[["n", "log_loss", "brier", "rps"]].round(4).to_string())
    b = boot[(boot.league == "tutti") & (boot.season == "tutte")]
    print(b[["source", "against", "metric", "mean_diff", "ci_low", "ci_high"]].round(5).to_string(index=False))
    return test
