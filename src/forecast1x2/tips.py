"""Registro dei consigli in prova: over/under 2.5 quando la quota migliore supera di molto la media.

Regola (congelata il 2026-10-23, versione RULE_VERSION; non va cambiata guardando i risultati):
  - quote del venerdì di football-data (fixtures.csv): migliore (Max) e media (Avg) per over e under 2.5;
  - probabilità "giusta" = quote medie senza margine (normalizzazione proporzionale);
  - vantaggio = quota migliore × probabilità giusta − 1, per over e per under; si tiene il lato migliore;
  - consiglio se il vantaggio supera THRESHOLD, nei 10 campionati del progetto (prima e seconda divisione).
THRESHOLD è stata scelta tra 0, 1, 2, 3, 5, 7,5, 10 e 15% come la migliore sulle stagioni 2015/16-2023/24
(almeno 300 giocate); 2024/25 e 2025/26 servono solo da verifica (`python -m forecast1x2 consigli-backtest`).

Il registro è solo in aggiunta: una partita entra una volta sola, almeno MIN_LEAD prima del calcio d'inizio,
con le quote di quel momento, e non viene più modificata se non per segnare il risultato.
Sono giocate simulate da 1 unità, non consigli di scommessa.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (LEAGUES, ODDS_MAX_AGE, RAW, ROOT, SEASONS, SECOND_DIV, fetched_at_of, parse_dates,
                   parse_fd_fixtures, parse_results_csv, results_path, write_csv_atomic)
from .evaluation import _roi_ci

RULE_VERSION = "ou25-v1"
RULE_FROZEN = "2026-10-23"
THRESHOLD = 0.02
DIVS = list(LEAGUES) + list(SECOND_DIV.values())
TIER = {**{d: 1 for d in LEAGUES}, **{d: 2 for d in SECOND_DIV.values()}}
MIN_LEAD = pd.Timedelta(hours=1)   # margine anche se l'orario di football-data fosse in ora italiana
FD_TZ = "Europe/London"
VOID_AFTER = pd.Timedelta(days=7)  # partita senza risultato una settimana dopo: rinviata, giocata annullata

REGISTER = ROOT / "data" / "tips" / "ou_register.csv"
BACKTEST_OUT = ROOT / "outputs" / "tips_backtest.csv"
CAL_SEASONS = [s for s in SEASONS if s <= "2324"]
TEST_SEASONS = ["2425", "2526"]
KEYS = ["league", "date", "home", "away"]
COLUMNS = ["rule", "logged_at", "odds_fetched_at", "league", "tier", "kickoff_utc", "date", "home", "away",
           "pick", "odds_best", "avg_over", "avg_under", "fair_prob", "edge", "status", "goals", "profit"]


def _num(df: pd.DataFrame, *names) -> pd.Series:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def ou_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Best side (Over/Under 2.5) and its edge from football-data columns (Max/Avg, or BbMx/BbAv in old files)."""
    max_o, max_u = _num(df, "Max>2.5", "BbMx>2.5"), _num(df, "Max<2.5", "BbMx<2.5")
    avg_o, avg_u = _num(df, "Avg>2.5", "BbAv>2.5"), _num(df, "Avg<2.5", "BbAv<2.5")
    fair_o = (1 / avg_o) / (1 / avg_o + 1 / avg_u)
    e_o, e_u = max_o * fair_o - 1, max_u * (1 - fair_o) - 1
    over = e_o >= e_u
    return pd.DataFrame({
        "pick": np.where(over, "Over", "Under"),
        "odds_best": np.where(over, max_o, max_u),
        "avg_over": avg_o, "avg_under": avg_u,
        "fair_prob": np.where(over, fair_o, 1 - fair_o),
        "edge": np.where(over, e_o, e_u),
    }, index=df.index)


def kickoff_utc(df: pd.DataFrame) -> pd.Series:
    """football-data Date + Time (UK time); midnight if the time is missing."""
    day = parse_dates(df["Date"])
    t = df["Time"].fillna("00:00").str.strip() if "Time" in df.columns else pd.Series("00:00", index=df.index)
    t = t.where(t.str.fullmatch(r"\d{1,2}:\d{2}"), "00:00")
    local = day + pd.to_timedelta(t + ":00")
    return local.dt.tz_localize(FD_TZ, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")


def candidates(fd: pd.DataFrame, now: pd.Timestamp, fetched_at: pd.Timestamp) -> pd.DataFrame:
    """Matches in football-data fixtures.csv that pass the rule and can still be logged before kickoff."""
    fd = fd[fd["Div"].str.strip().isin(DIVS)].copy()
    if fd.empty or now - fetched_at > ODDS_MAX_AGE:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.DataFrame({
        "rule": RULE_VERSION, "logged_at": now.isoformat(timespec="seconds"),
        "odds_fetched_at": fetched_at.isoformat(timespec="seconds"),
        "league": fd["Div"].str.strip(), "tier": fd["Div"].str.strip().map(TIER),
        "kickoff_utc": kickoff_utc(fd), "date": parse_dates(fd["Date"]).dt.strftime("%Y-%m-%d"),
        "home": fd["HomeTeam"].str.strip(), "away": fd["AwayTeam"].str.strip(),
    }).join(ou_edges(fd))
    out = out[(out.edge > THRESHOLD) & out.kickoff_utc.notna() & (out.kickoff_utc - MIN_LEAD > now)].copy()
    out["kickoff_utc"] = out.kickoff_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["status"], out["goals"], out["profit"] = "aperta", np.nan, np.nan
    return out[COLUMNS].reset_index(drop=True)


def settle(reg: pd.DataFrame, matches: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Mark open tips as won/lost from the results, or void if still unplayed a week after kickoff."""
    reg = reg.copy()
    res = matches.dropna(subset=["hg", "ag"])
    res = pd.DataFrame({"league": res.league, "date": pd.to_datetime(res.date).dt.strftime("%Y-%m-%d"),
                        "home": res.home, "away": res.away, "total": res.hg + res.ag}).drop_duplicates(KEYS)
    open_idx = reg.index[reg.status == "aperta"]
    if len(open_idx) == 0:
        return reg
    m = reg.loc[open_idx, KEYS + ["pick", "odds_best", "kickoff_utc"]].reset_index().merge(
        res, on=KEYS, how="left").set_index("index")
    played = m[m.total.notna()]
    won = np.where(played.pick == "Over", played.total > 2.5, played.total < 2.5)
    reg.loc[played.index, "goals"] = played.total
    reg.loc[played.index, "status"] = np.where(won, "vinta", "persa")
    reg.loc[played.index, "profit"] = np.where(won, played.odds_best - 1, -1.0)
    unplayed = m[m.total.isna()]
    stale = unplayed.index[pd.to_datetime(unplayed.kickoff_utc, utc=True) + VOID_AFTER < now]
    reg.loc[stale, "status"] = "annullata"
    reg.loc[stale, "profit"] = 0.0
    return reg


def load_register(path: Path = REGISTER) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"date": str, "league": str, "home": str, "away": str, "status": str})


def update_register(matches: pd.DataFrame, now: pd.Timestamp = None, fd_path: Path = RAW / "fd_fixtures.csv",
                    path: Path = REGISTER) -> list:
    """Add new tips from the Friday odds and settle the open ones. Returns warnings."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    reg = load_register(path)
    warnings = []
    if fd_path.exists():
        fetched_at = fetched_at_of(fd_path)
        fd = parse_fd_fixtures(fd_path.read_bytes())
        if "Max>2.5" not in fd.columns:
            warnings.append("registro consigli: fixtures.csv senza quote over/under, nessun nuovo consiglio")
        else:
            new = candidates(fd, now, fetched_at)
            seen = set(map(tuple, reg[KEYS].astype(str).to_numpy()))
            new = new[[tuple(k) not in seen for k in new[KEYS].astype(str).to_numpy()]]
            if len(new):
                reg = pd.concat([reg, new], ignore_index=True) if len(reg) else new
                print(f"registro consigli: {len(new)} nuovi consigli over/under")
    reg = settle(reg, matches, now)
    for c in ["fair_prob", "edge", "profit"]:   # stable text: re-reading floats must not change the file
        reg[c] = pd.to_numeric(reg[c]).round(6)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(reg[COLUMNS], path)
    return warnings


def summary(reg: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Results of settled tips (1 unit each): all, first and second division, with a week-bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    done = reg[reg.status.isin(["vinta", "persa"])]
    weeks = pd.to_datetime(done.date).dt.to_period("W-SUN").astype(str)
    rows = []
    for label, mask in [("tutti", done.tier > 0), ("prima divisione", done.tier == 1),
                        ("seconda divisione", done.tier == 2)]:
        p = done.profit[mask].to_numpy(dtype=float)
        lo, hi = _roi_ci(p, weeks[mask].to_numpy(), n_boot, rng)
        rows.append({"group": label, "bets": len(p), "wins": int((done.status[mask] == "vinta").sum()),
                     "profit": p.sum(), "roi": p.mean() if len(p) else np.nan, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def rule_backtest(n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """The frozen rule on the historical Friday odds: calibration, test and current season so far."""
    rows = []
    for div in DIVS:
        for season in SEASONS:
            raw = parse_results_csv(results_path(season, div).read_bytes()).dropna(subset=["FTHG", "FTAG"])
            e = ou_edges(raw)
            total = pd.to_numeric(raw.FTHG) + pd.to_numeric(raw.FTAG)
            won = np.where(e.pick == "Over", total > 2.5, total < 2.5)
            rows.append(pd.DataFrame({
                "tier": TIER[div], "season": season, "date": parse_dates(raw.Date).dt.strftime("%Y-%m-%d"),
                "edge": e.edge, "odds_best": e.odds_best,
                "status": np.where(won, "vinta", "persa"), "profit": np.where(won, e.odds_best - 1, -1.0),
            }))
    hist = pd.concat(rows, ignore_index=True)
    hist = hist[(hist.edge > THRESHOLD) & hist.odds_best.between(1.01, 50)]
    out = []
    for label, seasons in [("calibrazione 2015/16-2023/24", CAL_SEASONS), ("verifica 2024/25-2025/26", TEST_SEASONS),
                           ("2026/27 finora", ["2627"])]:
        out.append(summary(hist[hist.season.isin(seasons)], n_boot, seed).assign(period=label))
    return pd.concat(out, ignore_index=True)[["period", "group", "bets", "wins", "profit", "roi", "ci_low", "ci_high"]]


def run_rule_backtest() -> None:
    res = rule_backtest()
    BACKTEST_OUT.parent.mkdir(exist_ok=True)
    write_csv_atomic(res, BACKTEST_OUT)
    pd.set_option("display.width", 160)
    print(f"regola {RULE_VERSION}: over/under 2.5, quota migliore oltre il {100 * THRESHOLD:.0f}% sopra la media")
    print(res.round(4).to_string(index=False))
