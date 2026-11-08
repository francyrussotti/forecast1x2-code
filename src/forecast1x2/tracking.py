"""Registro dei consigli: giocate e schedine salvate prima delle partite, poi esito e CLV.

Ogni aggiornamento salva, per ogni partita con quote, la giocata più probabile e la più conveniente
(advisor.py) e, una volta al giorno, le schedine pronte (advisor.PRESETS). Una riga entra una volta sola,
almeno MIN_LEAD prima del calcio d'inizio, e non viene più modificata se non per l'esito.

Dopo la partita:
  - esito: dal risultato finale e del primo tempo (football-data), con le stesse funzioni dei mercati;
  - CLV (closing line value): probabilità del mercato alla chiusura (quote medie di chiusura senza margine,
    stesso motore delle quote giuste) × quota presa − 1. Se in media è positivo, le quote prese erano
    migliori di quelle finali: è il segnale più rapido (bastano poche settimane) che i consigli hanno valore.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (CURRENT_SEASON, LEAGUES, ROOT, SECOND_DIV, parse_dates, parse_results_csv, results_path,
                   write_csv_atomic)

PICKS = ROOT / "data" / "tips" / "picks_register.csv"
SLIPS = ROOT / "data" / "tips" / "slips_register.csv"
MIN_LEAD = pd.Timedelta(hours=1)
VOID_AFTER = pd.Timedelta(days=7)
UK = "Europe/London"   # football-data dates are UK dates
PICK_COLS = ["logged_at", "kind", "league", "kickoff_utc", "home", "away", "group", "market", "bookmaker", "odds",
             "prob", "value", "status", "profit", "close_prob", "clv"]
SLIP_COLS = ["logged_at", "preset", "legs", "odds", "prob", "value", "status", "profit", "clv"]
KEYS = ["league", "home", "away"]


def _load(path: Path, cols: list) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path, dtype={"status": str, "home": str, "away": str, "league": str})


def _uk_date(kickoff_utc) -> pd.Series:
    return pd.to_datetime(pd.Series(list(kickoff_utc)), utc=True).dt.tz_convert(UK).dt.strftime("%Y-%m-%d")


# ------------------------------------------------------------------ record

def record_picks(vt: pd.DataFrame, reg: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Add the most probable and the cheapest pick of every match not yet in the register (one per match and kind)."""
    from .advisor import best_picks, cheapest_picks
    new = pd.concat([best_picks(vt).assign(kind="più probabile"), cheapest_picks(vt).assign(kind="più conveniente")],
                    ignore_index=True)
    if new.empty:
        return reg
    new = new[pd.to_datetime(new.kickoff_utc, utc=True) - MIN_LEAD > now]
    seen = set(map(tuple, reg[["kind"] + KEYS].astype(str).to_numpy()))
    new = new[[tuple(k) not in seen for k in new[["kind"] + KEYS].astype(str).to_numpy()]]
    if new.empty:
        return reg
    new = new.assign(logged_at=now.isoformat(timespec="seconds"), status="aperta", profit=np.nan,
                     close_prob=np.nan, clv=np.nan)[PICK_COLS]
    return pd.concat([reg, new], ignore_index=True) if len(reg) else new.reset_index(drop=True)


def record_slips(vt: pd.DataFrame, reg: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Once a day, add the ready-made slips (advisor.PRESETS) whose matches all start at least MIN_LEAD later."""
    from .advisor import PRESETS, best_multiple
    soon = vt[pd.to_datetime(vt.kickoff_utc, utc=True) - MIN_LEAD > now] if len(vt) else vt
    today = now.strftime("%Y-%m-%d")
    done = {(p, str(t)[:10]) for p, t in zip(reg.preset, reg.logged_at)}
    rows = []
    for name, (k, target) in PRESETS.items():
        if (name, today) in done or soon.empty:
            continue
        m = best_multiple(soon, k, target)
        if m is None:
            continue
        legs = m["legs"][["league", "kickoff_utc", "home", "away", "group", "market", "bookmaker", "odds", "prob"]]
        rows.append({"logged_at": now.isoformat(timespec="seconds"), "preset": name,
                     "legs": json.dumps(legs.to_dict("records"), ensure_ascii=False), "odds": round(m["odds"], 4),
                     "prob": round(m["prob"], 5), "value": round(m["value"], 5), "status": "aperta",
                     "profit": np.nan, "clv": np.nan})
    if not rows:
        return reg
    new = pd.DataFrame(rows, columns=SLIP_COLS)
    return pd.concat([reg, new], ignore_index=True) if len(reg) else new


# ----------------------------------------------------------------- results

def results_table(seasons=None) -> pd.DataFrame:
    """Played matches with half-time goals and margin-free closing probabilities (1X2 and Over 2.5)."""
    from .markets import _num, devig
    seasons = seasons or [CURRENT_SEASON]
    rows = []
    for div in list(LEAGUES) + list(SECOND_DIV.values()):
        for s in seasons:
            path = results_path(s, div)
            if not path.exists():
                continue
            d = parse_results_csv(path.read_bytes()).dropna(subset=["FTHG", "FTAG"])
            h, dr, a = (_num(d, f"AvgC{c}").fillna(_num(d, f"B365C{c}")) for c in "HDA")
            o = _num(d, "AvgC>2.5").fillna(_num(d, "B365C>2.5"))
            u = _num(d, "AvgC<2.5").fillna(_num(d, "B365C<2.5"))
            p1, _, p2 = devig(h, dr, a)
            po, _ = devig(o, u)
            rows.append(pd.DataFrame({
                "league": div, "date": parse_dates(d.Date).dt.strftime("%Y-%m-%d").to_numpy(),
                "home": d.HomeTeam.str.strip().to_numpy(), "away": d.AwayTeam.str.strip().to_numpy(),
                "hg": pd.to_numeric(d.FTHG).to_numpy(), "ag": pd.to_numeric(d.FTAG).to_numpy(),
                "h1": pd.to_numeric(d.HTHG, errors="coerce").to_numpy(),
                "a1": pd.to_numeric(d.HTAG, errors="coerce").to_numpy(),
                "c1": np.asarray(p1), "c2": np.asarray(p2), "cover": np.asarray(po)}))
    cols = ["league", "date", "home", "away", "hg", "ag", "h1", "a1", "c1", "c2", "cover"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.concat(rows, ignore_index=True).drop_duplicates(["league", "date", "home", "away"])[cols]


def leg_results(legs: pd.DataFrame, res: pd.DataFrame, params: dict) -> pd.DataFrame:
    """For each leg (league, kickoff_utc, home, away, group, market): played, won (1/0; NaN if refunded)
    and the closing probability of that market (NaN without closing odds)."""
    from .markets import fit_rates, market_outcomes, market_probs, override_1x2
    legs = legs.reset_index(drop=True)
    legs = legs.assign(date=_uk_date(legs.kickoff_utc).to_numpy())
    m = legs.merge(res, on=["league", "date", "home", "away"], how="left")
    played = (m.hg.notna() & m.h1.notna()).to_numpy()
    out = pd.DataFrame({"played": played, "won": np.nan, "close_prob": np.nan})
    if not played.any():
        return out
    p = m[played]
    Y = market_outcomes(p.hg, p.ag, p.h1, p.a1)
    Y.index = p.index
    q = p[p[["c1", "c2", "cover"]].notna().all(axis=1)]
    P = None
    if len(q):
        nu = params.get("nu", 1.0)
        lh, la, _ = fit_rates(q.c1.to_numpy(), q.c2.to_numpy(), q.cover.to_numpy(), params["rho"], nu)
        P = override_1x2(market_probs(lh, la, params["rho"], params["first_half_share"], nu), q.c1, q.c2, params)
        P.index = q.index
    for idx, r in p.iterrows():
        key = (r.group, r.market)
        if key in Y.columns:
            out.loc[idx, "won"] = Y.loc[idx, key]
            if P is not None and idx in P.index:
                out.loc[idx, "close_prob"] = P.loc[idx, key]
    return out


def settle_picks(reg: pd.DataFrame, res: pd.DataFrame, params: dict, now: pd.Timestamp) -> pd.DataFrame:
    reg = reg.copy()
    open_ = reg.index[reg.status == "aperta"]
    if len(open_) == 0:
        return reg
    lr = leg_results(reg.loc[open_], res, params)
    lr.index = open_
    for idx in open_:
        r, x = reg.loc[idx], lr.loc[idx]
        if x.played:
            if np.isnan(x.won):
                reg.loc[idx, "status"], reg.loc[idx, "profit"] = "rimborsata", 0.0
            else:
                reg.loc[idx, "status"] = "vinta" if x.won else "persa"
                reg.loc[idx, "profit"] = float(r.odds) - 1 if x.won else -1.0
            if not np.isnan(x.close_prob):
                reg.loc[idx, "close_prob"] = round(float(x.close_prob), 5)
                reg.loc[idx, "clv"] = round(float(r.odds) * float(x.close_prob) - 1, 5)
        elif pd.Timestamp(r.kickoff_utc) + VOID_AFTER < now:
            reg.loc[idx, "status"], reg.loc[idx, "profit"] = "annullata", 0.0
    return reg


def settle_slips(reg: pd.DataFrame, res: pd.DataFrame, params: dict, now: pd.Timestamp) -> pd.DataFrame:
    """A slip wins if every leg wins; a refunded or void leg counts at odds 1.00 (as in the betting rules)."""
    reg = reg.copy()
    for idx in reg.index[reg.status == "aperta"]:
        legs = pd.DataFrame(json.loads(reg.loc[idx, "legs"]))
        lr = leg_results(legs, res, params)
        late = (pd.to_datetime(legs.kickoff_utc, utc=True) + VOID_AFTER < now).to_numpy()
        if not (lr.played.to_numpy() | late).all():
            continue
        odds = np.where(lr.played & lr.won.notna(), legs.odds, 1.0)
        if (lr.won == 0).any():
            reg.loc[idx, "status"], reg.loc[idx, "profit"] = "persa", -1.0
        else:
            reg.loc[idx, "status"], reg.loc[idx, "profit"] = "vinta", float(np.prod(odds)) - 1
        if lr.close_prob.notna().all():
            reg.loc[idx, "clv"] = round(float(np.prod(legs.odds.to_numpy() * lr.close_prob.to_numpy())) - 1, 5)
    return reg


# ----------------------------------------------------------------- summary

def summary(reg: pd.DataFrame, by: str = None, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Settled bets: count, wins, return and average CLV, with week-bootstrap 95% CIs."""
    from .evaluation import _roi_ci
    rng = np.random.default_rng(seed)
    done = reg[reg.status.isin(["vinta", "persa", "rimborsata"])].copy()
    when = done.kickoff_utc if "kickoff_utc" in done else done.logged_at
    done["week"] = pd.to_datetime(when, utc=True).dt.tz_convert(None).dt.to_period("W-SUN").astype(str)
    groups = [("tutte", done)] + (list(done.groupby(by)) if by else [])
    rows = []
    for name, g in groups:
        p = g.profit.to_numpy(dtype=float)
        lo, hi = _roi_ci(p, g.week.to_numpy(), n_boot, rng)
        c = g.dropna(subset=["clv"])
        clo, chi = _roi_ci(c.clv.to_numpy(dtype=float), c.week.to_numpy(), n_boot, rng)
        rows.append({"group": name, "bets": len(g), "wins": int((g.status == "vinta").sum()),
                     "roi": p.mean() if len(p) else np.nan, "roi_lo": lo, "roi_hi": hi,
                     "clv_n": len(c), "clv": c.clv.mean() if len(c) else np.nan, "clv_lo": clo, "clv_hi": chi})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ update

def update_tracking(now: pd.Timestamp = None, picks_path: Path = PICKS, slips_path: Path = SLIPS) -> list:
    """Record today's advice and settle what has been played. Returns warnings."""
    from .markets import FAIR_ODDS_OUT, load_params
    from .oddspapi import OUT as BOOKMAKER_ODDS, value_table
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    params = load_params()
    picks, slips = _load(picks_path, PICK_COLS), _load(slips_path, SLIP_COLS)
    if FAIR_ODDS_OUT.exists() and BOOKMAKER_ODDS.exists():
        up, bk = pd.read_csv(FAIR_ODDS_OUT), pd.read_csv(BOOKMAKER_ODDS)
        up = up[pd.to_datetime(up.kickoff_utc, utc=True) > now]
        if len(up) and len(bk):
            vt = value_table(up, bk, params, {})
            if len(vt):
                picks, slips = record_picks(vt, picks, now), record_slips(vt, slips, now)
    if (picks.status == "aperta").any() or (slips.status == "aperta").any():
        res = results_table()
        picks, slips = settle_picks(picks, res, params, now), settle_slips(slips, res, params, now)
    picks_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(picks[PICK_COLS], picks_path)
    write_csv_atomic(slips[SLIP_COLS], slips_path)
    print(f"registro consigli: {len(picks)} giocate ({(picks.status == 'aperta').sum()} aperte), "
          f"{len(slips)} schedine ({(slips.status == 'aperta').sum()} aperte)")
    return []
