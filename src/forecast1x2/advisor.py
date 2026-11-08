"""Consigliere: la giocata più probabile di ogni partita e le schedine, con regole tratte dallo storico.

Regole (verifica 2024/25-2025/26 alle quote Bet365, vedi README «Schedina consigliata»):
  - la giocata più probabile di ogni partita perde in media solo il margine del bookmaker (−3,6%),
    quasi nulla con quote basse (1,01-1,30: +0,1%);
  - ogni partita in più in una multipla costa altri 3-4 punti (doppia −4%, tripla −10%, cinquina −16%);
  - il «valore» contro il consenso su un singolo bookmaker non ha retto (−50% sulla verifica).
Quindi: giocata più probabile, nei mercati dove il bookmaker trattiene meno, in singola o al massimo in doppia.
Per lo stesso incontro il regolamento prevede i mercati Combo: due esiti della stessa partita non vanno in multipla.
"""
from functools import reduce
from itertools import combinations

import numpy as np
import pandas as pd

MIN_ODDS = 1.20        # sotto, la vincita non ripaga il rischio di sbagliare
MAX_ODDS = 2.50        # sopra, il margine del bookmaker cresce (fascia 3-5: −12%, oltre 5: −29%)
MAX_COST = 0.06        # rendimento atteso non peggiore di −6% (il margine tipico dei mercati principali)
EXCLUDED_GROUPS = {"Draw no bet"}   # probabilità condizionata al non pareggio: non confrontabile con le altre
EXOTIC_GROUPS = ["Multigoal", "Combo DC + U/O", "Combo DC + Multigoal", "Combo 1X2 + U/O", "Combo DC + GG/NG"]
POOL = 15              # le schedine si cercano tra le migliori POOL giocate


MIN_PROB_CHEAP = 0.60  # la giocata «più conveniente» deve comunque essere probabile
# le schedine pronte: nome -> (partite, quota totale minima)
PRESETS = {"Singola (quota ≥ 1,50)": (1, 1.5), "Doppia prudente (≥ 1,50)": (2, 1.5),
           "Doppia (≥ 2,00)": (2, 2.0), "Tripla (≥ 3,00)": (3, 3.0)}
KEYS = ["league", "home", "away"]


def candidates(vt: pd.DataFrame, min_odds: float = MIN_ODDS, max_odds: float = MAX_ODDS) -> pd.DataFrame:
    """Bookmaker prices worth considering: sensible odds, not much worse than fair, no suspicious quotes."""
    if vt.empty:
        return vt
    c = vt[(vt.odds >= min_odds) & (vt.odds <= max_odds) & (vt.value >= -MAX_COST)
           & (vt.verdict != "quota sospetta") & ~vt.group.isin(EXCLUDED_GROUPS)]
    return c.copy()


def best_picks(vt: pd.DataFrame, min_odds: float = MIN_ODDS, max_odds: float = MAX_ODDS) -> pd.DataFrame:
    """One pick per match: the most probable candidate (ties: higher value)."""
    c = candidates(vt, min_odds, max_odds)
    if c.empty:
        return c
    c = c.sort_values(["prob", "value"], ascending=False)
    return c.drop_duplicates(KEYS).sort_values("prob", ascending=False).reset_index(drop=True)


def cheapest_picks(vt: pd.DataFrame, min_prob: float = MIN_PROB_CHEAP) -> pd.DataFrame:
    """One pick per match: the candidate with the smallest bookmaker margin among those with prob >= min_prob."""
    c = candidates(vt)
    c = c[c.prob >= min_prob] if not c.empty else c
    if c.empty:
        return c
    c = c.sort_values(["value", "prob"], ascending=False)
    return c.drop_duplicates(KEYS).sort_values("value", ascending=False).reset_index(drop=True)


def exotic_suggestions(fair: pd.DataFrame, lo: float = 1.30, hi: float = 2.50) -> pd.DataFrame:
    """Per match, the most probable Combo/Multigoal with fair odds in [lo, hi]: odds to ask the bookmaker for.

    fair: rows of markets.fair_odds_for for one match (group, market, prob, fair_odds)."""
    f = fair[fair.group.isin(EXOTIC_GROUPS) & fair.fair_odds.between(lo, hi)]
    return f.sort_values("prob", ascending=False).drop_duplicates("group").reset_index(drop=True)


def pareto_options(c: pd.DataFrame) -> list:
    """Per match, the candidates not beaten on both probability and odds (sorted by probability)."""
    out = []
    for _, g in c.sort_values(["prob", "odds"], ascending=False).groupby(KEYS, sort=False):
        keep, best_odds = [], 0.0
        for _, r in g.iterrows():
            if r.odds > best_odds:   # higher odds than every more probable option
                keep.append(r)
                best_odds = r.odds
        out.append(pd.DataFrame(keep))
    return sorted(out, key=lambda d: -d.prob.iloc[0])


POOL_BY_LEGS = {1: 80, 2: 30, 3: 16, 4: 11, 5: 9}   # matches searched, so that any search stays under ~1 s
OBJECTIVES = ("prob", "value")


def best_multiple(vt: pd.DataFrame, k: int, target_odds: float = 1.0, objective: str = "prob",
                  exclude: set = frozenset(), pool: int = None):
    """Best k-fold (different matches, one option each) with total odds >= target_odds.

    objective 'prob': most likely to win; 'value': highest expected return (probability x odds).
    Exact search over the Pareto options of the best `pool` matches (by most probable pick, or by best expected
    return for 'value'); matches in `exclude` (league, home, away) are skipped. Returns a dict or None."""
    if objective not in OBJECTIVES:
        raise ValueError(objective)
    options = [o for o in pareto_options(candidates(vt)) if tuple(o.iloc[0][KEYS]) not in exclude]
    if objective == "value":
        options.sort(key=lambda o: -(o.prob * o.odds).max())
    options = options[:pool or POOL_BY_LEGS.get(k, 8)]
    log_target = np.log(target_odds) - 1e-12
    best = None
    for idx in combinations(range(len(options)), k):
        sets = [options[i] for i in idx]
        lq = reduce(np.add, np.ix_(*[np.log(s.odds.to_numpy()) for s in sets]))   # every option combination
        lp = reduce(np.add, np.ix_(*[np.log(s.prob.to_numpy()) for s in sets]))
        score = np.where(lq >= log_target, lp if objective == "prob" else lp + lq, -np.inf)
        j = np.unravel_index(np.argmax(score), score.shape)
        if np.isfinite(score[j]) and (best is None or score[j] > best[0] + 1e-12):
            best = (score[j], [s.iloc[[jj]] for s, jj in zip(sets, np.atleast_1d(j))])
    if best is None:
        return None
    legs = pd.concat(best[1]).sort_values("kickoff_utc").reset_index(drop=True)
    q, p = float(legs.odds.prod()), float(legs.prob.prod())
    return {"legs": legs, "odds": q, "prob": p, "value": p * q - 1}


def alternative_slips(vt: pd.DataFrame, k: int, target_odds: float, n: int = 3, objective: str = "prob") -> list:
    """Up to n slips with no match in common, each the best of what is left."""
    out, used = [], set()
    for _ in range(n):
        m = best_multiple(vt, k, target_odds, objective, exclude=used)
        if m is None:
            break
        out.append(m)
        used |= {tuple(r) for r in m["legs"][KEYS].itertuples(index=False)}
    return out


def expected_outcome(prob: float, odds: float, stake: float = 10.0) -> dict:
    """What a bet means in money: win chance, payout if it wins, average result over many bets."""
    return {"stake": stake, "win_pct": 100 * prob, "payout": stake * odds, "avg_result": stake * (prob * odds - 1)}
