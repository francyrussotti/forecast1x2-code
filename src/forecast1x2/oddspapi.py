"""Quote di Planetwin365, Sisal/Snai e Bet365 da OddsPapi (API v4, piano gratuito), confrontate con le quote giuste.

Facoltativo: senza chiave (variabile ODDSPAPI_API_KEY o file KEY_FILE, fuori dal progetto) non fa nulla.
Pochi request:
  - al primo uso e ogni 30 giorni, 3 request di riferimento (bookmaker, campionati, codici dei mercati) in REFERENCE;
  - poi, al massimo ogni MIN_INTERVAL, 2 request per bookmaker (/odds-by-tournaments accetta un solo bookmaker
    e al massimo 5 campionati), più 1 per i nomi delle squadre solo quando compare una squadra nuova.
Il contatore mensile in STATE si ferma a MONTHLY_CAP (il piano gratuito dichiara 250 al mese).

Le partite di OddsPapi si abbinano alle nostre per campionato, orario (±3 ore) e somiglianza dei nomi.
"""
import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import ROOT, DataError, write_atomic, write_csv_atomic

BASE_URL = "https://api.oddspapi.io/v4"
# One request = one bookmaker (API rule). Many Italian brands share a platform: Planetwin365 = Goldbet =
# Lottomatica = Betflag, Snai = Sisal = PokerStars; so these three cover the main different price lists.
BOOKMAKERS = {"planetwin365.it": "Planetwin365", "sisal.it": "Sisal/Snai", "bet365.it": "Bet365"}
BOOKMAKER = "planetwin365.it"   # the first one: kept for old callers
BOOKMAKER_NAME = "Planetwin365, Sisal/Snai e Bet365"
SOCCER = 10
KEY_ENV = "ODDSPAPI_API_KEY"
KEY_FILE = Path.home() / ".config" / "forecast1x2" / "oddspapi_key"
REFERENCE = ROOT / "data" / "oddspapi_reference.json"
STATE = ROOT / "data" / "oddspapi_state.json"
OUT = ROOT / "outputs" / "bookmaker_odds.csv"
MIN_INTERVAL = pd.Timedelta(hours=23)   # 3 bookmakers x 2 requests a day ≈ 180 a month, under MONTHLY_CAP
MONTHLY_CAP = 200
HORIZON = pd.Timedelta(days=7)
REFERENCE_MAX_AGE = pd.Timedelta(days=30)
MAX_KICKOFF_GAP = pd.Timedelta(hours=3)
MIN_NAME_SCORE = 0.45
COOLDOWN_S = 1.1   # the API asks for 1 s between calls to the same endpoint
MAX_TOURNAMENTS_PER_CALL = 5   # API limit of /odds-by-tournaments
OUT_COLS = ["league", "kickoff_utc", "home", "away", "bookmaker", "group", "market", "odds", "fetched_at"]

# our league code -> (OddsPapi categoryName, accepted tournamentName values)
TOURNAMENTS = {
    "I1": ("Italy", ["Serie A"]), "I2": ("Italy", ["Serie B"]),
    "E0": ("England", ["Premier League"]), "E1": ("England", ["Championship"]),
    "SP1": ("Spain", ["LaLiga", "La Liga", "Primera Division", "LaLiga EA Sports"]),
    "SP2": ("Spain", ["LaLiga 2", "La Liga 2", "LaLiga Hypermotion", "Segunda Division", "Segunda División"]),
    "D1": ("Germany", ["Bundesliga"]), "D2": ("Germany", ["2. Bundesliga", "2nd Bundesliga"]),
    "F1": ("France", ["Ligue 1"]), "F2": ("France", ["Ligue 2"]),
}


# ------------------------------------------------------------------ helpers

def get_key():
    key = os.environ.get(KEY_ENV, "").strip()
    if not key and KEY_FILE.exists():
        lines = KEY_FILE.read_text().strip().splitlines()
        key = lines[0].strip() if lines else ""
    return key or None


def _norm(s: str) -> str:
    """Lowercase ASCII name without punctuation, for loose comparisons."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


_DROP = {"fc", "ac", "as", "ssc", "sc", "cf", "ud", "rc", "sd", "cd", "afc", "us", "calcio", "club", "sv", "vfb",
         "vfl", "tsg", "bsc", "fk", "ss", "rcd", "sco", "ogc", "aj", "stade", "olympique", "1", "05", "04", "1899",
         "1846", "1907", "1910", "1913", "1919", "1929", "hsc", "ca", "real", "de", "la", "el"}
_EXPAND = {"man": "manchester", "utd": "united", "ath": "athletic", "ein": "eintracht", "nott m": "nottingham",
           "m gladbach": "monchengladbach", "st": "saint", "sp": "sporting", "wolves": "wolverhampton",
           "qpr": "queens park rangers", "psg": "paris saint germain", "paris sg": "paris saint germain",
           "sheff": "sheffield", "wed": "wednesday", "inter": "internazionale", "verona": "hellas verona"}


def _tokens(name: str) -> set:
    s = _norm(name)
    for k, v in _EXPAND.items():
        s = re.sub(rf"\b{k}\b", v, s)
    return {t for t in s.split() if t and t not in _DROP}


def name_score(a: str, b: str) -> float:
    """Similarity 0-1 between two team names written differently (e.g. 'Man City' / 'Manchester City')."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return SequenceMatcher(None, _norm(a), _norm(b)).ratio()
    common = len(ta & tb)   # extra words on one side lower the score: 'Real Madrid' beats 'Atletico Madrid'
    overlap = 0.7 * common / min(len(ta), len(tb)) + 0.3 * common / max(len(ta), len(tb))
    seq = SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    return max(overlap, seq)


# ------------------------------------------------------------------ client

class Client:
    """Counts every request in the state dict, refuses to go over MONTHLY_CAP, waits COOLDOWN_S between calls."""

    def __init__(self, key: str, state: dict, now: pd.Timestamp, get=requests.get, sleep=time.sleep):
        self.key, self.state, self.get_fn, self.sleep = key, state, get, sleep
        self.calls = 0
        month = now.strftime("%Y-%m")
        if state.get("month") != month:
            state.update(month=month, requests=0)

    def get(self, path: str, **params):
        if self.state["requests"] >= MONTHLY_CAP:
            raise DataError(f"OddsPapi: raggiunto il limite di {MONTHLY_CAP} richieste per questo mese")
        if self.calls:
            self.sleep(COOLDOWN_S)
        self.calls += 1
        self.state["requests"] += 1
        try:
            r = self.get_fn(BASE_URL + path, params={**params, "apiKey": self.key}, timeout=60)
        except requests.RequestException as e:
            raise DataError(f"OddsPapi {path}: richiesta fallita ({type(e).__name__})")
        if r.status_code != 200:   # message without the URL, which contains the key
            raise DataError(f"OddsPapi {path}: HTTP {r.status_code} {r.text[:200]}")
        return r.json()


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


# --------------------------------------------------------------- reference

def resolve_tournaments(rows: list) -> dict:
    """Our league code -> OddsPapi tournamentId, matching country and exact (normalised) league name."""
    out = {}
    for div, (country, names) in TOURNAMENTS.items():
        want = {_norm(n) for n in names}
        hits = [r for r in rows if _norm(r.get("categoryName", "")) == _norm(country)
                and _norm(r.get("tournamentName", "")) in want]
        if hits:
            out[div] = int(hits[0]["tournamentId"])
    return out


def market_key(market_type: str, period: str, handicap: float, outcome: str, market_name: str = ""):
    """(group, market) of markets.MARKETS for an OddsPapi outcome, or None if we do not price it."""
    h, o = f"{float(handicap):g}", str(outcome).strip()
    full = period in ("fulltime", "result")
    if o in ("Yes", "No") and (market_type == "bothteamsscore" or "both teams" in market_name.lower()):
        if full:
            return ("Goal/No goal", "Goal" if o == "Yes" else "No goal")
        return ("Primo tempo", "Goal 1T") if period == "p1" and o == "Yes" else None
    if market_type == "1x2" and float(handicap) == 0 and o in ("1", "X", "2"):
        if full:
            return ("1X2", o)
        return {"p1": ("Primo tempo", f"1X2 1T: {o}"), "p2": ("Secondo tempo", f"1X2 2T: {o}")}.get(period)
    if market_type == "totals" and o in ("Over", "Under"):
        if full:
            return ("Under/Over", f"{o} {h}")
        return {"p1": ("Primo tempo", f"{o} {h} 1T"), "p2": ("Secondo tempo", f"{o} {h} 2T")}.get(period)
    if not full:
        return None
    if market_type == "drawnobet" and o in ("1", "2"):
        return ("Draw no bet", f"{o} (rimborso se X)")
    if market_type == "oddeven" and o in ("Odd", "Even"):
        return ("Pari/Dispari", "Dispari" if o == "Odd" else "Pari")
    if market_type in ("teamtotals-team1", "teamtotals-team2") and o in ("Over", "Under"):
        return ("Under/Over squadra", f"{o} {h} {'casa' if market_type.endswith('1') else 'ospite'}")
    if market_type == "doublechance":
        dc = o.replace("/", "").replace(" ", "").upper()
        return ("Doppia chance", dc) if dc in ("1X", "X2", "12") else None
    return None


def outcome_map(markets: list) -> dict:
    """outcomeId -> [group, market] for the OddsPapi soccer markets we can price."""
    from .markets import MARKETS
    known = {(g, m) for g, m, _, _ in MARKETS}
    out = {}
    for mk in markets:
        if mk.get("playerProp") or mk.get("sportId", SOCCER) != SOCCER:
            continue
        for oc in mk.get("outcomes", []):
            k = market_key(mk.get("marketType", ""), mk.get("period", ""), mk.get("handicap", 0),
                           oc.get("outcomeName", ""), mk.get("marketName", ""))
            if k in known:
                out[str(oc["outcomeId"])] = list(k)
    return out


def refresh_reference(client: Client, now: pd.Timestamp, path: Path = REFERENCE, old: dict = None) -> dict:
    slugs = {b.get("slug") for b in client.get("/bookmakers")}
    books = [b for b in BOOKMAKERS if b in slugs]
    if not books:
        raise DataError(f"OddsPapi: nessuno dei bookmaker {list(BOOKMAKERS)} è disponibile per questa chiave")
    tournaments = resolve_tournaments(client.get("/tournaments", sportId=SOCCER))
    outcomes = outcome_map(client.get("/markets"))
    ref = {"bookmakers": books, "missing_bookmakers": sorted(set(BOOKMAKERS) - set(books)), "tournaments": tournaments,
           "missing_tournaments": sorted(set(TOURNAMENTS) - set(tournaments)), "outcomes": outcomes,
           "participants": (old or {}).get("participants", {}), "fetched_at": now.isoformat(timespec="seconds")}
    write_atomic(path, json.dumps(ref, indent=1, ensure_ascii=False))
    return ref


# ---------------------------------------------------------------- fixtures

def normalize_fixture(fx: dict, names: dict, bookmaker: str = BOOKMAKER) -> dict:
    """v4 fixture -> {fixtureId, tournamentId, start, home, away, prices {outcomeId: price}} (active prices only)."""
    prices = {}
    bo = (fx.get("bookmakerOdds") or {}).get(bookmaker) or {}
    if bo.get("bookmakerIsActive", True):
        for market in (bo.get("markets") or {}).values():
            for oid, oc in (market.get("outcomes") or {}).items():
                p = (oc.get("players") or {}).get("0") or {}
                if p.get("active", True) and p.get("price") and float(p["price"]) > 1:
                    prices[str(oid)] = float(p["price"])
    return {"fixtureId": fx["fixtureId"], "tournamentId": int(fx.get("tournamentId", -1)),
            "start": pd.Timestamp(fx["startTime"]).tz_convert("UTC") if pd.Timestamp(fx["startTime"]).tzinfo
            else pd.Timestamp(fx["startTime"], tz="UTC"),
            "home": names.get(str(fx.get("participant1Id")), ""), "away": names.get(str(fx.get("participant2Id")), ""),
            "prices": prices}


def match_fixtures(ours: pd.DataFrame, fixtures: list, tournaments: dict) -> list:
    """Pairs (index in ours, normalized fixture): same league, kickoff within MAX_KICKOFF_GAP, both names similar.

    One-to-one, best pairs first."""
    league_of = {v: k for k, v in tournaments.items()}
    cands = []
    for fx in fixtures:
        div = league_of.get(fx["tournamentId"])
        if div is None or not fx["home"] or not fx["away"]:
            continue
        for i, r in ours[ours.league == div].iterrows():
            gap = abs(pd.Timestamp(r.kickoff_utc) - fx["start"])
            if gap > MAX_KICKOFF_GAP:
                continue
            s1, s2 = name_score(r.home, fx["home"]), name_score(r.away, fx["away"])
            if min(s1, s2) < MIN_NAME_SCORE:
                continue
            cands.append((s1 + s2 - gap / pd.Timedelta(hours=12), i, fx["fixtureId"], fx))
    pairs, used_i, used_f = [], set(), set()
    for _, i, fid, fx in sorted(cands, key=lambda c: -c[0]):
        if i not in used_i and fid not in used_f:
            pairs.append((i, fx))
            used_i.add(i)
            used_f.add(fid)
    return pairs


# ------------------------------------------------------------------ update

def update_bookmaker_odds(now: pd.Timestamp = None, get=requests.get, fair_path: Path = None,
                          state_path: Path = STATE, reference_path: Path = REFERENCE, out: Path = OUT,
                          sleep=time.sleep, force: bool = False) -> list:
    """Download the BOOKMAKERS odds for the upcoming matches in outputs/fair_odds.csv. Returns warnings."""
    from .markets import FAIR_ODDS_OUT
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    key = get_key()
    if not key:
        print(f"quote {BOOKMAKER_NAME} disattivate: nessuna chiave ({KEY_ENV} o {KEY_FILE})")
        return []
    state = load_json(state_path, {})
    last = state.get("last_fetch")
    if not force and last and now - pd.Timestamp(last) < MIN_INTERVAL:
        print(f"quote {BOOKMAKER_NAME}: ultimo download {last}, il prossimo almeno {MIN_INTERVAL} dopo")
        return []
    fair_path = fair_path or FAIR_ODDS_OUT
    ours = pd.read_csv(fair_path) if fair_path.exists() else pd.DataFrame()
    if ours.empty:
        return []
    ours["kickoff_utc"] = pd.to_datetime(ours.kickoff_utc, utc=True)
    client = Client(key, state, now, get, sleep)
    warnings, raw = [], {}
    try:
        ref = load_json(reference_path, {})
        if not ref.get("fetched_at") or now - pd.Timestamp(ref["fetched_at"]) > REFERENCE_MAX_AGE:
            ref = refresh_reference(client, now, reference_path, ref)
        if not ref["tournaments"]:
            raise DataError("OddsPapi: nessuno dei nostri campionati trovato")
        tids = [str(t) for t in ref["tournaments"].values()]
        calls = -(-len(tids) // MAX_TOURNAMENTS_PER_CALL)
        books = [b for b in BOOKMAKERS if b not in ref.get("missing_bookmakers", [])]
        for book in books:
            if state["requests"] + calls + 1 > MONTHLY_CAP:   # never half a bookmaker; 1 kept for team names
                warnings.append(f"OddsPapi: richieste del mese quasi finite, saltato {BOOKMAKERS[book]}")
                continue
            raw[book] = []
            for k in range(0, len(tids), MAX_TOURNAMENTS_PER_CALL):
                raw[book] += client.get("/odds-by-tournaments", bookmaker=book,
                                        tournamentIds=",".join(tids[k:k + MAX_TOURNAMENTS_PER_CALL]))
        ids = {str(fx.get(k)) for lst in raw.values() for fx in lst for k in ("participant1Id", "participant2Id")}
        if ids - set(ref.get("participants", {})):   # a team we have never seen: refresh the names once
            try:
                names = client.get("/participants", sportId=SOCCER)
                ref["participants"] = {**ref.get("participants", {}), **{i: names[i] for i in ids if i in names}}
                write_atomic(reference_path, json.dumps(ref, indent=1, ensure_ascii=False))
            except DataError as e:   # keep the odds already downloaded; unknown teams stay unmatched
                warnings.append(f"OddsPapi: nomi delle squadre non aggiornati ({e})")
    finally:
        write_atomic(state_path, json.dumps(state, indent=1))   # count requests even when something fails
    rows, matched = [], {}
    for book, lst in raw.items():
        fixtures = [normalize_fixture(fx, ref["participants"], book) for fx in lst]
        fixtures = [f for f in fixtures if now < f["start"] <= now + HORIZON and f["prices"]]
        pairs = match_fixtures(ours, fixtures, ref["tournaments"])
        matched[book] = len(pairs)
        for i, fx in pairs:
            r = ours.loc[i]
            for oid, price in fx["prices"].items():
                k = ref["outcomes"].get(oid)
                if k:
                    rows.append({"league": r.league, "kickoff_utc": r.kickoff_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "home": r.home, "away": r.away, "bookmaker": book, "group": k[0],
                                 "market": k[1], "odds": price, "fetched_at": now.isoformat(timespec="seconds")})
    if not raw:   # nothing downloaded (monthly budget): keep the previous file and do not move last_fetch
        return warnings
    state["last_fetch"] = now.isoformat(timespec="seconds")
    write_atomic(state_path, json.dumps(state, indent=1))
    write_csv_atomic(pd.DataFrame(rows, columns=OUT_COLS), out)
    print(f"quote bookmaker: {len(rows)} quote; partite abbinate "
          + ", ".join(f"{BOOKMAKERS[b]} {n}" for b, n in matched.items())
          + f" (su {len(ours)} nostre); richieste questo mese: {state['requests']}/{MONTHLY_CAP}")
    if ref.get("missing_tournaments"):
        warnings.append(f"OddsPapi: campionati non trovati: {', '.join(ref['missing_tournaments'])}")
    if ref.get("missing_bookmakers"):
        warnings.append(f"OddsPapi: bookmaker non disponibili: {', '.join(ref['missing_bookmakers'])}")
    return warnings


def best_prices(bk: pd.DataFrame) -> pd.DataFrame:
    """For every match and market, the highest odds among the bookmakers (and which bookmaker offers it)."""
    if bk.empty:
        return bk
    return (bk.sort_values("odds", ascending=False)
              .drop_duplicates(["league", "home", "away", "group", "market"]).reset_index(drop=True))


# ----------------------------------------------------------------- values

def value_table(upcoming: pd.DataFrame, bk: pd.DataFrame, params: dict, err: dict, best: bool = True) -> pd.DataFrame:
    """Bookmaker prices next to their fair odds: value, quarter-Kelly stake (max 5%), and a verdict.

    best=True keeps, for every market, only the bookmaker with the highest odds (line shopping)."""
    from .markets import fair_odds_for
    if "bookmaker" not in bk.columns:
        bk = bk.assign(bookmaker=BOOKMAKER)
    if best:
        bk = best_prices(bk)
    rows = []
    for _, r in upcoming.iterrows():
        b = bk[(bk.league == r.league) & (bk.home == r.home) & (bk.away == r.away)]
        if b.empty:
            continue
        fo = fair_odds_for(r, params).merge(b[["group", "market", "odds", "bookmaker"]], on=["group", "market"])
        for f in fo.itertuples():
            value = f.odds * f.prob - 1
            e = abs(err.get((f.group, f.market), 0.0)) / 100 * f.odds
            suspicious = f.odds > 2 * f.fair_odds
            stake = 0.0 if suspicious else float(np.clip(0.25 * value / (f.odds - 1), 0, 0.05))
            verdict = ("quota sospetta" if suspicious else "nessun valore" if value <= 0
                       else "valore entro l'errore" if value <= e else "valore")
            rows.append({"league": r.league, "kickoff_utc": r.kickoff_utc, "home": r.home, "away": r.away,
                         "group": f.group, "market": f.market, "prob": f.prob, "fair_odds": f.fair_odds,
                         "odds": f.odds, "bookmaker": f.bookmaker, "value": value, "stake": stake,
                         "verdict": verdict})
    return pd.DataFrame(rows)
