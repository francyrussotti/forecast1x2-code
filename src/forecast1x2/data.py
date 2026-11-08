"""Download, validation and normalisation of results, odds and fixtures."""
import fcntl
import io
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TEAM_MAP = ROOT / "data" / "team_names.csv"

# div -> (name, fixturedownload slug, openfootball code, local timezone)
LEAGUES = {
    "I1": ("Serie A", "serie-a", "it.1", "Europe/Rome"),
    "E0": ("Premier League", "epl", "en.1", "Europe/London"),
    "SP1": ("La Liga", "la-liga", "es.1", "Europe/Madrid"),
    "D1": ("Bundesliga", "bundesliga", "de.1", "Europe/Berlin"),
    "F1": ("Ligue 1", "ligue-1", "fr.1", "Europe/Paris"),
}
# second division of each league: used only to rate promoted teams
SECOND_DIV = {"I1": "I2", "E0": "E1", "SP1": "SP2", "D1": "D2", "F1": "F2"}
SEASONS = ["1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
CURRENT_SEASON = "2627"

RESULTS_URL = "https://football-data.co.uk/mmz4281/{season}/{div}.csv"
FIXTUREDOWNLOAD_URL = "https://fixturedownload.com/feed/json/{slug}-2026"
FD_FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"
OPENFOOTBALL_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/2026-27/{code}.json"

RESULT_COLS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
FD_KEYS = {"RoundNumber", "DateUtc", "HomeTeam", "AwayTeam", "HomeTeamScore"}
OF_KEYS = {"round", "date", "team1", "team2"}


class DataError(Exception):
    pass


class PipelineBusy(DataError):
    pass


LOCK_PATH = ROOT / "outputs" / ".pipeline.lock"
MAX_MISSING_CLOSED = 2   # closed season: tolerate a couple of abandoned/unrecorded matches, no more
# seasons known to be shorter than a full double round robin (checked on the data)
KNOWN_SHORT_SEASONS = {("F1", "1920"): "Ligue 1 interrotta per il COVID", ("F2", "1920"): "Ligue 2 interrotta per il COVID"}
ODDS_MAX_AGE = pd.Timedelta(days=7)


@contextmanager
def pipeline_lock():
    """One pipeline run at a time (dashboard button, CLI and scheduled job share this lock)."""
    LOCK_PATH.parent.mkdir(exist_ok=True)
    with open(LOCK_PATH, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PipelineBusy("un aggiornamento è già in corso: riprova tra qualche minuto")
        yield


def write_atomic(path: Path, text: str) -> None:
    """Write via a temporary file and rename, so readers never see a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    write_atomic(path, df.to_csv(index=False))


# ---------------------------------------------------------------- validation

def parse_results_csv(content: bytes) -> pd.DataFrame:
    """Parse a football-data CSV (with or without BOM); raise DataError if the format is unexpected."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        # some older football-data files are Windows-1252 (e.g. Serie B 2018/19: one stray 0xA0 in an odds column)
        text = content.decode("cp1252", errors="strict")
    df = pd.read_csv(io.StringIO(text), dtype=str)
    missing = [c for c in RESULT_COLS if c not in df.columns]
    if missing:
        raise DataError(f"colonne mancanti nel CSV: {missing}")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    if df.empty:
        raise DataError("CSV senza righe di partite")
    return df


def parse_fixturedownload(content: bytes) -> list:
    data = json.loads(content)
    if not isinstance(data, list) or not data:
        raise DataError("feed fixturedownload vuoto o non è una lista")
    missing = FD_KEYS - set(data[0])
    if missing:
        raise DataError(f"chiavi mancanti nel feed fixturedownload: {sorted(missing)}")
    return data


def parse_openfootball(content: bytes) -> list:
    data = json.loads(content)
    matches = data.get("matches") if isinstance(data, dict) else None
    if not matches:
        raise DataError("file openfootball senza 'matches'")
    missing = OF_KEYS - set(matches[0])
    if missing:
        raise DataError(f"chiavi mancanti nel file openfootball: {sorted(missing)}")
    return matches


def download(url: str, dest: Path, parse, min_rows: int = 0) -> None:
    """Download url, validate with parse(), and only then overwrite dest.

    min_rows: refuse a file with fewer rows than this (e.g. a current-season file shorter than the one we have).
    """
    try:
        r = requests.get(url, timeout=30, allow_redirects=True)
    except requests.RequestException as e:
        raise DataError(f"{url}: richiesta fallita ({e})")
    if r.status_code != 200:
        raise DataError(f"{url}: HTTP {r.status_code}")
    try:
        parsed = parse(r.content)
    except (DataError, ValueError) as e:
        raise DataError(f"{url}: formato non valido ({e})")
    if len(parsed) < min_rows:
        raise DataError(f"{url}: {len(parsed)} righe, meno delle {min_rows} già scaricate; tengo il file precedente")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(dest)


# ------------------------------------------------------------- normalisation

def parse_dates(s: pd.Series) -> pd.Series:
    """football-data dates: dd/mm/yyyy, or dd/mm/yy in older seasons."""
    s = s.str.strip()
    long = s.str.fullmatch(r"\d{2}/\d{2}/\d{4}")
    short = s.str.fullmatch(r"\d{2}/\d{2}/\d{2}")
    bad = s[~(long | short)]
    if not bad.empty:
        raise DataError(f"formato data non riconosciuto: {bad.iloc[0]!r}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out[long] = pd.to_datetime(s[long], format="%d/%m/%Y")
    out[short] = pd.to_datetime(s[short], format="%d/%m/%y")
    return out


def pick_odds(df: pd.DataFrame, closing: bool):
    """1X2 odds: market average, falling back row by row to Bet365. Returns (n x 3 array, source labels)."""
    c = "C" if closing else ""

    def cols(prefix):   # a missing column counts as missing odds
        return np.column_stack([pd.to_numeric(df[f"{prefix}{c}{k}"], errors="coerce").to_numpy(dtype=float)
                                if f"{prefix}{c}{k}" in df else np.full(len(df), np.nan) for k in "HDA"])

    avg, b365 = cols("Avg"), cols("B365")
    use_avg = ~np.isnan(avg).any(axis=1)
    use_b365 = ~use_avg & ~np.isnan(b365).any(axis=1)
    values = np.where(use_avg[:, None], avg, np.where(use_b365[:, None], b365, np.nan))
    kind = "closing" if closing else "open"
    source = np.where(use_avg, f"avg_{kind}", np.where(use_b365, f"b365_{kind}", None))
    return values, source


def normalise_results(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Select and normalise results and closing odds (market average, B365 fallback)."""
    out = pd.DataFrame({
        "league": df["Div"].str.strip(),
        "season": season,
        "date": parse_dates(df["Date"]),
        "home": df["HomeTeam"].str.strip(),
        "away": df["AwayTeam"].str.strip(),
        "hg": pd.to_numeric(df["FTHG"]),
        "ag": pd.to_numeric(df["FTAG"]),
        "result": df["FTR"].str.strip(),
    })
    out.insert(1, "tier", np.where(out.league.isin(SECOND_DIV.values()), 2, 1))

    # closing odds (AvgC*, fallback B365C*) and early odds collected before the round (Avg*, fallback B365*)
    for closing, cols, label in [(True, ["odds_h", "odds_d", "odds_a"], "closing"),
                                 (False, ["open_h", "open_d", "open_a"], "open")]:
        values, source = pick_odds(df, closing)
        for i, col in enumerate(cols):
            out[col] = values[:, i]
        out[f"{'odds' if closing else 'open'}_source"] = source
    bad = out[["hg", "ag"]].isna().any(axis=1) | ~out["result"].isin(["H", "D", "A"])
    if bad.any():
        desc = [f"{r.date.date()} {r.home}-{r.away}" for r in out[bad].itertuples()]
        if (out.tier[bad] == 1).any():
            raise DataError(f"{out.league.iloc[0]} {season}: risultato mancante o non valido: {desc}")
        print(f"ATTENZIONE {out.league.iloc[0]} {season}: scartate partite senza risultato: {desc}")
        out = out[~bad]
    return out


def load_team_map(path: Path = TEAM_MAP) -> pd.DataFrame:
    m = pd.read_csv(path, dtype=str)
    dup = m[m.duplicated(["league", "source", "source_name"], keep=False)]
    if not dup.empty:
        raise DataError(f"mappatura nomi duplicata: {dup.to_dict('records')}")
    return m


def map_names(names: pd.Series, league: str, source: str, team_map: pd.DataFrame) -> pd.Series:
    """Map source team names to football-data names. Fails on any unmapped name."""
    sub = team_map[(team_map.league == league) & (team_map.source == source)]
    lookup = dict(zip(sub.source_name, sub.canonical))
    unknown = sorted(set(names) - set(lookup))
    if unknown:
        raise DataError(
            f"squadre non mappate ({league}, fonte {source}): {unknown}. "
            f"Aggiungile a data/team_names.csv."
        )
    return names.map(lookup)


def normalise_fixturedownload(rows: list, league: str, team_map: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "league": league,
        "round": df["RoundNumber"].astype(int),
        "kickoff_utc": pd.to_datetime(df["DateUtc"], format="%Y-%m-%d %H:%M:%SZ", utc=True),
        "home": map_names(df["HomeTeam"], league, "fixturedownload", team_map),
        "away": map_names(df["AwayTeam"], league, "fixturedownload", team_map),
        "source": "fixturedownload",
        "source_played": df["HomeTeamScore"].notna(),
    })


def normalise_openfootball(rows: list, league: str, team_map: pd.DataFrame) -> pd.DataFrame:
    """openfootball times are local league time (checked against fixturedownload UTC times)."""
    df = pd.DataFrame(rows)
    rnd = df["round"].str.extract(r"(\d+)$")[0]
    if rnd.isna().any():
        raise DataError(f"openfootball {league}: round non numerico: {df['round'][rnd.isna()].iloc[0]!r}")
    time = df["time"].fillna("00:00") if "time" in df else "00:00"
    local = pd.to_datetime(df["date"] + " " + time, format="%Y-%m-%d %H:%M")
    return pd.DataFrame({
        "league": league,
        "round": rnd.astype(int),
        "kickoff_utc": local.dt.tz_localize(LEAGUES[league][3]).dt.tz_convert("UTC"),
        "home": map_names(df["team1"], league, "openfootball", team_map),
        "away": map_names(df["team2"], league, "openfootball", team_map),
        "source": "openfootball",
        "source_played": [isinstance(sc, dict) and "ft" in sc for sc in df.get("score", [None] * len(df))],
    })


# ------------------------------------------------------------------- update

def results_path(season: str, div: str) -> Path:
    return RAW / f"{season}_{div}.csv"


def check_calendar(fx: pd.DataFrame, known_teams: set, name: str) -> None:
    """Double round robin: no repeated home/away pair, every team n-1 home and n-1 away matches."""
    unknown = sorted((set(fx.home) | set(fx.away)) - known_teams)
    if unknown:
        raise DataError(f"{name}: nomi mappati che non compaiono in football-data {CURRENT_SEASON}: {unknown}")
    dup = fx[fx.duplicated(["home", "away"], keep=False)]
    if not dup.empty:
        raise DataError(f"{name}: coppie casa/trasferta ripetute nel calendario: "
                        f"{sorted(set(zip(dup.home, dup.away)))}")
    n = len(set(fx.home) | set(fx.away))
    counts = pd.concat([fx.home.value_counts(), fx.away.value_counts()], axis=1)
    if not (counts == n - 1).all().all():
        raise DataError(f"{name}: calendario incompleto, partite in casa/trasferta per squadra diverse da {n - 1}")


def check_complete(raw: pd.DataFrame, div: str, season: str) -> int:
    """Closed season: every team plays every other twice. Returns how many matches are missing."""
    n = len(set(raw.HomeTeam) | set(raw.AwayTeam))
    missing = n * (n - 1) - len(raw)
    if (div, season) in KNOWN_SHORT_SEASONS:
        return missing
    if missing > MAX_MISSING_CLOSED:
        raise DataError(f"{div} {season}: mancano {missing} partite su {n * (n - 1)}; "
                        "file incompleto, riscaricalo con `update-data --force`")
    return missing


def update_data(force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    warnings = []

    all_divs = list(LEAGUES) + list(SECOND_DIV.values())
    for div in all_divs:
        for season in SEASONS:
            dest = results_path(season, div)
            if season != CURRENT_SEASON and dest.exists() and not force:
                continue
            old_rows = len(parse_results_csv(dest.read_bytes())) if season == CURRENT_SEASON and dest.exists() else 0
            download(RESULTS_URL.format(season=season, div=div), dest, parse_results_csv, min_rows=old_rows)
            print(f"scaricato {dest.name}")

    frames = []
    for div in all_divs:
        for season in SEASONS:
            raw = parse_results_csv(results_path(season, div).read_bytes())
            if season != CURRENT_SEASON:
                missing = check_complete(raw, div, season)
                if missing:
                    print(f"nota {div} {season}: {missing} partite non presenti nel file (tollerate)")
            frames.append(normalise_results(raw, season))
    matches = pd.concat(frames, ignore_index=True).sort_values(["date", "league"], kind="stable")
    top = matches[matches.tier == 1]
    print(f"partite: {len(top)} prima divisione + {len(matches) - len(top)} seconda divisione")
    print(f"prima divisione - senza quote di chiusura: {top.odds_source.isna().sum()}, "
          f"fallback B365 chiusura: {(top.odds_source == 'b365_closing').sum()}, "
          f"senza quote di apertura: {top.open_source.isna().sum()}, "
          f"fallback B365 apertura: {(top.open_source == 'b365_open').sum()}")

    team_map = load_team_map()
    previous = load_fixtures() if (PROCESSED / "fixtures.csv").exists() else None
    fixtures = []
    for div, (name, slug, code, _) in LEAGUES.items():
        fd_dest = RAW / f"fixtures_fixturedownload_{div}.json"
        of_dest = RAW / f"fixtures_openfootball_{div}.json"
        current = set(matches.loc[(matches.league == div) & (matches.season == CURRENT_SEASON), "home"])
        try:
            download(FIXTUREDOWNLOAD_URL.format(slug=slug), fd_dest, parse_fixturedownload)
            fx = normalise_fixturedownload(parse_fixturedownload(fd_dest.read_bytes()), div, team_map)
            check_calendar(fx, current, name)
        except DataError as e:
            print(f"ATTENZIONE {name}: fonte primaria scartata ({e}); uso openfootball")
            try:
                download(OPENFOOTBALL_URL.format(code=code), of_dest, parse_openfootball)
                fx = normalise_openfootball(parse_openfootball(of_dest.read_bytes()), div, team_map)
                check_calendar(fx, current, name)
                warnings.append(f"{name}: calendario dalla fonte di riserva (openfootball). Motivo: {e}")
            except DataError as e2:
                if previous is None or (previous.league == div).sum() == 0:
                    raise
                fx = previous[previous.league == div].copy()
                warnings.append(f"{name}: entrambe le fonti del calendario non disponibili ({e2}); "
                                "uso il calendario dell'aggiornamento precedente")
        fixtures.append(fx)
        print(f"calendario {name}: {len(fx)} partite da {fx.source.iloc[0]}")

    warnings += update_upcoming_odds()
    from .xg import attach_xg, update_xg   # local import: xg imports this module
    try:
        xg = update_xg(force)
    except DataError as e:   # optional source: never blocks results and predictions
        xg_path = PROCESSED / "xg.csv"
        xg = pd.read_csv(xg_path, parse_dates=["date"], dtype={"season": str}) if xg_path.exists() else None
        warnings.append(f"xG non aggiornati ({e}); " + ("uso gli ultimi disponibili" if xg is not None else "modello sui soli gol"))
    matches = attach_xg(matches, xg)
    from .tips import update_register   # local import: tips imports this module
    try:   # trial tips register: optional, never blocks results and predictions
        warnings += update_register(matches)
    except (DataError, ValueError, KeyError) as e:
        warnings.append(f"registro consigli non aggiornato ({e})")
    from .markets import update_fair_odds   # local import: markets imports this module
    try:   # fair odds of all result markets: optional, never blocks results and predictions
        warnings += update_fair_odds()
    except (DataError, ValueError, KeyError) as e:
        warnings.append(f"quote giuste non aggiornate ({e})")
    from .oddspapi import update_bookmaker_odds   # local import: oddspapi imports this module
    try:   # Planetwin365 odds from OddsPapi: optional (needs a key), never blocks the rest
        warnings += update_bookmaker_odds()
    except (DataError, ValueError, KeyError, TypeError) as e:
        warnings.append(f"quote dei bookmaker non aggiornate ({e})")
    from .tracking import update_tracking   # local import: tracking imports this module
    try:   # advice register with results and CLV: optional, never blocks the rest
        warnings += update_tracking()
    except (DataError, ValueError, KeyError, TypeError) as e:
        warnings.append(f"registro dei consigli non aggiornato ({e})")
    for w in warnings:
        print(f"ATTENZIONE: {w}")
    write_csv_atomic(matches, PROCESSED / "matches.csv")
    write_csv_atomic(pd.concat(fixtures, ignore_index=True), PROCESSED / "fixtures.csv")
    meta = {"data_updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "last_match_date": {d: str(matches[matches.league == d].date.max().date()) for d in LEAGUES},
            "warnings": warnings}
    write_atomic(PROCESSED / "update_meta.json", json.dumps(meta, indent=2))


def parse_fd_fixtures(content: bytes) -> pd.DataFrame:
    """football-data fixtures.csv: may legitimately contain none of our leagues, but needs the expected header."""
    text = content.decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(text), dtype=str)
    missing = [c for c in ["Div", "Date", "HomeTeam", "AwayTeam", "AvgH", "AvgD", "AvgA"] if c not in df.columns]
    if missing:
        raise DataError(f"colonne mancanti in fixtures.csv: {missing}")
    return df


def fetched_at_path(path: Path) -> Path:
    return path.with_suffix(".fetched_at")


def fetched_at_of(path: Path) -> pd.Timestamp:
    """When a downloaded file was fetched: saved next to it, because a fresh git checkout resets file times."""
    side = fetched_at_path(path)
    if side.exists():
        return pd.Timestamp(side.read_text().strip())
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")


def update_upcoming_odds() -> list:
    """Early odds for upcoming matches (football-data publishes them a few days before each round).

    Optional input: if the file is unavailable, predictions fall back to the model alone.
    Returns warnings. Each row carries `fetched_at`, so stale odds can be discarded at prediction time.
    """
    dest = RAW / "fd_fixtures.csv"
    warnings = []
    try:
        download(FD_FIXTURES_URL, dest, parse_fd_fixtures)
        write_atomic(fetched_at_path(dest), pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))
    except DataError as e:
        warnings.append(f"quote pre-partita non aggiornate ({e}); uso l'ultimo file se recente")
    if not dest.exists():
        return warnings
    fetched_at = fetched_at_of(dest)
    df = parse_fd_fixtures(dest.read_bytes())
    df = df[df.Div.str.strip().isin(LEAGUES)]
    values, source = pick_odds(df, closing=False)
    out = pd.DataFrame({"league": df.Div.str.strip().to_numpy(), "date": parse_dates(df.Date).to_numpy(),
                        "home": df.HomeTeam.str.strip().to_numpy(), "away": df.AwayTeam.str.strip().to_numpy(),
                        "open_h": values[:, 0], "open_d": values[:, 1], "open_a": values[:, 2],
                        "open_source": source, "fetched_at": fetched_at})
    write_csv_atomic(out, PROCESSED / "upcoming_odds.csv")
    print(f"quote pre-partita per {len(out)} partite dei 5 campionati in fixtures.csv")
    return warnings


def usable_odds(odds: pd.DataFrame, fixtures: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Odds rows that match a fixture's date (±1 day) and were fetched within ODDS_MAX_AGE."""
    if odds.empty:
        return odds
    odds = odds.copy()
    fetched = pd.to_datetime(odds["fetched_at"], utc=True) if "fetched_at" in odds else pd.Series(pd.NaT, index=odds.index)
    odds = odds[(now - fetched) <= ODDS_MAX_AGE]
    fx = fixtures[["league", "home", "away", "kickoff_utc"]]
    m = odds.merge(fx, on=["league", "home", "away"], how="inner")
    gap = (m.kickoff_utc.dt.tz_localize(None).dt.normalize() - pd.to_datetime(m.date)).abs()
    return m[gap <= pd.Timedelta(days=1)].drop(columns="kickoff_utc")


def load_upcoming_odds() -> pd.DataFrame:
    path = PROCESSED / "upcoming_odds.csv"
    cols = ["league", "date", "home", "away", "open_h", "open_d", "open_a", "open_source"]
    return pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame(columns=cols)


def load_matches() -> pd.DataFrame:
    path = PROCESSED / "matches.csv"
    if not path.exists():
        raise DataError("dati mancanti: lancia prima `python -m forecast1x2 update-data`")
    return pd.read_csv(path, parse_dates=["date"], dtype={"season": str})


def load_fixtures() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "fixtures.csv")
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    return df


def upcoming_round(fixtures: pd.DataFrame, played: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Matches of the next round for one league, plus earlier unplayed matches.

    A fixture is played if (home, away) appears in the current-season football-data results,
    or if the calendar source itself already reports a score (football-data can lag a day or two).
    Next round = lowest RoundNumber with at least half of its matches unplayed.
    Unplayed matches of earlier rounds are flagged 'recupero' (future date) or
    'rinviata' (date already passed without a result).
    """
    keys = set(zip(played.home, played.away))
    fx = fixtures.copy()
    fx["played"] = [(h, a) in keys for h, a in zip(fx.home, fx.away)]
    if "source_played" in fx:
        fx["played"] |= fx.source_played.astype(bool)
    share_unplayed = (~fx.played).groupby(fx["round"]).mean()
    candidates = share_unplayed[share_unplayed >= 0.5]
    if candidates.empty:
        return fx.iloc[0:0].drop(columns="played").assign(status=pd.Series(dtype=str))
    nxt = candidates.index.min()
    out = fx[~fx.played & (fx["round"] <= nxt)].copy()
    out["status"] = "giornata"
    early = out["round"] < nxt
    out.loc[early & (out.kickoff_utc >= now), "status"] = "recupero"
    out.loc[early & (out.kickoff_utc < now), "status"] = "rinviata"
    return out.sort_values("kickoff_utc").drop(columns="played")


def promoted_teams(history: pd.DataFrame, season: str) -> set:
    """Top-flight teams of `season` that played the previous season in the second division.

    Uses only who is in which division (known from the calendar before the season starts).
    """
    i = SEASONS.index(season)
    if i == 0:
        return set()
    def teams(mask):
        return set(history.loc[mask, "home"]) | set(history.loc[mask, "away"])

    top = teams((history.season == season) & (history.tier == 1))
    second_prev = teams((history.season == SEASONS[i - 1]) & (history.tier == 2))
    return top & second_prev


def league_history(matches: pd.DataFrame, league: str) -> pd.DataFrame:
    """Top-flight matches of `league` plus its second division, with promoted-team flags on top-flight matches."""
    h = matches[matches.league.isin([league, SECOND_DIV[league]])].copy()
    promoted = {s: promoted_teams(h, s) for s in SEASONS}
    is_top = h.tier == 1
    h["home_promoted"] = is_top & [t in promoted[s] for t, s in zip(h.home, h.season)]
    h["away_promoted"] = is_top & [t in promoted[s] for t, s in zip(h.away, h.season)]
    return h
