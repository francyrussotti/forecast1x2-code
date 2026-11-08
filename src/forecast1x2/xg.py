"""Per-match expected goals from a licensed provider (Sportmonks API v3, xG add-on).

Enabled only when the environment variable SPORTMONKS_API_TOKEN is set; otherwise the
pipeline runs on goals alone. Written from the public v3 docs (fixtures/between endpoint,
include=participants;xGFixture, type_id 5304 = xG); not yet run against the live API.
"""
import json
import os
from datetime import date, timedelta

from pathlib import Path

import pandas as pd
import requests

from .data import CURRENT_SEASON, PROCESSED, RAW, DataError, load_team_map, map_names, write_csv_atomic

TOKEN_ENV = "SPORTMONKS_API_TOKEN"
# private file outside the project, so the scheduled job and the dashboard see the key too
TOKEN_FILE = Path.home() / ".config" / "forecast1x2" / "sportmonks_token"
BASE_URL = "https://api.sportmonks.com/v3/football/fixtures/between/{start}/{end}"
LEAGUE_IDS = {"E0": 8, "I1": 384, "SP1": 564, "D1": 82, "F1": 301}
XG_TYPE_ID = 5304
# Sportmonks has fixture xG for these leagues only from 2024/25 (checked Oct 2026: earlier seasons
# return no fixtures at all on our subscription)
XG_SEASONS = ["2425", "2526", "2627"]
MAX_RANGE_DAYS = 100   # API limit for fixtures/between


def season_bounds(season: str):
    """'2425' -> (2024-07-01, 2025-06-30)."""
    start = 2000 + int(season[:2])
    return date(start, 7, 1), date(start + 1, 6, 30)


def season_of(d: pd.Series) -> pd.Series:
    """Season code of each date, seasons starting on 1 July."""
    start = d.dt.year.where(d.dt.month >= 7, d.dt.year - 1)
    return (start % 100).map("{:02d}".format) + ((start + 1) % 100).map("{:02d}".format)


def fetch_range(token: str, start: date, end: date) -> list:
    """All top-five-league fixtures between start and end (inclusive), following pagination."""
    url = BASE_URL.format(start=start.isoformat(), end=end.isoformat())
    params = {"api_token": token, "include": "participants;xGFixture", "per_page": 50,
              "filters": "fixtureLeagues:" + ",".join(str(i) for i in LEAGUE_IDS.values())}
    out, page = [], 1
    while True:
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            raise DataError(f"Sportmonks {start}..{end}: richiesta fallita ({type(e).__name__})")
        if r.status_code != 200:   # message without the URL, which contains the token
            raise DataError(f"Sportmonks {start}..{end}: HTTP {r.status_code} {r.text[:200]}")
        body = r.json()
        if not isinstance(body.get("data", []), list):
            raise DataError(f"Sportmonks {start}..{end}: risposta senza lista 'data'")
        out.extend(body.get("data", []))
        pag = body.get("pagination") or {}
        if not pag.get("has_more"):
            return out
        # page-based pagination: the API's next_cursor is a full URL, and per_page is not allowed with it
        page += 1
        params["page"] = page


def fetch_season(token: str, season: str) -> list:
    start, end = season_bounds(season)
    fixtures, s = [], start
    while s <= end:
        e = min(s + timedelta(days=MAX_RANGE_DAYS - 1), end)
        fixtures.extend(fetch_range(token, s, e))
        s = e + timedelta(days=1)
    return fixtures


def parse_fixtures(fixtures: list, team_map: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture with xG for both teams; fixtures without xG (not played yet) are skipped."""
    div_of = {v: k for k, v in LEAGUE_IDS.items()}
    rows = []
    for fx in fixtures:
        div = div_of.get(fx.get("league_id"))
        if div is None:
            continue
        teams = {p.get("meta", {}).get("location"): p.get("name") for p in fx.get("participants", [])}
        if not {"home", "away"} <= set(teams):
            raise DataError(f"Sportmonks fixture {fx.get('id')}: partecipanti senza meta.location home/away")
        xg = {e.get("location"): e.get("data", {}).get("value")
              for e in (fx.get("expected") or fx.get("xgfixture") or []) if e.get("type_id") == XG_TYPE_ID}
        if xg.get("home") is None or xg.get("away") is None:
            continue
        rows.append({"league": div, "date": fx["starting_at"][:10], "home_src": teams["home"],
                     "away_src": teams["away"], "hxg": float(xg["home"]), "axg": float(xg["away"])})
    df = pd.DataFrame(rows, columns=["league", "date", "home_src", "away_src", "hxg", "axg"])
    df["date"] = pd.to_datetime(df["date"])
    out = []
    for div, g in df.groupby("league"):
        g = g.copy()
        g["home"] = map_names(g.home_src, div, "sportmonks", team_map)
        g["away"] = map_names(g.away_src, div, "sportmonks", team_map)
        out.append(g)
    df = pd.concat(out) if out else df.assign(home=[], away=[])
    df["season"] = season_of(df.date)
    return df[["league", "season", "date", "home", "away", "hxg", "axg"]]


def get_token():
    """API key from the environment variable, else from TOKEN_FILE (first line). None if neither is set."""
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip().splitlines()[0].strip() if TOKEN_FILE.read_text().strip() else ""
    return token or None


def update_xg(force: bool = False):
    """Download and normalise xG. Returns the xG table, or None if no token is configured."""
    token = get_token()
    if not token:
        print(f"xG disattivati: nessuna chiave ({TOKEN_ENV} o {TOKEN_FILE}); il modello usa solo i gol")
        return None
    for season in XG_SEASONS:
        dest = RAW / f"xg_sportmonks_{season}.json"
        if season != CURRENT_SEASON and dest.exists() and not force:
            continue
        fixtures = fetch_season(token, season)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(fixtures))
        tmp.replace(dest)
        print(f"xG {season}: {len(fixtures)} partite scaricate")
    team_map = load_team_map()
    parts = []
    for s in XG_SEASONS:
        part = parse_fixtures(json.loads((RAW / f"xg_sportmonks_{s}.json").read_text()), team_map)
        if s != CURRENT_SEASON and part.empty:   # e.g. response format differs from the docs
            raise DataError(f"xG {s}: nessuna partita con xG letta dalla risposta di Sportmonks; "
                            "controlla l'add-on xG e il formato (campo 'expected', type_id 5304)")
        parts.append(part)
    xg = pd.concat(parts, ignore_index=True)
    write_csv_atomic(xg, PROCESSED / "xg.csv")
    return xg


def attach_xg(matches: pd.DataFrame, xg) -> pd.DataFrame:
    """Add hxg/axg to top-flight matches (NaN where missing), keyed by league, season and teams."""
    if xg is None or xg.empty:
        return matches.assign(hxg=float("nan"), axg=float("nan"))
    if xg.duplicated(["league", "season", "home", "away"]).any():
        raise DataError("xG: partite duplicate per campionato/stagione/squadre")
    out = matches.merge(xg[["league", "season", "home", "away", "hxg", "axg"]],
                        on=["league", "season", "home", "away"], how="left")
    top = out[out.tier == 1]
    print("copertura xG per stagione (prima divisione): "
          + ", ".join(f"{s}: {c:.0%}" for s, c in top.groupby("season").hxg.apply(lambda v: v.notna().mean()).items()))
    return out
