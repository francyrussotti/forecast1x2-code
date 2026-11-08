import json

import pandas as pd
import pytest

from forecast1x2.data import (
    DataError, check_calendar, league_history, load_team_map, map_names, normalise_fixturedownload,
    normalise_openfootball, normalise_results, parse_dates, parse_fixturedownload,
    parse_openfootball, parse_results_csv, promoted_teams, upcoming_round,
)

HEADER = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365CH,B365CD,B365CA,AvgCH,AvgCD,AvgCA\n"


def test_csv_with_bom_and_odds_fallback():
    body = (HEADER
            + "I1,23/08/2025,18:30,Inter,Torino,5,0,H,1.3,5.5,10,1.31,5.4,9.5\n"
            + "I1,24/08/2025,20:45,Milan,Lecce,1,1,D,1.4,4.8,8,,,\n"
            + ",,,,,,,,,,,,,\n")
    df = parse_results_csv(b"\xef\xbb\xbf" + body.encode())
    assert list(df.columns)[0] == "Div"  # BOM stripped
    assert len(df) == 2                  # empty trailing row dropped
    out = normalise_results(df, "2526")
    assert out.date.tolist() == [pd.Timestamp("2025-08-23"), pd.Timestamp("2025-08-24")]
    assert out.odds_source.tolist() == ["avg_closing", "b365_closing"]
    assert out.open_source.isna().all() and out.open_h.isna().all()   # no early-odds columns in this file
    assert (out.tier == 1).all()
    assert out.loc[0, "odds_h"] == 1.31 and out.loc[1, "odds_h"] == 1.4


def test_csv_missing_columns_fails():
    with pytest.raises(DataError, match="colonne mancanti"):
        parse_results_csv(b"Div,Date,HomeTeam\nI1,01/01/2020,Inter\n")


def test_dates_both_formats_and_bad():
    s = parse_dates(pd.Series(["01/09/2020", "15/05/19"]))
    assert s.tolist() == [pd.Timestamp("2020-09-01"), pd.Timestamp("2019-05-15")]
    with pytest.raises(DataError):
        parse_dates(pd.Series(["2020-09-01"]))


TEAM_MAP = pd.DataFrame({
    "league": ["I1"] * 4,
    "source": ["fixturedownload", "fixturedownload", "openfootball", "openfootball"],
    "source_name": ["Internazionale", "Milan", "FC Internazionale Milano", "AC Milan"],
    "canonical": ["Inter", "Milan", "Inter", "Milan"],
})


def test_map_names_and_unknown_fails():
    assert map_names(pd.Series(["Internazionale"]), "I1", "fixturedownload", TEAM_MAP).tolist() == ["Inter"]
    with pytest.raises(DataError, match="non mappate"):
        map_names(pd.Series(["Inter Milan"]), "I1", "fixturedownload", TEAM_MAP)
    # a name mapped for another source must not leak
    with pytest.raises(DataError):
        map_names(pd.Series(["AC Milan"]), "I1", "fixturedownload", TEAM_MAP)


def test_versioned_team_map_has_no_duplicates():
    m = load_team_map()
    assert set(m.source) == {"fixturedownload", "openfootball", "sportmonks"}
    assert set(m.league) == {"I1", "E0", "SP1", "D1", "F1"}


def test_fixturedownload_parsing():
    raw = json.dumps([{"MatchNumber": 1, "RoundNumber": 7, "DateUtc": "2026-10-18 18:45:00Z",
                       "HomeTeam": "Internazionale", "AwayTeam": "Milan", "HomeTeamScore": None}]).encode()
    fx = normalise_fixturedownload(parse_fixturedownload(raw), "I1", TEAM_MAP)
    assert fx.loc[0, "home"] == "Inter" and fx.loc[0, "round"] == 7
    assert fx.loc[0, "kickoff_utc"] == pd.Timestamp("2026-10-18 18:45", tz="UTC")
    with pytest.raises(DataError):
        parse_fixturedownload(b'[{"Round": 1}]')


def test_openfootball_local_time_to_utc():
    raw = json.dumps({"matches": [{"round": "Matchday 7", "date": "2026-10-18", "time": "20:45",
                                   "team1": "FC Internazionale Milano", "team2": "AC Milan"}]}).encode()
    fx = normalise_openfootball(parse_openfootball(raw), "I1", TEAM_MAP)
    assert fx.loc[0, "round"] == 7
    assert fx.loc[0, "kickoff_utc"] == pd.Timestamp("2026-10-18 18:45", tz="UTC")  # CEST = UTC+2


def _fixtures(pairs):
    return pd.DataFrame([{"league": "I1", "round": r, "kickoff_utc": pd.Timestamp(t, tz="UTC"),
                          "home": h, "away": a} for r, t, h, a in pairs])


def test_check_calendar():
    ok = _fixtures([(1, "2026-08-01", "A", "B"), (1, "2026-08-01", "C", "D"),
                    (2, "2026-08-08", "B", "C"), (2, "2026-08-08", "D", "A"),
                    (3, "2026-08-15", "A", "C"), (3, "2026-08-15", "B", "D"),
                    (4, "2026-08-22", "B", "A"), (4, "2026-08-22", "D", "C"),
                    (5, "2026-08-29", "C", "B"), (5, "2026-08-29", "A", "D"),
                    (6, "2026-09-05", "C", "A"), (6, "2026-09-05", "D", "B")])
    check_calendar(ok, {"A", "B", "C", "D"}, "test")
    bad = ok.copy()
    bad.loc[11, ["home", "away"]] = ["B", "D"]  # repeated pair, D-B missing
    with pytest.raises(DataError, match="ripetute"):
        check_calendar(bad, {"A", "B", "C", "D"}, "test")
    with pytest.raises(DataError, match="non compaiono"):
        check_calendar(ok, {"A", "B", "C"}, "test")


def test_upcoming_round_uses_source_score_when_results_lag():
    fx = _fixtures([(1, "2026-10-10", "A", "B"), (1, "2026-10-11", "C", "D"),
                    (2, "2026-10-17", "B", "C"), (2, "2026-10-18", "D", "A")])
    fx["source_played"] = [True, True, False, False]   # football-data has not published round 1 yet
    out = upcoming_round(fx, pd.DataFrame({"home": [], "away": []}), pd.Timestamp("2026-10-14", tz="UTC"))
    assert set(out["round"]) == {2} and (out.status == "giornata").all()


def test_promoted_teams_and_flags():
    rows = [("I1", 1, "2425", "A", "B"), ("I2", 2, "2425", "C", "D"),
            ("I1", 1, "2526", "A", "C"), ("I2", 2, "2526", "B", "D")]
    m = pd.DataFrame(rows, columns=["league", "tier", "season", "home", "away"])
    m["date"] = pd.Timestamp("2025-01-01")
    h = league_history(m, "I1")
    assert promoted_teams(h, "2526") == {"C"}       # C was in I2 in 2024/25, now in I1
    row = h[(h.season == "2526") & (h.tier == 1)].iloc[0]
    assert not row.home_promoted and row.away_promoted
    assert not h[h.tier == 2].home_promoted.any()     # flags only on top-flight matches


def test_upcoming_round_with_postponed_and_recovery():
    fx = _fixtures([
        (6, "2026-09-20", "A", "B"), (6, "2026-09-20", "E", "F"),
        (6, "2026-10-21", "C", "D"),                                 # C-D rescheduled to the future
        (7, "2026-09-27", "A", "C"), (7, "2026-09-27", "F", "E"),
        (7, "2026-10-10", "B", "D"),                                 # B-D date passed, not played
        (8, "2026-10-17", "D", "A"), (8, "2026-10-18", "B", "C"), (8, "2026-10-18", "E", "A"),
    ])
    played = pd.DataFrame({"home": ["A", "E", "A", "F"], "away": ["B", "F", "C", "E"]})
    out = upcoming_round(fx, played, pd.Timestamp("2026-10-14", tz="UTC"))
    status = dict(zip(zip(out.home, out.away), out.status))
    assert status == {("C", "D"): "recupero", ("B", "D"): "rinviata",
                      ("D", "A"): "giornata", ("B", "C"): "giornata", ("E", "A"): "giornata"}
