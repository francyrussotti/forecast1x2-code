import pandas as pd
import pytest

from forecast1x2 import data as datamod
from forecast1x2 import xg as xgmod
from forecast1x2.data import (DataError, PipelineBusy, check_complete, download, parse_results_csv,
                              pipeline_lock, usable_odds)

HEADER = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"


def _season(pairs):
    return pd.DataFrame(pairs, columns=["HomeTeam", "AwayTeam"])


def test_check_complete():
    teams = ["A", "B", "C", "D"]
    full = [(h, a) for h in teams for a in teams if h != a]   # 12 matches
    assert check_complete(_season(full), "I1", "2425") == 0
    assert check_complete(_season(full[:-1]), "I1", "2425") == 1        # tolerated
    with pytest.raises(DataError, match="mancano 3"):
        check_complete(_season(full[:-3]), "I1", "2425")


class _Resp:
    status_code = 200

    def __init__(self, text):
        self.content = text.encode()


def test_download_refuses_shorter_file(tmp_path, monkeypatch):
    dest = tmp_path / "2627_I1.csv"
    old = HEADER + "I1,01/09/2026,A,B,1,0,H\nI1,02/09/2026,C,D,0,0,D\n"
    dest.write_text(old)
    monkeypatch.setattr(datamod.requests, "get", lambda *a, **k: _Resp(HEADER + "I1,01/09/2026,A,B,1,0,H\n"))
    with pytest.raises(DataError, match="meno delle 2"):
        download("https://example.test/x.csv", dest, parse_results_csv, min_rows=2)
    assert dest.read_text() == old   # previous file kept


def test_pipeline_lock_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(datamod, "LOCK_PATH", tmp_path / ".lock")
    with pipeline_lock():
        with pytest.raises(PipelineBusy):
            with pipeline_lock():
                pass
    with pipeline_lock():   # released after the first block
        pass


def test_usable_odds_filters_stale_and_mismatched():
    now = pd.Timestamp("2026-10-16 12:00", tz="UTC")
    fixtures = pd.DataFrame({"league": ["I1", "I1", "I1"], "home": ["A", "C", "E"], "away": ["B", "D", "F"],
                             "kickoff_utc": pd.to_datetime(["2026-10-17 18:45", "2026-10-18 13:00",
                                                            "2026-10-18 16:00"], utc=True)})
    odds = pd.DataFrame({"league": ["I1", "I1", "I1"], "home": ["A", "C", "E"], "away": ["B", "D", "F"],
                         "date": pd.to_datetime(["2026-10-17", "2026-10-18", "2026-04-12"]),   # E-F: last season's odds
                         "open_h": [2.0, 2.5, 1.8], "open_d": [3.3, 3.2, 3.5], "open_a": [3.8, 2.9, 4.5],
                         "fetched_at": pd.to_datetime(["2026-10-16 10:00", "2026-10-01 10:00",
                                                       "2026-10-16 10:00"], utc=True)})   # C-D: fetched too long ago
    out = usable_odds(odds, fixtures, now)
    assert list(zip(out.home, out.away)) == [("A", "B")]


def test_xg_closed_season_without_data_fails(tmp_path, monkeypatch):
    monkeypatch.setenv(xgmod.TOKEN_ENV, "test-token")
    monkeypatch.setattr(xgmod, "RAW", tmp_path)
    monkeypatch.setattr(xgmod, "PROCESSED", tmp_path)
    monkeypatch.setattr(xgmod, "XG_SEASONS", ["2425", "2627"])
    monkeypatch.setattr(xgmod, "fetch_season", lambda token, season: [])   # e.g. unexpected response format
    with pytest.raises(DataError, match="2425: nessuna partita con xG"):
        xgmod.update_xg()
