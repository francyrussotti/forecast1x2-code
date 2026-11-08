import json

import pandas as pd
import pytest

from forecast1x2 import oddspapi as op
from forecast1x2.data import DataError

NOW = pd.Timestamp("2026-10-23 12:00", tz="UTC")
KICK = pd.Timestamp("2026-10-24 14:00", tz="UTC")

TOURNAMENTS = [
    {"tournamentId": 23, "tournamentName": "Serie A", "categoryName": "Italy"},
    {"tournamentId": 18, "tournamentName": "Serie A Women", "categoryName": "Italy"},
    {"tournamentId": 35, "tournamentName": "Bundesliga", "categoryName": "Germany"},
    {"tournamentId": 44, "tournamentName": "2. Bundesliga", "categoryName": "Germany"},
    {"tournamentId": 8, "tournamentName": "LaLiga", "categoryName": "Spain"},
    {"tournamentId": 99, "tournamentName": "Bundesliga", "categoryName": "Austria"},
    {"tournamentId": 17, "tournamentName": "Premier League", "categoryName": "England"},
    {"tournamentId": 34, "tournamentName": "Ligue 1", "categoryName": "France"},
]
MARKETS = [   # v4 /markets: all sports, BTTS sometimes typed as "totals"
    {"marketId": 101, "sportId": 10, "marketType": "1x2", "marketName": "Full Time Result", "period": "fulltime",
     "handicap": 0, "playerProp": False, "outcomes": [{"outcomeId": 101, "outcomeName": "1"},
                                                     {"outcomeId": 102, "outcomeName": "X"},
                                                     {"outcomeId": 103, "outcomeName": "2"}]},
    {"marketId": 104, "sportId": 10, "marketType": "totals", "marketName": "Both Teams To Score", "period": "fulltime",
     "handicap": 0, "playerProp": False, "outcomes": [{"outcomeId": 104, "outcomeName": "Yes"},
                                                     {"outcomeId": 105, "outcomeName": "No"}]},
    {"marketId": 1010, "sportId": 10, "marketType": "totals", "marketName": "Over Under Full Time", "period": "fulltime",
     "handicap": 2.5, "playerProp": False, "outcomes": [{"outcomeId": 1010, "outcomeName": "Over"},
                                                       {"outcomeId": 1011, "outcomeName": "Under"}]},
    {"marketId": 1012, "sportId": 10, "marketType": "totals", "marketName": "Over Under Full Time", "period": "fulltime",
     "handicap": 2.25, "playerProp": False, "outcomes": [{"outcomeId": 1012, "outcomeName": "Over"},
                                                        {"outcomeId": 1013, "outcomeName": "Under"}]},
    {"marketId": 10208, "sportId": 10, "marketType": "1x2", "marketName": "First Half Result", "period": "p1",
     "handicap": 0, "playerProp": False, "outcomes": [{"outcomeId": 10208, "outcomeName": "1"},
                                                     {"outcomeId": 10209, "outcomeName": "X"},
                                                     {"outcomeId": 10210, "outcomeName": "2"}]},
    {"marketId": 10224, "sportId": 10, "marketType": "teamtotals-team1", "marketName": "Over Under Team 1",
     "period": "fulltime", "handicap": 0.5, "playerProp": False,
     "outcomes": [{"outcomeId": 10224, "outcomeName": "Over"}, {"outcomeId": 10225, "outcomeName": "Under"}]},
    {"marketId": 1500, "sportId": 10, "marketType": "anytimegoalscorer", "marketName": "Goalscorer", "period": "fulltime",
     "handicap": 0, "playerProp": True, "outcomes": [{"outcomeId": 1500, "outcomeName": "Yes"}]},
    {"marketId": 111, "sportId": 11, "marketType": "moneyline", "marketName": "Winner", "period": "result",
     "handicap": 0, "playerProp": False, "outcomes": [{"outcomeId": 111, "outcomeName": "1"}]},
]
NAMES = {"1": "Real Madrid CF", "2": "Valencia CF", "3": "Atletico Madrid", "4": "Sevilla FC",
         "5": "Inter Milano", "6": "AC Milan", "7": "Getafe CF", "8": "Girona FC"}


def v4_fixture(fid, tid, p1, p2, start=KICK, prices=None, active=True):
    markets = {}
    for oid, price in (prices or {}).items():
        mid = str(oid)
        markets.setdefault(mid, {"outcomes": {}, "bookmakerMarketId": "x"})
        markets[mid]["outcomes"][str(oid)] = {"players": {"0": {"active": active, "price": price,
                                                                "bookmakerOutcomeId": "x", "limit": 100}}}
    return {"fixtureId": fid, "participant1Id": p1, "participant2Id": p2, "sportId": 10, "tournamentId": tid,
            "startTime": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "bookmakerOdds": {op.BOOKMAKER: {"bookmakerIsActive": True, "markets": markets}}}


ODDS = {
    "8": [v4_fixture("idA", 8, 1, 2, prices={101: 1.5, 102: 4.2, 103: 6.5, 1010: 1.7, 1011: 2.15, 1012: 1.8,
                                             10208: 2.1, 104: 1.6, 999: 3.0}),
          v4_fixture("idB", 8, 3, 4, prices={101: 1.9}),
          v4_fixture("idP", 8, 7, 8, start=NOW - pd.Timedelta(hours=1), prices={101: 2.0})],   # already started
    "23": [v4_fixture("idC", 23, 5, 6, start=KICK + pd.Timedelta(hours=1), prices={101: 2.3, 10224: 1.25})],
}


class Resp:
    def __init__(self, data, status=200):
        self.data, self.status_code, self.text = data, status, json.dumps(data)

    def json(self):
        return self.data


BOOK_FACTOR = {"planetwin365.it": 1.0, "sisal.it": 1.02, "bet365.it": 0.98}


class FakeApi:
    """Answers like OddsPapi v4 (shapes from the official docs); `books` are the bookmakers the key can see."""

    def __init__(self, fail=None, books=("planetwin365.it",)):
        self.calls, self.fail, self.books = [], fail, books

    def __call__(self, url, params, timeout):
        path = url.replace(op.BASE_URL, "")
        self.calls.append((path, params))
        if self.fail == path:
            return Resp({"error": 401, "message": "invalid apiKey"}, 401)
        if path == "/bookmakers":
            return Resp([{"bookmakerName": b, "slug": b} for b in self.books])
        if path == "/tournaments":
            return Resp(TOURNAMENTS)
        if path == "/markets":
            return Resp(MARKETS)
        if path == "/participants":
            return Resp(NAMES)
        if path == "/odds-by-tournaments":
            ids, book = params["tournamentIds"].split(","), params["bookmaker"]
            if len(ids) > 5 or "," in book:
                return Resp({"error": {"message": "Invalid parameter."}}, 400)
            out = []
            for fx in (fx for t in ids for fx in ODDS.get(t, [])):
                mk = json.loads(json.dumps(fx["bookmakerOdds"][op.BOOKMAKER]))
                for m in mk["markets"].values():
                    for oc in m["outcomes"].values():
                        oc["players"]["0"]["price"] = round(oc["players"]["0"]["price"] * BOOK_FACTOR[book], 3)
                out.append({**fx, "bookmakerOdds": {book: mk}})
            return Resp(out)
        raise AssertionError(path)


def ours():
    return pd.DataFrame({"league": ["SP1", "SP1", "I1"], "kickoff_utc": [KICK, KICK, KICK + pd.Timedelta(hours=1)],
                         "home": ["Real Madrid", "Ath Madrid", "Inter"], "away": ["Valencia", "Sevilla", "Milan"]})


def test_tournaments_need_country_and_exact_name():
    t = op.resolve_tournaments(TOURNAMENTS)
    assert t == {"I1": 23, "D1": 35, "D2": 44, "SP1": 8, "E0": 17, "F1": 34}   # no women, no Austria


def test_outcome_map_keeps_only_soccer_markets_we_price():
    m = op.outcome_map(MARKETS)
    assert m["101"] == ["1X2", "1"] and m["104"] == ["Goal/No goal", "Goal"]
    assert m["1011"] == ["Under/Over", "Under 2.5"] and m["10209"] == ["Primo tempo", "1X2 1T: X"]
    assert m["10224"] == ["Under/Over squadra", "Over 0.5 casa"]
    assert "1012" not in m and "1500" not in m and "111" not in m   # quarter line, player prop, basketball


def test_normalize_and_match():
    fx = [op.normalize_fixture(f, NAMES) for t in ODDS.values() for f in t]
    a = next(f for f in fx if f["fixtureId"] == "idA")
    assert a["home"] == "Real Madrid CF" and a["start"] == KICK and a["prices"]["1011"] == 2.15
    pairs = op.match_fixtures(ours(), fx, op.resolve_tournaments(TOURNAMENTS))
    got = {ours().loc[i, "home"]: f["fixtureId"] for i, f in pairs}
    assert got == {"Real Madrid": "idA", "Ath Madrid": "idB", "Inter": "idC"}


def test_matching_rejects_far_kickoff_wrong_teams_and_inactive_prices():
    t = op.resolve_tournaments(TOURNAMENTS)
    far = op.normalize_fixture(v4_fixture("idA", 8, 1, 2, start=KICK + pd.Timedelta(hours=5), prices={101: 2}), NAMES)
    wrong = op.normalize_fixture(v4_fixture("idZ", 8, 7, 8, prices={101: 2}), NAMES)
    assert op.match_fixtures(ours(), [far, wrong], t) == []
    off = op.normalize_fixture(v4_fixture("idA", 8, 1, 2, prices={101: 2}, active=False), NAMES)
    assert off["prices"] == {}


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setenv(op.KEY_ENV, "SECRETKEY")
    fair = tmp_path / "fair_odds.csv"
    o = ours()
    o.assign(kickoff_utc=o.kickoff_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_csv(fair, index=False)
    return {"fair_path": fair, "state_path": tmp_path / "state.json", "reference_path": tmp_path / "ref.json",
            "out": tmp_path / "bk.csv", "sleep": lambda s: None}


def test_update_downloads_matches_and_counts_requests(paths):
    api = FakeApi()
    warnings = op.update_bookmaker_odds(NOW, api, **paths)
    assert warnings == ["OddsPapi: campionati non trovati: E1, F2, I2, SP2",
                        "OddsPapi: bookmaker non disponibili: bet365.it, sisal.it"]
    assert [c[0] for c in api.calls] == ["/bookmakers", "/tournaments", "/markets", "/odds-by-tournaments",
                                         "/odds-by-tournaments", "/participants"]   # 6 leagues -> 2 calls of max 5
    assert all(c[1]["apiKey"] == "SECRETKEY" and len(c[1].get("tournamentIds", "").split(",")) <= 5 for c in api.calls)
    bk = pd.read_csv(paths["out"]).set_index(["home", "group", "market"])
    assert bk.loc[("Real Madrid", "1X2", "1"), "odds"] == 1.5
    assert bk.loc[("Real Madrid", "Under/Over", "Under 2.5"), "odds"] == 2.15
    assert bk.loc[("Real Madrid", "Goal/No goal", "Goal"), "odds"] == 1.6
    assert bk.loc[("Inter", "Under/Over squadra", "Over 0.5 casa"), "odds"] == 1.25
    assert len(bk) == 10   # 7 + 1 + 2: 999 (unknown), 1012 (quarter line) and the started match dropped
    state = json.loads(paths["state_path"].read_text())
    assert state["requests"] == 6 and state["month"] == "2026-10"

    api2 = FakeApi()   # 13 hours later: nothing is requested (one download a day)
    op.update_bookmaker_odds(NOW + pd.Timedelta(hours=13), api2, **paths)
    assert api2.calls == []
    api3 = FakeApi()   # 24 hours later: reference and team names cached, only the odds
    op.update_bookmaker_odds(NOW + pd.Timedelta(hours=24), api3, **paths)
    assert [c[0] for c in api3.calls] == ["/odds-by-tournaments", "/odds-by-tournaments"]
    assert json.loads(paths["state_path"].read_text())["requests"] == 8


def test_three_bookmakers_and_best_price(paths):
    api = FakeApi(books=tuple(op.BOOKMAKERS))
    assert op.update_bookmaker_odds(NOW, api, **paths) == ["OddsPapi: campionati non trovati: E1, F2, I2, SP2"]
    odds_calls = [c[1]["bookmaker"] for c in api.calls if c[0] == "/odds-by-tournaments"]
    assert odds_calls == ["planetwin365.it"] * 2 + ["sisal.it"] * 2 + ["bet365.it"] * 2   # one bookmaker per call
    bk = pd.read_csv(paths["out"])
    assert bk.groupby("bookmaker").size().to_dict() == {"bet365.it": 10, "planetwin365.it": 10, "sisal.it": 10}
    best = op.best_prices(bk).set_index(["home", "group", "market"])
    assert best.loc[("Real Madrid", "1X2", "1"), "bookmaker"] == "sisal.it"   # 1.5 x 1.02 is the highest
    assert best.loc[("Real Madrid", "1X2", "1"), "odds"] == pytest.approx(1.53)
    assert len(best) == 10


def test_budget_skips_whole_bookmakers(paths):
    paths["state_path"].write_text(json.dumps({"month": "2026-10", "requests": op.MONTHLY_CAP - 7}))
    api = FakeApi(books=tuple(op.BOOKMAKERS))
    warnings = op.update_bookmaker_odds(NOW, api, **paths)   # 3 reference + 2 odds = 5; Sisal would need 2 more
    assert "saltato Sisal/Snai" in " ".join(warnings) and "saltato Bet365" in " ".join(warnings)
    assert json.loads(paths["state_path"].read_text())["requests"] <= op.MONTHLY_CAP


def test_monthly_cap_and_errors_without_the_key(paths):
    paths["state_path"].write_text(json.dumps({"month": "2026-10", "requests": op.MONTHLY_CAP}))
    with pytest.raises(DataError, match="limite"):
        op.update_bookmaker_odds(NOW, FakeApi(), **paths)
    paths["state_path"].write_text(json.dumps({"month": "2026-09", "requests": op.MONTHLY_CAP}))   # new month
    with pytest.raises(DataError) as e:
        op.update_bookmaker_odds(NOW, FakeApi(fail="/bookmakers"), **paths)
    assert "401" in str(e.value) and "SECRETKEY" not in str(e.value)
    assert json.loads(paths["state_path"].read_text())["requests"] == 1   # a failed request still counts


def test_no_key_no_requests(paths, monkeypatch):
    monkeypatch.delenv(op.KEY_ENV)
    monkeypatch.setattr(op, "KEY_FILE", paths["out"].parent / "missing")
    api = FakeApi()
    assert op.update_bookmaker_odds(NOW, api, **paths) == [] and api.calls == []


def test_value_table():
    up = pd.DataFrame({"league": ["I1"], "kickoff_utc": ["2026-10-24T14:00:00Z"], "home": ["Roma"], "away": ["Lazio"],
                       "rate_home": [1.4], "rate_away": [1.1]})
    params = {"rho": -0.05, "nu": 1.1, "first_half_share": 0.44}
    from forecast1x2.markets import fair_odds_for
    fair = fair_odds_for(up.iloc[0], params).set_index(["group", "market"]).fair_odds
    bk = pd.DataFrame({"league": ["I1"] * 3, "home": ["Roma"] * 3, "away": ["Lazio"] * 3,
                       "group": ["1X2"] * 3, "market": ["1", "X", "2"],
                       "odds": [fair[("1X2", "1")] * 1.10, fair[("1X2", "X")] * 0.9, fair[("1X2", "2")] * 3]})
    v = op.value_table(up, bk, params, {("1X2", "1"): 1.0}).set_index("market")
    assert v.loc["1", "verdict"] == "valore" and v.loc["1", "value"] == pytest.approx(0.10)
    assert 0 < v.loc["1", "stake"] <= 0.05
    assert v.loc["X", "verdict"] == "nessun valore" and v.loc["X", "stake"] == 0
    assert v.loc["2", "verdict"] == "quota sospetta" and v.loc["2", "stake"] == 0
