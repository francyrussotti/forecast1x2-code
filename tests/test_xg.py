import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from forecast1x2 import xg as xgmod
from forecast1x2.data import DataError
from forecast1x2.model import N_GLOBAL, fit, neg_loglik, score_matrix
from forecast1x2.xg import attach_xg, fetch_range, parse_fixtures, season_of, update_xg

TEAM_MAP = pd.DataFrame({"league": ["I1", "I1"], "source": ["sportmonks", "sportmonks"],
                         "source_name": ["Inter", "AC Milan"], "canonical": ["Inter", "Milan"]})


def _fixture(fid, home, away, hxg=None, axg=None, league_id=384, when="2024-03-03 19:45:00"):
    expected = []
    if hxg is not None:
        expected = [{"id": 1, "fixture_id": fid, "type_id": 5304, "participant_id": 1,
                     "data": {"value": hxg}, "location": "home"},
                    {"id": 2, "fixture_id": fid, "type_id": 5304, "participant_id": 2,
                     "data": {"value": axg}, "location": "away"},
                    {"id": 3, "fixture_id": fid, "type_id": 5305, "participant_id": 1,
                     "data": {"value": 9.9}, "location": "home"}]   # other types are ignored
    return {"id": fid, "league_id": league_id, "starting_at": when,
            "participants": [{"id": 1, "name": home, "meta": {"location": "home"}},
                             {"id": 2, "name": away, "meta": {"location": "away"}}],
            "expected": expected}


def test_parse_fixtures():
    fx = [_fixture(1, "Inter", "AC Milan", 2.1, 0.7),
          _fixture(2, "AC Milan", "Inter"),                       # not played yet: no xG
          _fixture(3, "Arsenal", "Chelsea", 1.0, 1.0, league_id=999)]  # league we do not use
    df = parse_fixtures(fx, TEAM_MAP)
    assert len(df) == 1
    r = df.iloc[0]
    assert (r.league, r.season, r.home, r.away, r.hxg, r.axg) == ("I1", "2324", "Inter", "Milan", 2.1, 0.7)


def test_parse_fixtures_unmapped_team_fails():
    with pytest.raises(DataError, match="non mappate"):
        parse_fixtures([_fixture(1, "Internazionale FC", "AC Milan", 1.0, 1.0)], TEAM_MAP)


def test_parse_fixtures_without_location_fails():
    fx = _fixture(1, "Inter", "AC Milan", 1.0, 1.0)
    fx["participants"][0]["meta"] = {}
    with pytest.raises(DataError, match="meta.location"):
        parse_fixtures([fx], TEAM_MAP)


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


def test_fetch_range_follows_pages_and_hides_token(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(dict(params))
        if params.get("page", 1) == 1:   # shape of the real response (next_cursor is a full URL)
            return _Resp(200, {"data": [{"id": 1}], "pagination": {"has_more": True, "current_page": 1,
                                                                   "next_cursor": "https://api.example/...&cursor=x"}})
        return _Resp(200, {"data": [{"id": 2}], "pagination": {"has_more": False, "current_page": 2}})

    monkeypatch.setattr(xgmod.requests, "get", fake_get)
    out = fetch_range("SECRET", pd.Timestamp("2024-08-01").date(), pd.Timestamp("2024-08-31").date())
    assert [f["id"] for f in out] == [1, 2] and calls[1]["page"] == 2
    assert "cursor" not in calls[1] and calls[1]["per_page"] == 50
    assert "fixtureLeagues:8,384,564,82,301" == calls[0]["filters"]

    monkeypatch.setattr(xgmod.requests, "get", lambda url, params, timeout: _Resp(401, {"message": "Unauthenticated"}))
    with pytest.raises(DataError) as e:
        fetch_range("SECRET", pd.Timestamp("2024-08-01").date(), pd.Timestamp("2024-08-31").date())
    assert "SECRET" not in str(e.value)


def test_update_xg_without_token_is_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv(xgmod.TOKEN_ENV, raising=False)
    monkeypatch.setattr(xgmod, "TOKEN_FILE", tmp_path / "missing")
    assert update_xg() is None


def test_token_from_private_file(monkeypatch, tmp_path):
    monkeypatch.delenv(xgmod.TOKEN_ENV, raising=False)
    f = tmp_path / "sportmonks_token"
    f.write_text("  abc123  \n")
    monkeypatch.setattr(xgmod, "TOKEN_FILE", f)
    assert xgmod.get_token() == "abc123"
    monkeypatch.setenv(xgmod.TOKEN_ENV, "from-env")   # environment variable wins
    assert xgmod.get_token() == "from-env"


def test_parse_accepts_xgfixture_key():
    fx = _fixture(1, "Inter", "AC Milan", 2.1, 0.7)
    fx["xgfixture"] = fx.pop("expected")
    assert parse_fixtures([fx], TEAM_MAP).hxg.tolist() == [2.1]


def test_attach_xg():
    m = pd.DataFrame({"league": ["I1", "I1", "I2"], "tier": [1, 1, 2], "season": ["2324"] * 3,
                      "home": ["Inter", "Milan", "Pisa"], "away": ["Milan", "Inter", "Bari"]})
    x = pd.DataFrame({"league": ["I1"], "season": ["2324"], "home": ["Inter"], "away": ["Milan"],
                      "hxg": [2.1], "axg": [0.7]})
    out = attach_xg(m, x)
    assert out.hxg.tolist()[0] == 2.1 and np.isnan(out.hxg.tolist()[1]) and np.isnan(out.hxg.tolist()[2])
    assert attach_xg(m, None).hxg.isna().all()


def test_season_of():
    d = pd.Series(pd.to_datetime(["2024-08-20", "2025-05-30", "2026-07-01"]))
    assert season_of(d).tolist() == ["2425", "2425", "2627"]


def test_gradient_with_effective_goals():
    rng = np.random.default_rng(0)
    n, k = 5, 40
    hi, ai = rng.integers(0, n, k), rng.integers(0, n, k)
    x, y = rng.poisson(1.4, k).astype(float), rng.poisson(1.1, k).astype(float)
    xe, ye = 0.5 * x + 0.5 * rng.gamma(2, 0.7, k), 0.5 * y + 0.5 * rng.gamma(2, 0.6, k)
    args = (hi, ai, x, y, np.ones(k), n, 1.0, None, None, xe, ye)
    theta = rng.normal(0, 0.2, N_GLOBAL + 2 * (n - 1))
    theta[2] = -0.05
    assert check_grad(lambda t: neg_loglik(t, *args)[0], lambda t: neg_loglik(t, *args)[1], theta) < 1e-5


def test_xg_weight_one_fits_on_xg():
    # xG equal to the true scoring rates: with xg_weight=1 strengths are recovered almost exactly,
    # whereas goals alone (same matches) are much noisier
    rng = np.random.default_rng(5)
    teams = [f"T{i}" for i in range(8)]
    att = rng.normal(0, 0.3, 8); att -= att.mean()
    dfn = rng.normal(0, 0.25, 8); dfn -= dfn.mean()
    rows = []
    for r in range(2):
        for h in range(8):
            for a in range(8):
                if h != a:
                    lam, m = np.exp(0.1 + 0.25 + att[h] + dfn[a]), np.exp(0.1 + att[a] + dfn[h])
                    M = score_matrix(lam, m, -0.1, max_goals=15)
                    g = rng.choice(M.size, p=M.ravel())
                    rows.append({"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=r), "home": teams[h],
                                 "away": teams[a], "hg": g // 16, "ag": g % 16, "hxg": lam, "axg": m})
    df = pd.DataFrame(rows)
    ref = df.date.max() + pd.Timedelta(days=1)
    err = lambda p: np.abs(p.attack[[p.teams.index(t) for t in teams]] - att).max()
    assert err(fit(df, ref, 0.0, l2=0.0, xg_weight=1.0)) < 0.05
    assert err(fit(df, ref, 0.0, l2=0.0, xg_weight=0.0)) > err(fit(df, ref, 0.0, l2=0.0, xg_weight=1.0))
