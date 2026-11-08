import json

import pytest

from forecast1x2 import site
from forecast1x2.data import DataError


def test_encrypt_round_trip_and_wrong_password():
    blob = site.encrypt(b'{"a": 1}', "segreta", iterations=1000)
    assert json.loads(site.decrypt(blob, "segreta")) == {"a": 1}
    assert set(blob) == {"v", "kdf", "iter", "salt", "iv", "ct"}
    with pytest.raises(Exception):
        site.decrypt(blob, "sbagliata")
    other = site.encrypt(b'{"a": 1}', "segreta", iterations=1000)
    assert other["ct"] != blob["ct"]   # new salt and nonce every time


def test_build_requires_a_password(tmp_path, monkeypatch):
    monkeypatch.delenv(site.PASSWORD_ENV, raising=False)
    monkeypatch.setattr(site, "PASSWORD_FILE", tmp_path / "missing")
    with pytest.raises(DataError, match="password"):
        site.build_site(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_built_site_has_no_plain_data(tmp_path, monkeypatch):
    fake = {"updated": "2026-11-08T10:15:00+00:00", "matches": [{"home": "Squadra Segreta", "away": "Altra"}],
            "presets": [], "custom": [], "tracking": {}}
    monkeypatch.setattr(site, "site_data", lambda now=None: fake)
    monkeypatch.setattr(site, "ITERATIONS", 1000)
    digest = site.build_site(tmp_path, password="prova")
    files = {p.name for p in tmp_path.iterdir()}
    assert {"index.html", "data.json", "manifest.webmanifest", "apple-touch-icon.png", ".nojekyll",
            "version.txt"} <= files
    for p in tmp_path.iterdir():
        assert b"Squadra Segreta" not in p.read_bytes(), p.name
    assert json.loads(site.decrypt(json.loads((tmp_path / "data.json").read_text()), "prova")) == fake
    assert (tmp_path / "version.txt").read_text().strip() == digest
    assert site.build_site(tmp_path, password="altra") == digest   # same content, same version: nothing to publish


def test_reliability_in_plain_numbers():
    import pandas as pd
    picks = pd.DataFrame({"status": ["vinta"] * 7 + ["persa"] * 3 + ["aperta"],
                          "prob": [0.65] * 4 + [0.85] * 6 + [0.7],
                          "clv": [-0.02] * 10 + [None], "kickoff_utc": ["2026-10-24T14:00:00Z"] * 11})
    r = site.reliability(picks)
    assert r["n"] == 10 and r["actual"] == pytest.approx(0.7) and r["pred"] == pytest.approx(0.77)
    assert [b["label"] for b in r["bands"]] == ["60-70%", "oltre l'80%"]
    assert r["strength"] == "ancora poche partite" and r["clv"] == pytest.approx(-0.02)
    assert r["verdict"] in ("ottima", "buona", "da verificare") and r["since"] == "2026-10-24"
    assert site.reliability(picks.iloc[[10]]) == {"n": 0}


def test_icons_are_png():
    data = site.png(32)
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and b"IHDR" in data and data.endswith(b"IEND\xaeB`\x82")
