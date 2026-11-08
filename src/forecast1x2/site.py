"""Sito statico per GitHub Pages, installabile su iPhone, con i dati cifrati da una password.

    python -m forecast1x2 sito          # costruisce site/ (serve SITE_PASSWORD o ~/.config/forecast1x2/site_password)

La pagina (site/index.html) è pubblica, ma i dati stanno in site/data.json cifrati con AES-GCM: la chiave si
ricava dalla password con PBKDF2-SHA256 e il browser li decifra solo dopo che l'utente l'ha scritta (WebCrypto).
Senza password il sito mostra solo la richiesta di password. I dati non vengono mai scritti in chiaro in site/.
"""
import base64
import hashlib
import json
import os
import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LEAGUES, ROOT, DataError, SECOND_DIV, write_atomic

SITE = ROOT / "site"
PASSWORD_ENV = "SITE_PASSWORD"
PASSWORD_FILE = Path.home() / ".config" / "forecast1x2" / "site_password"
ITERATIONS = 250_000
LEAGUE_NAMES = {**{d: v[0] for d, v in LEAGUES.items()}, "I2": "Serie B", "E1": "Championship",
                "SP2": "La Liga 2", "D2": "2. Bundesliga", "F2": "Ligue 2"}
SHOWN_GROUPS = ["1X2", "Doppia chance", "Under/Over", "Goal/No goal", "Multigoal", "Combo 1X2 + U/O",
                "Combo DC + U/O", "Combo DC + GG/NG", "Combo DC + Multigoal", "Primo tempo", "Under/Over squadra"]
PW_BOOK = "planetwin365.it"   # the bookmaker the user plays on: the default version of the advice
CUSTOM_SLIPS = [(1, 1.5), (1, 2.0), (2, 1.5), (2, 2.0), (2, 3.0), (3, 2.5), (3, 3.0), (3, 5.0), (4, 5.0), (5, 8.0)]
DISCLAIMER = ("Le previsioni non sono consigli di scommessa: nessun modo di giocare ha guadagnato in modo affidabile "
              "nello storico. Il gioco può causare dipendenza.")
HISTORY = [["Singola: la più probabile di ogni partita", "60%", "−3,6%"], ["…con quota 1,01-1,30", "83%", "+0,1%"],
           ["Doppia delle più probabili", "37%", "−4,4%"], ["Tripla", "21%", "−10,0%"],
           ["Quadrupla", "13%", "−11,6%"], ["Cinquina", "8%", "−15,9%"],
           ["«Valore» su un solo bookmaker (oltre +5%)", "10%", "−49,5%"]]


# ------------------------------------------------------------------ helpers

def get_password():
    pw = os.environ.get(PASSWORD_ENV, "").strip()
    if not pw and PASSWORD_FILE.exists():
        lines = PASSWORD_FILE.read_text().strip().splitlines()
        pw = lines[0].strip() if lines else ""
    return pw or None


def _f(x, nd=4):
    """JSON-friendly float (None for NaN)."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _read(path: Path, **kw):
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()


def encrypt(plaintext: bytes, password: str, salt: bytes = None, iv: bytes = None, iterations: int = ITERATIONS) -> dict:
    """AES-256-GCM with a PBKDF2-SHA256 key: the same scheme the page decrypts with WebCrypto."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, iv = salt or os.urandom(16), iv or os.urandom(12)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations, dklen=32)
    ct = AESGCM(key).encrypt(iv, plaintext, None)
    b64 = lambda b: base64.b64encode(b).decode()
    return {"v": 1, "kdf": "PBKDF2-SHA256", "iter": iterations, "salt": b64(salt), "iv": b64(iv), "ct": b64(ct)}


def decrypt(blob: dict, password: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    d = lambda k: base64.b64decode(blob[k])
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), d("salt"), blob["iter"], dklen=32)
    return AESGCM(key).decrypt(d("iv"), d("ct"), None)


def png(size: int) -> bytes:
    """App icon: green field with a white ball (pure Python, no image library)."""
    rows = []
    c, r = (size - 1) / 2, size * 0.30
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            d = ((x - c) ** 2 + (y - c) ** 2) ** 0.5
            if d < r:
                inner = d < r * 0.35 or (abs(x - c) < size * 0.02) or (abs(y - c) < size * 0.02)
                row += bytes((20, 60, 40) if inner else (255, 255, 255))
            else:
                row += bytes((22, 128, 76))
        rows.append(bytes(row))
    chunk = lambda t, b: struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


# --------------------------------------------------------------------- data

def _leg(r) -> dict:
    return {"league": LEAGUE_NAMES.get(r.league, r.league), "kickoff": r.kickoff_utc, "home": r.home, "away": r.away,
            "group": r.group, "market": r.market, "odds": _f(r.odds, 3), "bookmaker": r.bookmaker,
            "prob": _f(r.prob)}


def _slip(m) -> dict:
    return {"legs": [_leg(r) for r in m["legs"].itertuples()], "odds": _f(m["odds"], 3), "prob": _f(m["prob"]),
            "value": _f(m["value"])}


BANDS = [(0.0, 0.6, "sotto il 60%"), (0.6, 0.7, "60-70%"), (0.7, 0.8, "70-80%"), (0.8, 1.01, "oltre l'80%")]


def reliability(picks: pd.DataFrame, slips: pd.DataFrame = None) -> dict:
    """«When we say X%, it happens Y%»: predicted vs actual win rate of the settled advice, in plain words.

    gap = average distance (points) between predicted and actual, weighted by bets; the verdict gets stronger
    as more matches are settled, because the sample size is part of it."""
    done = picks[picks.status.isin(["vinta", "persa"])] if len(picks) else picks
    if len(done) == 0:
        return {"n": 0}
    y = (done.status == "vinta").to_numpy(float)
    p = done.prob.to_numpy(float)
    bands = []
    for lo, hi, label in BANDS:
        m = (p >= lo) & (p < hi)
        if m.sum():
            n, k = int(m.sum()), y[m].mean()
            half = 1.96 * np.sqrt(max(k * (1 - k), 0.01) / n)   # 95% interval of the actual rate
            bands.append({"label": label, "n": n, "pred": _f(p[m].mean()), "actual": _f(k), "lo": _f(max(0, k - half)),
                          "hi": _f(min(1, k + half))})
    gap = sum(abs(b["pred"] - b["actual"]) * b["n"] for b in bands) / len(done)
    verdict = "ottima" if gap < 0.03 else "buona" if gap < 0.06 else "da verificare"
    n = len(done)
    strength = ("ancora poche partite" if n < 100 else "indicativa" if n < 500 else "solida" if n < 2000
                else "molto solida")
    clv = done.clv.dropna().to_numpy(float) if "clv" in done else np.array([])
    out = {"n": n, "pred": _f(p.mean()), "actual": _f(y.mean()), "gap": _f(gap), "verdict": verdict,
           "strength": strength, "bands": bands, "clv": _f(clv.mean()) if len(clv) else None, "clv_n": int(len(clv)),
           "since": str(done.kickoff_utc.min())[:10]}
    if slips is not None and len(slips):
        s = slips[slips.status.isin(["vinta", "persa"])]
        if len(s):
            out["slips"] = {"n": int(len(s)), "pred": _f(s.prob.mean()), "actual": _f((s.status == "vinta").mean())}
    return out


def _summary(df: pd.DataFrame, by: str = None) -> list:
    from .tracking import summary
    if df.empty:
        return []
    s = summary(df, by)
    return [{"group": str(r.group), "bets": int(r.bets), "wins": int(r.wins), "roi": _f(r.roi), "roi_lo": _f(r.roi_lo),
             "roi_hi": _f(r.roi_hi), "clv": _f(r.clv), "clv_lo": _f(r.clv_lo), "clv_hi": _f(r.clv_hi)}
            for r in s.itertuples()]


def site_data(now: pd.Timestamp = None) -> dict:
    """Everything the page shows, as plain JSON (encrypted before writing)."""
    from . import advisor
    from .markets import FAIR_ODDS_OUT, fair_odds_for, load_params
    from .oddspapi import BOOKMAKERS, OUT as BOOKMAKER_ODDS, best_prices, value_table
    from .tips import load_register
    from .tracking import PICKS, SLIPS
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    outputs = ROOT / "outputs"
    meta = json.loads((outputs / "meta.json").read_text()) if (outputs / "meta.json").exists() else {}
    params = load_params()
    cal = _read(outputs / "markets_calibration.csv")
    err = {}
    if len(cal):
        t = cal[cal.period.str.startswith("verifica")]
        err = {(g, m): d for g, m, d in zip(t.group, t.market, t.diff_pp)}
    up, bk = _read(FAIR_ODDS_OUT), _read(BOOKMAKER_ODDS)
    if len(up):
        up = up[pd.to_datetime(up.kickoff_utc, utc=True) > now].sort_values("kickoff_utc")
    # two versions of every piece of advice: only Planetwin365 (the user's bookmaker) and the best of all
    key = lambda df: {(r.league, r.home, r.away): r for r in df.itertuples()} if len(df) else {}
    V = {}
    for name, b_ in {"pw": bk[bk.bookmaker == PW_BOOK] if len(bk) else bk, "all": bk}.items():
        vt_ = value_table(up, b_, params, err) if len(up) and len(b_) else pd.DataFrame()
        V[name] = {"vt": vt_, "best": best_prices(b_) if len(b_) else b_,
                   "bp": key(advisor.best_picks(vt_)) if len(vt_) else {},
                   "cp": key(advisor.cheapest_picks(vt_)) if len(vt_) else {}}
    pick = lambda p: None if p is None else {"group": p.group, "market": p.market, "odds": _f(p.odds, 3),
                                              "bookmaker": p.bookmaker, "prob": _f(p.prob), "value": _f(p.value)}
    matches = []
    for i, r in enumerate(up.itertuples()):
        fo = fair_odds_for(pd.Series(r._asdict()), params)
        fo = fo[fo.group.isin(SHOWN_GROUPS)]
        prices = {}
        for name, v in V.items():
            b = v["best"]
            b = b[(b.league == r.league) & (b.home == r.home) & (b.away == r.away)] if len(b) else b
            prices[name] = {(g, m): (o, k) for g, m, o, k in zip(b.group, b.market, b.odds, b.bookmaker)} if len(b) else {}
        mk = []
        for f in fo.itertuples():
            o, k = prices["all"].get((f.group, f.market), (None, None))
            mk.append({"g": f.group, "m": f.market, "p": _f(f.prob), "fair": _f(f.fair_odds, 3),
                       "err": _f(err.get((f.group, f.market)), 2), "odds": _f(o, 3), "bk": k,
                       "pw": _f(prices["pw"].get((f.group, f.market), (None, None))[0], 3)})
        ex = advisor.exotic_suggestions(fair_odds_for(pd.Series(r._asdict()), params))
        k3 = (r.league, r.home, r.away)
        matches.append({"id": i, "league": LEAGUE_NAMES.get(r.league, r.league), "kickoff": r.kickoff_utc,
                        "home": r.home, "away": r.away, "xg": [_f(r.exp_home, 2), _f(r.exp_away, 2)],
                        "best": pick(V["all"]["bp"].get(k3)), "cheap": pick(V["all"]["cp"].get(k3)),
                        "best_pw": pick(V["pw"]["bp"].get(k3)), "cheap_pw": pick(V["pw"]["cp"].get(k3)),
                        "markets": mk,
                        "exotic": [{"g": e.group, "m": e.market, "p": _f(e.prob), "fair": _f(e.fair_odds, 3),
                                    "min": _f(e.fair_odds * (1 - advisor.MAX_COST), 2)} for e in ex.itertuples()]})

    variants = {}
    for name, v in V.items():
        presets, custom = [], []
        if len(v["vt"]):
            for pname, (k, target) in advisor.PRESETS.items():
                m = advisor.best_multiple(v["vt"], k, target)
                presets.append({"name": pname, "slip": _slip(m) if m else None})
            for k, target in CUSTOM_SLIPS:
                for obj in ("prob", "value"):
                    custom.append({"k": k, "target": target, "objective": obj,
                                   "slips": [_slip(m) for m in advisor.alternative_slips(v["vt"], k, target, 3, obj)]})
        variants[name] = {"presets": presets, "custom": custom}
    presets, custom = variants["all"]["presets"], variants["all"]["custom"]

    picks, slips = _read(PICKS), _read(SLIPS)
    ou = load_register()
    ou = ou[ou.status.isin(["vinta", "persa"])] if len(ou) else ou
    if len(ou):   # CLV of the over/under register, from the closing odds (as for the other registers)
        from .tracking import leg_results, results_table
        legs = pd.DataFrame({"league": ou.league, "kickoff_utc": ou.kickoff_utc, "home": ou.home, "away": ou.away,
                             "group": "Under/Over", "market": ou.pick + " 2.5"})
        lr = leg_results(legs, results_table(), params)
        ou = ou.reset_index(drop=True).assign(clv=ou.odds_best.to_numpy() * lr.close_prob.to_numpy() - 1)
    done = picks[picks.status.isin(["vinta", "persa", "rimborsata"])] if len(picks) else picks
    recent = done.sort_values("kickoff_utc", ascending=False).head(40) if len(done) else done
    tracking = {
        "reliability": reliability(picks, slips),
        "logged": int(len(picks)), "settled": int(len(done)), "open": int((picks.status == "aperta").sum()) if len(picks) else 0,
        "by_kind": _summary(picks, "kind"), "by_group": _summary(picks, "group")[1:],
        "by_book": _summary(picks.assign(book=picks.bookmaker.map(lambda x: BOOKMAKERS.get(x, x))), "book")[1:]
        if len(picks) else [],
        "slips": _summary(slips.assign(kickoff_utc=slips.logged_at), "preset") if len(slips) else [],
        "ou": _summary(ou) if len(ou) else [],
        "recent": [{"kickoff": r.kickoff_utc, "match": f"{r.home} – {r.away}", "market": r.market,
                    "odds": _f(r.odds, 3), "bookmaker": BOOKMAKERS.get(r.bookmaker, r.bookmaker),
                    "status": r.status, "clv": _f(r.clv)} for r in recent.itertuples()],
    }
    return {"updated": meta.get("data_updated_at"), "odds_fetched": bk.fetched_at.max() if len(bk) else None,
            "bookmakers": BOOKMAKERS, "disclaimer": DISCLAIMER, "history": HISTORY,
            "rules": {"min_odds": advisor.MIN_ODDS, "max_odds": advisor.MAX_ODDS, "max_cost": advisor.MAX_COST},
            "matches": matches, "presets": presets, "custom": custom, "variants": variants, "tracking": tracking}


# -------------------------------------------------------------------- build

def build_site(out: Path = SITE, password: str = None, now: pd.Timestamp = None) -> str:
    """Write the encrypted site to `out`. Returns the content hash (unchanged hash = nothing new to publish)."""
    password = password or get_password()
    if not password:
        raise DataError(f"manca la password del sito ({PASSWORD_ENV} o {PASSWORD_FILE}): senza non si pubblica nulla")
    data = site_data(now)
    plain = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    template = (Path(__file__).parent / "site_template.html").read_text()
    digest = hashlib.sha256(plain + template.encode()).hexdigest()[:16]   # new data or new page: republish
    out.mkdir(parents=True, exist_ok=True)
    write_atomic(out / "data.json", json.dumps(encrypt(plain, password)))
    write_atomic(out / "index.html", template)
    write_atomic(out / "manifest.webmanifest", json.dumps({
        "name": "Forecast 1X2", "short_name": "Forecast 1X2", "start_url": ".", "display": "standalone",
        "background_color": "#0f1712", "theme_color": "#16804c",
        "icons": [{"src": "icon-192.png", "sizes": "192x192", "type": "image/png"},
                  {"src": "icon-512.png", "sizes": "512x512", "type": "image/png"}]}, indent=1))
    for size, name in [(180, "apple-touch-icon.png"), (192, "icon-192.png"), (512, "icon-512.png")]:
        (out / name).write_bytes(png(size))
    (out / ".nojekyll").write_text("")
    write_atomic(out / "version.txt", digest + "\n")
    print(f"sito: {len(data['matches'])} partite, {len(data['presets'])} schedine pronte, contenuto {digest}")
    return digest
