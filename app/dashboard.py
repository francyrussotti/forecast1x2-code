"""Streamlit dashboard. Reads only from outputs/; refits only when the user presses "Aggiorna"."""
import json
import subprocess
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Su Streamlit Cloud il pacchetto non è installato con pip install -e: lo si prende da src/.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Streamlit rilancia questo file a ogni aggiornamento ma tiene in memoria i moduli già importati: se il codice
# del pacchetto è cambiato (es. dopo un push su GitHub), li si scarta così vengono reimportati nuovi.
_sig = tuple(sorted((p.name, p.stat().st_mtime_ns) for p in (_SRC / "forecast1x2").glob("*.py")))
if getattr(sys, "_forecast1x2_src_sig", None) != _sig:
    for _name in [m for m in sys.modules if m == "forecast1x2" or m.startswith("forecast1x2.")]:
        del sys.modules[_name]
    sys._forecast1x2_src_sig = _sig

from forecast1x2.auto import last_update_time, load_kickoffs, next_update
from forecast1x2.data import LEAGUES
from forecast1x2.evaluation import SOURCES, calibration_table, outcome_index, probs, rps
from forecast1x2 import advisor
from forecast1x2.markets import fair_odds_for, load_params
from forecast1x2.oddspapi import BOOKMAKER_NAME, BOOKMAKERS, best_prices, value_table
from forecast1x2.tips import RULE_FROZEN, RULE_VERSION, THRESHOLD, load_register, summary

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
STALE_DAYS = 3
RISK_LABELS = {1: "🟢 1 · basso", 2: "🟡 2 · medio", 3: "🔴 3 · alto"}
DISCLAIMER = ("Le previsioni non sono consigli di scommessa. Nel backtest 2024-26 il modello non batte le quote "
              "di mercato, e puntare dove «vede valore» avrebbe perso tra il 13% e il 22% di quanto puntato "
              "(dettagli nella pagina Backtest). Il gioco può causare dipendenza.")
AGENT_PLIST = Path.home() / "Library" / "LaunchAgents" / "com.forecast1x2.auto.plist"
TZ = "Europe/Rome"
LEAGUE_NAMES = {d: v[0] for d, v in LEAGUES.items()}
SEASON_NAMES = {"2425": "2024/25", "2526": "2025/26", "tutte": "Tutte"}
DAYS = ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]
STATUS = {"giornata": "", "recupero": "recupero", "rinviata": "rinviata, data da definire"}

CHART_SOURCES = ["model", "market_open", "market", "baseline"]
COLOR = alt.Color("Fonte", scale=alt.Scale(domain=[SOURCES[s] for s in CHART_SOURCES],
                                           range=["#e4572e", "#8e44ad", "#2e86de", "#9e9e9e"]))

st.set_page_config(page_title="Forecast 1X2", layout="wide")


@st.cache_data
def _read_csv_cached(name, mtime, **kw):
    path = OUTPUTS / name
    return pd.read_csv(path, **kw) if mtime is not None else None


def read_csv(name, **kw):
    """Cached read that follows the file: a new version (e.g. pushed by the GitHub update) is read again."""
    path = OUTPUTS / name
    return _read_csv_cached(name, path.stat().st_mtime_ns if path.exists() else None, **kw)


def read_meta():
    path = OUTPUTS / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def fmt_time(iso):
    return pd.Timestamp(iso).tz_convert(TZ).strftime("%d/%m/%Y %H:%M")


def sidebar_update(meta):
    """Runs update-data + predict in a separate process on demand (a few seconds to ~1 minute)."""
    st.sidebar.divider()
    if meta.get("data_updated_at"):
        st.sidebar.caption(f"Dati aggiornati il {fmt_time(meta['data_updated_at'])}")
    if AGENT_PLIST.exists():
        nxt = next_update(pd.Timestamp.now(tz="UTC"), load_kickoffs(), last_update_time())
        st.sidebar.caption(f"Aggiornamento automatico attivo · prossimo verso il {nxt.tz_convert(TZ):%d/%m %H:%M}")
    else:
        st.sidebar.caption("Aggiornamento automatico su GitHub: controllo ogni ora, aggiorna attorno alle partite")
    if st.sidebar.button("Aggiorna dati e previsioni", width="stretch"):
        with st.spinner("Scarico risultati, quote e calendari e rifaccio le previsioni..."):
            try:
                r = subprocess.run([sys.executable, "-W", "ignore", "-m", "forecast1x2", "aggiorna"], cwd=ROOT,
                                   capture_output=True, text=True, timeout=1800)
                ok, log = r.returncode == 0, r.stdout + r.stderr
            except subprocess.TimeoutExpired:
                ok, log = False, "Tempo scaduto (30 minuti)."
        st.session_state["update_result"] = (ok, log)
        st.cache_data.clear()
        st.rerun()
    if "update_result" in st.session_state:
        ok, log = st.session_state["update_result"]
        (st.sidebar.success if ok else st.sidebar.error)(
            "Aggiornamento completato." if ok else "Aggiornamento fallito: dettagli qui sotto.")
        with st.sidebar.expander("Dettagli aggiornamento", expanded=not ok):
            st.code(log or "(nessun output)", language=None)


def model_vs_market(p: pd.DataFrame) -> list:
    """Largest gap (percentage points) where the model is above the margin-free market, e.g. '+6 su 2'."""
    out = []
    for r in p.itertuples():
        if pd.isna(r.market_1):
            out.append("")
            continue
        gaps = {o: 100 * (getattr(r, mp) - getattr(r, mk)) for o, mp, mk in
                [("1", "p1", "market_1"), ("X", "px", "market_X"), ("2", "p2", "market_2")]}
        o = max(gaps, key=gaps.get)
        out.append(f"+{gaps[o]:.0f} pp su {o}" if gaps[o] >= 1 else "allineato")
    return out


def source_in_use(meta, available) -> str:
    """The forecast shown on the predictions page: ensemble, with xG if they are active."""
    available = set(available)
    return "ensemble_xg" if meta.get("xg_weight") and "ensemble_xg" in available else "ensemble"


def reliability_panel(meta):
    rel = read_csv("reliability.csv")
    if rel is None:
        return
    r = rel.set_index("source")
    src = source_in_use(meta, r.index)
    if src not in r.index or "market" not in r.index:
        return
    m, k = r.loc[src], r.loc["market"]
    st.markdown("**Affidabilità del modello** (backtest 2024/25 e 2025/26, 3.504 partite)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Esito più probabile indovinato", f"{m.accuracy_pct:.0f}%",
              f"{m.accuracy_pct - k.accuracy_pct:+.1f} punti rispetto al mercato", delta_color="off")
    c2.metric("Errore medio delle probabilità", f"±{m.calib_error_pp:.1f} punti %",
              help="Quando il modello dice 40%, l'esito si verifica in media tra il 40 − x e il 40 + x %.")
    c3.metric("Capacità predittiva rispetto al mercato", f"{m.market_share_pct:.0f}%",
              help="100% = prevede bene quanto le quote di chiusura; 0% = come le sole frequenze storiche 1/X/2.")
    st.caption(f"Per confronto, le quote di chiusura indovinano l'esito più probabile il {k.accuracy_pct:.0f}% "
               "delle volte. Nel calcio anche il miglior pronostico sbaglia quasi una partita su due.")


def risk_legend(meta):
    """How often the favourite actually won in each risk level (backtest of the model in use)."""
    rt = read_csv("risk_tiers.csv")
    if rt is None:
        return
    source = source_in_use(meta, rt.source)
    rt = rt[rt.source == source].set_index("level")
    st.caption("Rischio = quanto è netto il pronostico (probabilità dell'esito più probabile: "
               "≥ 60% basso, 45–60% medio, < 45% alto). Nel backtest 2024-26 il favorito ha vinto: "
               + " · ".join(f"{RISK_LABELS[l]} {100 * rt.loc[l, 'fav_win_rate']:.0f}% delle volte"
                            for l in rt.index) + ".")


def page_predictions():
    st.title("Previsioni")
    st.info(DISCLAIMER)
    reliability_panel(read_meta())
    preds, meta = read_csv("predictions.csv"), read_meta()
    if preds is None:
        st.error("Nessuna previsione ancora. Premi «Aggiorna dati e previsioni» nella barra a sinistra.")
        return
    now = pd.Timestamp.now(tz="UTC")
    age = now - pd.Timestamp(meta["data_updated_at"])
    if age > pd.Timedelta(days=STALE_DAYS):
        st.warning(f"Dati aggiornati {age.days} giorni fa: premi «Aggiorna dati e previsioni» prima di usarle.")
    for w in meta.get("warnings", []):   # fallback sources, stale calendar/odds/xG from the last update
        st.warning(w)
    div = st.selectbox("Campionato", list(LEAGUE_NAMES), format_func=LEAGUE_NAMES.get)
    p = preds[preds.league == div]
    if p.empty:
        st.info("Nessuna partita in programma per questo campionato.")
        return
    st.subheader(f"Giornata {p.loc[p.status == 'giornata', 'round'].max()}")

    kickoff_utc = pd.to_datetime(p.kickoff_utc, utc=True)
    kickoff = kickoff_utc.dt.tz_convert(TZ)
    started = (kickoff_utc <= now) & (p.status != "rinviata")
    note = [" · ".join(x for x in [STATUS[s] + (f" (giornata {r})" if s != "giornata" else ""),
                                   "già iniziata" if st_ else ""] if x)
            for s, r, st_ in zip(p.status, p["round"], started)]
    if started.all():
        st.warning("Tutte le partite di questa giornata sono già iniziate: aggiorna per passare alla prossima.")
    table = pd.DataFrame({
        "Data e ora (Roma)": kickoff.dt.dayofweek.map(dict(enumerate(DAYS))) + " " + kickoff.dt.strftime("%d/%m %H:%M"),
        "Partita": p.home + " – " + p.away,
        "1 %": (100 * p.p1).round(1),
        "X %": (100 * p.px).round(1),
        "2 %": (100 * p.p2).round(1),
        "Gol attesi casa": p.xg_home.round(2),
        "Gol attesi trasf.": p.xg_away.round(2),
        "Risultato più probabile": p.top_score + "  (" + (100 * p.top_score_prob).round(1).astype(str) + "%)",
        "Rischio": p.risk_level.map(RISK_LABELS),
        "Favorito": p.fav,
        "Mercato 1 %": (100 * p.market_1).round(1),
        "Mercato X %": (100 * p.market_X).round(1),
        "Mercato 2 %": (100 * p.market_2).round(1),
        "Modello − mercato": model_vs_market(p),
        "Note": note,
    })
    st.dataframe(table, hide_index=True, width="stretch")
    if (p.status != "giornata").any():
        st.caption("Recupero/rinviata: partite di giornate precedenti non ancora giocate.")
    risk_legend(meta)
    if p.market_1.isna().all():
        st.caption("Quote pre-partita non ancora disponibili: football-data le pubblica pochi giorni prima della giornata.")
    else:
        st.caption("Mercato = quote medie pre-partita (football-data) senza margine. "
                   f"Peso del modello nella combinazione calibrato sul passato: w = {meta.get('w')}"
                   + (" (il modello non aggiunge informazione alle quote)." if meta.get("w") == 0 else "."))

    m = meta.get("model", {}).get(div, {})
    st.caption(
        f"Ultimo aggiornamento dati: {fmt_time(meta['data_updated_at'])} · "
        f"ultima partita nei dati: {meta['last_match_date'][div]} · "
        f"previsioni generate: {fmt_time(meta['predictions_generated_at'])} · "
        f"vantaggio casa {m.get('home', float('nan')):.3f}, ρ {m.get('rho', float('nan')):.3f}, "
        f"κ neopromosse {m.get('kappa', float('nan')):.3f}, ξ {meta.get('xi')}, λ {meta.get('l2')}"
    )


def xg_section(league, season):
    """Cross-season test of the xG model (2024/25 and 2025/26), if the backtest produced it."""
    xm, xb = read_csv("xg_metrics.csv", dtype={"season": str}), read_csv("xg_bootstrap.csv", dtype={"season": str})
    if xm is None or xb is None:
        return
    meta = read_meta()
    st.subheader("Test degli xG (2024/25 e 2025/26)")
    st.caption("Test incrociato: ogni stagione è prevista con il peso degli xG scelto sull'altra stagione, "
               "sulle stesse partite del mercato. Le previsioni usano il peso scelto su entrambe: "
               f"{meta.get('xg_weight')}.")
    sel = xm[(xm.league == league) & (xm.season == season)].set_index("source")
    sel = sel.loc[[s for s in SOURCES if s in sel.index]]
    st.dataframe(pd.DataFrame({
        "Fonte": [SOURCES[s] for s in sel.index], "Partite": sel.n.to_numpy(),
        "Log-loss": sel.log_loss.round(4).to_numpy(), "Brier": sel.brier.round(4).to_numpy(),
        "RPS": sel.rps.round(4).to_numpy(),
    }), hide_index=True, width="stretch")
    b = xb[(xb.league == league) & (xb.season == season)]
    st.dataframe(pd.DataFrame({
        "Confronto": ["Modello + xG − " + {"model": "Modello (solo gol)", "market": "Mercato chiusura"}[a]
                      for a in b.against],
        "Metrica": b.metric.to_numpy(), "Differenza media": b.mean_diff.round(4).to_numpy(),
        "IC 95%": [f"{lo:+.4f} … {hi:+.4f}" for lo, hi in zip(b.ci_low, b.ci_high)],
    }), hide_index=True, width="stretch")
    st.caption("Valori negativi = Modello + xG migliore.")


def betting_section():
    """Historical simulation of betting on the model's 'value' picks at pre-match odds."""
    sim = read_csv("betting_sim.csv")
    if sim is None:
        return
    st.subheader("Simulazione scommesse (2024/25 e 2025/26)")
    st.caption("1 unità per partita, quote medie pre-partita, al massimo una giocata per partita. "
               "«Valore»: si gioca l'esito con valore atteso (probabilità × quota − 1) sopra la soglia. "
               "«Favorito del mercato»: si gioca sempre il favorito delle quote, come riferimento.")
    names = {k: v for k, v in SOURCES.items()}
    st.dataframe(pd.DataFrame({
        "Modello": sim.source.map(names).to_numpy(),
        "Strategia": ["Favorito del mercato" if s == "favorito" else f"Valore > {100 * t:.0f}%"
                      for s, t in zip(sim.strategy, sim.threshold)],
        "Giocate": sim.bets.to_numpy(), "Vinte": sim.wins.to_numpy(),
        "Guadagno (unità)": sim.profit.round(1).to_numpy(),
        "Rendimento": [f"{100 * r:+.1f}%" for r in sim.roi],
        "IC 95%": [f"{100 * lo:+.1f}% … {100 * hi:+.1f}%" for lo, hi in zip(sim.ci_low, sim.ci_high)],
    }), hide_index=True, width="stretch")
    st.caption("Rendimento negativo = perdita per unità giocata. Tutte le strategie del modello perdono più "
               "del margine del bookmaker: il modello «vede valore» soprattutto dove sbaglia.")


def page_backtest():
    st.title("Backtest walk-forward")
    metrics = read_csv("backtest_metrics.csv", dtype={"season": str})
    boot = read_csv("backtest_bootstrap.csv", dtype={"season": str})
    matches = read_csv("backtest_matches.csv", parse_dates=["date", "week_start"], dtype={"season": str})
    if metrics is None or boot is None or matches is None:
        st.error("Nessun backtest in outputs/. Lancia `python -m forecast1x2 backtest`.")
        return

    c1, c2 = st.columns(2)
    league = c1.selectbox("Campionato", ["tutti"] + list(LEAGUE_NAMES),
                          format_func=lambda d: "Tutti" if d == "tutti" else LEAGUE_NAMES[d])
    season = c2.selectbox("Stagione", ["tutte", "2425", "2526"], format_func=SEASON_NAMES.get)

    sel = metrics[(metrics.league == league) & (metrics.season == season)].set_index("source")
    sel = sel.loc[[s for s in SOURCES if s in sel.index]]
    st.dataframe(pd.DataFrame({
        "Fonte": [SOURCES[s] for s in sel.index], "Partite": sel.n.to_numpy(),
        "Log-loss": sel.log_loss.round(4).to_numpy(), "Brier": sel.brier.round(4).to_numpy(),
        "RPS": sel.rps.round(4).to_numpy(),
    }), hide_index=True, width="stretch")

    st.caption("Differenze rispetto al mercato di chiusura (IC 95% bootstrap per settimane). "
               "Valori positivi = peggiore del mercato di chiusura.")
    b = boot[(boot.league == league) & (boot.season == season)]
    st.dataframe(pd.DataFrame({
        "Fonte": b.source.map(SOURCES).to_numpy(), "Metrica": b.metric.to_numpy(),
        "Differenza media": b.mean_diff.round(4).to_numpy(),
        "IC 95%": [f"{lo:+.4f} … {hi:+.4f}" for lo, hi in zip(b.ci_low, b.ci_high)],
    }), hide_index=True, width="stretch")

    df = matches
    if league != "tutti":
        df = df[df.league == league]
    if season != "tutte":
        df = df[df.season == season]
    df = df.sort_values("date").reset_index(drop=True)

    xg_section(league, season)
    betting_section()
    rel = read_csv("reliability.csv")
    if rel is not None:
        st.subheader("Affidabilità di tutti i modelli (2024/25 e 2025/26)")
        rel = rel.set_index("source").loc[[s for s in SOURCES if s in set(rel.source)]]
        st.dataframe(pd.DataFrame({
            "Fonte": [SOURCES[s] for s in rel.index],
            "Esito più probabile indovinato": [f"{v:.1f}%" for v in rel.accuracy_pct],
            "Errore medio probabilità": [f"±{v:.1f} pp" for v in rel.calib_error_pp],
            "Miglioramento sulla baseline": [f"{v:.1f}%" for v in rel.skill_vs_baseline_pct],
            "Capacità vs mercato": [f"{v:.0f}%" for v in rel.market_share_pct],
            "Log-loss": rel.log_loss.round(4).to_numpy(),
        }), hide_index=True, width="stretch")
        st.caption("«Capacità vs mercato»: quanta parte del vantaggio delle quote di chiusura sulla baseline "
                   "il modello riesce a ottenere (100% = come il mercato).")

    st.subheader("Calibrazione")
    outcome = st.radio("Esito", ["1", "X", "2"], horizontal=True)
    cal = calibration_table(df)
    cal = cal[(cal.outcome == outcome) & cal.source.isin(CHART_SOURCES)].copy()
    cal["Fonte"] = cal.source.map(SOURCES)
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[4, 4], color="gray").encode(x="x", y="y")
    pts = alt.Chart(cal).mark_line(point=True).encode(
        x=alt.X("mean_pred", title="Probabilità prevista", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("obs_freq", title="Frequenza osservata", scale=alt.Scale(domain=[0, 1])),
        color=COLOR, tooltip=["Fonte", alt.Tooltip("mean_pred", format=".3f"),
                                alt.Tooltip("obs_freq", format=".3f"), "n"])
    st.altair_chart(diag + pts, use_container_width=True)

    st.subheader("RPS medio cumulato")
    y = outcome_index(df)
    long = [pd.DataFrame({"Data": df.date, "Fonte": SOURCES[s],
                          "RPS cumulato": pd.Series(rps(probs(df, s), y)).expanding().mean()})
            for s in CHART_SOURCES]
    line = alt.Chart(pd.concat(long)).mark_line().encode(
        x="Data:T", y=alt.Y("RPS cumulato", scale=alt.Scale(zero=False)), color=COLOR)
    st.altair_chart(line, use_container_width=True)


ALL_LEAGUE_NAMES = {**LEAGUE_NAMES, "I2": "Serie B", "E1": "Championship", "SP2": "La Liga 2",
                    "D2": "2. Bundesliga", "F2": "Ligue 2"}


def tips_results_table(s: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Gruppo": s.group.str.capitalize().to_numpy(), "Giocate": s.bets.to_numpy(), "Vinte": s.wins.to_numpy(),
        "Guadagno (unità)": s.profit.round(1).to_numpy(),
        "Rendimento": ["—" if pd.isna(r) else f"{100 * r:+.1f}%" for r in s.roi],
        "IC 95%": ["—" if pd.isna(lo) else f"{100 * lo:+.1f}% … {100 * hi:+.1f}%" for lo, hi in zip(s.ci_low, s.ci_high)],
    })


def page_tips():
    st.title("Consigli in prova: over/under 2.5")
    st.info(DISCLAIMER)
    st.warning(
        "Questa è una **prova a soldi finti**, non un sistema che vince. Si segnalano le partite in cui un "
        f"bookmaker paga l'over o l'under 2.5 almeno il {100 * THRESHOLD:.0f}% più della media del mercato, "
        "con le quote del venerdì. Nello storico la regola ha funzionato in seconda divisione e non in prima, "
        "con pochi casi: serve almeno una stagione intera di prova prima di trarre conclusioni.")
    reg = load_register()
    now = pd.Timestamp.now(tz="UTC")
    kickoff = pd.to_datetime(reg.kickoff_utc, utc=True)

    st.subheader("Prossime partite segnalate")
    nxt = reg[(reg.status == "aperta") & (kickoff > now)].assign(k=kickoff).sort_values("k")
    if nxt.empty:
        st.caption("Nessuna partita segnalata al momento. Le quote escono di solito il venerdì.")
    else:
        k = nxt.k.dt.tz_convert(TZ)
        st.dataframe(pd.DataFrame({
            "Data e ora (Roma)": k.dt.dayofweek.map(dict(enumerate(DAYS))) + " " + k.dt.strftime("%d/%m %H:%M"),
            "Campionato": nxt.league.map(ALL_LEAGUE_NAMES).to_numpy(),
            "Partita": (nxt.home + " – " + nxt.away).to_numpy(),
            "Giocata": (nxt.pick + " 2.5").to_numpy(),
            "Quota migliore": nxt.odds_best.round(2).to_numpy(),
            "Quota media": [round(o if p == "Over" else u, 2) for p, o, u in zip(nxt.pick, nxt.avg_over, nxt.avg_under)],
            "Vantaggio stimato": [f"+{100 * e:.1f}%" for e in nxt.edge],
        }), hide_index=True, width="stretch")
        st.caption("Quota migliore = la più alta tra i bookmaker raccolti da football-data, che non dice quale sia: "
                   "nella realtà può non essere disponibile o essere già cambiata.")
    waiting = ((reg.status == "aperta") & (kickoff <= now)).sum()
    if waiting:
        st.caption(f"{waiting} partite già iniziate in attesa del risultato.")

    st.subheader("Risultati della prova")
    s = summary(reg)
    st.dataframe(tips_results_table(s), hide_index=True, width="stretch")
    st.caption(f"1 unità per giocata alla quota migliore registrata. Regola {RULE_VERSION}, congelata il "
               f"{pd.Timestamp(RULE_FROZEN):%d/%m/%Y}: ogni partita entra nel registro almeno un'ora prima "
               "dell'inizio e non viene più modificata. IC 95% bootstrap per settimane.")
    if s.bets.iloc[0] < 100:
        st.caption("Meno di 100 giocate concluse: il risultato dipende ancora soprattutto dal caso.")

    hist = read_csv("tips_backtest.csv")
    if hist is not None:
        st.subheader("La stessa regola sullo storico")
        st.dataframe(tips_results_table(hist).assign(Periodo=hist.period.to_numpy())[
            ["Periodo", "Gruppo", "Giocate", "Vinte", "Guadagno (unità)", "Rendimento", "IC 95%"]],
            hide_index=True, width="stretch")
        st.caption("La soglia è stata scelta sul periodo di calibrazione; 2024/25-2025/26 è la verifica. "
                   "Il 2026/27 fino al congelamento della regola non è una prova: la prova vera è il registro qui sopra.")

    if len(reg):
        with st.expander("Tutte le giocate registrate"):
            st.dataframe(reg.sort_values("kickoff_utc", ascending=False), hide_index=True, width="stretch")


KELLY_FRACTION = 0.25
MAX_STAKE = 0.05   # never more than 5% of the budget on one bet
SUSPICIOUS = 2.0   # odds above twice the fair odds: almost surely a typo (e.g. 24 instead of 2,4)


def parse_odds(s: pd.Series) -> np.ndarray:
    """Odds typed by hand: '2,40' or '2.40'; NaN if not a number above 1."""
    v = pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce").to_numpy(float)
    return np.where(v > 1, v, np.nan)


def kelly_stake(prob, odds):
    """Fraction of the budget: a quarter of Kelly, capped; 0 when the bet has no value."""
    f = (odds * prob - 1) / (odds - 1)
    return np.clip(KELLY_FRACTION * f, 0, MAX_STAKE)


def page_fair_odds():
    st.title("Quote giuste e strategia")
    st.info(DISCLAIMER)
    up = read_csv("fair_odds.csv")
    if up is None or up.empty:
        st.warning("Nessuna partita con quote disponibili: football-data pubblica le quote di solito il venerdì.")
        return
    params = load_params()
    cal = read_csv("markets_calibration.csv")
    err = {}
    if cal is not None:
        t = cal[cal.period.str.startswith("verifica")]
        err = {(g, m): d for g, m, d in zip(t.group, t.market, t.diff_pp)}
    st.caption("Le probabilità vengono dalle quote medie 1X2 e Over/Under 2.5 senza margine (i mercati più "
               "precisi), trasformate in gol attesi e poi in ogni altro mercato. Scrivi la quota del tuo bookmaker: "
               "c'è **valore** solo se è più alta della quota giusta.")

    up = up.assign(k=pd.to_datetime(up.kickoff_utc, utc=True)).sort_values("k")
    up = up[up.k > pd.Timestamp.now(tz="UTC")]
    if up.empty:
        st.warning("Tutte le partite con quote sono già iniziate: le prossime quote escono di solito il venerdì.")
        return
    bk = read_csv("bookmaker_odds.csv")
    bk = bk if bk is not None and not bk.empty else None
    if bk is not None:
        bookmaker_opportunities(up, bk, params, err)
    c1, c2 = st.columns([1, 2])
    leagues = [d for d in ALL_LEAGUE_NAMES if d in set(up.league)]
    div = c1.selectbox("Campionato", leagues, format_func=ALL_LEAGUE_NAMES.get, key="fo_league")
    games = up[up.league == div]
    labels = [f"{k.tz_convert(TZ):%d/%m %H:%M} · {h} – {a}" for k, h, a in zip(games.k, games.home, games.away)]
    idx = c2.selectbox("Partita", range(len(games)), format_func=lambda i: labels[i], key="fo_game")
    row = games.iloc[idx]
    st.markdown(f"**Gol attesi:** {row.home} {row.exp_home:.2f} – {row.exp_away:.2f} {row.away} · "
                f"mercato: 1 {100 * row.p1:.0f}%, 2 {100 * row.p2:.0f}%, Over 2.5 {100 * row.p_over:.0f}%")

    fo = fair_odds_for(row, params)
    groups = list(dict.fromkeys(fo.group))
    default = ["1X2", "Doppia chance", "Under/Over", "Goal/No goal", "Multigoal", "Combo 1X2 + U/O"]
    chosen = st.multiselect("Mercati", groups, default=[g for g in default if g in groups], key="fo_groups")
    fo = fo[fo.group.isin(chosen)].reset_index(drop=True)
    table = pd.DataFrame({
        "Gruppo": fo.group, "Mercato": fo.market, "Probabilità %": (100 * fo.prob).round(1),
        "Quota giusta": fo.fair_odds.round(2),
        "Errore storico (pp)": [round(err.get((g, m), np.nan), 1) for g, m in zip(fo.group, fo.market)],
        "Quota bookmaker": "",
    })
    if bk is not None:   # prefill with the best of the bookmakers where available; the user can overwrite
        mine = best_prices(bk[(bk.league == row.league) & (bk.home == row.home) & (bk.away == row.away)])
        pw = {(g, m): (o, b) for g, m, o, b in zip(mine.group, mine.market, mine.odds, mine.bookmaker)}
        table["Quota bookmaker"] = [f"{pw[(g, m)][0]:.2f}".replace(".", ",") if (g, m) in pw else ""
                                    for g, m in zip(fo.group, fo.market)]
        table["Dove"] = [BOOKMAKERS.get(pw[(g, m)][1], pw[(g, m)][1]) if (g, m) in pw else ""
                         for g, m in zip(fo.group, fo.market)]
        if pw:
            st.caption(f"Colonna «Quota bookmaker» già compilata con la quota migliore tra {BOOKMAKER_NAME} "
                       f"del {fmt_time(mine.fetched_at.iloc[0])} («Dove» dice quale): puoi sovrascriverla.")
    edited = st.data_editor(
        table, hide_index=True, width="stretch", key=f"fo_edit_{div}_{row.home}_{row.away}",
        disabled=["Gruppo", "Mercato", "Probabilità %", "Quota giusta", "Errore storico (pp)", "Dove"],
        column_config={"Quota bookmaker": st.column_config.TextColumn(
                           help="La quota del tuo bookmaker, con la virgola o con il punto: 2,40 oppure 2.40."),
                       "Errore storico (pp)": st.column_config.NumberColumn(
                           help="Verifica 2024/25-2025/26: frequenza osservata − probabilità prevista, in punti %.")})
    st.caption("Errore storico: quanto, in media, la probabilità del mercato si è discostata dalla frequenza reale "
               "nelle stagioni di verifica. Un valore stimato più piccolo di questo errore non è affidabile.")

    entered = edited["Quota bookmaker"].fillna("").astype(str).str.strip()
    q = edited[entered != ""].copy()
    if not q.empty:
        odds = parse_odds(q["Quota bookmaker"])
        bad = np.isnan(odds)
        if bad.any():
            st.warning("Quote non valide (servono numeri maggiori di 1, es. 2,40): "
                       + ", ".join(q["Mercato"][bad]))
        q, odds = q[~bad], odds[~bad]
    if not q.empty:
        p = q["Probabilità %"].to_numpy() / 100
        value = odds * p - 1
        margin_pp = np.abs(q["Errore storico (pp)"].fillna(0).to_numpy()) / 100 * odds   # value error from model error
        suspicious = odds > SUSPICIOUS * q["Quota giusta"].to_numpy()
        q["Quota bookmaker"] = odds
        q["Valore %"] = (100 * value).round(1)
        q["Puntata (% budget)"] = np.where(suspicious, 0.0, (100 * kelly_stake(p, odds)).round(2))
        q["Giudizio"] = np.where(suspicious, "quota sospetta: ricontrolla",
                                 np.where(value <= 0, "nessun valore",
                                          np.where(value <= margin_pp, "valore entro l'errore", "valore")))
        value = np.where(suspicious, -1.0, value)
        st.subheader("Le tue quote")
        st.dataframe(q[["Mercato", "Quota giusta", "Quota bookmaker", "Valore %", "Puntata (% budget)", "Giudizio"]]
                     .sort_values("Valore %", ascending=False), hide_index=True, width="stretch")
        st.caption(f"Puntata = un quarto del criterio di Kelly, al massimo il {100 * MAX_STAKE:.0f}% del budget: "
                   "cresce con il valore e si azzera senza valore. Con quote sbagliate o valore stimato male, "
                   "puntate più alte fanno perdere più in fretta.")
        if st.button("Aggiungi le giocate con valore alla schedina", key="fo_add"):
            slip = st.session_state.setdefault("slip", [])
            have = {(b["Partita"], b["Mercato"]) for b in slip}
            for market, prob, o, v in zip(q["Mercato"], p, odds, value):
                if v > 0 and (f"{row.home} – {row.away}", market) not in have:
                    slip.append({"Partita": f"{row.home} – {row.away}", "Mercato": market,
                                 "prob": float(prob), "Quota": float(o)})
    slip_section()


def bookmaker_opportunities(up, bk, params, err):
    """All Planetwin365 prices above fair odds by more than the model's error, across the upcoming matches."""
    vt = value_table(up, bk, params, err)
    st.subheader(f"Occasioni {BOOKMAKER_NAME}")
    if vt.empty:
        st.caption(f"Nessuna quota {BOOKMAKER_NAME} abbinata alle prossime partite.")
        return
    good = vt[vt.verdict == "valore"].sort_values("value", ascending=False)
    fetched = fmt_time(bk.fetched_at.max())
    if good.empty:
        st.caption(f"Nessuna quota {BOOKMAKER_NAME} sopra la quota giusta oltre l'errore del modello "
                   f"({len(vt)} quote controllate, aggiornate il {fetched}).")
        return
    k = pd.to_datetime(good.kickoff_utc, utc=True).dt.tz_convert(TZ)
    st.dataframe(pd.DataFrame({
        "Data (Roma)": k.dt.dayofweek.map(dict(enumerate(DAYS))).to_numpy() + " " + k.dt.strftime("%d/%m %H:%M").to_numpy(),
        "Campionato": good.league.map(ALL_LEAGUE_NAMES).to_numpy(),
        "Partita": (good.home + " – " + good.away).to_numpy(),
        "Mercato": (good.group + ": " + good.market).to_numpy(),
        "Miglior quota": good.odds.round(2).to_numpy(),
        "Dove": good.bookmaker.map(lambda b: BOOKMAKERS.get(b, b)).to_numpy(),
        "Quota giusta": good.fair_odds.round(2).to_numpy(),
        "Valore %": (100 * good.value).round(1).to_numpy(),
        "Puntata (% budget)": (100 * good.stake).round(2).to_numpy(),
    }), hide_index=True, width="stretch")
    st.caption(f"{len(good)} occasioni su {len(vt)} quote {BOOKMAKER_NAME} controllate (aggiornate il {fetched}). "
               "Controlla sempre la quota sul sito prima di giocare: può essere cambiata.")


def slip_section():
    """Singles vs accumulator on the selected bets (assumed independent: different matches only)."""
    slip = st.session_state.get("slip", [])
    st.subheader("Schedina")
    if not slip:
        st.caption("Vuota. Scrivi le quote del tuo bookmaker e aggiungi le giocate con valore.")
        return
    s = pd.DataFrame(slip)
    st.dataframe(s.assign(**{"Probabilità %": (100 * s.prob).round(1)})[["Partita", "Mercato", "Probabilità %", "Quota"]],
                 hide_index=True, width="stretch")
    singles_ev = (s.prob * s.Quota - 1).mean()
    same_match = s.Partita.duplicated().any()
    acc_p, acc_q = s.prob.prod(), s.Quota.prod()
    c1, c2, c3 = st.columns(3)
    c1.metric("Valore medio delle singole", f"{100 * singles_ev:+.1f}%")
    c2.metric("Multipla: quota", f"{acc_q:.2f}", f"probabilità {100 * acc_p:.1f}%", delta_color="off")
    c3.metric("Multipla: valore", f"{100 * (acc_p * acc_q - 1):+.1f}%")
    if same_match:
        st.warning("Ci sono due giocate sulla stessa partita: non sono indipendenti e il calcolo della multipla "
                   "non vale (molti bookmaker non le accettano insieme).")
    st.caption("In una multipla i valori si moltiplicano: se ogni giocata ha valore la multipla ne ha di più, ma se "
               "anche una sola è sotto la quota giusta il margine del bookmaker si accumula. La multipla vince "
               "molto più di rado: a parità di valore atteso, le singole oscillano molto meno.")
    if st.button("Svuota la schedina", key="slip_clear"):
        st.session_state["slip"] = []
        st.rerun()


HISTORY = pd.DataFrame({   # strategy_research: Bet365, 10 campionati, verifica 2024/25-2025/26
    "Come giocare": ["Singola: la più probabile di ogni partita", "…con quota 1,01-1,30", "Doppia delle più probabili",
                     "Tripla", "Quadrupla", "Cinquina", "«Valore» su un solo bookmaker (oltre +5%)"],
    "Vinte": ["60%", "83%", "37%", "21%", "13%", "8%", "10%"],
    "Rendimento": ["−3,6%", "+0,1%", "−4,4%", "−10,0%", "−11,6%", "−15,9%", "−49,5%"],
})
SLIPS = advisor.PRESETS


def matches_label(df):
    k = pd.to_datetime(df.kickoff_utc, utc=True).dt.tz_convert(TZ)
    return (k.dt.dayofweek.map(dict(enumerate(DAYS))) + " " + k.dt.strftime("%d/%m %H:%M")).to_numpy()


def advice_data():
    """Upcoming matches, fair-odds parameters and the value table of the bookmaker prices (None if missing)."""
    up = read_csv("fair_odds.csv")
    bk = read_csv("bookmaker_odds.csv")
    if up is None or bk is None or up.empty or bk.empty:
        st.warning(f"Servono le quote {BOOKMAKER_NAME}: arrivano con l'aggiornamento automatico (di solito dal venerdì).")
        return None
    up = up[pd.to_datetime(up.kickoff_utc, utc=True) > pd.Timestamp.now(tz="UTC")]
    params = load_params()
    cal = read_csv("markets_calibration.csv")
    err = {}
    if cal is not None:
        t = cal[cal.period.str.startswith("verifica")]
        err = {(g, m): d for g, m, d in zip(t.group, t.market, t.diff_pp)}
    vt = value_table(up, bk, params, err)
    if vt.empty:
        st.warning(f"Nessuna quota {BOOKMAKER_NAME} per le prossime partite.")
        return None
    return up, params, vt, bk


def how_we_choose():
    with st.expander("Come scelgo le giocate (e perché poche partite per schedina)", expanded=False):
        st.markdown(
            "- **La giocata più probabile** di ogni partita, con quota tra 1,20 e 2,50, tra i mercati dove "
            f"{BOOKMAKER_NAME} trattiene meno (Under/Over, Doppia chance, Goal, 1X2: circa il 6%; oltre quota 5 "
            "il bookmaker trattiene il 29%).\n"
            "- **Mai due giocate della stessa partita** in una multipla: il regolamento prevede per questo i mercati "
            "Combo, che vanno giocati come una sola scommessa.\n"
            "- **Al massimo due o tre partite**: ogni partita in più moltiplica anche il margine del bookmaker.\n"
            "- Nel regolamento una partita rinviata o annullata vale **quota 1,00** nella multipla.")
        st.dataframe(HISTORY, hide_index=True, width="stretch")
        st.caption("Verifica 2024/25-2025/26 alle quote Bet365, 10 campionati. Nessun modo di giocare ha guadagnato in "
                   "modo affidabile: il più probabile, a quota bassa e in singola, è quello che perde meno.")


def slip_card(m, stake, title=None):
    """One slip: legs, total odds, chance to win, payout and average result."""
    if title:
        st.markdown(f"**{title}**")
    for leg, when in zip(m["legs"].itertuples(), matches_label(m["legs"])):
        st.markdown(f"{when} · {ALL_LEAGUE_NAMES.get(leg.league, leg.league)}  \n{leg.home} – {leg.away}: "
                    f"**{leg.market}** @ {leg.odds:.2f} ({BOOKMAKERS.get(leg.bookmaker, leg.bookmaker)})")
    o = advisor.expected_outcome(m["prob"], m["odds"], stake)
    st.metric("Quota totale", f"{m['odds']:.2f}", f"vince {o['win_pct']:.0f}% delle volte", delta_color="off")
    st.caption(f"Con {stake:.0f} € incassi {o['payout']:.2f} € se vince. In media, ripetendola tante volte: "
               f"{o['avg_result']:+.2f} € a giocata ({100 * m['value']:+.1f}%).")


PERIODS = ["Prossimi 7 giorni", "Oggi", "Domani", "Fino a domenica"]


def in_period(kickoff_utc: pd.Series, period: str) -> pd.Series:
    k = pd.to_datetime(kickoff_utc, utc=True).dt.tz_convert(TZ)
    today = pd.Timestamp.now(tz=TZ).normalize()
    if period == "Oggi":
        return k.dt.normalize() == today
    if period == "Domani":
        return k.dt.normalize() == today + pd.Timedelta(days=1)
    if period == "Fino a domenica":
        return k.dt.normalize() <= today + pd.Timedelta(days=6 - today.dayofweek)
    return k <= today + pd.Timedelta(days=8)


def page_slips():
    st.title("Schedine pronte")
    st.info(DISCLAIMER)
    data = advice_data()
    if data is None:
        return
    up, params, vt, bk = data
    how_we_choose()

    c1, c2 = st.columns(2)
    period = c1.selectbox("Partite", PERIODS, key="sl_period")
    stake = c2.number_input("Puntata per schedina (€)", min_value=1.0, max_value=1000.0, value=10.0, step=1.0,
                            key="sl_stake")
    leagues = [d for d in ALL_LEAGUE_NAMES if d in set(vt.league)]
    chosen = st.multiselect("Campionati", leagues, default=leagues, format_func=ALL_LEAGUE_NAMES.get, key="sl_leagues")
    pool = vt[vt.league.isin(chosen) & in_period(vt.kickoff_utc, period).to_numpy()]
    n_matches = pool[["league", "home", "away"]].drop_duplicates().shape[0]
    st.caption(f"{n_matches} partite con quote {BOOKMAKER_NAME} in {len(chosen)} campionati "
               f"(quote del {fmt_time(bk.fetched_at.max())}). Controlla sempre la quota sul sito prima di giocare.")
    if pool.empty:
        st.warning("Nessuna partita con questi filtri.")
        return

    st.subheader("Le più probabili")
    cols = st.columns(len(SLIPS))
    for col, (name, (k, target)) in zip(cols, SLIPS.items()):
        with col:
            m = advisor.best_multiple(pool, k, target)
            if m is None:
                st.markdown(f"**{name}**")
                st.caption("Nessuna combinazione possibile.")
            else:
                slip_card(m, stake, name)

    st.subheader("Costruisci la tua")
    c1, c2, c3 = st.columns(3)
    k = c1.slider("Partite", 1, 5, 2, key="sl_k")
    target = c2.number_input("Quota totale almeno", min_value=1.2, max_value=50.0, value=2.0, step=0.1, key="sl_q")
    objective = c3.radio("Obiettivo", ["Più probabile", "Miglior rendimento"], key="sl_obj")
    if k >= 4:
        st.warning(f"Con {k} partite il margine del bookmaker si moltiplica: nello storico le quadruple hanno reso "
                   "−12% e le cinquine −16%, contro il −4% delle doppie.")
    slips = advisor.alternative_slips(pool, k, target, 3, "prob" if objective == "Più probabile" else "value")
    if not slips:
        st.caption("Nessuna combinazione raggiunge questa quota con le regole del consigliere (quota per partita "
                   "1,20-2,50): prova meno partite in più o una quota più bassa.")
    cols = st.columns(3)
    for i, (col, m) in enumerate(zip(cols, slips)):
        with col:
            slip_card(m, stake, f"Alternativa {i + 1}")
    if slips:
        st.caption("Le alternative non hanno partite in comune: si possono giocare anche tutte. «Miglior rendimento» "
                   "cerca la combinazione che perde meno in media (o guadagna, se capita), di solito con quota "
                   "più alta e probabilità più bassa.")


def page_advice():
    st.title("Giocate per partita")
    st.info(DISCLAIMER)
    data = advice_data()
    if data is None:
        return
    up, params, vt, bk = data
    how_we_choose()

    st.subheader("Giocata consigliata per ogni partita")
    bp, cp = advisor.best_picks(vt), advisor.cheapest_picks(vt)
    if bp.empty:
        st.caption("Nessuna giocata rispetta le regole.")
    else:
        table = bp[["league", "kickoff_utc", "home", "away", "market", "odds", "prob", "bookmaker"]].merge(
            cp[["league", "home", "away", "market", "odds", "prob", "value"]], on=["league", "home", "away"],
            how="left", suffixes=("", "_c")).sort_values("kickoff_utc")
        st.dataframe(pd.DataFrame({
            "Data (Roma)": matches_label(table),
            "Campionato": table.league.map(ALL_LEAGUE_NAMES).to_numpy(),
            "Partita": (table.home + " – " + table.away).to_numpy(),
            "Più probabile": (table.market + " @ " + table.odds.map("{:.2f}".format) + " ("
                              + table.bookmaker.map(lambda b: BOOKMAKERS.get(b, b)) + ")").to_numpy(),
            "Vince (%)": (100 * table.prob).round(0).to_numpy(),
            "Più conveniente": [f"{m} @ {o:.2f}" if isinstance(m, str) else "—" for m, o in zip(table.market_c, table.odds_c)],
            "Vince (%) ": [round(100 * p) if p == p else None for p in table.prob_c],
            "Costo atteso": [f"{100 * v:+.1f}%" if v == v else "—" for v in table.value],
        }), hide_index=True, width="stretch")
        st.caption("Più probabile: la giocata che vince più spesso. Più conveniente: tra quelle con almeno il 60% di "
                   f"probabilità, quella su cui {BOOKMAKER_NAME} trattiene meno (costo atteso più vicino a zero).")

    st.subheader("Combo e Multigoal: quote da cercare")
    st.caption(f"{BOOKMAKER_NAME} non pubblica Combo e Multigoal su OddsPapi: ecco le più probabili di ogni partita e "
               "la quota sotto cui non conviene giocarle (quota giusta − 6%, lo stesso costo massimo delle altre).")
    labels = [f"{w} · {h} – {a}" for w, h, a in zip(matches_label(up), up.home, up.away)]
    i = st.selectbox("Partita", range(len(up)), format_func=lambda j: labels[j], key="adv_game")
    ex = advisor.exotic_suggestions(fair_odds_for(up.iloc[i], params))
    if ex.empty:
        st.caption("Nessun Combo o Multigoal con quota giusta tra 1,30 e 2,50.")
    else:
        st.dataframe(pd.DataFrame({
            "Mercato": (ex.group + ": " + ex.market).to_numpy(), "Probabilità %": (100 * ex.prob).round(0).to_numpy(),
            "Quota giusta": ex.fair_odds.round(2).to_numpy(),
            "Gioca solo se la quota è almeno": (ex.fair_odds * (1 - advisor.MAX_COST)).round(2).to_numpy(),
        }), hide_index=True, width="stretch")


def tracking_table(s: pd.DataFrame, label: str) -> pd.DataFrame:
    pct = lambda v: "—" if pd.isna(v) else f"{100 * v:+.1f}%"
    ci = lambda lo, hi: "—" if pd.isna(lo) else f"{100 * lo:+.1f}% … {100 * hi:+.1f}%"
    return pd.DataFrame({
        label: s.group.astype(str).to_numpy(), "Giocate": s.bets.to_numpy(), "Vinte": s.wins.to_numpy(),
        "Rendimento": [pct(v) for v in s.roi], "IC 95%": [ci(a, b) for a, b in zip(s.roi_lo, s.roi_hi)],
        "CLV medio": [pct(v) for v in s.clv], "IC 95% CLV": [ci(a, b) for a, b in zip(s.clv_lo, s.clv_hi)],
    })


def page_tracking():
    st.title("Verifica dei consigli")
    st.info(DISCLAIMER)
    st.markdown(
        "Ogni consiglio viene salvato **prima della partita** e non viene più modificato. Dopo la partita si "
        "segna l'esito e si calcola il **CLV** (closing line value): la quota presa confrontata con la probabilità "
        "del mercato alla chiusura. **CLV positivo in media = le quote prese erano migliori di quelle finali**, il "
        "segnale più rapido che i consigli valgono qualcosa: bastano poche settimane, mentre per il rendimento "
        "servono centinaia di giocate.")
    from forecast1x2.tracking import PICKS, SLIPS as SLIPS_PATH, leg_results, results_table, summary as tr_summary
    picks = pd.read_csv(PICKS) if PICKS.exists() else pd.DataFrame()
    slips = pd.read_csv(SLIPS_PATH) if SLIPS_PATH.exists() else pd.DataFrame()
    if picks.empty:
        st.warning("Il registro è vuoto: si riempie da solo con gli aggiornamenti automatici.")
        return
    done = picks[picks.status.isin(["vinta", "persa", "rimborsata"])]
    c1, c2, c3 = st.columns(3)
    c1.metric("Giocate registrate", len(picks))
    c2.metric("Concluse", len(done))
    c3.metric("In attesa del risultato", int((picks.status == "aperta").sum()))
    if len(done) < 100:
        st.caption("Meno di 100 giocate concluse: i numeri qui sotto dipendono ancora molto dal caso. "
                   "Il CLV diventa leggibile dopo 3-4 settimane, il rendimento dopo una stagione.")

    st.subheader("Giocate singole")
    st.dataframe(tracking_table(tr_summary(picks, "kind"), "Tipo"), hide_index=True, width="stretch")
    with st.expander("Per mercato e per bookmaker"):
        st.dataframe(tracking_table(tr_summary(picks, "group").iloc[1:], "Mercato"), hide_index=True, width="stretch")
        b = tr_summary(picks.assign(book=picks.bookmaker.map(lambda x: BOOKMAKERS.get(x, x))), "book").iloc[1:]
        st.dataframe(tracking_table(b, "Bookmaker"), hide_index=True, width="stretch")

    st.subheader("Schedine pronte")
    if slips.empty:
        st.caption("Nessuna schedina registrata.")
    else:
        st.dataframe(tracking_table(tr_summary(slips.assign(kickoff_utc=slips.logged_at), "preset"), "Schedina"),
                     hide_index=True, width="stretch")

    ou = load_register()
    ou = ou[ou.status.isin(["vinta", "persa"])]
    if len(ou):
        st.subheader("Consigli in prova over/under (regola ou25-v1)")
        legs = pd.DataFrame({"league": ou.league, "kickoff_utc": ou.kickoff_utc, "home": ou.home, "away": ou.away,
                             "group": "Under/Over", "market": ou.pick + " 2.5"})
        lr = leg_results(legs, results_table(), load_params())
        ou = ou.reset_index(drop=True).assign(clv=ou.odds_best.to_numpy() * lr.close_prob.to_numpy() - 1)
        st.dataframe(tracking_table(tr_summary(ou), "Registro"), hide_index=True, width="stretch")

    if len(done):
        st.subheader("Ultime giocate concluse")
        last = done.sort_values("kickoff_utc", ascending=False).head(30)
        st.dataframe(pd.DataFrame({
            "Data (Roma)": matches_label(last), "Partita": (last.home + " – " + last.away).to_numpy(),
            "Giocata": (last.market + " @ " + last.odds.map("{:.2f}".format)).to_numpy(),
            "Dove": last.bookmaker.map(lambda x: BOOKMAKERS.get(x, x)).to_numpy(),
            "Esito": last.status.to_numpy(),
            "CLV": [("—" if pd.isna(v) else f"{100 * v:+.1f}%") for v in last.clv],
        }), hide_index=True, width="stretch")


page = st.sidebar.radio("Pagina", ["Schedine pronte", "Giocate per partita", "Verifica", "Previsioni",
                                   "Quote giuste e strategia", "Consigli in prova", "Backtest"])
sidebar_update(read_meta())
if page == "Previsioni":
    page_predictions()
elif page == "Verifica":
    page_tracking()
elif page == "Schedine pronte":
    page_slips()
elif page == "Giocate per partita":
    page_advice()
elif page == "Quote giuste e strategia":
    page_fair_odds()
elif page == "Consigli in prova":
    page_tips()
else:
    page_backtest()
