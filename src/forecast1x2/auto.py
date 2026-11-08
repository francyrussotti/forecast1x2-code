"""Match-driven automatic updates.

`python -m forecast1x2 auto` is meant to run every 30 minutes (launchd). It updates data and
predictions only when a trigger has passed since the last update:
  - 24h and 3h before each kickoff (early odds and fresh predictions),
  - 3h and 24h after each kickoff (results; football-data can lag by a day),
  - at least once every 24h.
Otherwise it exits immediately without network access.
"""
import json
from datetime import datetime, timezone

import pandas as pd

from .data import PROCESSED, ROOT

OFFSETS = [pd.Timedelta(hours=-24), pd.Timedelta(hours=-3), pd.Timedelta(hours=3), pd.Timedelta(hours=24)]
HEARTBEAT = pd.Timedelta(hours=24)
META = ROOT / "outputs" / "meta.json"


def triggers(kickoffs: pd.Series) -> pd.Series:
    """All trigger times derived from the kickoff times (UTC), sorted."""
    k = pd.to_datetime(kickoffs, utc=True).dropna()
    return pd.concat([k + off for off in OFFSETS]).sort_values().reset_index(drop=True)


def update_due(now: pd.Timestamp, kickoffs: pd.Series, last_update) -> str:
    """Reason for updating now, or '' if nothing is due."""
    if last_update is None:
        return "nessun aggiornamento precedente"
    if now - last_update >= HEARTBEAT:
        return "ultimo aggiornamento più vecchio di 24 ore"
    t = triggers(kickoffs)
    passed = t[(t > last_update) & (t <= now)]
    if not passed.empty:
        return f"evento partita alle {passed.iloc[-1]:%Y-%m-%d %H:%M} UTC"
    return ""


def next_update(now: pd.Timestamp, kickoffs: pd.Series, last_update) -> pd.Timestamp:
    """When the next update will happen (first trigger after now, or the heartbeat)."""
    t = triggers(kickoffs)
    candidates = [t[t > now].min()] if (t > now).any() else []
    if last_update is not None:
        candidates.append(last_update + HEARTBEAT)
    return min(candidates) if candidates else now


def last_update_time():
    if not META.exists():
        return None
    return pd.Timestamp(json.loads(META.read_text())["data_updated_at"])


def load_kickoffs() -> pd.Series:
    path = PROCESSED / "fixtures.csv"
    return pd.read_csv(path).kickoff_utc if path.exists() else pd.Series(dtype=str)


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC] {msg}", flush=True)


def run_auto() -> int:
    from .data import PipelineBusy, pipeline_lock
    try:
        with pipeline_lock():
            return _run_locked()
    except PipelineBusy:
        log("aggiornamento già in corso, salto")
        return 0


def _run_locked() -> int:
    now = pd.Timestamp.now(tz="UTC")
    reason = update_due(now, load_kickoffs(), last_update_time())
    if not reason:
        return 0
    log(f"aggiorno: {reason}")
    from .data import DataError, update_data
    from .model import predict_all
    try:
        update_data()
        predict_all()
    except DataError as e:   # retried at the next run, since data_updated_at did not move
        log(f"ERRORE: {e}")
        return 1
    log("aggiornamento completato")
    return 0
