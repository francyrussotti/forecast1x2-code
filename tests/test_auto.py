import pandas as pd

from forecast1x2.auto import next_update, triggers, update_due

T = lambda s: pd.Timestamp(s, tz="UTC")
KICKOFFS = pd.Series(["2026-10-17 18:45:00+00:00", "2026-10-18 13:00:00+00:00"])


def test_triggers_around_each_kickoff():
    t = triggers(KICKOFFS)
    assert T("2026-10-16 18:45") in t.tolist()   # 24h before
    assert T("2026-10-17 15:45") in t.tolist()   # 3h before
    assert T("2026-10-17 21:45") in t.tolist()   # 3h after
    assert T("2026-10-19 13:00") in t.tolist()   # 24h after
    assert len(t) == 8


def test_nothing_due_between_events():
    # updated right after the 24h-before trigger; next trigger is 3h before kickoff
    assert update_due(T("2026-10-17 10:00"), KICKOFFS, T("2026-10-16 19:00")) == ""


def test_due_after_a_trigger_passes():
    reason = update_due(T("2026-10-17 16:00"), KICKOFFS, T("2026-10-16 19:00"))
    assert "2026-10-17 15:45" in reason


def test_trigger_consumed_once_updated():
    assert update_due(T("2026-10-17 16:30"), KICKOFFS, T("2026-10-17 16:00")) == ""


def test_heartbeat_and_first_run():
    assert "24 ore" in update_due(T("2026-11-20 10:00"), KICKOFFS, T("2026-11-19 09:00"))
    assert update_due(T("2026-11-20 10:00"), KICKOFFS, None) != ""


def test_next_update():
    # 24h before the second kickoff (18/10 13:00) comes before 3h before the first one (17/10 15:45)
    assert next_update(T("2026-10-17 10:00"), KICKOFFS, T("2026-10-16 19:00")) == T("2026-10-17 13:00")
    # far from any match: the daily heartbeat
    assert next_update(T("2026-12-01 10:00"), KICKOFFS, T("2026-12-01 09:00")) == T("2026-12-02 09:00")
