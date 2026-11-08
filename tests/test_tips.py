import numpy as np
import pandas as pd
import pytest

from forecast1x2 import tips

NOW = pd.Timestamp("2026-10-23 12:00", tz="UTC")
HEADER = "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA,Max>2.5,Max<2.5,Avg>2.5,Avg<2.5\n"


def fixtures_file(tmp_path, rows):
    p = tmp_path / "fd_fixtures.csv"
    p.write_text(HEADER + "".join(r + "\n" for r in rows))
    return p


def matches(rows):
    return pd.DataFrame(rows, columns=["league", "date", "home", "away", "hg", "ag"]).assign(
        date=lambda d: pd.to_datetime(d.date))


# fair over = (1/2)/(1/2 + 1/2) = 0.5; best over 2.2 -> edge +10%; best under 1.9 -> -5%
GOOD = "I2,24/10/2026,15:00,Bari,Pisa,2,3,4,2.2,1.9,2.0,2.0"
SMALL = "I1,24/10/2026,15:00,Roma,Lazio,2,3,4,2.03,1.9,2.0,2.0"      # edge +1.5%: below threshold
OTHER = "B1,24/10/2026,15:00,Genk,Gent,2,3,4,3.0,1.9,2.0,2.0"        # league not in the project
STARTED = "E0,23/10/2026,12:30,Leeds,Hull,2,3,4,2.2,1.9,2.0,2.0"     # less than an hour before kickoff


def test_edges_pick_the_better_side():
    df = pd.DataFrame({"Max>2.5": [2.2, 1.8], "Max<2.5": [1.9, 2.3], "Avg>2.5": [2.0, 2.0], "Avg<2.5": [2.0, 2.0]})
    e = tips.ou_edges(df)
    assert e.pick.tolist() == ["Over", "Under"]
    assert e.edge.round(4).tolist() == [0.10, 0.15]
    assert e.fair_prob.tolist() == [0.5, 0.5]


def test_kickoff_is_uk_time():
    k = tips.kickoff_utc(pd.DataFrame({"Date": ["24/10/2026", "01/11/2026"], "Time": ["15:00", "15:00"]}))
    assert k.tolist() == [pd.Timestamp("2026-10-24 14:00", tz="UTC"), pd.Timestamp("2026-11-01 15:00", tz="UTC")]


def test_register_adds_only_rule_matches_before_kickoff(tmp_path):
    reg_path = tmp_path / "reg.csv"
    fx = fixtures_file(tmp_path, [GOOD, SMALL, OTHER, STARTED])
    tips.update_register(matches([]), NOW, fx, reg_path)
    reg = tips.load_register(reg_path)
    assert reg[["league", "home", "pick", "status"]].values.tolist() == [["I2", "Bari", "Over", "aperta"]]
    assert reg.odds_best.iloc[0] == 2.2 and reg.tier.iloc[0] == 2 and reg.rule.iloc[0] == tips.RULE_VERSION


def test_register_never_rewrites_a_logged_tip(tmp_path):
    reg_path = tmp_path / "reg.csv"
    tips.update_register(matches([]), NOW, fixtures_file(tmp_path, [GOOD]), reg_path)
    changed = GOOD.replace("2.2,1.9", "2.5,1.9")   # odds move later: the logged odds must stay
    tips.update_register(matches([]), NOW + pd.Timedelta(hours=2), fixtures_file(tmp_path, [changed]), reg_path)
    reg = tips.load_register(reg_path)
    assert len(reg) == 1 and reg.odds_best.iloc[0] == 2.2


def test_stale_odds_are_ignored(tmp_path):
    fd = pd.read_csv(fixtures_file(tmp_path, [GOOD]), dtype=str)
    assert len(tips.candidates(fd, NOW, NOW - pd.Timedelta(days=1))) == 1
    assert tips.candidates(fd, NOW, NOW - pd.Timedelta(days=8)).empty


def test_fetch_time_saved_next_to_the_file_wins_over_file_time(tmp_path):
    from forecast1x2.data import fetched_at_of, fetched_at_path
    fx = fixtures_file(tmp_path, [GOOD])
    assert abs(fetched_at_of(fx) - pd.Timestamp.now(tz="UTC")) < pd.Timedelta(minutes=5)   # fallback: file time
    fetched_at_path(fx).write_text("2026-10-01T10:00:00+00:00")   # e.g. after a fresh git checkout
    assert fetched_at_of(fx) == pd.Timestamp("2026-10-01 10:00", tz="UTC")
    tips.update_register(matches([]), NOW, fx, tmp_path / "reg.csv")   # 22 days old: no new tips
    assert tips.load_register(tmp_path / "reg.csv").empty


def test_settlement_won_lost_and_void(tmp_path):
    reg_path = tmp_path / "reg.csv"
    under = "F2,24/10/2026,19:00,Metz,Caen,2,3,4,1.8,2.3,2.0,2.0"   # under +15%
    late = "D2,25/10/2026,12:00,Kiel,Ulm,2,3,4,2.2,1.9,2.0,2.0"     # never played
    tips.update_register(matches([]), NOW, fixtures_file(tmp_path, [GOOD, under, late]), reg_path)
    res = matches([["I2", "2026-10-24", "Bari", "Pisa", 2, 1],      # 3 goals: over wins at 2.2
                   ["F2", "2026-10-24", "Metz", "Caen", 2, 2]])     # 4 goals: under loses
    tips.update_register(res, NOW + pd.Timedelta(days=10), tmp_path / "missing.csv", reg_path)
    reg = tips.load_register(reg_path).set_index("home")
    assert reg.loc["Bari", "status"] == "vinta" and reg.loc["Bari", "profit"] == pytest.approx(1.2)
    assert reg.loc["Metz", "status"] == "persa" and reg.loc["Metz", "profit"] == -1
    assert reg.loc["Kiel", "status"] == "annullata" and reg.loc["Kiel", "profit"] == 0
    s = tips.summary(reg.reset_index(), n_boot=100).set_index("group")
    assert (s.loc["tutti", "bets"], s.loc["tutti", "wins"]) == (2, 1)
    assert s.loc["tutti", "roi"] == pytest.approx(0.1)
    assert s.loc["prima divisione", "bets"] == 0 and np.isnan(s.loc["prima divisione", "roi"])
