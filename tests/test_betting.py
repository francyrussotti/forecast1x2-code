import numpy as np
import pandas as pd

from forecast1x2.evaluation import betting_simulation, risk_level, risk_table


def test_risk_level_boundaries():
    assert risk_level([0.75, 0.60, 0.5999, 0.45, 0.4499, 0.34]).tolist() == [1, 1, 2, 2, 3, 3]


def _two_matches():
    # match 1: EV home = 0.6*2.0-1 = +0.20 (best), result H -> +1.00
    # match 2: EV away = 0.4*2.7-1 = +0.08 (best), result D -> -1.00
    return pd.DataFrame({
        "result": ["H", "D"], "week_start": pd.to_datetime(["2025-01-06", "2025-01-13"]),
        "model_1": [0.6, 0.3], "model_X": [0.25, 0.3], "model_2": [0.15, 0.4],
        "open_h": [2.0, 2.5], "open_d": [3.5, 3.2], "open_a": [5.0, 2.7],
    })


def test_betting_simulation_by_hand():
    sim = betting_simulation(_two_matches(), "model", thresholds=[0.0, 0.10], n_boot=200).set_index(["strategy", "threshold"])
    r = sim.loc[("valore", 0.0)]
    assert (r.bets, r.wins, r.profit, r.roi) == (2, 1, 0.0, 0.0)
    r = sim.loc[("valore", 0.10)]              # only match 1 has EV above 10%
    assert (r.bets, r.wins, r.profit, r.roi) == (1, 1, 1.0, 1.0)
    fav = sim.xs("favorito", level="strategy").iloc[0]   # market favourite: home both times (2.0, 2.5)
    assert (fav.bets, fav.wins, fav.profit) == (2, 1, 0.0)


def test_risk_table():
    t = risk_table(_two_matches(), "model").set_index("level")
    assert t.loc[1, "n"] == 1 and t.loc[1, "fav_win_rate"] == 1.0   # 0.60 favourite, home won
    assert t.loc[3, "n"] == 1 and t.loc[3, "fav_win_rate"] == 0.0   # 0.40 favourite (away), draw
    assert t.loc[2, "n"] == 0 and np.isnan(t.loc[2, "fav_win_rate"])
