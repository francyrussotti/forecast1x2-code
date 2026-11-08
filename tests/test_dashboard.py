from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
NEEDED = ["predictions.csv", "meta.json", "backtest_metrics.csv", "backtest_bootstrap.csv", "backtest_matches.csv"]


@pytest.mark.skipif(not all((ROOT / "outputs" / f).exists() for f in NEEDED),
                    reason="outputs/ non ancora generati (lancia aggiorna e backtest)")
def test_dashboard_pages_render_without_errors():
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=60).run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Backtest").run()
    assert not at.exception
    at.selectbox[1].set_value("2526").run()   # season filter
    assert not at.exception
