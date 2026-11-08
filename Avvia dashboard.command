#!/bin/bash
# Apre la dashboard nel browser (http://localhost:8501). Chiudi questa finestra per fermarla.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/streamlit ]; then
  ./installa.command
fi
exec .venv/bin/streamlit run app/dashboard.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false
