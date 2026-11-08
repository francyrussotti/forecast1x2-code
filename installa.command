#!/bin/bash
# Prima installazione (o reinstallazione): crea l'ambiente Python e scarica dati e previsioni.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Creo l'ambiente Python in .venv ..."
  python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/pip install -q -e .
echo "Scarico i dati e calcolo le previsioni (la prima volta può richiedere qualche minuto)..."
.venv/bin/python -W ignore -m forecast1x2 aggiorna
echo
echo "Installazione completata. Per aprire la dashboard fai doppio clic su «Avvia dashboard.command»."
