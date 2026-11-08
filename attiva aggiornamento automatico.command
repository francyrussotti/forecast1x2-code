#!/bin/bash
# Attiva il controllo ogni 30 minuti: aggiorna dati e previsioni solo attorno alle partite
# (24h e 3h prima, 3h e 24h dopo) e almeno una volta al giorno. Log in logs/auto.log.
set -e
cd "$(dirname "$0")"
DIR="$(pwd)"
LABEL="com.forecast1x2.auto"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
if [ ! -x .venv/bin/python ]; then ./installa.command; fi
mkdir -p logs "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$DIR/.venv/bin/python</string>
    <string>-W</string><string>ignore</string>
    <string>-m</string><string>forecast1x2</string><string>auto</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>StartInterval</key><integer>1800</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$DIR/logs/auto.log</string>
  <key>StandardErrorPath</key><string>$DIR/logs/auto.log</string>
</dict>
</plist>
PLIST
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Aggiornamento automatico attivo (controllo ogni 30 minuti). Log: $DIR/logs/auto.log"
echo "Per disattivarlo: doppio clic su «disattiva aggiornamento automatico.command»."
