#!/bin/bash
# Disattiva e rimuove l'aggiornamento automatico.
LABEL="com.forecast1x2.auto"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
echo "Aggiornamento automatico disattivato."
