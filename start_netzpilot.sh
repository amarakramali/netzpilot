#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

# ============================================================
#  NetzPilot starten (macOS / Linux).
#  Richtet beim ersten Mal alles ein (venv + Pakete), startet das
#  echte Python-Backend und oeffnet den Browser.
#    chmod +x start_netzpilot.sh   # einmalig
#    ./start_netzpilot.sh
# ============================================================
set -e
cd "$(dirname "$0")"

echo
echo "===== NetzPilot wird gestartet ====="
echo "(Beim ersten Start dauert die Einrichtung 1-3 Minuten.)"
echo

PY=python3
command -v "$PY" >/dev/null 2>&1 || { echo "[FEHLER] python3 nicht gefunden. Bitte Python 3.10+ installieren."; exit 1; }

if [ ! -x ".venv/bin/python" ]; then
  echo "[1/3] Erstelle virtuelle Umgebung .venv ..."
  "$PY" -m venv .venv
fi
VENV_PY=".venv/bin/python"

echo "[2/3] Installiere/pruefe Pakete (requirements.txt) ..."
"$VENV_PY" -m pip install --disable-pip-version-check -q -r requirements.txt

URL="http://127.0.0.1:8000/"
echo "[3/3] Starte NetzPilot-Server auf $URL ..."
echo "Browser oeffnet sich gleich. Zum BEENDEN: Strg+C."
( sleep 2; (command -v open >/dev/null && open "$URL") || (command -v xdg-open >/dev/null && xdg-open "$URL") || true ) &
exec "$VENV_PY" -m uvicorn netzpilot.service.app:app --host 127.0.0.1 --port 8000
