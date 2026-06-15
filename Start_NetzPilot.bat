@echo off
REM ============================================================
REM  NetzPilot starten — ein Doppelklick.
REM  Richtet beim ersten Mal automatisch alles ein (venv + Pakete),
REM  startet das echte Python-Backend und oeffnet den Browser.
REM ============================================================
setlocal
cd /d "%~dp0"
title NetzPilot

echo.
echo ===== NetzPilot wird gestartet =====
echo (Beim ersten Start dauert die Einrichtung 1-3 Minuten.)
echo.

REM --- Python finden ---
where py >nul 2>nul
if %errorlevel%==0 ( set "PYLAUNCH=py -3" ) else (
  where python >nul 2>nul
  if %errorlevel%==0 ( set "PYLAUNCH=python" ) else (
    echo [FEHLER] Python wurde nicht gefunden.
    echo Bitte Python 3.10+ von https://www.python.org/downloads/ installieren
    echo und beim Setup "Add Python to PATH" anhaken. Danach diese Datei erneut doppelklicken.
    pause
    exit /b 1
  )
)

REM --- venv anlegen (nur beim ersten Mal) ---
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Erstelle virtuelle Umgebung .venv ...
  %PYLAUNCH% -m venv .venv
)
set "VENV_PY=.venv\Scripts\python.exe"

REM --- Pakete installieren (idempotent; schnell wenn schon da) ---
echo [2/3] Installiere/pruefe Pakete (requirements.txt) ...
"%VENV_PY%" -m pip install --disable-pip-version-check -q -r requirements.txt
if %errorlevel% neq 0 (
  echo [FEHLER] Paketinstallation fehlgeschlagen. Internetverbindung pruefen.
  pause
  exit /b 1
)

REM --- Backend starten + Browser oeffnen ---
echo [3/3] Starte NetzPilot-Server auf http://127.0.0.1:8000 ...
echo.
echo Der Browser oeffnet sich gleich. Zum BEENDEN dieses Fenster schliessen.
echo.
start "" "http://127.0.0.1:8000/"
"%VENV_PY%" -m uvicorn netzpilot.service.app:app --host 127.0.0.1 --port 8000

echo.
echo NetzPilot wurde beendet.
pause
