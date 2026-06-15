# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""NetzPilot Service-Schicht: macht aus der Batch-Engine einen betreibbaren Dienst.

- app.py:   FastAPI-REST-API (Prognose, §14a-Fahrplan, Persistenz-Abruf)
- store.py: schlanke Datei-Persistenz (JSON je Lauf, kein DB-Server nötig)
- runner.py: orchestriert Loader -> forecast_next_day -> §14a-Fahrplan (echte Engine, keine Nachbildung)
"""
