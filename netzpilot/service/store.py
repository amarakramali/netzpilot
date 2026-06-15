# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Schlanke Datei-Persistenz für Prognoseläufe — kein DB-Server nötig.

Jeder Lauf je Mandant (utility) wird als JSON unter store_dir/<utility>/<date>.json abgelegt;
`latest.json` zeigt auf den jüngsten. Bewusst dateibasiert: läuft offline beim Stadtwerk,
leicht auditierbar, kein Betriebsaufwand. Tauschbar gegen echte DB, wenn nötig.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone


class ForecastStore:
    def __init__(self, base_dir: str = "data_cache/service_store"):
        self.base = base_dir
        os.makedirs(self.base, exist_ok=True)

    def _utility_dir(self, utility: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in utility) or "default"
        d = os.path.join(self.base, safe)
        os.makedirs(d, exist_ok=True)
        return d

    def save(self, utility: str, result: dict) -> str:
        d = self._utility_dir(utility)
        date = result.get("forecast_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        record = dict(result)
        record["stored_utc"] = datetime.now(timezone.utc).isoformat()
        path = os.path.join(d, f"{date}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        with open(os.path.join(d, "latest.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        return path

    def latest(self, utility: str) -> dict | None:
        p = os.path.join(self._utility_dir(utility), "latest.json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def get(self, utility: str, date: str) -> dict | None:
        p = os.path.join(self._utility_dir(utility), f"{date}.json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def history(self, utility: str) -> list[str]:
        d = self._utility_dir(utility)
        return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json") and f != "latest.json")

    def list_utilities(self) -> list[str]:
        if not os.path.isdir(self.base):
            return []
        return sorted(n for n in os.listdir(self.base) if os.path.isdir(os.path.join(self.base, n)))
