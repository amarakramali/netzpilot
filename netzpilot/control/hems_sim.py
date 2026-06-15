"""Simuliertes HEMS (Heimenergiemanagement).

Pollt den aEMT fuer seine MaLo, holt die aktive §14a-Wirkleistungsgrenze und wendet sie auf
eine steuerbare Verbrauchseinrichtung (z. B. Wallbox) an. Repraesentiert die Rolle, die im
Feld ein echtes HEMS hinter dem SMGW spielt. Wunsch des Nutzers = Nennleistung des Geraets.
"""
from __future__ import annotations
import json
from urllib.request import urlopen
from urllib.parse import quote


class Hems:
    def __init__(self, aemt_url: str, malo: str, device_nominal_kw: float):
        self.aemt_url = aemt_url.rstrip("/")
        self.malo = malo
        self.nominal = float(device_nominal_kw)

    def query_limit_kw(self, ts_iso: str):
        url = f"{self.aemt_url}/fahrplan/{self.malo}?at={quote(ts_iso, safe='')}"
        with urlopen(url, timeout=5) as r:
            return json.loads(r.read()).get("p_limit_kw")

    def applied_power_kw(self, ts_iso: str) -> dict:
        limit = self.query_limit_kw(ts_iso)
        desired = self.nominal
        applied = desired if limit is None else min(desired, float(limit))
        return {"ts": ts_iso, "desired_kw": desired, "limit_kw": limit,
                "applied_kw": round(applied, 2),
                "curtailed": limit is not None and applied < desired - 1e-9}
