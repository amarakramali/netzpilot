#!/usr/bin/env python3
"""Verify fetch_smard-CLI OHNE Netz: fetch_series wird gemockt, geprüft wird die komplette
CSV-/Provenance-/Generation-Logik (das, was zwischen SMARD-API und Engine liegt).

Der Netz-Connector selbst (netzpilot/data/smard.py) ist T2-verifiziert; URL-Format zuletzt
2026-06-04 live gegengeprüft. Hier geht es um die CLI-Schicht: Spalten, Einheiten-Doku,
Provenance-Sidecar, PV+Wind-Summe, cache_dir=None (keine pyarrow-Abhängigkeit),
Feiertags-Region-Mapping. Exit!=0 bei Fehler.
"""
from __future__ import annotations
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

import netzpilot.data.smard as smard
import scripts.fetch_smard as fs

N = [0]
def check(ok, msg):
    N[0] += 1
    print(("ok  " if ok else "FAIL"), f"{N[0]:2d}:", msg)
    if not ok:
        sys.exit(1)

# --- Mock: 70 Tage synthetische Stundenreihe (deterministisch, GW-Skala) ---
idx = pd.date_range("2026-03-01", periods=70 * 24, freq="h", tz="UTC")
base = 18000 + 4000 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi)
REAL = pd.Series(base, index=idx, name="value")

calls = []
def fake_fetch_series(start, end, filter_id=410, region="DE", resolution="hour",
                      cache_dir=None, force=False):
    calls.append((filter_id, region, resolution, cache_dir))
    check(cache_dir is None, f"fetch_series(filter {filter_id}) mit cache_dir=None aufgerufen (kein pyarrow)")
    if filter_id == smard.FILTERS["load"]:
        return REAL
    if filter_id == smard.FILTERS["wind_offshore"]:
        return REAL.iloc[0:0]          # leere Reihe: Amprion hat kein Offshore -> muss sauber uebersprungen werden
    return REAL * 0.25                  # PV / Wind onshore

fs.fetch_series = fake_fetch_series     # Modul-lokaler Name in fetch_smard

with tempfile.TemporaryDirectory() as td:
    csv, gen, prov = fs.fetch_to_csv("Amprion", 60, td, with_generation=True)

    df = pd.read_csv(csv)
    check(list(df.columns) == ["timestamp_utc", "load_mw"], "Last-CSV: Spalten timestamp_utc,load_mw")
    check(len(df) == len(REAL), f"Last-CSV: alle {len(REAL)} Stunden geschrieben")
    check(abs(float(df['load_mw'].mean()) - float(REAL.mean())) < 1e-6, "Last-CSV: Werte unveraendert (Mittel exakt)")

    gdf = pd.read_csv(gen)
    check(list(gdf.columns) == ["timestamp_utc", "generation_mw"], "Gen-CSV: Spalten timestamp_utc,generation_mw")
    check(abs(float(gdf['generation_mw'].iloc[0]) - 2 * 0.25 * float(REAL.iloc[0])) < 1e-6,
          "Gen-CSV: PV+Wind_onshore summiert, leeres Offshore uebersprungen")

    sj = json.load(open(csv.replace(".csv", "_source.json"), encoding="utf-8"))
    for k in ("source", "region", "n_hours_load", "unit", "retrieved_utc", "level_note"):
        check(k in sj, f"Provenance-Sidecar enthaelt '{k}'")
    check(sj["region"] == "Amprion" and sj["unit"] == "MW" and sj["n_hours_load"] == len(REAL),
          "Provenance: Region/Einheit/Stundenzahl korrekt")
    check("Regelzonen" in sj["level_note"], "Provenance: ehrliche Ebenen-Einordnung (Regelzone, kein Stadtwerk)")
    check(isinstance(sj["generation_parts"], dict) and sj["generation_parts"].get("pv") == len(REAL),
          "Provenance: generation_parts dokumentiert")

filters_called = [c[0] for c in calls]
check(filters_called == [smard.FILTERS["load"], smard.FILTERS["pv"],
                         smard.FILTERS["wind_onshore"], smard.FILTERS["wind_offshore"]],
      "Reihenfolge der fetch_series-Aufrufe: load, pv, wind_onshore, wind_offshore")
check(all(c[1] == "Amprion" and c[2] == "hour" for c in calls), "alle Aufrufe mit Region/Resolution der CLI")

import argparse  # CLI-Choices gegen HOLIDAY_REGION pruefen (jede waehlbare Region hat einen Kalender)
choices = ["DE", "Amprion", "TenneT", "TransnetBW", "50Hertz", "AT", "LU"]
check(all(r in fs.HOLIDAY_REGION for r in choices), "HOLIDAY_REGION deckt alle CLI-Regionen ab")

print(f"ALLE {N[0]} CHECKS GRUEN — fetch_smard-CLI (ohne Netz) verifiziert.")
