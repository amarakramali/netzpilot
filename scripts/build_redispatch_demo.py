#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Build a reproducible rolling redispatch demo on real public DSO rows."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.service.runner import run_forecast

OUT = Path("data_cache/benchmark/redispatch_demo.md")
MALO = "DE-DEMO-REDISPATCH-0001"

CASES = [
    {
        "name": "Hilden Netzumsatz 2025",
        "csv": "data_cache/real/Netzumsatz-Lastgang-2025.csv",
        "unit": "kW",
        "ts_col": "Text",
        "load_col": "Reihe1",
        "threshold_drop_mw": 0.8,
        "steuve_demands_kw": [1500.0, 1000.0, 800.0, 500.0],
    },
    {
        "name": "Herne Bezug vorgelagerte Ebene 2024 (110/10 kV)",
        "csv": "data_cache/real/herne_bezug_vorgelagerte_ebene_2024.csv",
        "unit": "kW",
        "ts_col": "Datum+von",
        "load_col": "Load_1",
        "threshold_drop_mw": 1.2,
        "steuve_demands_kw": [2500.0, 1600.0, 1200.0, 800.0],
    },
    {
        "name": "EVDB Jahreshoechstlast NS 2024",
        "csv": "data_cache/real/evdb_lastgang_ns_2024.csv",
        "unit": "kW",
        "ts_col": "Datum+von",
        "load_col": "Wert",
        "threshold_drop_mw": 0.3,
        "steuve_demands_kw": [300.0, 220.0, 180.0],
    },
]


def _constraints_ok(redispatch: dict, floor_kw: float = 4.2) -> tuple[bool, bool]:
    cap_ok = True
    floor_ok = True
    for h in redispatch["hourly"]:
        if not h["intervention"]:
            continue
        if h["feasible"] and sum(h["limits_kw"]) > h["cap_kw"] + 1e-6:
            cap_ok = False
        if any(l < floor_kw - 1e-9 for l in h["limits_kw"]):
            floor_ok = False
    return cap_ok, floor_ok


def _run_case(case: dict) -> dict:
    base = run_forecast(
        case["csv"],
        utility=case["name"],
        unit=case["unit"],
        ts_col=case["ts_col"],
        load_col=case["load_col"],
    )
    max_p50 = max(float(h["p50"]) for h in base["forecast"])
    threshold_mw = round(max_p50 - case["threshold_drop_mw"], 3)
    result = run_forecast(
        case["csv"],
        utility=case["name"],
        unit=case["unit"],
        ts_col=case["ts_col"],
        load_col=case["load_col"],
        congestion_threshold_mw=threshold_mw,
        steuve_malo=MALO,
        steuve_demands_kw=case["steuve_demands_kw"],
        rolling_redispatch=True,
    )
    rd = result["redispatch"]
    cap_ok, floor_ok = _constraints_ok(rd)
    naive = float(rd["naive_shed_kwh"])
    saved = float(rd["saved_vs_naive_kwh"])
    saved_pct = (saved / naive * 100.0) if naive > 0 else 0.0
    return {
        "name": case["name"],
        "forecast_date": result["forecast_date"],
        "load_column": result["load_column"],
        "load_level": result.get("load_level"),
        "threshold_mw": threshold_mw,
        "max_p50_mw": round(max_p50, 3),
        "intervention_hours": rd["intervention_hours"],
        "rolling_shed_kwh": rd["total_shed_kwh"],
        "naive_shed_kwh": rd["naive_shed_kwh"],
        "saved_kwh": rd["saved_vs_naive_kwh"],
        "saved_pct": round(saved_pct, 1),
        "forecast_basis": rd["forecast_basis"],
        "cap_ok": cap_ok,
        "floor_ok": floor_ok,
        "heterogeneous": rd["heterogeneous"],
    }


def main() -> None:
    rows = [_run_case(case) for case in CASES]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Rolling Paragraph-14a redispatch demo",
        "",
        "Basis: echte oeffentliche DSO-Lastreihen. Die Engpassschwelle ist synthetisch und",
        "knapp unter die prognostizierte P50-Tagesspitze gesetzt. Echte Schwelle =",
        "Netzkapazitaet im Pilot. Forecast-Basis ist ehrlich als `day_ahead_p50_static`",
        "gelabelt; das ist keine echte Intraday-Nachfuehrung.",
        "",
        "| Reihe | Prognosetag | Schwelle MW | Eingriff h | rollierend kWh | pauschal kWh | gespart kWh | gespart % | Constraints | Basis |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        constraints = "Netzgrenze ja; Floors ja" if row["cap_ok"] and row["floor_ok"] else "PRUEFEN"
        lines.append(
            f"| {row['name']} | {row['forecast_date']} | {row['threshold_mw']:.3f} "
            f"| {row['intervention_hours']} | {row['rolling_shed_kwh']:.1f} "
            f"| {row['naive_shed_kwh']:.1f} | {row['saved_kwh']:.1f} "
            f"| {row['saved_pct']:.1f} | {constraints} | {row['forecast_basis']} |"
        )
    mean_saved = sum(row["saved_pct"] for row in rows) / len(rows)
    n_ok = sum(1 for row in rows if row["cap_ok"] and row["floor_ok"])
    lines += [
        "",
        "## Fazit",
        "",
        f"Rollierender Paragraph-14a-Re-Dispatch ist im Dienst angebunden. Auf {len(rows)} echten",
        f"Reihen spart die rollierende Berechnung im Mittel {mean_saved:.1f} % Abregelenergie",
        f"gegenueber pauschaler Dauerdimmung; Constraints gehalten in {n_ok}/{len(rows)} Reihen.",
        "",
        "Hinweis: Bei statischer Day-ahead-P50-Basis entsteht der Vorteil nur aus minimaler",
        "stundenweiser Abregelung gegenueber pauschaler Dimmung. Der groessere Intraday-Vorteil",
        "muss mit echten rolling-origin Forecast-Bahnen separat belegt werden.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
