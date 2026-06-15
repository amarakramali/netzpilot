#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Build IEC-style transformer thermal-risk demo on a real public DSO row."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.service.runner import run_forecast

OUT = Path("data_cache/benchmark/thermal_demo.md")
WEATHER_CSV = Path("data_cache/benchmark/thermal_weather_openmeteo_demo.csv")
WEATHER_PARQUET = Path(
    "data_cache/t2_2022-01-01_2024-01-01/raw/"
    "openmeteo_historical_forecast_berlin_east_2022-01-01_2023-12-31.parquet"
)
CASE = {
    "name": "Stadtwerke Neuruppin 2022 (NA MS demo column)",
    "csv": "data_cache/real/neuruppin_lgl_strom_2022.csv",
    "unit": "kW",
    "ts_col": "Datum+von",
    "load_col": "Wert.11",
}


def ensure_weather_csv() -> str:
    if WEATHER_CSV.exists():
        return str(WEATHER_CSV)
    weather = pd.read_parquet(WEATHER_PARQUET).copy()
    weather = weather.reset_index().rename(columns={"index": "time"})
    weather["time"] = pd.to_datetime(weather["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    WEATHER_CSV.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(WEATHER_CSV, index=False)
    return str(WEATHER_CSV)


def main() -> None:
    weather_csv = ensure_weather_csv()
    base = run_forecast(
        CASE["csv"],
        utility="ThermalDemo",
        unit=CASE["unit"],
        ts_col=CASE["ts_col"],
        load_col=CASE["load_col"],
    )
    peak_kw = max(float(h["p50"]) for h in base["forecast"]) * 1000.0
    rating_kw = peak_kw * 0.85
    result = run_forecast(
        CASE["csv"],
        utility="ThermalDemo",
        unit=CASE["unit"],
        ts_col=CASE["ts_col"],
        load_col=CASE["load_col"],
        weather_csv=weather_csv,
        thermal_rating_kw=rating_kw,
        thermal_hotspot_limit_c=120.0,
        thermal_risk_alpha=0.05,
    )
    thermal = result["thermal"]
    risky_hours = [h["hour"] for h in thermal["hourly"] if h["at_risk"]]
    ambient_min = min(thermal["ambient_temperature_c"])
    ambient_max = max(thermal["ambient_temperature_c"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Transformer thermal-risk demo",
        "",
        "Basis: echte oeffentliche DSO-Lastreihe plus vorhandene Open-Meteo Historical-Forecast-",
        "Wetterdaten im bestehenden Wetter-CSV-Format (`temperature_2m`). Die Trafo-Parameter und",
        "Bemessungsleistung sind Demo-Annahmen; echte Werte kommen im Pilot vom Netzbetreiber.",
        "",
        f"Reihe: {CASE['name']} (`{CASE['load_col']}`).",
        f"Forecast date: {result['forecast_date']}.",
        f"Angenommene Trafo-Bemessung: {rating_kw:.1f} kW (85 % des P50-Peaks {peak_kw:.1f} kW).",
        f"Umgebungstemperatur-Quelle: {thermal['ambient_source']} ({ambient_min:.1f}..{ambient_max:.1f} C).",
        "Wetterproxy: Open-Meteo Historical Forecast Berlin East aus dem vorhandenen T2-Cache.",
        f"Hotspot-Grenze: {thermal['hotspot_limit_c']:.1f} C; alpha={thermal['risk_alpha']:.2f};",
        f"Residuenbasis: {thermal['n_test_days']} trailing rolling-origin Testtage.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| hours at risk | {thermal['hours_at_risk']} |",
        f"| max hotspot exceedance probability | {thermal['max_exceedance_prob'] * 100:.1f} % |",
        f"| peak risk hour | {thermal['peak_risk_hour']} |",
        f"| expected loss of life | {thermal['expected_loss_of_life_h_total']:.3f} h |",
        f"| P90 loss of life | {thermal['p90_loss_of_life_h_total']:.3f} h |",
        f"| expected equivalent aging factor | {thermal['equivalent_aging_factor_expected']:.3f} |",
        f"| P90 max hotspot | {thermal['p90_max_hotspot_c']:.1f} C |",
        "",
        "## Hourly thermal risk",
        "",
        "| Hour | P50 hotspot C | P90 hotspot C | exceedance prob | expected aging h | at risk |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for h in thermal["hourly"]:
        lines.append(
            f"| {h['hour']:02d} | {h['p50_hotspot_c']:.1f} | {h['p90_hotspot_c']:.1f} "
            f"| {h['exceedance_prob'] * 100:.1f} % | {h['expected_loss_of_life_h']:.4f} "
            f"| {str(h['at_risk']).lower()} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"Risikostunden: {risky_hours if risky_hours else 'keine ueber alpha'}. Gegenueber der reinen",
        "`load > rating`-Ampel bewertet diese Sicht die thermische Traegheit und den Lebensdauerverbrauch.",
        "Das ist naeher an IEC/IEEE-Trafo-Betrieb, bleibt aber Einzelasset-Thermik mit Standardparametern",
        "und ersetzt keine Netzlastflussrechnung.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
