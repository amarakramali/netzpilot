#!/usr/bin/env python3
"""Build probabilistic asset-overload demo on a real public DSO row."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.service.runner import run_forecast

OUT = Path("data_cache/benchmark/overload_demo.md")
CASE = {
    "name": "Herne Bezug vorgelagerte Ebene 2024 (110/10 kV)",
    "csv": "data_cache/real/herne_bezug_vorgelagerte_ebene_2024.csv",
    "unit": "kW",
    "ts_col": "Datum+von",
    "load_col": "Load_1",
}
RISK_ALPHA = 0.05


def main() -> None:
    base = run_forecast(
        CASE["csv"],
        utility="OverloadDemo",
        unit=CASE["unit"],
        ts_col=CASE["ts_col"],
        load_col=CASE["load_col"],
    )
    peak_mw = max(float(h["p50"]) for h in base["forecast"])
    rating_mw = max(0.1, peak_mw - 0.5)
    result = run_forecast(
        CASE["csv"],
        utility="OverloadDemo",
        unit=CASE["unit"],
        ts_col=CASE["ts_col"],
        load_col=CASE["load_col"],
        congestion_threshold_mw=rating_mw,
        asset_rating_kw=rating_mw * 1000.0,
        overload_risk_alpha=RISK_ALPHA,
    )
    overload = result["overload"]
    hosting = result["hosting_capacity"]
    risky_hours = [h["hour"] for h in overload["hourly"] if h["at_risk"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Probabilistic asset-overload demo",
        "",
        "Basis: echte oeffentliche DSO-Lastreihe und echter NetzPilot-Day-ahead-Forecast.",
        "Die Bemessungsgrenze ist fuer diese Demo angenommen, nicht aus einem Netzplan extrahiert.",
        "Scope: Einzelasset gegen rating_kw; kein Netzlastfluss, keine Topologie, keine Spannungsebene-Simulation.",
        "",
        f"Reihe: {CASE['name']} (`{CASE['load_col']}`).",
        f"Angenommenes Asset-Rating: {rating_mw:.2f} MW (P50-Peak {peak_mw:.2f} MW minus 0.5 MW).",
        f"Risikoschwelle alpha: {RISK_ALPHA:.2f}; Residuenbasis: {overload['n_test_days']} trailing rolling-origin Testtage.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| hours at risk | {overload['hours_at_risk']} |",
        f"| max exceedance probability | {overload['max_exceedance_prob'] * 100:.1f} % |",
        f"| peak risk hour | {overload['peak_risk_hour']} |",
        f"| expected overload energy | {overload['expected_overload_kwh_total']:.1f} kWh |",
        f"| prob any overload (indep approx) | {overload['prob_any_overload_indep'] * 100:.1f} % |",
        f"| hosting capacity | {hosting['hosting_capacity_kw']:.1f} kW |",
        f"| binding hour | {hosting['binding_hour']} |",
        "",
        "## Hourly risk",
        "",
        "| Hour | P50 load kW | P90 load kW | exceedance prob | expected overload kWh | at risk |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for h in overload["hourly"]:
        lines.append(
            f"| {h['hour']:02d} | {h['p50_load_kw']:.1f} | {h['p90_load_kw']:.1f} "
            f"| {h['exceedance_prob'] * 100:.1f} % | {h['expected_overload_kwh']:.2f} "
            f"| {str(h['at_risk']).lower()} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"Risikostunden: {risky_hours if risky_hours else 'keine ueber alpha'}; freie koinzidente",
        f"Zusatzlast bei alpha={RISK_ALPHA:.2f}: {hosting['hosting_capacity_kw']:.1f} kW.",
        "Das ist eine probabilistische Asset-Ampel. Sie ersetzt keine Lastflussrechnung, kann aber",
        "als operativer Trigger fuer Paragraph-14a-Redispatch und Anschlusskapazitaets-Pruefungen dienen.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
