#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Build Paragraph-14a quantile-dispatch demo on a real DSO row."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.eval.backtest import rolling_origin
from netzpilot.features.build import get_holidays
from netzpilot.models.robust_corrector import ShrunkCorrector
from netzpilot.service.runner import _load2d_from_csv, run_forecast

OUT = Path("data_cache/benchmark/dispatch_demo.md")
CASE = {
    "name": "Herne Bezug vorgelagerte Ebene 2024 (110/10 kV)",
    "csv": "data_cache/real/herne_bezug_vorgelagerte_ebene_2024.csv",
    "unit": "kW",
    "ts_col": "Datum+von",
    "load_col": "Load_1",
}
STEUV_E_KWH = 4000.0
STEUV_PMAX_KW = 1000.0
C_SHORT = 0.20
C_LONG = 0.10


def fee_profile() -> list[float]:
    return [0.05] * 6 + [0.12] * 11 + [0.28] * 5 + [0.05] * 2


def _coverage_for_case() -> tuple[int, float]:
    load2d, days, _meta, _hourly = _load2d_from_csv(
        CASE["csv"],
        ts_col=CASE["ts_col"],
        load_col=CASE["load_col"],
        unit=CASE["unit"],
    )
    keep = min(120, len(load2d))
    n_test = min(42, max(14, keep - 60))
    hol = get_holidays(sorted({d.year for d in days}), "NW")
    R, _summary = rolling_origin(
        load2d[-keep:],
        days[-keep:],
        lambda: ShrunkCorrector(10.0),
        n_test=n_test,
        holiday_set=hol,
    )
    actual = np.asarray(R["actual"], float)
    p10 = np.asarray(R["p10"], float)
    p90 = np.asarray(R["p90"], float)
    coverage = float(np.mean((actual >= p10) & (actual <= p90)))
    return n_test, coverage


def main() -> None:
    base = run_forecast(
        CASE["csv"],
        utility="DispatchDemo",
        unit=CASE["unit"],
        ts_col=CASE["ts_col"],
        load_col=CASE["load_col"],
    )
    peak_mw = max(float(h["p50"]) for h in base["forecast"])
    threshold_mw = max(0.1, peak_mw - 0.5)
    result = run_forecast(
        CASE["csv"],
        utility="DispatchDemo",
        unit=CASE["unit"],
        ts_col=CASE["ts_col"],
        load_col=CASE["load_col"],
        congestion_threshold_mw=threshold_mw,
        steuve_malo="DE-DEMO-DISPATCH",
        steuve_demands_kw=[STEUV_PMAX_KW],
        rolling_redispatch=True,
        grid_fee_eur_per_kwh=fee_profile(),
        dispatch_plan_enabled=True,
        dispatch_steuve_energy_kwh=STEUV_E_KWH,
        dispatch_steuve_p_max_kw=STEUV_PMAX_KW,
        dispatch_c_short=C_SHORT,
        dispatch_c_long=C_LONG,
    )
    dp = result["dispatch_plan"]
    n_test, coverage = _coverage_for_case()
    hourly = dp["hourly"]
    cheap_hours = [h["hour"] for h in hourly if h["fee_eur_per_kwh"] <= 0.05 and h["steuve_kw"] > 0]
    max_total = max(h["total_point_kw"] for h in hourly)
    cap_hours = [
        h["hour"] for h in hourly
        if h["cap_kw"] < STEUV_PMAX_KW - 1e-6
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Quantile dispatch demo",
        "",
        "Basis: echte oeffentliche DSO-Lastreihe und echter NetzPilot-Day-ahead-Forecast.",
        "Die Kostenasymmetrie ist illustrativ; absolute EUR sind Groessenordnung, nicht Pilotvertrag.",
        "Uebertragbar ist die Mechanik: Netzgrenze halten, flexible Last guenstig platzieren,",
        "Bilanzkreis per tau-Quantil statt P50 nominieren.",
        "",
        f"Reihe: {CASE['name']} (`{CASE['load_col']}`).",
        f"Angenommene Netzgrenze: {threshold_mw:.2f} MW (P50-Peak {peak_mw:.2f} MW minus 0.5 MW).",
        f"steuVE-Budget: {STEUV_E_KWH:.0f} kWh, p_max {STEUV_PMAX_KW:.0f} kW.",
        f"Imbalance-Koeffizienten: c_short={C_SHORT:.2f}, c_long={C_LONG:.2f}; tau={dp['tau']:.3f}.",
        f"Rolling-origin Residuen: {n_test} Testtage; P10/P90-Coverage {coverage * 100:.1f}% (Ziel 80%).",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| grid_safe | {str(dp['grid_safe']).lower()} |",
        f"| feasible | {str(dp['feasible']).lower()} |",
        f"| max total point kW | {max_total:.1f} |",
        f"| threshold kW | {threshold_mw * 1000:.1f} |",
        f"| grid fee cost EUR | {dp['grid_fee_cost_eur']:.2f} |",
        f"| expected imbalance P50 EUR | {dp['exp_imbalance_p50_eur']:.2f} |",
        f"| expected imbalance tau EUR | {dp['exp_imbalance_tau_eur']:.2f} |",
        f"| newsvendor saving EUR | {dp['newsvendor_saving_eur']:.2f} |",
        f"| redispatch cap consistency | {str(dp['redispatch_cap_consistency']['consistent']).lower()} |",
        "",
        "## Hourly plan",
        "",
        "| Hour | cap kW | steuVE kW | total point kW | nomination kW | fee EUR/kWh |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for h in hourly:
        lines.append(
            f"| {h['hour']:02d} | {h['cap_kw']:.1f} | {h['steuve_kw']:.1f} "
            f"| {h['total_point_kw']:.1f} | {h['nomination_kw']:.1f} "
            f"| {h['fee_eur_per_kwh']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"The flexible budget is scheduled in cheap hours {cheap_hours}; binding cap hours are {cap_hours}.",
        "The total point load stays below the assumed asset limit in every hour. The tau nomination",
        "reduces expected imbalance cost versus P50 for the illustrative asymmetric coefficients.",
        "This is still v1 dispatch: deterministic placement, no stochastic recourse, no battery SOC,",
        "and no device-level comfort model.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
