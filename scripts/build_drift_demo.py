#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Build drift-monitoring demo on a real DSO row plus labeled synthetic drift."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.eval.backtest import rolling_origin
from netzpilot.eval.drift import coverage_report, drift_report
from netzpilot.features.build import get_holidays
from netzpilot.models.robust_corrector import ShrunkCorrector
from netzpilot.service.runner import _load2d_from_csv

CASE = {
    "name": "Herne Bezug vorgelagerte Ebene 2024 (110/10 kV)",
    "csv": "data_cache/real/herne_bezug_vorgelagerte_ebene_2024.csv",
    "unit": "kW",
    "ts_col": "Datum+von",
    "load_col": "Load_1",
}
OUT = Path("data_cache/benchmark/drift_demo.md")
TRAINING_DAYS = 60
REFERENCE_DAYS = 120
RECENT_DAYS = 30


def _slice(values, start_day: int, n_days: int):
    lo = start_day * 24
    hi = (start_day + n_days) * 24
    return list(np.asarray(values)[lo:hi])


def _errors(R: dict, start_day: int, n_days: int) -> list[float]:
    actual = np.asarray(R["actual"], float)
    model = np.asarray(R["model"], float)
    lo = start_day * 24
    hi = (start_day + n_days) * 24
    return list(actual[lo:hi] - model[lo:hi])


def _coverage(R: dict, start_day: int, n_days: int, *, actual_override=None) -> dict:
    actual = actual_override if actual_override is not None else _slice(R["actual"], start_day, n_days)
    return coverage_report(
        _slice(R["p10"], start_day, n_days),
        _slice(R["p90"], start_day, n_days),
        actual,
        nominal=0.8,
    )


def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _date_range(test_days, start_day: int, n_days: int) -> str:
    start = test_days[start_day]
    end = test_days[start_day + n_days - 1]
    return f"{start.date()} bis {end.date()}"


def main() -> None:
    load2d, days, meta, _hourly = _load2d_from_csv(
        CASE["csv"],
        ts_col=CASE["ts_col"],
        load_col=CASE["load_col"],
        unit=CASE["unit"],
    )
    if len(load2d) <= TRAINING_DAYS + REFERENCE_DAYS + RECENT_DAYS:
        raise ValueError(f"zu wenig Tage fuer Drift-Demo: {len(load2d)}")

    n_test = len(load2d) - TRAINING_DAYS
    hol = get_holidays(sorted({d.year for d in days}), "NW")
    print(f"{CASE['name']}: rolling-origin n_test={n_test}, history={len(load2d)} days")
    R, summary = rolling_origin(
        load2d,
        days,
        lambda: ShrunkCorrector(10.0),
        n_test=n_test,
        holiday_set=hol,
    )
    test_days = list(days)[-n_test:]
    reference = _errors(R, 0, REFERENCE_DAYS)
    real_recent = _errors(R, REFERENCE_DAYS, RECENT_DAYS)
    real_report = drift_report(reference, real_recent)
    real_coverage = _coverage(R, REFERENCE_DAYS, RECENT_DAYS)

    ref_std = float(np.std(reference)) or 1.0
    synthetic_recent = [e * 1.8 + 1.2 * ref_std for e in real_recent]
    synthetic_report = drift_report(reference, synthetic_recent)
    model_recent = _slice(R["model"], REFERENCE_DAYS, RECENT_DAYS)
    synthetic_actual = [float(m) + float(e) for m, e in zip(model_recent, synthetic_recent)]
    synthetic_coverage = _coverage(
        R,
        REFERENCE_DAYS,
        RECENT_DAYS,
        actual_override=synthetic_actual,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Drift monitoring demo",
        "",
        "Basis: echte oeffentliche DSO-Lastreihe, leakage-sauberer rolling-origin Backtest.",
        "Fehlerdefinition: `actual - forecast` in MW. Die Referenz ist die fruehe Testperiode;",
        "Recent ist eine spaetere realisierte Periode. Der synthetische Fall ist klar gelabelt",
        "und nutzt dieselben Recent-Fehler mit kuenstlichem Bias plus Skalierung.",
        "",
        f"Reihe: {CASE['name']} (`{CASE['load_col']}`, {meta.get('load_level') or 'load level n/a'}).",
        f"Backtest: {n_test} Testtage nach {TRAINING_DAYS} Start-Training; "
        f"MAE Modell {summary['metriken']['model']['MAE_MW']} MW.",
        f"Referenz: {_date_range(test_days, 0, REFERENCE_DAYS)} ({REFERENCE_DAYS} Tage).",
        f"Recent real: {_date_range(test_days, REFERENCE_DAYS, RECENT_DAYS)} ({RECENT_DAYS} Tage).",
        "",
        "| Fall | Status | needs_recalibration | n_ref | n_recent | PSI | KS | MAE-Ratio | Bias-Shift/ref-Std | Coverage | Gruende |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    rows = [
        ("Echt: spaetere Herne-Periode", real_report, real_coverage),
        ("Synthetisch: Bias + Scale auf Recent", synthetic_report, synthetic_coverage),
    ]
    for label, report, cov in rows:
        needs = report["status"] in {"watch", "drift"} or cov["status"] == "drift"
        lines.append(
            f"| {label} | {report['status']} | {str(needs).lower()} "
            f"| {report['n_ref']} | {report['n_recent']} | {_fmt(report['psi'])} "
            f"| {_fmt(report['ks'])} | {_fmt(report['mae_ratio'])} "
            f"| {_fmt(report['bias_shift_in_ref_std'])} | {_fmt(cov['coverage'])} "
            f"| {'; '.join(report['reasons'])} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Der echte Herne-Ausschnitt bleibt stabil; das ist kein Problem, sondern die ehrliche Aussage:",
        "im beobachteten Fenster gibt es keinen belastbaren Drift-Alarm. Der synthetische Bias/Scale-",
        "Fall feuert dagegen deutlich und zeigt, dass der Alarm-Hook bei Verteilungs- und Genauigkeitsdrift",
        "anspringt. Drift ist eine Warnung zur Pruefung/Re-Kalibrierung, kein Ursachenbeweis und kein",
        "Auto-Retrain.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
