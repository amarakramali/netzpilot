#!/usr/bin/env python3
"""Build input-validation demo on a real public DSO row with injected defects."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.data.validate import validate_load
from netzpilot.service.input_validation import summarize_validation
from scripts.pilot_in_a_box import robust_load_csv

OUT = Path("data_cache/benchmark/validate_demo.md")
CASE = {
    "name": "Herne Bezug vorgelagerte Ebene 2024 (110/10 kV)",
    "csv": "data_cache/real/herne_bezug_vorgelagerte_ebene_2024.csv",
    "unit": "kW",
    "ts_col": "Datum+von",
    "load_col": "Load_1",
}


def main() -> None:
    hourly, _ts, _lc, _meta = robust_load_csv(
        CASE["csv"],
        ts_col=CASE["ts_col"],
        load_col=CASE["load_col"],
        unit=CASE["unit"],
        return_meta=True,
    )
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="1h", tz=hourly.index.tz)
    original = hourly.reindex(full_index)
    values = original.tolist()
    clean_max = max(v for v in values if v is not None and v == v)

    injected = {
        "missing": 200,
        "negative": 500,
        "out_of_range": 800,
        "frozen_start": 1100,
        "frozen_len": 8,
    }
    values[injected["missing"]] = None
    values[injected["negative"]] = -abs(float(values[injected["negative"]]))
    values[injected["out_of_range"]] = clean_max * 20.0
    frozen_value = float(values[injected["frozen_start"] - 1])
    for i in range(injected["frozen_start"], injected["frozen_start"] + injected["frozen_len"]):
        values[i] = frozen_value

    report = validate_load(values, period_per_day=24, max_plausible=clean_max * 2.0)
    summary = summarize_validation(
        report,
        full_index,
        cleaned_values_used=False,
        would_apply_cleaned_values=(
            report["n_replaced"] > 0
            and report["n_unreplaceable"] == 0
            and all(v is not None for v in report["cleaned"])
        ),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Input-validation demo",
        "",
        "Basis: echte oeffentliche DSO-Lastreihe; Defekte wurden kuenstlich injiziert, damit",
        "die Plausibilisierung sichtbar wird. Die Originaldatei bleibt unangetastet.",
        "Die Summen koennen groesser als die Injektionen sein, weil die echte Zeitachse zusaetzliche",
        "DST-/Luecken- und Ausreisserpunkte mitbringen kann.",
        "",
        f"Reihe: {CASE['name']} (`{CASE['load_col']}`), Werte nach Loader in MW.",
        "",
        "| Injected defect | Index | Timestamp |",
        "|---|---:|---|",
        f"| missing | {injected['missing']} | {full_index[injected['missing']].isoformat()} |",
        f"| negative | {injected['negative']} | {full_index[injected['negative']].isoformat()} |",
        f"| out_of_range | {injected['out_of_range']} | {full_index[injected['out_of_range']].isoformat()} |",
        f"| frozen run | {injected['frozen_start']}..{injected['frozen_start'] + injected['frozen_len'] - 1} | {full_index[injected['frozen_start']].isoformat()} |",
        "",
        "| Validation metric | Value |",
        "|---|---:|",
        f"| quality_score | {summary['quality_score'] * 100:.2f} % |",
        f"| n_missing | {summary['n_missing']} |",
        f"| n_negative | {summary['n_negative']} |",
        f"| n_out_of_range | {summary['n_out_of_range']} |",
        f"| n_outlier | {summary['n_outlier']} |",
        f"| n_frozen | {summary['n_frozen']} |",
        f"| n_replaced | {summary['n_replaced']} |",
        f"| n_unreplaceable | {summary['n_unreplaceable']} |",
        f"| would_apply_cleaned_values | {str(summary['would_apply_cleaned_values']).lower()} |",
        "",
        "## Replacement sample",
        "",
        "| Index | Timestamp | Method | Value MW |",
        "|---:|---|---|---:|",
    ]
    for row in summary["replacements_sample"][:10]:
        lines.append(
            f"| {row['index']} | {row.get('timestamp', '')} | {row['method']} | {row['value']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Luecken, negative Werte und ausserhalb der plausiblen Grenze liegende Werte werden ersetzt.",
        "Eingefrorene Phasen werden nur gemeldet, nicht automatisch ueberschrieben, weil flache Last",
        "in echten Netzzeitreihen legitim sein kann. Genau so laeuft das Gate jetzt vor der Forecast-Engine.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
