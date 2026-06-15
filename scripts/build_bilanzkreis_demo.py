#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Build long-window realized Bilanzkreis settlement demo on public DSO rows."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.eval.backtest import rolling_origin
from netzpilot.eval.bilanzkreis_realized import (
    official_rebap_asymmetry_count,
    realized_settlement_from_backtest,
)
from netzpilot.features.build import get_holidays
from netzpilot.models.robust_corrector import ShrunkCorrector
from netzpilot.service.runner import _load2d_from_csv

REBAP_CSV = "data_cache/real/rebap_2024.csv"
REBAP_OFFICIAL_CSV = "data_cache/real/rebap_2024_official.csv"
SPOT_CSV = "data_cache/real/spot_da_2024.csv"
OUT = Path("data_cache/benchmark/bilanzkreis_demo.md")
TRAINING_DAYS = 60
BLOCK_DAYS = 30

CASES = [
    {
        "name": "Herne Bezug vorgelagerte Ebene 2024 (110/10 kV)",
        "csv": "data_cache/real/herne_bezug_vorgelagerte_ebene_2024.csv",
        "unit": "kW",
        "ts_col": "Datum+von",
        "load_col": "Load_1",
    },
    {
        "name": "EVDB Jahreshoechstlast NS 2024",
        "csv": "data_cache/real/evdb_lastgang_ns_2024.csv",
        "unit": "kW",
        "ts_col": "Datum+von",
        "load_col": "Wert",
    },
]


def _slice_result(R: dict, start_day: int, n_days: int) -> dict:
    lo = start_day * 24
    hi = (start_day + n_days) * 24
    return {k: np.asarray(v)[lo:hi] for k, v in R.items()}


def _block_summary(R: dict, test_days, block_days: int = BLOCK_DAYS) -> list[dict]:
    out = []
    n_test = len(test_days)
    for start in range(0, n_test, block_days):
        n = min(block_days, n_test - start)
        if n < block_days:
            continue
        block_R = _slice_result(R, start, n)
        block_days_idx = list(test_days)[start:start + n]
        e = realized_settlement_from_backtest(
            block_R,
            block_days_idx,
            n,
            REBAP_CSV,
            SPOT_CSV,
            include_band=False,
        )
        out.append({
            "start": str(block_days_idx[0].date()),
            "end": str(block_days_idx[-1].date()),
            "n_days": int(n),
            "savings_eur_per_year": e["savings_eur_per_year"],
            "linear_expected_eur_per_year": e["linear_expected_eur_per_year"],
            "bias_eur_per_year": e["savings_bias_term_eur_per_year"],
            "correlation_eur_per_year": e["savings_correlation_term_eur_per_year"],
        })
    return out


def _run_case(case: dict) -> dict:
    load2d, days, meta, _hourly = _load2d_from_csv(
        case["csv"],
        ts_col=case["ts_col"],
        load_col=case["load_col"],
        unit=case["unit"],
    )
    if len(load2d) <= TRAINING_DAYS + 30:
        raise ValueError(f"{case['name']}: zu wenig Tage fuer Langfenster ({len(load2d)}).")
    n_test = len(load2d) - TRAINING_DAYS
    hol = get_holidays(sorted({d.year for d in days}), "NW")
    print(f"{case['name']}: rolling-origin n_test={n_test}, history={len(load2d)} days")
    R, summary = rolling_origin(
        load2d,
        days,
        lambda: ShrunkCorrector(10.0),
        n_test=n_test,
        holiday_set=hol,
    )
    economics = realized_settlement_from_backtest(R, days, n_test, REBAP_CSV, SPOT_CSV)
    test_days = list(days)[-n_test:]
    blocks = _block_summary(R, test_days)
    return {
        "case": case,
        "meta": meta,
        "n_history_days": int(len(load2d)),
        "summary": summary,
        "economics_realized": economics,
        "blocks": blocks,
    }


def _gap_label(e: dict) -> str:
    real = abs(float(e["savings_eur_per_year"]))
    linear = abs(float(e["linear_expected_eur_per_year"]))
    if linear == 0:
        return "linear ~0; Gap nicht sinnvoll quotierbar"
    ratio = real / linear
    if ratio >= 1.5:
        return f"Gap haelt: realisiert {ratio:.1f}x linear"
    if ratio >= 0.75:
        return f"Gap schrumpft: realisiert {ratio:.1f}x linear"
    return f"Gap verschwindet weitgehend: realisiert {ratio:.1f}x linear"


def main() -> None:
    asym = official_rebap_asymmetry_count(REBAP_OFFICIAL_CSV)
    rows = [_run_case(case) for case in CASES]
    OUT.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Bilanzkreis settlement demo",
        "",
        "Basis: echte oeffentliche DSO-Lastreihen 2024, langer rolling-origin Backtest gegen",
        "Saisonal-Naiv. reBAP und Spot-DA sind echte 2024-QH-Reihen; fuer diese Demo",
        "werden sie je lokaler Stunde gemittelt. Last/Fahrplaene sind stuendlich, daher",
        "Abrechnung in `MWh per hour (MW * 1h)`. Anders als der verworfene T33/T34-",
        "Kurzlauf nutzt diese Fassung ein langes 2024-Testfenster; die Werte sind ueber N",
        "gemessene Tage auf 365 Tage normiert.",
        "",
        "Verworfen: Die fruehere 14-Tage-Demo wurde mit Faktor ca. 26 annualisiert und ist",
        "nicht mehr Headline-faehig.",
        "",
        f"Offizielle reBAP-Spalten: {asym['n_asymmetric_qh']} von {asym['n_qh']} QH haben",
        f"`unterdeckt != ueberdeckt` (max. Differenz {asym['max_abs_diff_eur_mwh']} EUR/MWh).",
        "Damit ist fuer 2024 keine asymmetrische Zusatzabrechnung noetig.",
        "",
        "| Reihe | n_days | scale | realisiert EUR/Jahr | linear EUR/Jahr | Bias EUR/Jahr | Korrelation EUR/Jahr | MC P5 | MC P50 | MC P95 | P(spart) | Gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        e = row["economics_realized"]
        b = e["band"]
        lines.append(
            f"| {row['case']['name']} | {e['n_days']:.0f} | {e['annualization_factor']:.3f} "
            f"| {e['savings_eur_per_year']:.0f} | {e['linear_expected_eur_per_year']:.0f} "
            f"| {e['savings_bias_term_eur_per_year']:.0f} "
            f"| {e['savings_correlation_term_eur_per_year']:.0f} "
            f"| {b['p5_eur']:.0f} | {b['p50_eur']:.0f} | {b['p95_eur']:.0f} "
            f"| {b['prob_positive'] * 100:.1f}% | {_gap_label(e)} |"
        )

    lines += [
        "",
        "## 30-Tage-Bloecke",
        "",
        "Die folgende Tabelle zeigt disjunkte 30-Tage-Bloecke innerhalb des langen Testfensters.",
        "Alle Werte sind je Block auf EUR/Jahr normiert; sie zeigen die Streuung des",
        "Korrelations-Gaps statt nur eine Jahressumme.",
        "Die Bloecke zaehlen vollstaendige lokale 24h-Tage; DST-Umstelltage sind in der",
        "Lastaufbereitung verworfen, deshalb koennen Kalenderintervalle leicht laenger wirken.",
        "",
        "| Reihe | Block | realisiert EUR/Jahr | linear EUR/Jahr | Bias EUR/Jahr | Korrelation EUR/Jahr |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        for b in row["blocks"]:
            lines.append(
                f"| {row['case']['name']} | {b['start']} bis {b['end']} "
                f"| {b['savings_eur_per_year']:.0f} | {b['linear_expected_eur_per_year']:.0f} "
                f"| {b['bias_eur_per_year']:.0f} | {b['correlation_eur_per_year']:.0f} |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "Der realized-vs-linear-Gap ist auf dem langen Fenster reihenabhaengig: Bei Herne",
        "verschwindet der 14-Tage-Korrelationsvorteil weitgehend und das MC-Band umfasst klar",
        "negative Jahre; bei EVDB bleibt der Gap positiv. Die Korrelationsterme schwanken ueber",
        "30-Tage-Bloecke stark; genau deshalb stehen n_days, scale und MC-Band prominent in",
        "der Tabelle.",
        "",
        "Das Monte-Carlo-Band ist ein Tages-Block-Bootstrap innerhalb des beobachteten",
        "Preisregimes 2024. Es ist keine Aussage ueber ein anderes Jahr und modelliert keinen",
        "Intraday-Handel; es zeigt die Settlement-Exposure ohne Glattstellung. DSO-Last bleibt",
        "ein Proxy fuer die Bilanzkreis-Entnahme bis echte Pilotdaten vorliegen.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
