#!/usr/bin/env python3
"""Build the T43 temporal reconciliation demo on Hilden Netzumsatz 2025."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from netzpilot.service.reconcile_temporal import build_temporal_reconciliation_payload


REAL_CSV = os.path.join(ROOT, "data_cache", "real", "Netzumsatz-Lastgang-2025.csv")
OUT_MD = os.path.join(ROOT, "data_cache", "benchmark", "reconcile_temporal_demo.md")


def _fmt(x, digits=3):
    return f"{float(x):.{digits}f}"


def _row(label: str, metrics: dict, coh: dict) -> str:
    return (
        f"| {label} | {_fmt(metrics['base_mae'])} | {_fmt(metrics['reconciled_mae'])} | "
        f"{_fmt(metrics['delta_pct'])} | {_fmt(metrics['base_mape_pct'])} | "
        f"{_fmt(metrics['reconciled_mape_pct'])} | {_fmt(coh['before_max'], 6)} | "
        f"{_fmt(coh['after_max'], 6)} |"
    )


def main() -> None:
    payload = build_temporal_reconciliation_payload(
        REAL_CSV,
        ts_col="Text",
        load_col="Reihe1",
        unit="kW",
        region="NW",
        method="wls_struct",
        n_test=14,
    )
    holdout = payload["holdout"]
    metrics = holdout["metrics"]
    coh = holdout["coherence"]
    forecast = payload["forecast"]
    input_meta = payload["input"]

    lines = [
        "# T43 - Temporale MinT-Reconciliation auf Hilden Netzumsatz 2025",
        "",
        "Quelle: `data_cache/real/Netzumsatz-Lastgang-2025.csv`, Spalte `Reihe1`, "
        "4 Werte pro Stunde, echte Messdaten. Die bekannte Hilden-Headline aus dem "
        "Pilot bleibt MAPE 4.36 % fuer die Stundenprognose; diese Demo misst zusaetzlich "
        "die temporale Koharenz von Tag/Stunde/Viertelstunde.",
        "",
        f"- Methode: `{holdout['method']}` mit `build_temporal_summing_matrix(96, [96, 4])`.",
        f"- Holdout: {holdout['n_test_days']} leakage-sichere Tage "
        f"({holdout['target_start']} bis {holdout['target_end']}).",
        f"- Einheit: MWh je Knotenperiode; Viertelstundenleistung aus kW wurde mit dt_h=0.25 in MWh umgerechnet.",
        f"- Vollstaendige 96er-Tage: {input_meta['complete_days']}; verworfene Nicht-96-Tage: "
        f"{', '.join(input_meta['dropped_non_96_days']) or 'keine'}.",
        "",
        "| Ebene | Base MAE (MWh) | Reconciled MAE (MWh) | Delta % | Base MAPE % | Reconciled MAPE % | coherence before max | coherence after max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _row("Tag", metrics["day"], coh),
        _row("Stunde", metrics["hour"], coh),
        _row("Viertelstunde", metrics["quarter"], coh),
        "",
        f"Koharenz global: before mean {_fmt(coh['before_mean'], 6)} MWh, "
        f"before max {_fmt(coh['before_max'], 6)} MWh; after mean {_fmt(coh['after_mean'], 6)} MWh, "
        f"after max {_fmt(coh['after_max'], 6)} MWh.",
        "",
        "Lesart: Reconciliation garantiert, dass 96 Viertelstunden, 24 Stunden und die Tagesenergie "
        "exakt zusammenpassen. Die Genauigkeit wird je Ebene ehrlich berichtet; wenn eine Ebene "
        "nicht profitiert, bleibt der operative Nutzen die geschlossene Nominierung ohne Volumenluecke.",
        "",
        "Naechster koharenter P50-Fahrplan:",
        f"- Forecast-Date: {forecast['date']}",
        f"- Base coherence error: {_fmt(forecast['coherence_before'], 6)} MWh",
        f"- Reconciled coherence error: {_fmt(forecast['coherence_after'], 6)} MWh",
        f"- Reconciled day energy: {_fmt(forecast['reconciled']['day_mwh'], 3)} MWh",
        "",
        "Caveat: Keine Reconciliation ueber Spannungsebenen. Die veroeffentlichten vertikalen "
        "Netzlast-je-Netzebene-Reihen wurden an echten Daten als Kaskade statt additive Hierarchie "
        "identifiziert; MinT laeuft hier deshalb nur auf der temporal exakten Achse einer Reihe.",
        "",
    ]
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
