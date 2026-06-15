#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Build Mehr-/Mindermengen demo on a real public DSO row."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.service.runner import run_forecast

OUT = Path("data_cache/benchmark/mmm_demo.md")
CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"
PRICE = 60.0


def main() -> None:
    result = run_forecast(
        CSV,
        utility="MmmDemo",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        mmm_price_eur_mwh=PRICE,
    )
    mmm = result["mmm"]
    snaive = mmm["report_snaive"]
    netzpilot = mmm["report_netzpilot"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mehr-/Mindermengen demo",
        "",
        "Basis: echter Hilden-Netzumsatz-Lastgang 2025. Die Rechnung nutzt den trailing",
        "rolling-origin Backtest: Prognose vs. Ist je Stunde, MW als MWh mit dt_h=1.0.",
        f"MMM-Preis-Annahme: {PRICE:.2f} EUR/MWh. Im Produkt muss dieser regulierte Preis",
        "als Mandanten-/Abrechnungsconfig gesetzt werden; er wird hier nicht als Marktnutzen",
        "erfunden.",
        "",
        f"Forecast date: {result['forecast_date']}.",
        f"Backtest-Tage: {mmm['n_test_days']} bei {mmm['history_days_used']} Historientagen.",
        "",
        "| Forecast | Mehrmenge MWh | Mindermenge MWh | Netto MWh | abs. Volumen MWh | Netto EUR |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Saisonal-naiv | {snaive['mehrmenge_mwh']:.4f} | {snaive['mindermenge_mwh']:.4f} | "
            f"{snaive['netto_mwh']:.4f} | {snaive['abs_volumen_mwh']:.4f} | {snaive['netto_eur']:.2f} |"
        ),
        (
            f"| NetzPilot | {netzpilot['mehrmenge_mwh']:.4f} | {netzpilot['mindermenge_mwh']:.4f} | "
            f"{netzpilot['netto_mwh']:.4f} | {netzpilot['abs_volumen_mwh']:.4f} | "
            f"{netzpilot['netto_eur']:.2f} |"
        ),
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| abs. Volumenreduktion vs. Saisonal-naiv | {mmm['abs_volumen_reduktion_mwh']:.4f} MWh |",
        f"| Volumenreduktion am Preis | {mmm['abs_volumen_reduktion_at_price_eur']:.2f} EUR |",
        "",
        "## Interpretation",
        "",
        "MMM ist die EDM-Reconciliation-Sicht: Mehr- und Mindermengen werden als Brutto-Volumina",
        "ausgewiesen und zum regulierten MMM-Preis bewertet. Das ist nicht identisch mit",
        "QH-Ausgleichsenergie/reBAP; es steht bewusst neben `economics_realized`.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
