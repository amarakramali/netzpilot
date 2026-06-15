#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Build a service-level EEBUS-LPC mapping demo on a real public DSO row."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.service.runner import run_forecast

OUT = Path("data_cache/benchmark/lpc_demo.md")
CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"


def main() -> None:
    result = run_forecast(
        CSV,
        utility="LpcDemo",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        congestion_threshold_mw=33.0,
        steuve_malo="DE0001234567890",
        submit_to_aemt=True,
        aemt_adapter="eebus_lpc",
    )
    lpc = result["fahrplan_lpc"]
    ack = result["aemt_ack"]
    first = lpc["limits"][0]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# EEBUS-LPC handover demo",
        "",
        "Basis: echter Hilden-Netzumsatz-Lastgang, bewusst niedrige Demo-Netzgrenze 33 MW,",
        "damit ein Paragraph-14a-Fahrplan entsteht. NetzPilot baut nur die Datenabbildung;",
        "EEBUS-Transport, SHIP/SPINE, SMGW und Anlagensteuerung bleiben extern beim aEMT/CEM.",
        "",
        f"Forecast date: {result['forecast_date']}.",
        f"aEMT adapter: eebus_lpc; ack status: {ack['status']}.",
        f"LPC use case: {lpc['use_case']}.",
        f"Transport label: {lpc['transport']}.",
        "",
        "| Mapping | Value |",
        "|---|---:|",
        f"| n_limits | {lpc['n_limits']} |",
        f"| consumption_limit_w | {first['consumption_limit_w']:.0f} W |",
        f"| failsafe_value_w | {first['failsafe_value_w']:.0f} W |",
        f"| duration_s | {first['duration_s']:.0f} s |",
        f"| is_limit_active | {str(first['is_limit_active']).lower()} |",
        "",
        "## First LPC limit payload",
        "",
        "```json",
        json.dumps(first, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Interpretation",
        "",
        "Die Wirkleistungsgrenze aus dem Paragraph-14a-Fahrplan wird von kW nach W uebersetzt.",
        "Der Failsafe entspricht dem Paragraph-14a-Floor und ist die Rueckfallgrenze bei",
        "Kommunikationsausfall. NetzPilot beruehrt kein SMGW und sendet nicht selbst ueber EEBUS.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
