#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Build illustrative Paragraph-14a Module-3 tariff demo."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.service.tariff_schedule import build_tariff_schedule

OUT = Path("data_cache/benchmark/tariff_demo.md")
ENERGY_KWH = 40.0
P_MAX_KW = 11.0
START_HOUR = 18
END_HOUR = 0


def illustrative_fee_profile() -> list[float]:
    fee = [0.12] * 24
    for h in [0, 1, 2, 3, 4, 5, 22, 23]:
        fee[h] = 0.05
    for h in [17, 18, 19, 20, 21]:
        fee[h] = 0.28
    return fee


def illustrative_redispatch_cap() -> dict:
    hourly = []
    for h in range(24):
        cap = None
        if h == 22:
            cap = 0.0
        elif h == 23:
            cap = 4.0
        hourly.append({"hour": h, "cap_kw": cap})
    return {
        "forecast_basis": "illustrative_day_ahead_p50_static",
        "hourly": hourly,
    }


def _hours_with_energy(schedule: list[float]) -> str:
    return ", ".join(f"{h:02d}: {kwh:.1f} kWh" for h, kwh in enumerate(schedule) if kwh > 1e-9)


def main() -> None:
    fee = illustrative_fee_profile()
    base = build_tariff_schedule(
        fee,
        ENERGY_KWH,
        P_MAX_KW,
        available_start_hour=START_HOUR,
        available_end_hour=END_HOUR,
    )
    capped = build_tariff_schedule(
        fee,
        ENERGY_KWH,
        P_MAX_KW,
        redispatch=illustrative_redispatch_cap(),
        available_start_hour=START_HOUR,
        available_end_hour=END_HOUR,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tariff schedule demo",
        "",
        "Illustrative §14a Module-3 profile. The grid-fee levels below are not claimed as a real",
        "network operator tariff; they are only a transparent test profile. Savings are only the",
        "grid-fee component versus naive immediate charging on the same profile.",
        "",
        f"Flexible load: EV-like demand {ENERGY_KWH:.0f} kWh, {P_MAX_KW:.0f} kW max, available "
        f"{START_HOUR:02d}:00-24:00.",
        "",
        "| Scenario | feasible | scheduled kWh | shortfall kWh | total cost EUR | naive cost EUR | saving EUR | cap source | binding cap hours | planned hours |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        f"| No congestion cap | {str(base['feasible']).lower()} | {base['scheduled_kwh']:.1f} "
        f"| {base['shortfall_kwh']:.1f} | {base['total_cost_eur']:.2f} "
        f"| {base['naive_cost_eur']:.2f} | {base['saving_eur']:.2f} "
        f"| {base['cap_source']} | - | {_hours_with_energy(base['schedule_kwh'])} |",
        f"| Redispatch cap in cheap hour | {str(capped['feasible']).lower()} | {capped['scheduled_kwh']:.1f} "
        f"| {capped['shortfall_kwh']:.1f} | {capped['total_cost_eur']:.2f} "
        f"| {capped['naive_cost_eur']:.2f} | {capped['saving_eur']:.2f} "
        f"| {capped['cap_source']} | {', '.join(str(h) for h in capped['binding_cap_hours'])} "
        f"| {_hours_with_energy(capped['schedule_kwh'])} |",
        "",
        "## Fee profile",
        "",
        "| Hour | Fee EUR/kWh |",
        "|---:|---:|",
    ]
    for h, value in enumerate(fee):
        lines.append(f"| {h:02d} | {value:.2f} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "The optimizer first fills the cheapest available hours. When the redispatch cap blocks",
        "hour 22 and limits hour 23, the schedule moves energy to the next cheapest allowed hours.",
        "This is the intended composition: §14a network safety is hard, Module-3 tariff optimization",
        "only decides within that feasible space. The model is an energy-budget schedule, not a",
        "thermal dynamics or charging-comfort model.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
