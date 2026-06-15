#!/usr/bin/env python3
"""Build a VPP/pool dispatch demo on a real public DSO row."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.service.runner import run_forecast

OUT = Path("data_cache/benchmark/vpp_demo.md")
CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"


def _profile(value: float) -> list[float]:
    return [value] * 24


def main() -> None:
    assets = [
        {"id": "WP-1", "demand_kw": _profile(10.0), "floor_kw": 4.2, "weight": 1.0},
        {"id": "WB-1", "demand_kw": _profile(10.0), "floor_kw": 4.2, "weight": 1.0},
        {"id": "BAT-1", "demand_kw": _profile(10.0), "floor_kw": 4.2, "weight": 1.0},
    ]
    cap = [40.0] * 24
    cap[12] = 24.0
    result = run_forecast(
        CSV,
        utility="VppDemo",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        rating_kw=33000.0,
        pool_assets=assets,
        pool_shared_cap_kw=cap,
    )
    pool = result["pool_dispatch"]
    peak_hour = max(pool["hourly"], key=lambda h: h["pool_shed_kw"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VPP / pool dispatch demo",
        "",
        "Basis: echter Hilden-Netzumsatz-Lastgang plus drei illustrative steuVE-Assets hinter",
        "einer gemeinsamen Pool-Grenze. Die Pool-Assets sind Demo-Annahmen; der Forecast-Lauf",
        "nutzt eine einzige Rating-Wahrheit (`rating_kw=33000`) fuer Netzgrenze und Asset-Risiko.",
        "",
        f"Forecast date: {result['forecast_date']}.",
        f"Rating truth: {result['asset_limit']['rating_kw']:.0f} kW.",
        f"Pool grid safe: {str(pool['grid_safe']).lower()}; all feasible: {str(pool['all_feasible']).lower()}.",
        f"Pool shed total: {pool['pool_shed_kwh']:.3f} kWh.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| n_assets | {pool['n_assets']} |",
        f"| pool demand | {pool['pool_demand_kwh']:.3f} kWh |",
        f"| pool granted | {pool['pool_granted_kwh']:.3f} kWh |",
        f"| pool shed | {pool['pool_shed_kwh']:.3f} kWh |",
        f"| peak shed hour | {peak_hour['hour']} |",
        f"| peak hour cap | {peak_hour['cap_kw']:.3f} kW |",
        f"| peak hour pool limit | {peak_hour['pool_limit_kw']:.3f} kW |",
        "",
        "## Hourly pool dispatch",
        "",
        "| Hour | Pool demand kW | Cap kW | Pool limit kW | Shed kW | Asset limits kW | Feasible |",
        "|---:|---:|---:|---:|---:|---|---:|",
    ]
    for h in pool["hourly"]:
        lines.append(
            f"| {h['hour']:02d} | {h['pool_demand_kw']:.1f} | {h['cap_kw']:.1f} | "
            f"{h['pool_limit_kw']:.1f} | {h['pool_shed_kw']:.1f} | "
            f"{', '.join(f'{x:.1f}' for x in h['asset_limits_kw'])} | "
            f"{str(h['feasible']).lower()} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "In der Engpassstunde wird der Pool fair und minimal auf 24 kW gekappt: drei Assets",
        "erhalten je 8 kW und bleiben ueber ihrem 4,2-kW-Floor. Das ist Periodenkappung und",
        "Aggregation, keine zeituebergreifende Speicher-/SOC-Optimierung.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
