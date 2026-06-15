#!/usr/bin/env python3
"""Build a rich operational cockpit JSON demo from real service outputs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.service.runner import run_forecast

OUT = Path("data_cache/benchmark/cockpit_demo.json")
CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"
REBAP = "data_cache/real/rebap_2024.csv"
SPOT = "data_cache/real/spot_da_2024.csv"


def _profile(value: float) -> list[float]:
    return [value] * 24


def main() -> None:
    pool_assets = [
        {"id": "WP-1", "demand_kw": _profile(10.0), "floor_kw": 4.2},
        {"id": "WB-1", "demand_kw": _profile(10.0), "floor_kw": 4.2},
        {"id": "BAT-1", "demand_kw": _profile(10.0), "floor_kw": 4.2},
    ]
    pool_cap = [40.0] * 24
    pool_cap[12] = 24.0
    result = run_forecast(
        CSV,
        utility="CockpitDemo",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        rating_kw=33000.0,
        steuve_malo="DE0001234567890",
        steuve_demands_kw=[1000.0, 800.0, 600.0],
        rolling_redispatch=True,
        rebap_csv=REBAP,
        spot_csv=SPOT,
        realized_economics=True,
        thermal_rating_kw=33000.0,
        dispatch_plan_enabled=True,
        dispatch_steuve_energy_kwh=40.0,
        dispatch_steuve_p_max_kw=1000.0,
        dispatch_risk_beta=0.6,
        grid_fee_eur_per_kwh=[
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
            0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
            0.35, 0.35, 0.35, 0.20, 0.05, 0.05,
        ],
        tariff_energy_kwh=40.0,
        tariff_p_max_kw=11.0,
        mmm_price_eur_mwh=60.0,
        pool_assets=pool_assets,
        pool_shared_cap_kw=pool_cap,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
