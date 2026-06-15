#!/usr/bin/env python3
"""Verify VPP-/Pool-Dispatch (control/vpp_pool.py) — reine stdlib, kein Internet.

Aufruf: python scripts/verify_vpp.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.control.vpp_pool import pool_dispatch

ok = True
def check(name, cond):
    global ok
    safe_name = str(name).encode("ascii", "replace").decode("ascii")
    print(f"  [{'PASS' if cond else 'FAIL'}] {safe_name}")
    ok = ok and cond

def assets3():
    return [{"id": "A", "demand_kw": [10.0, 10.0], "floor_kw": 4.2},
            {"id": "B", "demand_kw": [10.0, 10.0], "floor_kw": 4.2},
            {"id": "C", "demand_kw": [10.0, 10.0], "floor_kw": 4.2}]

# --- S1: kein Engpass -> voller Bedarf ---
r = pool_dispatch(assets3(), [40.0, 40.0])
check("S1: Pool-Limit == Σ Bedarf (30)", all(h["pool_limit_kw"] == 30.0 for h in r["hourly"]))
check("S1: keine Abregelung", r["pool_shed_kwh"] == 0.0 and r["grid_safe"] and r["all_feasible"])

# --- S2: Engpass -> Pool hält Cap, fair, jede Anlage >= Floor ---
r = pool_dispatch(assets3(), [24.0, 24.0])
check("S2: Pool-Limit == Cap (24)", all(abs(h["pool_limit_kw"] - 24.0) < 1e-6 for h in r["hourly"]))
check("S2: fair gleich (je 8 kW)", all(abs(x - 8.0) < 1e-3 for h in r["hourly"] for x in h["asset_limits_kw"]))
check("S2: jede Anlage >= 4,2", all(x >= 4.2 - 1e-9 for h in r["hourly"] for x in h["asset_limits_kw"]))
check("S2: Pool-Abregelung 12 kWh (6/Periode)", abs(r["pool_shed_kwh"] - 12.0) < 1e-6 and r["grid_safe"])

# --- S3: Infeasibilität (Cap < Σ Floors) -> alle auf Floor, nicht netzsicher ---
r = pool_dispatch(assets3(), [10.0, 10.0])
check("S3: all_feasible False", r["all_feasible"] is False)
check("S3: alle auf Floor 4,2", all(abs(x - 4.2) < 1e-9 for h in r["hourly"] for x in h["asset_limits_kw"]))
check("S3: grid_safe False (12,6 > 10)", r["grid_safe"] is False)

# --- S4: Pool-Flexibilitätsband ---
r = pool_dispatch(assets3(), [40.0, 40.0])
check("S4: Band min == Σ floor (12,6)", all(abs(b["min_kw"] - 12.6) < 1e-6 for b in r["pool_band"]))
check("S4: Band max == Σ demand (30)", all(b["max_kw"] == 30.0 for b in r["pool_band"]))

# --- S5: Komposition pool_limit == Σ asset_limits ---
r = pool_dispatch(assets3(), [24.0, 24.0])
check("S5: pool_limit == Σ asset_limits", all(abs(h["pool_limit_kw"] - sum(h["asset_limits_kw"])) < 1e-6 for h in r["hourly"]))

# --- S6: per_asset granted <= demand, shed >= 0 ---
r = pool_dispatch(assets3(), [24.0, 24.0])
check("S6: granted <= demand & shed >= 0", all(a["granted_kwh"] <= a["demand_kwh"] + 1e-9 and a["shed_kwh"] >= -1e-9 for a in r["per_asset"]))
check("S6: Pool-Bilanz: demand == granted + shed", abs(r["pool_demand_kwh"] - (r["pool_granted_kwh"] + r["pool_shed_kwh"])) < 1e-6)

# --- S7: heterogene Gewichte/Floors (eine Anlage trägt mehr) ---
mix = [{"id": "WP", "demand_kw": [20.0], "floor_kw": 7.0, "weight": 1.0},
       {"id": "WB", "demand_kw": [20.0], "floor_kw": 4.2, "weight": 2.0}]
r = pool_dispatch(mix, [30.0])
lim = r["hourly"][0]["asset_limits_kw"]
check("S7: Pool hält Cap 30", abs(r["hourly"][0]["pool_limit_kw"] - 30.0) < 1e-6)
check("S7: höheres Gewicht (WB) stärker gekappt", (20.0 - lim[1]) > (20.0 - lim[0]) - 1e-9)
check("S7: WP-Floor 7,0 gehalten", lim[0] >= 7.0 - 1e-9)

# --- S8: Validierung ---
def raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True
check("S8: keine Assets -> ValueError", raises(lambda: pool_dispatch([], [10.0])))
check("S8: demand-Länge != Horizont -> ValueError", raises(lambda: pool_dispatch([{"demand_kw": [1.0, 2.0]}], [10.0])))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
