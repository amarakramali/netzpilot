#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify EEBUS-LPC-Mapping (control/eebus_lpc.py) — reine stdlib, kein Internet.

Aufruf: python scripts/verify_eebus_lpc.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.control.schema import make_fahrplan
from netzpilot.control.eebus_lpc import fahrplan_to_lpc, EebusLpcAdapter

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

def sp(p_limit, floor, start, end):
    return {"p_limit_kw": p_limit, "floor_kw": floor, "start_utc": start, "end_utc": end}

T0 = "2026-01-01T12:00:00+00:00"
T1 = "2026-01-01T13:00:00+00:00"   # +1h
T2 = "2026-01-01T15:00:00+00:00"   # +3h von T0

# --- S1: Basis-Mapping ---
fp = make_fahrplan("MALO123456", [sp(10.0, 4.2, T0, T1)])
lpc = fahrplan_to_lpc(fp)
L0 = lpc["limits"][0]
check("S1: consumption_limit_w == 10000", L0["consumption_limit_w"] == 10000.0)
check("S1: failsafe_value_w == 4200", L0["failsafe_value_w"] == 4200.0)
check("S1: duration_s == 3600", L0["duration_s"] == 3600.0)
check("S1: is_limit_active True, n_limits 1", L0["is_limit_active"] and lpc["n_limits"] == 1)
check("S1: use_case LPC + transport extern", "LPC" in lpc["use_case"] and lpc["transport"] == "stub_external")

# --- S2: Einheiten-Umrechnung kW->W ---
lpc = fahrplan_to_lpc(make_fahrplan("MALO123456", [sp(7.5, 4.2, T0, T1)]))
check("S2: 7.5 kW -> 7500 W", lpc["limits"][0]["consumption_limit_w"] == 7500.0)

# --- S3: per-setpoint Failsafe = floor; Top-Level = min ---
fp = make_fahrplan("MALO123456", [sp(20.0, 7.0, T0, T1), sp(12.0, 4.2, T1, T2)])
lpc = fahrplan_to_lpc(fp)
check("S3: erstes Limit Failsafe 7000", lpc["limits"][0]["failsafe_value_w"] == 7000.0)
check("S3: Top-Level Failsafe == min (4200)", lpc["failsafe_value_w"] == 4200.0)
check("S3: n_limits 2", lpc["n_limits"] == 2)

# --- S4: §14a-Verstoß (p_limit < floor) -> ValueError ---
def raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True
bad_fp = {"malo": "MALO123456", "schedule_id": "x", "setpoints": [sp(3.0, 4.2, T0, T1)]}
check("S4: p_limit < Failsafe -> ValueError", raises(lambda: fahrplan_to_lpc(bad_fp)))

# --- S5: Dauer aus Zeitfenster (3h -> 10800 s) ---
lpc = fahrplan_to_lpc(make_fahrplan("MALO123456", [sp(10.0, 4.2, T0, T2)]))
check("S5: duration_s == 10800", lpc["limits"][0]["duration_s"] == 10800.0)

# --- S6: mehrere Setpoints nach start sortiert ---
fp = make_fahrplan("MALO123456", [sp(12.0, 4.2, T1, T2), sp(10.0, 4.2, T0, T1)])
lpc = fahrplan_to_lpc(fp)
check("S6: sortiert nach start_utc", lpc["limits"][0]["start_utc"] == T0 and lpc["limits"][1]["start_utc"] == T1)

# --- S7: Adapter-Round-Trip ---
ack = EebusLpcAdapter().submit(make_fahrplan("MALO123456", [sp(10.0, 4.2, T0, T1)]))
check("S7: Adapter status MAPPED", ack["status"] == "MAPPED")
check("S7: Adapter liefert LPC mit 1 Limit", ack["lpc"]["n_limits"] == 1)
check("S7: Transport extern (kein SMGW-Zugriff)", ack["lpc"]["transport"] == "stub_external")

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
