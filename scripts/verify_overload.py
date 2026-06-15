#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify probabilistische Überlast-Prognose (grid/overload.py) — reine stdlib, kein Internet.

Hand-Anker: exceedance 2/7, erwartete Überlast 15/7; Hosting-Capacity-Grenzeigenschaft + Monotonie.

Aufruf: python scripts/verify_overload.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.grid.overload import overload_forecast, hosting_capacity_kw, _exceedance_prob

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

# --- S1: keine Gefahr (Grenze weit über Last) ---
r = overload_forecast([100.0]*4, [[-10.0,-5.0,0.0,5.0,10.0]]*4, rating_kw=200.0)
check("S1: exceedance 0 überall", all(h["exceedance_prob"] == 0.0 for h in r["hourly"]))
check("S1: 0 erwartete Überlast", r["expected_overload_kwh_total"] == 0.0)
check("S1: 0 Stunden at risk", r["hours_at_risk"] == 0)
hc = hosting_capacity_kw([100.0]*4, [[-10.0,-5.0,0.0,5.0,10.0]]*4, rating_kw=200.0)
check("S1: Hosting-Capacity > 0", hc["hosting_capacity_kw"] > 0.0 and not hc["already_at_risk"])

# --- S2: sichere Überlast ---
r = overload_forecast([100.0]*4, [[-5.0,0.0,5.0]]*4, rating_kw=80.0)
check("S2: exceedance 1.0", all(h["exceedance_prob"] == 1.0 for h in r["hourly"]))
check("S2: erwartete Überlast 20 kWh/h", abs(r["hourly"][0]["expected_overload_kwh"] - 20.0) < 1e-9)
check("S2: Total 80 kWh", abs(r["expected_overload_kwh_total"] - 80.0) < 1e-9)

# --- S3: Exceedance + erwartete Überlast gegen Handwert ---
r = overload_forecast([100.0], [[-10.0,-5.0,0.0,5.0,10.0,15.0,20.0]], rating_kw=110.0)
h0 = r["hourly"][0]
check("S3: exceedance == 2/7", abs(h0["exceedance_prob"] - 2.0/7.0) < 1e-3)
check("S3: erwartete Überlast == 15/7", abs(h0["expected_overload_kwh"] - 15.0/7.0) < 1e-3)
check("S3: p90_load == 117", abs(h0["p90_load_kw"] - 117.0) < 1e-6)

# --- S4: Hosting-Capacity-Grenzeigenschaft (point 100, res ±20, rating 150, alpha 0.2) ---
pt = [100.0, 100.0]; res = [[-20.0,-10.0,0.0,10.0,20.0]]*2
hc = hosting_capacity_kw(pt, res, rating_kw=150.0, risk_alpha=0.2)
cap = hc["hosting_capacity_kw"]
def maxexc(extra):
    return max(_exceedance_prob(100.0, sorted(res[0]), 150.0, extra_kw=extra) for _ in range(2))
check("S4: bei cap ist max-Überlast <= alpha", maxexc(cap) <= 0.2 + 1e-9)
check("S4: knapp über cap reißt die Schwelle", maxexc(cap + 1.0) > 0.2 + 1e-9)
check("S4: cap ~ 40 (Handwert)", abs(cap - 40.0) < 0.5)

# --- S5: Hosting-Capacity monoton in alpha ---
caps = [hosting_capacity_kw(pt, res, 150.0, risk_alpha=a)["hosting_capacity_kw"] for a in (0.05, 0.2, 0.4)]
check("S5: höheres alpha -> mehr Reserve (monoton)", caps[0] <= caps[1] <= caps[2])

# --- S6: schon at risk -> Hosting 0 ---
hc = hosting_capacity_kw([100.0], [[-5.0,0.0,5.0]], rating_kw=90.0, risk_alpha=0.05)
check("S6: already_at_risk -> Hosting 0", hc["already_at_risk"] and hc["hosting_capacity_kw"] == 0.0)

# --- S7: NaN/None werden verworfen ---
r = overload_forecast([100.0], [[-10.0, float("nan"), 20.0, None, 0.0]], rating_kw=110.0)
check("S7: nicht-finite verworfen (exceedance aus 3 Werten = 1/3)", abs(r["hourly"][0]["exceedance_prob"] - 1.0/3.0) < 1e-3)

# --- S8: Validierung ---
def raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True
check("S8: leerer Horizont -> ValueError", raises(lambda: overload_forecast([], [], 100.0)))
check("S8: rating<=0 -> ValueError", raises(lambda: overload_forecast([100.0], [[0.0]], 0.0)))
check("S8: Längen-Mismatch -> ValueError", raises(lambda: overload_forecast([100.0,100.0], [[0.0]], 100.0)))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
