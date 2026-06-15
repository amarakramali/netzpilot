#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify CVaR risk-averse Nominierung (control/risk.py) — reine stdlib, kein Internet.

Kern: β=0 == Newsvendor-Erwartungswert (τ-Quantil); β>0 senkt CVaR (Tail) zu Lasten höheren
Erwartungswerts (Risk/Return). Plus CVaR-Handwert, Ternär-vs-Grid, Symmetrie, Validierung.

Aufruf: python scripts/verify_cvar.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.control.risk import (risk_averse_nomination, cvar, imbalance_costs,
                                     expected_value, _quantile)

ok = True
def check(name, cond):
    global ok
    safe_name = str(name).encode("ascii", "replace").decode("ascii")
    print(f"  [{'PASS' if cond else 'FAIL'}] {safe_name}")
    ok = ok and cond

res = [float(i - 50) for i in range(101)]      # Residuen -50..50
PT = 100.0

# --- S1: β=0 reproduziert das Newsvendor-τ-Quantil ---
r0 = risk_averse_nomination(PT, res, c_short=2.0, c_long=1.0, beta=0.0)
tau_q = PT + _quantile(sorted(res), 2.0/3.0)
check("S1: β=0 -> Nominierung == Newsvendor-τ-Quantil", abs(r0["nomination_kw"] - tau_q) < 0.5)
check("S1: tau_equiv == 2/3", abs(r0["tau_equiv"] - 2.0/3.0) < 1e-3)

# --- S2: CVaR >= Erwartungswert (gleiches q) ---
costs = imbalance_costs(100.0, PT, res, 2.0, 1.0)
check("S2: CVaR >= E", cvar(costs, 0.95) >= expected_value(costs) - 1e-9)

# --- S3: Risk/Return — β steuert Tail vs Erwartungswert (RECHTSSCHIEFE Fehler: seltene große
# Unterprognosen = teurer Short-Tail; nur hier weicht CVaR-Optimum klar vom Erwartungswert-Optimum ab) ---
res_skew = [float(i - 10) for i in range(21)] + [60.0, 70.0, 80.0, 90.0, 100.0]
noms, cvars, exps = [], [], []
for b in (0.0, 0.3, 0.6, 0.9, 1.0):
    r = risk_averse_nomination(PT, res_skew, c_short=3.0, c_long=1.0, beta=b, alpha=0.95)
    noms.append(r["nomination_kw"]); cvars.append(r["cvar_eur"]); exps.append(r["expected_cost_eur"])
check("S3: Nominierung steigt mit β (Tail-Schutz nominiert höher)",
      all(noms[i] <= noms[i+1] + 1e-6 for i in range(len(noms)-1)) and noms[-1] > noms[0] + 1e-6)
check("S3: CVaR sinkt mit β (Tail-Kosten runter)",
      all(cvars[i] >= cvars[i+1] - 1e-6 for i in range(len(cvars)-1)) and cvars[-1] < cvars[0] - 1e-6)
check("S3: Erwartungswert steigt mit β (Risk/Return-Tradeoff)", exps[-1] > exps[0] - 1e-6 and exps[-1] >= exps[0])

# --- S4: Ternärsuche == Grid-Minimum der Zielfunktion ---
def obj(q, b):
    c = imbalance_costs(q, PT, res, 3.0, 1.0)
    return (1-b)*expected_value(c) + b*cvar(c, 0.95)
r = risk_averse_nomination(PT, res, 3.0, 1.0, beta=0.7, alpha=0.95)
grid = [PT + x*0.1 for x in range(-500, 501)]
grid_min = min(grid, key=lambda q: obj(q, 0.7))
check("S4: Ternär-q nahe Grid-Minimum", abs(r["nomination_kw"] - grid_min) < 0.5)
check("S4: Ternär-Objective <= Grid-Objective + eps", obj(r["nomination_kw"], 0.7) <= obj(grid_min, 0.7) + 1e-6)

# --- S5: symmetrisch + c_short==c_long -> Nominierung ~ Median (point) ---
r = risk_averse_nomination(PT, res, c_short=1.0, c_long=1.0, beta=0.5, alpha=0.95)
check("S5: symmetrisch -> Nominierung ~ 100", abs(r["nomination_kw"] - 100.0) < 1.0)

# --- S6: CVaR-Handwert ---
check("S6: CVaR([0,0,0,0,10], 0.8) == 10", abs(cvar([0.0,0.0,0.0,0.0,10.0], 0.8) - 10.0) < 1e-6)

# --- S7: Validierung ---
def raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True
check("S7: β>1 -> ValueError", raises(lambda: risk_averse_nomination(PT, res, 2.0, 1.0, beta=1.5)))
check("S7: alpha=1 -> ValueError", raises(lambda: risk_averse_nomination(PT, res, 2.0, 1.0, alpha=1.0)))
check("S7: c_short=0 -> ValueError", raises(lambda: risk_averse_nomination(PT, res, 0.0, 1.0)))
check("S7: leere Residuen -> ValueError", raises(lambda: risk_averse_nomination(PT, [], 2.0, 1.0)))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
