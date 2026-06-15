#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify Drift-Erkennung (eval/drift.py) — reine stdlib, kein Internet.

Harte Anker: PSI gegen einen HANDGERECHNETEN Wert (0,054931) und KS gegen den exakten
Uniform-Shift-Fall (0,5). Dazu Status-Schwellen, MAE-/Bias-/Scale-Drift, Coverage, NaN, Validierung.

Aufruf: python scripts/verify_drift.py
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.eval.drift import (population_stability_index, ks_statistic,
                                   drift_report, coverage_report)

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

rng = random.Random(20260602)
gauss = [rng.gauss(0.0, 1.0) for _ in range(5000)]

# --- S1: identische Verteilung -> kein Drift ---
r = drift_report(gauss, list(gauss))
check("S1: PSI == 0 bei identisch", abs(r["psi"]) < 1e-9)
check("S1: KS == 0 bei identisch", abs(r["ks"]) < 1e-9)
check("S1: mae_ratio == 1", abs(r["mae_ratio"] - 1.0) < 1e-9)
check("S1: bias_shift == 0", abs(r["bias_shift_abs"]) < 1e-9)
check("S1: Status stable", r["status"] == "stable")

# --- S2: PSI gegen handgerechneten Wert ---
# ref = 0..999 (10 Bins a 100 -> p_ref=0,1). recent: Bin0=0,15, Bins1-8=0,1, Bin9=0,05.
ref = [float(i) for i in range(1000)]
recent = [50.0] * 150
for c in (150, 250, 350, 450, 550, 650, 750, 850):
    recent += [float(c)] * 100
recent += [950.0] * 50
psi = population_stability_index(ref, recent, n_bins=10)
# erwartet: 0,05*ln(1,5) + (-0,05)*ln(0,5) = 0,054931
check("S2: PSI == handgerechnet 0,054931", abs(psi - 0.054931) < 5e-3)

# --- S3: PSI Monotonie + Nichtnegativitaet ---
psi_small = population_stability_index(gauss, [x + 0.5 for x in gauss])
psi_big = population_stability_index(gauss, [x + 1.5 for x in gauss])
check("S3: PSI >= 0", psi_small >= 0 and psi_big >= 0)
check("S3: PSI monoton mit Verschiebung", psi_big > psi_small)

# --- S4: KS exakter Uniform-Shift-Fall ---
unif = [i / 1000.0 for i in range(1000)]
ks = ks_statistic(unif, [x + 0.5 for x in unif])
check("S4: KS Uniform-Shift um 0,5 == 0,5", abs(ks - 0.5) < 0.02)
check("S4: KS identisch == 0", abs(ks_statistic(unif, list(unif))) < 1e-9)
check("S4: KS monoton", ks_statistic(gauss, [x + 1.5 for x in gauss])
      > ks_statistic(gauss, [x + 0.5 for x in gauss]))

# --- S5: reiner Bias-Shift -> Drift (Bias-Grund) ---
r = drift_report(gauss, [x + 1.5 for x in gauss])
check("S5: bias_shift ~ 1,5", abs(r["bias_shift_abs"] - 1.5) < 0.05)
check("S5: bias_shift in ref-Std ~ 1,5", abs(r["bias_shift_in_ref_std"] - 1.5) < 0.1)
check("S5: Status drift", r["status"] == "drift")
check("S5: Grund nennt bias", any("bias" in s for s in r["reasons"]))

# --- S6: reiner Scale-Anstieg -> Drift (mae_ratio ~ 2) ---
r = drift_report(gauss, [x * 2.0 for x in gauss])
check("S6: mae_ratio ~ 2", abs(r["mae_ratio"] - 2.0) < 0.1)
check("S6: Status drift", r["status"] == "drift")
check("S6: Grund nennt mae_ratio", any("mae_ratio" in s for s in r["reasons"]))

# --- S7: WATCH-Fall (kleiner Bias-Shift 0,35 ref-Std, sonst ruhig) ---
r = drift_report(gauss, [x + 0.35 for x in gauss])
check("S7: Status watch (nicht drift)", r["status"] == "watch")
check("S7: PSI bleibt unter Drift-Schwelle", r["psi"] <= 0.25)

# --- S8: Coverage stable bei korrektem 80%-Intervall ---
gz = [rng.gauss(0.0, 1.0) for _ in range(8000)]
q80 = 1.2816  # 80%-Intervall von N(0,1): +/-1,2816
r = coverage_report([-q80] * len(gz), [q80] * len(gz), gz, nominal=0.8, tol=0.1)
check("S8: Coverage ~ 0,80", 0.77 <= r["coverage"] <= 0.83)
check("S8: Status stable", r["status"] == "stable")
# zu enges Intervall (50%) -> Drift
q50 = 0.6745
r = coverage_report([-q50] * len(gz), [q50] * len(gz), gz, nominal=0.8, tol=0.1)
check("S8: zu eng -> coverage ~ 0,5", 0.46 <= r["coverage"] <= 0.54)
check("S8: zu eng -> Status drift", r["status"] == "drift")
check("S8: Schwanz-Anteile ~ symmetrisch", abs(r["frac_below_lower"] - r["frac_above_upper"]) < 0.05)

# --- S9: NaN/None werden verworfen ---
r = drift_report(gauss, [1.0, 2.0, float("nan"), 3.0, None] + [rng.gauss(0, 1) for _ in range(50)])
check("S9: nicht-finite verworfen (n_recent == 53)", r["n_recent"] == 53)
check("S9: Ergebnis finit", math.isfinite(r["psi"]) and math.isfinite(r["ks"]))

# --- S10: Validierung ---
def raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True
check("S10: leere recent -> ValueError", raises(lambda: drift_report(gauss, [])))
check("S10: leere coverage -> ValueError", raises(lambda: coverage_report([], [], [])))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
