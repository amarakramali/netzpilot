#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify §14a-Quantil-Dispatch (control/dispatch.py) — reine stdlib, kein Internet.

Prüft die Integration: §14a-Sicherheit (Grundlast+steuVE <= Schwelle überall), Newsvendor-Nominierung
(symmetrisch -> 0 Ersparnis, asymmetrisch wachsend, τ <= P50), Komposition, Energiebudget, Validierung.

Aufruf: python scripts/verify_dispatch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.control.dispatch import plan_day, cost_optimal_nomination, _quantile

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

H = 24
res = [-8.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0]           # geteilte Residuen-Stichprobe je Stunde
base = [80.0] * H
base[12] = 100.0                                        # eine Stunde direkt an der Schwelle
residuals = [list(res) for _ in range(H)]
fee = [0.10] * H
THR = 100.0

# --- S1: §14a-Sicherheit — Grundlast + steuVE nie über Schwelle, Cap=0 wo Grundlast==Schwelle ---
r = plan_day(base, residuals, THR, steuve_energy_kwh=200.0, steuve_p_max_kw=20.0,
             grid_fee_eur_per_kwh=fee, c_short=2.0, c_long=1.0)
maxtot = max(h["total_point_kw"] for h in r["hourly"])
check("S1: Netzgrenze überall gehalten (max total <= Schwelle)", maxtot <= THR + 1e-6)
check("S1: grid_safe True", r["grid_safe"] is True)
check("S1: Cap=0 in der Schwellen-Stunde 12", r["hourly"][12]["cap_kw"] == 0.0
      and r["hourly"][12]["steuve_kw"] == 0.0)

# --- S2: Komposition total == base + steuve; Nominierung == total + Quantil(res, τ) ---
comp_ok = all(abs(h["total_point_kw"] - round(base[h["hour"]] + h["steuve_kw"], 4)) < 1e-6 for h in r["hourly"])
check("S2: total_point == base + steuve (alle Stunden)", comp_ok)
tau = 2.0 / 3.0
nom_ok = True
for h in r["hourly"]:
    expect = h["total_point_kw"] + _quantile(sorted(res), tau)
    if abs(h["nomination_kw"] - round(expect, 4)) > 1e-3:
        nom_ok = False
check("S2: Nominierung == total + Quantil(res, tau)", nom_ok)

# --- S3: Energiebudget gedeckt (feasible) ---
total_steuve = sum(h["steuve_kw"] for h in r["hourly"])     # dt=1 -> kWh
check("S3: steuVE-Budget gedeckt (200 kWh)", abs(total_steuve - 200.0) < 1e-3 and r["feasible"])

# --- S4: Infeasibilität ehrlich (Budget > Summe Caps) ---
# caps: 23 Stunden à 20 + Stunde12=0 -> 460 kWh max; 600 verlangt -> shortfall 140
r2 = plan_day(base, residuals, THR, steuve_energy_kwh=600.0, steuve_p_max_kw=20.0,
              grid_fee_eur_per_kwh=fee, c_short=2.0, c_long=1.0)
check("S4: infeasible -> feasible False", r2["feasible"] is False)
check("S4: shortfall == 140 kWh", abs(r2["shortfall_kwh"] - 140.0) < 1e-3)
check("S4: trotzdem netzsicher", r2["grid_safe"] is True)

# --- S5: Newsvendor — symmetrisch -> 0 Ersparnis; asymmetrisch -> > 0 ---
r_sym = plan_day(base, residuals, THR, 200.0, 20.0, fee, c_short=1.0, c_long=1.0)
check("S5: symmetrisch -> newsvendor_saving == 0", abs(r_sym["newsvendor_saving_eur"]) < 1e-9)
check("S5: symmetrisch -> exp_tau == exp_p50", abs(r_sym["exp_imbalance_tau_eur"]
      - r_sym["exp_imbalance_p50_eur"]) < 1e-9)
check("S5: asymmetrisch -> saving > 0", r["newsvendor_saving_eur"] > 0.0)
check("S5: asymmetrisch -> exp_tau <= exp_p50", r["exp_imbalance_tau_eur"]
      <= r["exp_imbalance_p50_eur"] + 1e-9)

# --- S6: Newsvendor-Ersparnis wächst monoton mit der Asymmetrie ---
savings = []
for ratio in (1.0, 2.0, 3.0, 5.0):
    rr = plan_day(base, residuals, THR, 200.0, 20.0, fee, c_short=ratio, c_long=1.0)
    savings.append(rr["newsvendor_saving_eur"])
check("S6: saving(1.0) == 0", abs(savings[0]) < 1e-9)
check("S6: monoton wachsend mit Asymmetrie", all(savings[i] <= savings[i+1] + 1e-9
      for i in range(len(savings)-1)) and savings[-1] > savings[0])

# --- S7: cost_optimal_nomination-Helper direkt ---
nom, t = cost_optimal_nomination(100.0, res, c_short=2.0, c_long=1.0)
check("S7: tau == 2/3", abs(t - 2.0/3.0) < 1e-9)
check("S7: Nominierung == 100 + Quantil(res, 2/3)", abs(nom - (100.0 + _quantile(sorted(res), 2.0/3.0))) < 1e-9)

# --- S8: Validierung ---
def raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True
check("S8: c_short<=0 -> ValueError", raises(lambda: cost_optimal_nomination(100.0, res, 0.0, 1.0)))
check("S8: Längen-Mismatch -> ValueError", raises(lambda: plan_day(base, residuals[:5], THR, 200.0, 20.0, fee, 2.0, 1.0)))
check("S8: leere Residuen -> ValueError", raises(lambda: cost_optimal_nomination(100.0, [], 2.0, 1.0)))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
