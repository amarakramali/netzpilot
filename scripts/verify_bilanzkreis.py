#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify Bilanzkreis-Settlement (eval/bilanzkreis.py) — reine stdlib, kein Internet.

Kernaussage des Moduls: die echte Viertelstunden-Abrechnung faengt die Korrelation zwischen
Prognosefehler und Preis ein, die die lineare Naeherung (|Fehler| * Mittel-Spread) verfehlt.

Test-Harness-Hinweis: wir setzen scheduled=0 und actual=e, rebap=spread, spot=None, sodass
e_t = actual-scheduled = e und spread_t = reBAP-Spot = spread direkt steuerbar sind.

Aufruf: python scripts/verify_bilanzkreis.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.eval.bilanzkreis import imbalance_premium_eur, compare_forecasts_eur

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

def prem(errors, spreads):
    z = [0.0] * len(errors)
    return imbalance_premium_eur(z, errors, spreads, None)   # e=actual-0, spread=reBAP-0


# --- S1: perfekte Prognose -> 0 Praemie ---
r = prem([0.0, 0.0, 0.0, 0.0], [50.0, -30.0, 80.0, 10.0])
check("S1: perfekte Prognose -> Praemie 0", r["total_premium_eur"] == 0.0)

# --- S2: unkorrelierter Nullmittel-Fehler bei KONSTANTEM Spread -> ~0 (lineare Naeherung sagt viel) ---
err = [3.0, -3.0, 3.0, -3.0]
spr = [100.0, 100.0, 100.0, 100.0]
r = prem(err, spr)
linear_approx = r["sum_abs_error_mwh"] * r["mean_spread_eur_mwh"]   # |Fehler|*Mittel-Spread
check("S2: echte Praemie ~ 0 trotz Fehlervolumen", abs(r["total_premium_eur"]) < 1e-6)
check("S2: lineare Naeherung ueberschaetzt (=1200, echt 0)", abs(linear_approx - 1200.0) < 1e-6)
check("S2: sum_abs_error = 12", abs(r["sum_abs_error_mwh"] - 12.0) < 1e-9)

# --- S3: Fehler PERFEKT mit Spread korreliert -> grosse Praemie, die der Erwartungswert verfehlt ---
# Nullmittel-Fehler UND Nullmittel-Spread, aber gleichgerichtet -> reiner Korrelationsterm.
err = [2.0, 2.0, -2.0, -2.0]
spr = [50.0, 50.0, -50.0, -50.0]
r = prem(err, spr)
check("S3: Praemie = 400 (reiner Korrelationsterm)", abs(r["total_premium_eur"] - 400.0) < 1e-6)
check("S3: Bias-Term ~ 0 (Nullmittel)", abs(r["bias_term_eur"]) < 1e-6)
check("S3: Korrelations-Term = 400", abs(r["correlation_term_eur"] - 400.0) < 1e-6)
# bessere Prognose B (perfekt) -> spart genau die 400
cmp = compare_forecasts_eur([2.0, 2.0, -2.0, -2.0], [0.0]*4, [2.0, 2.0, -2.0, -2.0],
                            [50.0, 50.0, -50.0, -50.0], None)
check("S3: Einsparung B(perfekt) ggue. A = 400", abs(cmp["savings_b_vs_a_eur"] - 400.0) < 1e-6)

# --- S4: Einsparungs-Linearitaet: savings == sum (e_A - e_B)*spread ---
actual = [10.0, 12.0, 9.0, 11.0, 8.0]
sa = [9.0, 13.0, 9.5, 10.0, 9.0]     # Prognose A
sb = [10.2, 11.8, 9.1, 11.0, 8.2]    # Prognose B (besser)
reb = [60.0, -20.0, 120.0, 5.0, -40.0]
spot = [40.0, 35.0, 45.0, 38.0, 30.0]
cmp = compare_forecasts_eur(actual, sa, sb, reb, spot)
direct = sum(((actual[i]-sa[i]) - (actual[i]-sb[i])) * (reb[i]-spot[i]) for i in range(5))
check("S4: savings == sum (e_A-e_B)*(reBAP-Spot)", abs(cmp["savings_b_vs_a_eur"] - round(direct, 2)) < 1e-6)
check("S4: savings == premium_a - premium_b", abs(cmp["savings_b_vs_a_eur"]
      - round(cmp["premium_a_eur"] - cmp["premium_b_eur"], 2)) < 1e-6)

# --- S5: Decomposition-Identitaet total == bias + correlation (beliebige Reihen) ---
err = [1.5, -0.5, 2.0, -3.0, 0.7, 4.0]
spr = [30.0, -10.0, 90.0, 12.0, -25.0, 7.0]
r = prem(err, spr)
check("S5: total == bias + correlation", abs(r["total_premium_eur"]
      - (r["bias_term_eur"] + r["correlation_term_eur"])) < 1e-6)

# --- S6: reiner Bias (konstanter Fehler) -> Korrelations-Term ~ 0 ---
r = prem([2.0, 2.0, 2.0, 2.0], [10.0, 20.0, 30.0, 40.0])
check("S6: reiner Bias -> Praemie = 2*sum(spread) = 200", abs(r["total_premium_eur"] - 200.0) < 1e-6)
check("S6: Korrelations-Term ~ 0", abs(r["correlation_term_eur"]) < 1e-6)

# --- S7: NaN/None werden verworfen, kein NaN im Ergebnis ---
r = imbalance_premium_eur([0.0, 0.0, 0.0, 0.0], [2.0, float("nan"), 3.0, None],
                          [10.0, 10.0, 10.0, 10.0], None)
check("S7: 2 QH verworfen", r["n"] == 2 and r["n_dropped"] == 2)
check("S7: Ergebnis finit (2*10 + 3*10 = 50)", math.isfinite(r["total_premium_eur"])
      and abs(r["total_premium_eur"] - 50.0) < 1e-6)

# --- S8: Laengen-Mismatch -> ValueError ---
try:
    imbalance_premium_eur([0.0, 0.0], [1.0], [10.0, 10.0], None)
    raised = False
except ValueError:
    raised = True
check("S8: Laengen-Mismatch -> ValueError", raised)

# --- S9: spot=None identisch zu spot=0; reBAP-only-Modus = sum e*reBAP ---
r0 = imbalance_premium_eur([0.0]*3, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0], None)
rz = imbalance_premium_eur([0.0]*3, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [0.0, 0.0, 0.0])
check("S9: spot=None == spot=0", r0["total_premium_eur"] == rz["total_premium_eur"])
check("S9: reBAP-only = 1*10+2*20+3*30 = 140", abs(r0["total_premium_eur"] - 140.0) < 1e-6)

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
