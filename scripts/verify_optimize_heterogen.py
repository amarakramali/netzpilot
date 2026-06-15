#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify optimize_setpoints_heterogen (control/optimize.py) — reine stdlib, kein Internet.

Prueft die Eigenschaften, die fuer eine HETEROGENE steuVE-Flotte (WP/Wallbox/Speicher) gelten muessen:
  1) Kein Engpass (cap >= Summe Bedarf) -> voller Bedarf, 0 Abregelung.
  2) Engpass, erfuellbar -> Summe der Limits == cap EXAKT (=> minimal: genau die Ueberlast wird
     abgeregelt, kein kW mehr), jede steuVE >= ihrem EIGENEN Floor, keine ueber ihrem Bedarf.
  3) Individuelle Mindestleistungen werden respektiert (Mix aus 4,2-kW- und hoeheren Floors).
  4) §14a vs. Netz unvereinbar (cap < Summe Floors) -> alle auf Floor, feasible=False.
  5) Gewichtung wirkt: bei gleichem Bedarf wird die Anlage mit hoeherem Gewicht STAERKER abgeregelt.
  6) Robustheit bei Gewicht < 1 (der zuvor gefixte Bisektions-Rand): Summe haelt cap weiterhin exakt.
  7) Regression: die bestehende homogene optimize_setpoints liefert unveraendert die
     Water-Filling-Loesung (additive Aenderung hat das Altverhalten NICHT angetastet).

Aufruf: python scripts/verify_optimize_heterogen.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.control.optimize import optimize_setpoints, optimize_setpoints_heterogen
from netzpilot.control.schema import MIN_GUARANTEED_KW

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

TOL = 1e-6

# --- Szenario 1: kein Engpass ---
devs = [{"demand_kw": 11.0}, {"demand_kw": 7.0}, {"demand_kw": 22.0}]   # default floor 4,2 / weight 1
total = sum(x["demand_kw"] for x in devs)
r = optimize_setpoints_heterogen(devs, cap_kw=total + 5.0)
check("S1: kein Engpass -> voller Bedarf", r["limits_kw"] == [11.0, 7.0, 22.0])
check("S1: 0 kWh abgeregelt", r["total_shed_kw"] == 0.0)
check("S1: feasible, kein Floor bindend", r["feasible"] and not r["binding_floor"])

# --- Szenario 2: Engpass, erfuellbar -> Summe==cap exakt, Constraints ---
cap = 28.0                       # < total(40) und > Summe Floors(12,6) -> erfuellbar
r = optimize_setpoints_heterogen(devs, cap_kw=cap)
s = sum(r["limits_kw"])
check("S2: Summe Limits == cap (exakt minimal)", abs(s - cap) < TOL)
check("S2: total_shed == total - cap", abs(r["total_shed_kw"] - (total - cap)) < 1e-3)
check("S2: jede steuVE >= Floor 4,2", all(l >= MIN_GUARANTEED_KW - 1e-9 for l in r["limits_kw"]))
check("S2: keine steuVE ueber ihrem Bedarf",
      all(l <= d["demand_kw"] + 1e-9 for l, d in zip(r["limits_kw"], devs)))
check("S2: feasible", r["feasible"])

# --- Szenario 3: individuelle Floors (Mix) ---
# Wallbox Floor 4,2; grosse WP Floor 7,0 (Verdichter-Mindestleistung); Speicher Floor 0 (darf ganz aus)
mix = [{"demand_kw": 11.0, "floor_kw": 4.2},
       {"demand_kw": 15.0, "floor_kw": 7.0},
       {"demand_kw": 10.0, "floor_kw": 0.0}]
tmix = sum(x["demand_kw"] for x in mix)
r = optimize_setpoints_heterogen(mix, cap_kw=18.0)   # > Summe Floors(11,2), < total(36)
check("S3: Wallbox >= 4,2", r["limits_kw"][0] >= 4.2 - 1e-9)
check("S3: grosse WP >= 7,0", r["limits_kw"][1] >= 7.0 - 1e-9)
check("S3: Speicher >= 0", r["limits_kw"][2] >= -1e-9)
check("S3: Summe == cap", abs(sum(r["limits_kw"]) - 18.0) < TOL)

# --- Szenario 4: §14a vs. Netz unvereinbar ---
r = optimize_setpoints_heterogen(mix, cap_kw=5.0)    # < Summe Floors(11,2) -> nicht erfuellbar
check("S4: nicht erfuellbar -> feasible=False", r["feasible"] is False)
check("S4: alle auf ihrem Floor", r["limits_kw"] == [4.2, 7.0, 0.0])

# --- Szenario 5: Gewichtung wirkt (gleicher Bedarf, unterschiedliches Gewicht) ---
w = [{"demand_kw": 20.0, "weight": 2.0},   # traegt mehr Abregelung
     {"demand_kw": 20.0, "weight": 1.0}]   # traegt weniger
r = optimize_setpoints_heterogen(w, cap_kw=30.0)     # 10 kW muessen runter
shed0 = 20.0 - r["limits_kw"][0]
shed1 = 20.0 - r["limits_kw"][1]
check("S5: hoeheres Gewicht -> staerker abgeregelt", shed0 > shed1 + 1e-6)
# Exakt 2:1; Toleranz 5e-3 deckt die 3-Nachkomma-Rundung von limits_kw an beiden Enden ab.
check("S5: Verhaeltnis ~2:1 (solange kein Floor bindet)", abs(shed0 - 2 * shed1) < 5e-3)
check("S5: Summe == cap", abs(sum(r["limits_kw"]) - 30.0) < TOL)

# --- Szenario 6: Robustheit bei Gewicht < 1 (gefixter Bisektions-Rand) ---
wl = [{"demand_kw": 100.0, "floor_kw": 4.2, "weight": 0.3}]
r = optimize_setpoints_heterogen(wl, cap_kw=20.0)
check("S6: Gewicht<1, einzelne Anlage -> Limit == cap", abs(r["limits_kw"][0] - 20.0) < TOL)
wl2 = [{"demand_kw": 100.0, "weight": 0.3}, {"demand_kw": 50.0, "weight": 0.3}]
r = optimize_setpoints_heterogen(wl2, cap_kw=40.0)
check("S6: Gewicht<1, mehrere -> Summe == cap", abs(sum(r["limits_kw"]) - 40.0) < TOL)
check("S6: Floors gehalten", all(l >= MIN_GUARANTEED_KW - 1e-9 for l in r["limits_kw"]))

# --- Szenario 7: Regression der homogenen optimize_setpoints (Altverhalten unveraendert) ---
# Bekannter Water-Filling-Fall: 3 steuVE [10,10,4], cap 18, floor 4,2.
# Erwartung: die 4-kW-Anlage (unter Level) bleibt bei 4; die beiden 10er teilen sich gleich ->
# Level L mit 2L + 4 = 18 -> L = 7,0. (4 < L, also nicht weiter gedimmt.)
h = optimize_setpoints([10.0, 10.0, 4.0], cap_kw=18.0)
check("S7: homogen -> Summe == cap", abs(sum(h["limits_kw"]) - 18.0) < TOL)
check("S7: homogen -> gleicher Level fuer gedimmte (7,0/7,0/4,0)",
      abs(h["limits_kw"][0] - 7.0) < 1e-3 and abs(h["limits_kw"][1] - 7.0) < 1e-3
      and abs(h["limits_kw"][2] - 4.0) < 1e-3)

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
