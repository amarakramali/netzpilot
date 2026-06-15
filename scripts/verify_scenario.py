#!/usr/bin/env python3
"""Verify Szenario-Engine W7 (grid/scenario.py) — reine stdlib, kein Internet.

Aufruf: python scripts/verify_scenario.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.grid.scenario import der_added_load_kw, simulate_scenario, coincidence_band

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond
def raises(fn):
    try:
        fn(); return False
    except (ValueError, KeyError):
        return True

H = 24
# Basis A: moderates Risiko (Punkt 80, Grenze 100, Residuen bis +40 -> 2/8 = 0.25 Überlast-Wkt)
ptA = [80.0] * H
resA = [[-30, -20, -10, 0, 10, 20, 30, 40] for _ in range(H)]
RATING = 100.0
wp = {"name": "WP", "count": 50, "rated_kw": 4.0, "coincidence": 0.8, "sign": +1}
pv = {"name": "PV", "count": 20, "rated_kw": 10.0, "coincidence": 0.85, "sign": -1}

# --- S1: Nullflotte -> Szenario identisch zur Basis ---
s0 = simulate_scenario(ptA, resA, RATING, [])
check("S1: Nullflotte added_peak == 0", s0["added_peak_kw"] == 0.0)
check("S1: Nullflotte delta alle 0",
      s0["delta"]["max_exceedance_prob"] == 0 and s0["delta"]["hours_at_risk"] == 0)
check("S1: Nullflotte scenario == base (Spitzen-Wkt)",
      s0["scenario"]["max_exceedance_prob"] == s0["base"]["max_exceedance_prob"])

# --- S2: Last hinzufügen -> Überlast-Wkt steigt (monoton) ---
sL = simulate_scenario(ptA, resA, RATING, [wp])
check("S2: Last erhöht Überlast-Wkt", sL["scenario"]["max_exceedance_prob"] >= sL["base"]["max_exceedance_prob"])
check("S2: 50 WP -> deutlicher Anstieg (delta > 0)", sL["delta"]["max_exceedance_prob"] > 0)
check("S2: added_peak == 50·4·0.8 = 160", abs(sL["added_peak_kw"] - 160.0) < 1e-6)

# --- S3: Erzeugung (PV) -> Überlast-Wkt sinkt ---
sG = simulate_scenario(ptA, resA, RATING, [pv])
check("S3: PV senkt Überlast-Wkt", sG["scenario"]["max_exceedance_prob"] <= sG["base"]["max_exceedance_prob"])
check("S3: PV added_peak == 20·10·0.85 = 170 (Betrag)", abs(sG["added_peak_kw"] - 170.0) < 1e-6)

# --- S4: Hosting-Capacity-Konsistenz: konstante Zusatzlast A senkt Kapazität um ~A ---
ptB = [40.0] * H
resB = [[-30, -20, -10, 0, 10, 20, 30, 40] for _ in range(H)]   # exceedance bei 40 -> 0; HC positiv
A = 8.0
constload = {"name": "const", "count": 1, "rated_kw": A, "coincidence": 1.0, "sign": +1}
sC = simulate_scenario(ptB, resB, RATING, [constload])
C0, C1 = sC["hosting_capacity_base_kw"], sC["hosting_capacity_scenario_kw"]
check(f"S4: Basis-Hosting-Capacity > 0 ({C0})", C0 > 0)
check(f"S4: konstante Last {A} senkt HC um ~{A} ({C0}->{C1})", abs((C0 - C1) - A) < 0.05)

# --- S5: Profil-Gewichtung (Last nur in Stunde 5) ---
prof = [0.0] * H; prof[5] = 1.0
wp5 = {**wp, "profile": prof}
added = der_added_load_kw([wp5], H)
check("S5: Profil -> Zusatzlast nur in Stunde 5", added[5] == 160.0 and sum(abs(a) for i, a in enumerate(added) if i != 5) == 0)

# --- S6: GLF-Band monoton (höherer Faktor -> mehr Last, >= Wkt, <= Kapazität) ---
band = coincidence_band(ptB, resB, RATING, [wp], factors=(0.25, 0.5, 1.0))["band"]
peaks = [r["added_peak_kw"] for r in band]
probs = [r["scenario_max_exceedance_prob"] for r in band]
caps = [r["hosting_capacity_scenario_kw"] for r in band]
check("S6: added_peak steigt mit Faktor", peaks[0] < peaks[1] < peaks[2])
check("S6: Überlast-Wkt nicht fallend", probs[0] <= probs[1] <= probs[2])
check("S6: Hosting-Capacity nicht steigend", caps[0] >= caps[1] >= caps[2])

# --- S7: Validierung ---
check("S7: coincidence > 1 -> ValueError",
      raises(lambda: der_added_load_kw([{"name": "x", "count": 1, "rated_kw": 1, "coincidence": 1.5}], H)))
check("S7: count < 0 -> ValueError",
      raises(lambda: der_added_load_kw([{"name": "x", "count": -1, "rated_kw": 1, "coincidence": 0.5}], H)))
check("S7: profile falsche Länge -> ValueError",
      raises(lambda: der_added_load_kw([{"name": "x", "count": 1, "rated_kw": 1, "coincidence": 0.5, "profile": [1, 1]}], H)))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
