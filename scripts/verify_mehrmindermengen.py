#!/usr/bin/env python3
"""Verify Mehr-/Mindermengen-Report (eval/mehrmindermengen.py) — reine stdlib, kein Internet.

Aufruf: python scripts/verify_mehrmindermengen.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.eval.mehrmindermengen import mehr_mindermengen, compare_forecasts_mmm

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

PRICE = 60.0

# --- S1: perfekte Prognose -> alles 0 ---
r = mehr_mindermengen([10.0]*4, [10.0]*4, PRICE)
check("S1: perfekt -> mehr/minder/netto 0", r["mehrmenge_mwh"] == 0 and r["mindermenge_mwh"] == 0 and r["netto_mwh"] == 0)
check("S1: abs_volumen 0", r["abs_volumen_mwh"] == 0)

# --- S2: konstante Mehrentnahme ---
r = mehr_mindermengen([10.0]*4, [12.0]*4, PRICE)
check("S2: Mehrmenge 8, Mindermenge 0", r["mehrmenge_mwh"] == 8.0 and r["mindermenge_mwh"] == 0.0)
check("S2: Netto 8, netto_eur 480", r["netto_mwh"] == 8.0 and abs(r["netto_eur"] - 480.0) < 1e-6)

# --- S3: Hand-Mischfall e=[2,2,-2,-2] ---
r = mehr_mindermengen([10.0,10.0,10.0,10.0], [12.0,12.0,8.0,8.0], PRICE)
check("S3: Mehr 4 / Minder 4 / Netto 0", r["mehrmenge_mwh"] == 4.0 and r["mindermenge_mwh"] == 4.0 and r["netto_mwh"] == 0.0)
check("S3: abs_volumen 8", r["abs_volumen_mwh"] == 8.0)
check("S3: mehr_eur 240, minder_eur 240, netto_eur 0", r["mehrmenge_eur"] == 240.0 and r["mindermenge_eur"] == 240.0 and r["netto_eur"] == 0.0)

# --- S4: Identitäten ---
r = mehr_mindermengen([5.0,8.0,3.0,10.0,7.0], [6.0,4.0,9.0,2.0,7.0], PRICE)
check("S4: netto == mehr - minder", abs(r["netto_mwh"] - (r["mehrmenge_mwh"] - r["mindermenge_mwh"])) < 1e-9)
check("S4: abs_volumen == mehr + minder", abs(r["abs_volumen_mwh"] - (r["mehrmenge_mwh"] + r["mindermenge_mwh"])) < 1e-9)

# --- S5: bessere Prognose senkt das absolute MMM-Volumen ---
actual = [10.0, 12.0, 9.0, 11.0]
cmp = compare_forecasts_mmm(actual, [8.0,15.0,12.0,8.0], list(actual), PRICE)
check("S5: B (perfekt) -> abs_volumen 0", cmp["abs_volumen_b_mwh"] == 0.0)
check("S5: A schlechter -> Reduktion == 11", abs(cmp["abs_volumen_reduktion_mwh"] - 11.0) < 1e-6)

# --- S6: dt_h-Skalierung (Leistung MW -> MWh) ---
r = mehr_mindermengen([10.0]*4, [12.0]*4, PRICE, dt_h=0.25)
check("S6: dt=0.25 -> Mehrmenge 2 MWh", abs(r["mehrmenge_mwh"] - 2.0) < 1e-9)

# --- S7: NaN/None verworfen ---
r = mehr_mindermengen([10.0, 10.0, 10.0], [12.0, float("nan"), None], PRICE)
check("S7: 2 Paare verworfen (n==1)", r["n"] == 1 and r["n_dropped"] == 2)

# --- S8: Validierung ---
def raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True
check("S8: Längen-Mismatch -> ValueError", raises(lambda: mehr_mindermengen([1.0,2.0], [1.0], PRICE)))
check("S8: leere Reihen -> ValueError", raises(lambda: mehr_mindermengen([], [], PRICE)))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
