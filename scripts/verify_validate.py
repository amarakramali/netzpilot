#!/usr/bin/env python3
"""Verify Messdaten-Validierung (data/validate.py) — reine stdlib, kein Internet.

Jeder Defekttyp einzeln: Lücke, Ausreißer (MAD), negativ, eingefroren (nur Meldung); plus
Ersatzwert-Methoden (saisonal / lineare Interpolation), Qualitätsscore, Validierung.

Aufruf: python scripts/verify_validate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.data.validate import validate_load

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

day = [50.0 + 10.0 * (h % 6) for h in range(24)]   # 50,60,70,80,90,100,... keine Gleichstands-Läufe
base = day * 3                                       # 3 Tage, alle sauber

# --- S1: sauberer Lastgang -> keine Issues, Qualität 1.0 ---
r = validate_load(list(base))
check("S1: keine Issues", r["n_missing"] == 0 and r["n_outlier"] == 0 and r["n_negative"] == 0 and r["n_frozen"] == 0)
check("S1: Qualität 1.0", r["quality_score"] == 1.0)
check("S1: cleaned == Eingang", all(abs(r["cleaned"][i] - base[i]) < 1e-9 for i in range(len(base))))

# --- S2: Lücke -> saisonaler Ersatz (gleicher Slot Vortag) ---
v = list(base); v[30] = None
r = validate_load(v)
check("S2: 1 Lücke erkannt", r["n_missing"] == 1)
check("S2: ersetzt, cleaned finit", r["cleaned"][30] is not None and r["n_replaced"] >= 1)
check("S2: saisonaler Wert == Slot Vortag (base[6]=50)", abs(r["cleaned"][30] - base[6]) < 1e-6)
check("S2: Methode saisonal", any(x["index"] == 30 and x["method"] == "seasonal_neighbor_day" for x in r["replacements"]))

# --- S3: Ausreißer (MAD) -> erkannt + ersetzt ---
v = list(base); v[40] = 100000.0
r = validate_load(v)
check("S3: Ausreißer erkannt", r["n_outlier"] == 1)
check("S3: ersetzt ~ Slot Vortag (base[16]=90)", abs(r["cleaned"][40] - base[16]) < 1e-6)

# --- S4: negativ -> erkannt; mit allow_negative aus/an ---
v = list(base); v[10] = -5.0
r = validate_load(v)
check("S4: negativ erkannt + ersetzt", r["n_negative"] == 1 and r["cleaned"][10] >= 0)
r2 = validate_load(v, allow_negative=True)
check("S4: allow_negative -> nicht geflaggt", r2["n_negative"] == 0)

# --- S5: eingefrorener Zähler -> nur Meldung, NICHT ersetzt ---
v = list(base)
for i in range(12, 19):
    v[i] = 77.0                                      # 7 identische Werte
r = validate_load(v, frozen_run=6)
check("S5: eingefroren erkannt (>=7)", r["n_frozen"] >= 7)
check("S5: eingefroren NICHT ersetzt (Wert bleibt 77)", abs(r["cleaned"][15] - 77.0) < 1e-9)

# --- S6: kurze Lücke ohne saisonalen Bezug -> lineare Interpolation ---
r = validate_load([10.0, 20.0, None, 40.0, 50.0], period_per_day=24, gap_interp_max=2)
check("S6: lineare Interpolation 20->40 ergibt 30", abs(r["cleaned"][2] - 30.0) < 1e-6)
check("S6: Methode linear_interpolation", any(x["index"] == 2 and x["method"] == "linear_interpolation" for x in r["replacements"]))

# --- S7: Qualitätsscore = sauber/n ---
v = list(base); v[5] = None; v[6] = -1.0; v[7] = 100000.0   # 3 fehlerhafte
r = validate_load(v)
check("S7: Qualität == (72-3)/72", abs(r["quality_score"] - (len(base) - 3) / len(base)) < 1e-3)

# --- S8: Validierung ---
def raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True
check("S8: leere Reihe -> ValueError", raises(lambda: validate_load([])))
check("S8: nur nicht-finite -> ValueError", raises(lambda: validate_load([None, float('nan')])))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
