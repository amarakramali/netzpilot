#!/usr/bin/env python3
"""Verify feiertagsbewusste Baseline (features/holiday_base.py). Deterministisch, schnell.
Aufruf: python scripts/verify_holiday_base.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.features.holiday_base import holiday_aware_base, holiday_aware_resid_target

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond
def raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True

# Synthetik: Zeile i hat Wert i -> Rückgabewert verrät die gewählte Referenz-Zeile.
N = 40
l2 = np.tile(np.arange(N).reshape(-1, 1), (1, 3)).astype(float)
days = pd.date_range("2025-01-06", periods=N, freq="D")  # Start Montag

# --- S1: rückwärtskompatibel (ohne days/holiday_set == load2d[d-7]) ---
check("S1: ohne Kalender == load2d[d-7]", np.array_equal(holiday_aware_base(l2, 21), l2[14]))

# --- S2: d-7 Feiertag -> d-14 ---
d = 21
hs = {days[d - 7].date()}                      # d-7 ist Feiertag
check("S2: d-7 Feiertag -> Referenz d-14", holiday_aware_base(l2, d, days, hs)[0] == d - 14)

# --- S3: d-7 UND d-14 Feiertag -> d-21 ---
hs2 = {days[d - 7].date(), days[d - 14].date()}
check("S3: d-7 & d-14 Feiertag -> d-21", holiday_aware_base(l2, d, days, hs2)[0] == d - 21)

# --- S4: d-7 KEIN Feiertag (aber Set nicht leer) -> unverändert d-7 ---
hs3 = {days[d - 8].date()}                      # anderer Tag Feiertag
check("S4: d-7 kein Feiertag -> d-7", holiday_aware_base(l2, d, days, hs3)[0] == d - 7)

# --- S5: Leakage — Referenz IMMER < d ---
allok = True
for dd in range(7, N):
    hsx = {days[dd - 7].date(), days[dd - 14].date()} if dd >= 14 else {days[dd - 7].date()}
    ref = holiday_aware_base(l2, dd, days, hsx)[0]
    if not (ref < dd):
        allok = False
check("S5: Referenz strikt < d (leakage-frei)", allok)

# --- S6: resid_target konsistent (load2d[d] - base) ---
rt = holiday_aware_resid_target(l2, d, days, hs)
check("S6: resid_target == load2d[d] - base", np.array_equal(rt, l2[d] - holiday_aware_base(l2, d, days, hs)))

# --- S7: Randfälle ---
check("S7: d-7<0 -> ValueError", raises(lambda: holiday_aware_base(l2, 5)))
# alle möglichen Vorwochen-Refs Feiertag -> fällt auf kleinste gültige (>=0) zurück, kein Crash/Negativindex
hs_all = {days[i].date() for i in range(N)}
ref_small = holiday_aware_base(l2, 13, days, hs_all)[0]   # d=13: d-7=6; d-14=-1 nicht erlaubt -> bleibt 6
check("S7: keine Rückwärts-Ref möglich -> bleibt d-7 (kein Negativindex)", ref_small == 6)

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
