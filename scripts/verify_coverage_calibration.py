#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify Coverage-Kalibrierung (eval/coverage_calibration.py).

Synthetische Kontrollfälle (deterministisch) + echter leakage-sicherer Transfer-Beweis auf DSO-Reihen.
Aufruf: python scripts/verify_coverage_calibration.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.eval.coverage_calibration import coverage_scale, apply_scale, _coverage

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond
def raises(fn):
    try:
        fn(); return False
    except (ValueError,):
        return True

# Deterministische Actuals: N(0, 0.78) -> P(|.|<=1)=0.80 (zweiseitig). Coverage monoton steigend in s.
rng = np.random.default_rng(0)
N = 20000
sig = 1.0 / 1.2815515594   # 80%-zweiseitiges Quantil
a = rng.normal(0.0, sig, N)
p50 = np.zeros(N)

# --- S1: gut kalibriert (Band -1..+1) -> s_opt ~ 1 ---
s = coverage_scale(a, p50 - 1, p50, p50 + 1, target=0.8, shrink=1.0)
check(f"S1: gut kalibriert -> s~1 ({s:.2f})", abs(s - 1.0) < 0.12)

# --- S2: zu WEIT (Band -2..+2, ~99% Coverage) -> s_opt < 1 (verschmälern) ---
s_wide = coverage_scale(a, p50 - 2, p50, p50 + 2, target=0.8, shrink=1.0)
check(f"S2: überdeckt -> s<1 ({s_wide:.2f}, ~0.5)", s_wide < 0.85 and abs(s_wide - 0.5) < 0.15)

# --- S3: zu ENG (Band -0.5..+0.5, ~48%) -> s_opt > 1 (verbreitern) ---
s_narrow = coverage_scale(a, p50 - 0.5, p50, p50 + 0.5, target=0.8, shrink=1.0)
check(f"S3: unterdeckt -> s>1 ({s_narrow:.2f}, ~2.0)", s_narrow > 1.3 and abs(s_narrow - 2.0) < 0.3)

# --- S4: Shrinkage-Formel s_used = 1 + shrink*(s_opt-1) ---
s_full = coverage_scale(a, p50 - 2, p50, p50 + 2, target=0.8, shrink=1.0)
s_half = coverage_scale(a, p50 - 2, p50, p50 + 2, target=0.8, shrink=0.5)
s_zero = coverage_scale(a, p50 - 2, p50, p50 + 2, target=0.8, shrink=0.0)
check("S4: shrink=0 -> exakt 1.0", abs(s_zero - 1.0) < 1e-9)
check("S4: shrink=0.5 = 1+0.5*(s_full-1)", abs(s_half - (1 + 0.5 * (s_full - 1))) < 1e-9)

# --- S5: apply_scale monoton + Monotonie lo<=p50<=hi ---
lo1, hi1 = apply_scale(p50 - 1, p50, p50 + 1, 1.0)
lo2, hi2 = apply_scale(p50 - 1, p50, p50 + 1, 2.0)
los, his = apply_scale(p50 - 1, p50, p50 + 1, 0.5)
check("S5: s=2 breiter als s=1 breiter als s=0.5",
      float(np.mean(hi2 - lo2)) > float(np.mean(hi1 - lo1)) > float(np.mean(his - los)))
check("S5: lo<=p50<=hi", bool(np.all(lo1 <= p50) and np.all(p50 <= hi1)))

# --- S6: Validierung ---
check("S6: Längen-Mismatch -> ValueError", raises(lambda: coverage_scale([1, 2], [0], [0], [0])))
check("S6: leer -> ValueError", raises(lambda: coverage_scale([], [], [], [])))
check("S6: target außerhalb (0,1) -> ValueError", raises(lambda: coverage_scale(a, p50, p50, p50, target=1.5)))
check("S6: s<0 in apply_scale -> ValueError", raises(lambda: apply_scale(p50, p50, p50, -1.0)))

# --- S7: ECHTER leakage-sicherer Transfer-Beweis (tunen auf Vergangenheit, anwenden auf Holdout) ---
print("  -- S7: echte DSO-Reihen (leakage-sicher, n_test=28) --")
try:
    from scripts.benchmark_suite import robust_load_csv
    from scripts.dataset_manifest import MANIFEST as DM
    from netzpilot.features.build import to_daily_local, get_holidays
    from netzpilot.eval.backtest import rolling_origin
    from netzpilot.models.robust_corrector import ShrunkCorrector
    fac = lambda: ShrunkCorrector(10.0)
    idx = {m["key"]: m for m in DM}
    keys = ["bitterfeld_ms_2024", "neuruppin_ns_2022", "hilden_netzumsatz_2025"]
    dev_naiv, dev_cal = [], []
    for key in keys:
        e = idx[key]
        hourly = robust_load_csv(e["csv"], ts_col=e["ts"], load_col=e["col"], unit=e["unit"], return_meta=True)[0]
        l2, days, _ = to_daily_local(hourly)
        hol = get_holidays(sorted({d.year for d in days}), "NW")
        NT = 28; ND = len(l2)
        Rt, _ = rolling_origin(l2, days, fac, n_test=NT, holiday_set=hol)               # Holdout
        Rv, _ = rolling_origin(l2[:ND - NT], days[:ND - NT], fac, n_test=NT, holiday_set=hol)  # Vergangenheit
        s = coverage_scale(Rv["actual"], Rv["p10"], Rv["model"], Rv["p90"], target=0.8, shrink=0.5)
        lo, hi = apply_scale(Rt["p10"], Rt["model"], Rt["p90"], s)
        c_naiv = _coverage(Rt["actual"], Rt["p10"], Rt["p90"]) * 100
        c_cal = _coverage(Rt["actual"], lo, hi) * 100
        dev_naiv.append(abs(c_naiv - 80)); dev_cal.append(abs(c_cal - 80))
        print(f"     {key:24s} naiv {c_naiv:5.1f}% -> kalibriert {c_cal:5.1f}% (s={s:.2f})")
    mn, mc = float(np.mean(dev_naiv)), float(np.mean(dev_cal))
    check(f"S7: mean|cov-80| sinkt ({mn:.2f} -> {mc:.2f})", mc < mn)
except Exception as ex:  # pragma: no cover
    check(f"S7: echte Reihen (Ausnahme: {ex!r})", False)

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
