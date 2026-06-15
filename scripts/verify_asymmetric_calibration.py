#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify asymmetrische Coverage-Kalibrierung (eval/coverage_calibration.py). Deterministisch, schnell.
Aufruf: python scripts/verify_asymmetric_calibration.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.eval.coverage_calibration import (asymmetric_coverage_scale, apply_asymmetric,
                                                 rolling_asymmetric_scale)

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

rng = np.random.default_rng(0)
N = 20000
# rechtsschief: verschobene Exponential -> mehr Masse über +1 als unter -1
r = rng.exponential(0.6, N) - 0.5
p50 = np.zeros(N); p10 = p50 - 1; p90 = p50 + 1
flo0 = np.mean(r < -1) * 100; fhi0 = np.mean(r > 1) * 100

check(f"S1: Setup rechtsschief (fhi>flo: {fhi0:.1f}>{flo0:.1f})", fhi0 > flo0)
s_lo, s_hi = asymmetric_coverage_scale(r, p10, p50, p90, target_tail=0.1, shrink=1.0)
check(f"S2: s_hi>s_lo (mehr Weitung oben: {s_hi:.2f}>{s_lo:.2f})", s_hi > s_lo)
lo, hi = apply_asymmetric(p10, p50, p90, s_lo, s_hi)
flo1 = np.mean(r < lo) * 100; fhi1 = np.mean(r > hi) * 100
check(f"S3: untere Tail näher 10 ({flo0:.1f}->{flo1:.1f})", abs(flo1 - 10) <= abs(flo0 - 10) + 0.5)
check(f"S3: obere Tail näher 10 ({fhi0:.1f}->{fhi1:.1f})", abs(fhi1 - 10) <= abs(fhi0 - 10) + 0.5)

g = rng.normal(0, 1 / 1.2815515594, N)   # symmetrisch
sl2, sh2 = asymmetric_coverage_scale(g, p50 - 1, p50, p50 + 1, 0.1, shrink=1.0)
check(f"S4: symmetrisch -> s_lo~s_hi ({sl2:.2f}~{sh2:.2f})", abs(sl2 - sh2) < 0.25)
check("S5: shrink=0 -> (1,1)", asymmetric_coverage_scale(r, p10, p50, p90, 0.1, shrink=0.0) == (1.0, 1.0))
check("S6: target_tail>=0.5 -> ValueError", raises(lambda: asymmetric_coverage_scale(r, p10, p50, p90, target_tail=0.6)))
check("S6: neg s in apply -> ValueError", raises(lambda: apply_asymmetric(p10, p50, p90, -1, 1)))

# --- Rolling: Kausalität/Leakage ---
n, H = 120, 24
a2 = rng.exponential(0.6, (n, H)) - 0.5
P50 = np.zeros((n, H)); P10 = P50 - 1; P90 = P50 + 1
slo, shi, _, _ = rolling_asymmetric_scale(a2, P10, P50, P90, window=28, min_window=14)
ap = a2.copy(); i0 = 80; ap[i0] = 999.0
slo2, shi2, _, _ = rolling_asymmetric_scale(ap, P10, P50, P90, window=28, min_window=14)
check("S7: rolling kausal (actual[i0] ändert s[:i0+1] nicht)",
      np.allclose(slo[:i0 + 1], slo2[:i0 + 1]) and np.allclose(shi[:i0 + 1], shi2[:i0 + 1]))
check("S7: erste min_window s_lo=s_hi=1", np.allclose(slo[:14], 1.0) and np.allclose(shi[:14], 1.0))
check("S7: rechtsschief -> mean s_hi>s_lo nach Warmup", shi[40:].mean() > slo[40:].mean())
check("S7: 1D-Eingabe -> ValueError", raises(lambda: rolling_asymmetric_scale(a2[:, 0], P10[:, 0], P50[:, 0], P90[:, 0])))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
