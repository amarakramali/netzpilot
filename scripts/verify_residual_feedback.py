#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify Online-Residuen-Feedback (models/residual_feedback.py). Deterministisch, schnell.
Aufruf: python scripts/verify_residual_feedback.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.models.residual_feedback import online_residual_feedback

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
n, H = 160, 4
phi = 0.6
r = np.zeros((n, H))
for d in range(1, n):
    r[d] = phi * r[d - 1] + rng.normal(0, 1, H)
f = np.full((n, H), 10.0)
a = f + r

rho, delta, corr = online_residual_feedback(f, a, window=28, shrink=1.0, min_window=14)
base = np.mean(np.abs(a[30:] - f[30:]))
cmae = np.mean(np.abs(a[30:] - corr[30:]))
check(f"S1: AR(1) corrected MAE < base ({cmae:.3f} < {base:.3f})", cmae < base)
check(f"S2: rho ~ phi nach Warmup (mean {rho[40:].mean():.2f} ~ 0.6)", abs(rho[40:].mean() - 0.6) < 0.25)

# Kausalität/Leakage: actual[i0] ändern -> corrected[:i0+1] unverändert; corrected[i0+1] reagiert
a2 = a.copy(); i0 = 100; a2[i0] += 50.0
_, _, corr2 = online_residual_feedback(f, a2, window=28, shrink=1.0)
check("S3: kausal — actual[i0] ändert corrected[:i0+1] nicht", np.allclose(corr[:i0 + 1], corr2[:i0 + 1]))
check("S3: corrected[i0+1] reagiert auf actual[i0]", not np.allclose(corr[i0 + 1], corr2[i0 + 1]))

# White noise -> rho ~ 0, kein Schaden
rw = rng.normal(0, 1, (n, H)); aw = f + rw
rho_w, _, corr_w = online_residual_feedback(f, aw, window=28, shrink=1.0)
bw = np.mean(np.abs(aw[30:] - f[30:])); cw = np.mean(np.abs(aw[30:] - corr_w[30:]))
check(f"S4: white-noise -> rho klein (mean {rho_w[40:].mean():.2f})", rho_w[40:].mean() < 0.2)
check("S4: white-noise -> kein Schaden (cw <= bw·1.02)", cw <= bw * 1.02)

# Warmup: erste min_window Tage rho=0 (ab d=min_window aktiv)
check("S5: erste min_window (14) Tage rho=0", np.allclose(rho[:14], 0.0))

# Validierung
check("S6: 1D -> ValueError", raises(lambda: online_residual_feedback(f[:, 0], a[:, 0])))
check("S6: Form-Mismatch -> ValueError", raises(lambda: online_residual_feedback(f, a[:50])))
check("S6: shrink außerhalb [0,1] -> ValueError", raises(lambda: online_residual_feedback(f, a, shrink=1.5)))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
