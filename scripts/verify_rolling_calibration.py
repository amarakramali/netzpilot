#!/usr/bin/env python3
"""Verify ONLINE-rollende Coverage-Kalibrierung (eval/coverage_calibration.rolling_coverage_scale).

Deterministisch, synthetisch, schnell. Kern: Kausalität/Leakage (s_i nutzt nie Tag i) + Adaption.
Aufruf: python scripts/verify_rolling_calibration.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.eval.coverage_calibration import rolling_coverage_scale

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

def cov_per_day(a, lo, hi):
    return np.mean((a >= lo) & (a <= hi), axis=1) * 100

H = 24
rng = np.random.default_rng(0)
sig = 1.0 / 1.2815515594        # N(0,sig): P(|.|<=1)=0.80
n = 120
a = rng.normal(0.0, sig, (n, H))
p50 = np.zeros((n, H))

# --- S1: KAUSALITÄT/LEAKAGE — s_i hängt nie von Tag i (oder späteren) ab ---
# Bänder ±2 (überdeckt) -> s adaptiert; dann actual[i0] massiv verfälschen, s_arr[:i0+1] muss gleich bleiben.
p10w, p90w = p50 - 2, p50 + 2
s_base, _, _ = rolling_coverage_scale(a, p10w, p50, p90w, window=28)
a2 = a.copy(); i0 = 80
a2[i0] = 999.0                  # nur Tag i0 zerstören
s_pert, _, _ = rolling_coverage_scale(a2, p10w, p50, p90w, window=28)
check("S1: actual[i0] aendert s_arr[:i0+1] NICHT (kausal/leakage-frei)",
      np.allclose(s_base[:i0 + 1], s_pert[:i0 + 1]))
check("S1: spaetere Tage reagieren (s_arr[i0+1:] veraendert)",
      not np.allclose(s_base[i0 + 1:], s_pert[i0 + 1:]))

# --- S2: vor min_window kein Eingriff (s=1) ---
s_arr, lo, hi = rolling_coverage_scale(a, p10w, p50, p90w, window=28, min_window=14)
check("S2: erste min_window Tage s==1", np.allclose(s_arr[:14], 1.0))

# --- S3: Adaption bei Überdeckung -> s<1 nach Warmup, Coverage Richtung 80 ---
naiv = cov_per_day(a, p10w, p90w)[28:].mean()       # ~99 %
calib = cov_per_day(a, lo, hi)[28:].mean()
check(f"S3: ueberdeckt -> s<1 nach Warmup (mean s[40:]={s_arr[40:].mean():.2f})", s_arr[40:].mean() < 0.85)
check(f"S3: online Coverage naeher an 80 ({naiv:.1f} -> {calib:.1f})", abs(calib - 80) < abs(naiv - 80))

# --- S4: Unterdeckung -> s>1 (Bänder ±0.5, ~48 %) ---
p10n, p90n = p50 - 0.5, p50 + 0.5
s_n, lon, hin = rolling_coverage_scale(a, p10n, p50, p90n, window=28)
naiv_u = cov_per_day(a, p10n, p90n)[28:].mean()
cal_u = cov_per_day(a, lon, hin)[28:].mean()
check(f"S4: unterdeckt -> s>1 nach Warmup (mean s[40:]={s_n[40:].mean():.2f})", s_n[40:].mean() > 1.3)
check(f"S4: online Coverage naeher an 80 ({naiv_u:.1f} -> {cal_u:.1f})", abs(cal_u - 80) < abs(naiv_u - 80))

# --- S5: Drift — erste Haelfte kalibriert (±1), zweite Haelfte ueberdeckt (±2): s faellt erst, wenn
#         das nachlaufende Fenster die ueberdeckte Phase erreicht (kausaler Lag, korrekt) ---
p10d = np.where(np.arange(n)[:, None] < n // 2, p50 - 1, p50 - 2)
p90d = np.where(np.arange(n)[:, None] < n // 2, p50 + 1, p50 + 2)
s_d, _, _ = rolling_coverage_scale(a, p10d, p50, p90d, window=28)
check("S5: Drift — s in kalibrierter Phase ~1, in ueberdeckter Phase <1",
      abs(s_d[40] - 1.0) < 0.15 and s_d[110] < 0.85)

# --- S6: Validierung ---
check("S6: 1D-Eingabe -> ValueError", raises(lambda: rolling_coverage_scale(a[:, 0], p10w[:, 0], p50[:, 0], p90w[:, 0])))
check("S6: Form-Mismatch -> ValueError", raises(lambda: rolling_coverage_scale(a, p10w[:50], p50, p90w)))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
