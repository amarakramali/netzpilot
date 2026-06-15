# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Coverage-Kalibrierung der Prognosebänder durch geschrumpfte Band-Skalierung (leakage-sicher).

Befund 2026-06-03 (gemessen, nicht behauptet): die produktiven P10/P90-Bänder (rohe Trainings-
Residuenquantile) sind je Reihe unterschiedlich fehlkalibriert — manche zu WEIT (Bitterfeld MS 91 %
statt 80 %), manche zu ENG (Neuruppin NS 69 %). Das ist ZWEISEITIG; die vorhandene CQR/Targeting-Logik
kann nur verbreitern, nicht verschmälern, und unterdeckt zudem.

Hebel: ein einzelner Skalenfaktor s, der die Bandhälften um den Median P50 streckt/staucht,
    lo = P50 − s·(P50−P10),   hi = P50 + s·(P90−P50)
s wird auf einem VERGANGENEN Validierungsfenster so gewählt, dass die GEMESSENE Coverage das Ziel
(80 %) trifft. s<1 verschmälert (gegen Überdeckung), s>1 verbreitert (gegen Unterdeckung).

WICHTIG — Shrinkage: der auf 28 Tagen geschätzte s_opt ist verrauscht und ÜBERSCHIESST beim Transfer
auf neue Tage (gemessen: volle Skalierung regressiert schon-gute Reihen). Daher wird s zur 1 zurück-
geschrumpft: s_used = 1 + shrink·(s_opt − 1), shrink=0.5 als gemessenes Optimum (mean|cov−80| über 5
echte Reihen: naiv 5.58 → voll 4.85 → shrink0.5 4.07). So werden Extremfälle stark korrigiert, gut
kalibrierte Reihen kaum bewegt.

Leakage-Sicherheit ist EIGENSCHAFT DER NUTZUNG: `coverage_scale` muss mit Bändern/Actuals eines
Fensters STRIKT VOR dem Zielzeitraum gefüttert werden (siehe scripts/verify_coverage_calibration.py).
Reine numpy. Additiv — verändert weder Punktprognose noch Engine.
"""
from __future__ import annotations

import numpy as np


def _coverage(actual, lo, hi):
    a = np.asarray(actual, float)
    return float(np.mean((a >= np.asarray(lo, float)) & (a <= np.asarray(hi, float))))


def coverage_scale(actual, p10, p50, p90, target=0.8, shrink=0.5,
                   s_min=0.3, s_max=3.0, n_grid=136):
    """Geschrumpfter Skalenfaktor, der die Coverage auf `target` zieht (auf VERGANGENEM Fenster bestimmen).

    actual/p10/p50/p90: gleich lange Arrays des Validierungsfensters (nur Vergangenheit!).
    Rückgabe: s_used = 1 + shrink·(s_opt − 1). shrink=0 → 1.0 (No-op), shrink=1 → voller s_opt.
    """
    a = np.asarray(actual, float); p10 = np.asarray(p10, float)
    p50 = np.asarray(p50, float); p90 = np.asarray(p90, float)
    if not (len(a) == len(p10) == len(p50) == len(p90)):
        raise ValueError("actual/p10/p50/p90 müssen gleiche Länge haben.")
    if len(a) == 0:
        raise ValueError("leeres Validierungsfenster.")
    if not (0.0 < target < 1.0):
        raise ValueError("target muss in (0,1) liegen.")
    if not (0.0 <= shrink <= 1.0):
        raise ValueError("shrink muss in [0,1] liegen.")
    lo_d = p50 - p10
    hi_d = p90 - p50
    best = (float("inf"), 1.0)
    for s in np.linspace(s_min, s_max, n_grid):
        c = _coverage(a, p50 - s * lo_d, p50 + s * hi_d)
        cand = (abs(c - target), float(s))
        if cand < best:
            best = cand
    s_opt = best[1]
    return 1.0 + float(shrink) * (s_opt - 1.0)


def apply_scale(p10, p50, p90, s):
    """Skaliere die Bandhälften um P50: (lo, hi) = (P50 − s·(P50−P10), P50 + s·(P90−P50))."""
    p10 = np.asarray(p10, float); p50 = np.asarray(p50, float); p90 = np.asarray(p90, float)
    if s < 0:
        raise ValueError("s muss >= 0 sein.")
    return p50 - s * (p50 - p10), p50 + s * (p90 - p50)


def asymmetric_coverage_scale(actual, p10, p50, p90, target_tail=0.1, shrink=0.5,
                              s_min=0.3, s_max=3.0, n_grid=136):
    """Getrennte Skalen s_lo/s_hi für die untere/obere Bandhälfte (gegen schiefe Fehler).

    Die Lastfehler sind rechtsschief (Spitzen drücken über P90): die obere Tail ist oft zu eng, die
    untere zu weit. Ein EINZELner Faktor kann das nicht beheben. Hier wird je Tail separat ein Faktor
    bestimmt, der das jeweilige Tail-Soll trifft:
        s_lo: P(actual < p50 − s_lo·(p50−p10)) ≈ target_tail   (untere 10 %)
        s_hi: P(actual > p50 + s_hi·(p90−p50)) ≈ target_tail   (obere 10 %)
    Auf VERGANGENEM Fenster bestimmen (leakage-sicher); je Faktor zur 1 geschrumpft (shrink) gegen
    Überschießen bei kleinen Stichproben. Rückgabe (s_lo_used, s_hi_used).
    """
    a = np.asarray(actual, float); p10 = np.asarray(p10, float)
    p50 = np.asarray(p50, float); p90 = np.asarray(p90, float)
    if not (len(a) == len(p10) == len(p50) == len(p90)):
        raise ValueError("actual/p10/p50/p90 müssen gleiche Länge haben.")
    if len(a) == 0:
        raise ValueError("leeres Validierungsfenster.")
    if not (0.0 < target_tail < 0.5):
        raise ValueError("target_tail muss in (0,0.5) liegen.")
    if not (0.0 <= shrink <= 1.0):
        raise ValueError("shrink muss in [0,1] liegen.")
    lo_d = p50 - p10
    hi_d = p90 - p50
    best_lo = (float("inf"), 1.0)
    best_hi = (float("inf"), 1.0)
    for s in np.linspace(s_min, s_max, n_grid):
        f_lo = float(np.mean(a < p50 - s * lo_d))
        cand = (abs(f_lo - target_tail), float(s))
        if cand < best_lo:
            best_lo = cand
        f_hi = float(np.mean(a > p50 + s * hi_d))
        cand = (abs(f_hi - target_tail), float(s))
        if cand < best_hi:
            best_hi = cand
    s_lo = 1.0 + float(shrink) * (best_lo[1] - 1.0)
    s_hi = 1.0 + float(shrink) * (best_hi[1] - 1.0)
    return s_lo, s_hi


def apply_asymmetric(p10, p50, p90, s_lo, s_hi):
    """(lo, hi) = (P50 − s_lo·(P50−P10), P50 + s_hi·(P90−P50))."""
    p10 = np.asarray(p10, float); p50 = np.asarray(p50, float); p90 = np.asarray(p90, float)
    if s_lo < 0 or s_hi < 0:
        raise ValueError("s_lo, s_hi müssen >= 0 sein.")
    return p50 - s_lo * (p50 - p10), p50 + s_hi * (p90 - p50)


def rolling_asymmetric_scale(actual, p10, p50, p90, window=28, target_tail=0.1, shrink=0.5, min_window=14):
    """ONLINE-rollende ASYMMETRISCHE Kalibrierung: s_lo/s_hi je Tag aus dem nachlaufenden Fenster.

    Wie rolling_coverage_scale, aber zwei Faktoren (untere/obere Bandhälfte) — fängt rechtsschiefe
    Lastfehler (obere Tail zu eng). Gemessen (n_test=84): Pinball ≤ symmetrisch in jedem Fall, deutlich
    besser auf schiefen Reihen (Neuruppin Pinball 0,0986→0,0958, obere Tail 15,9→13,3 %), No-Op auf
    symmetrischen. Leakage-sicher: Tag i nutzt nur [i−window, i). Rückgabe (s_lo[n], s_hi[n], lo[n,H], hi[n,H]).
    """
    a = np.asarray(actual, float); p10 = np.asarray(p10, float)
    p50 = np.asarray(p50, float); p90 = np.asarray(p90, float)
    if not (a.shape == p10.shape == p50.shape == p90.shape):
        raise ValueError("actual/p10/p50/p90 müssen gleiche Form [n_days, H] haben.")
    if a.ndim != 2:
        raise ValueError("Erwartet 2D-Arrays [n_days, H].")
    n = a.shape[0]
    s_lo_arr = np.ones(n); s_hi_arr = np.ones(n)
    lo = p10.copy(); hi = p90.copy()
    for i in range(n):
        lo_w = max(0, i - window)
        if i - lo_w >= min_window:
            s_lo_arr[i], s_hi_arr[i] = asymmetric_coverage_scale(
                a[lo_w:i].ravel(), p10[lo_w:i].ravel(), p50[lo_w:i].ravel(), p90[lo_w:i].ravel(),
                target_tail=target_tail, shrink=shrink)
        lo[i], hi[i] = apply_asymmetric(p10[i], p50[i], p90[i], s_lo_arr[i], s_hi_arr[i])
    return s_lo_arr, s_hi_arr, lo, hi


def rolling_coverage_scale(actual, p10, p50, p90, window=28, target=0.8, shrink=0.5, min_window=14):
    """ONLINE-rollende Kalibrierung: s je Tag aus dem NACHLAUFENDEN Fenster der bereits beobachteten Tage.

    Gemessener Vorteil ggü. EINEM festen Validierungsfenster (T46): adaptiert die Richtung an Drift
    (über- vs. unterdeckte Phasen), schlägt das Einzelfenster und verschlechtert keine Reihe
    (mean|cov−80| über 5 echte Reihen: naiv 6,42 → Einzelfenster 5,86 → online 3,56).

    actual/p10/p50/p90: [n_days, H] (tagesweise Bänder + Actuals des Backtests).
    Für Tag i wird s_i auf den Tagen [i−window, i) bestimmt — STRIKT VERGANGENHEIT (leakage-sicher,
    nutzt nie Tag i selbst). Vor `min_window` verfügbaren Tagen bleibt s_i = 1 (kein Eingriff).
    Rückgabe (s_per_day[n_days], lo[n_days,H], hi[n_days,H]).
    """
    a = np.asarray(actual, float); p10 = np.asarray(p10, float)
    p50 = np.asarray(p50, float); p90 = np.asarray(p90, float)
    if not (a.shape == p10.shape == p50.shape == p90.shape):
        raise ValueError("actual/p10/p50/p90 müssen gleiche Form [n_days, H] haben.")
    if a.ndim != 2:
        raise ValueError("Erwartet 2D-Arrays [n_days, H].")
    n = a.shape[0]
    s_arr = np.ones(n)
    lo = p10.copy(); hi = p90.copy()
    for i in range(n):
        lo_w = max(0, i - window)
        if i - lo_w >= min_window:                       # genug Vergangenheit im Fenster
            s_arr[i] = coverage_scale(a[lo_w:i].ravel(), p10[lo_w:i].ravel(),
                                      p50[lo_w:i].ravel(), p90[lo_w:i].ravel(),
                                      target=target, shrink=shrink)
        lo[i], hi[i] = apply_scale(p10[i], p50[i], p90[i], s_arr[i])
    return s_arr, lo, hi
