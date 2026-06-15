# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Drift-Erkennung fuer Live-Prognosefehler vs. Referenz-Verteilung (Achse C.1).

Operativer Zweck: NetzPilot prognostiziert taeglich. Ueber Zeit kann das Modell still degradieren
(neue Waermepumpen/Wallboxen, geaendertes Verbrauchsverhalten, Sensordrift, Regimewechsel). Eine im
Backtest kalibrierte Prognose wird dann unbemerkt schlechter. Dieses Modul vergleicht die JUENGSTEN
Prognosefehler gegen die REFERENZ-Verteilung (aus dem Backtest/der Kalibrierung) und meldet, wann sie
weglaufen — mit interpretierbaren, dokumentierten Schwellen, damit ein Re-Kalibrieren/Neutrainieren
ausgeloest werden kann.

Drei komplementaere Signale:
  - ACCURACY: mae_ratio (recent/ref) + Bias-Shift (mittlerer Fehler weg von 0). Sagen, OB und WAS
    degradiert ist (Streuung vs. systematische Verzerrung) — direkt handlungsweisend.
  - VERTEILUNG: PSI (Population Stability Index, Industriestandard) + KS (verteilungsfrei). Fangen
    Form-/Lage-/Skalen-Drift in einer Zahl mit etablierten Schwellen.
  - KALIBRIERUNG (separat, wenn Intervalle vorliegen): coverage_report prueft, ob die CQR-P10/P90-
    Intervalle ihre nominale Ueberdeckung noch halten.

Schwellen (Defaults, dokumentiert & parametrisierbar):
  PSI:        < 0,1 stabil | 0,1–0,25 beobachten | > 0,25 Drift   (gaengige PSI-Konvention)
  mae_ratio:  < 1,15 stabil | 1,15–1,30 beobachten | > 1,30 Drift
  bias_shift: |Delta Bias| in Einheiten der Referenz-Std: < 0,25 stabil | 0,25–0,5 beobachten | > 0,5 Drift
Status = schlimmster ausgeloester Einzelbefund; `reasons` listet die ausloesenden Signale.
KS wird als ergaenzende Evidenz BERICHTET (mit 5%-kritischem Wert), aber NICHT separat in den Status
gezaehlt — die Verteilungs-Drift deckt bereits PSI ab (kein Doppelzaehlen).

Ehrliche Vorbehalte (CLAUDE.md):
  - Drift-Erkennung ist eine WARNUNG, kein Beweis von Ursache. Sie sagt „pruefen/neu kalibrieren", nicht
    warum. Schwellen sind Heuristiken — pro Einsatz nachjustierbar.
  - Bei kleinen recent-Stichproben ist jedes Signal verrauscht; n_recent wird mitgegeben, kleine n
    ehrlich als unsicher behandeln (der Aufrufer sollte ein Mindest-n erzwingen).
  - reBAP/€ spielen hier keine Rolle — rein die Prognosegueteverteilung.

Reine stdlib (math) — kein numpy noetig, ueberall importierbar. Additiv: aendert nichts an bestehenden Modulen.
"""
from __future__ import annotations

import math


def _finite(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clean(xs) -> list:
    out = []
    for x in xs:
        v = _finite(x)
        if v is not None:
            out.append(v)
    return out


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    p = (len(sorted_vals) - 1) * max(0.0, min(1.0, q))
    lo, hi = int(p), min(int(p) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (p - lo)


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def population_stability_index(reference, recent, n_bins: int = 10, eps: float = 1e-4) -> float:
    """PSI zwischen Referenz- und Recent-Verteilung. Bins = Quantile der REFERENZ.

    PSI = sum_bins (p_recent - p_ref) * ln(p_recent / p_ref). 0 = identisch; gaengige Lesart:
    <0,1 stabil, 0,1-0,25 moderate Verschiebung, >0,25 deutliche Verschiebung. Leere Bins werden mit
    eps abgefangen (kein ln(0)/Division durch 0).
    """
    ref = sorted(_clean(reference))
    rec = _clean(recent)
    if len(ref) < n_bins or not rec:
        raise ValueError(f"Zu wenig Daten fuer PSI (ref={len(ref)}, recent={len(rec)}, n_bins={n_bins}).")
    edges = [_quantile(ref, k / n_bins) for k in range(1, n_bins)]

    def bin_idx(x):
        i = 0
        for e in edges:
            if x <= e:
                return i
            i += 1
        return i

    rc = [0] * n_bins
    cc = [0] * n_bins
    for x in ref:
        rc[bin_idx(x)] += 1
    for x in rec:
        cc[bin_idx(x)] += 1
    nref, nrec = len(ref), len(rec)
    psi = 0.0
    for k in range(n_bins):
        pr = max(rc[k] / nref, eps)
        pc = max(cc[k] / nrec, eps)
        psi += (pc - pr) * math.log(pc / pr)
    return psi


def ks_statistic(reference, recent) -> float:
    """Zwei-Stichproben-Kolmogorov-Smirnov-Statistik D = max_x |F_ref(x) - F_recent(x)|. Verteilungsfrei."""
    a = sorted(_clean(reference))
    b = sorted(_clean(recent))
    if not a or not b:
        raise ValueError("Leere Reihe fuer KS.")
    na, nb = len(a), len(b)
    i = j = 0
    d = 0.0
    while i < na and j < nb:
        # gemeinsamer Schwellenwert x = kleinerer Kopf; BEIDE Seiten ueber alle Gleichstaende von x
        # vorruecken, sonst zaehlt ein wertgleicher Punkt faelschlich als CDF-Luecke (KS != 0 bei identisch).
        x = a[i] if a[i] <= b[j] else b[j]
        while i < na and a[i] == x:
            i += 1
        while j < nb and b[j] == x:
            j += 1
        d = max(d, abs(i / na - j / nb))
    return d


def drift_report(reference_errors, recent_errors, *, n_bins: int = 10,
                 psi_watch: float = 0.1, psi_drift: float = 0.25,
                 mae_ratio_watch: float = 1.15, mae_ratio_drift: float = 1.30,
                 bias_shift_watch: float = 0.25, bias_shift_drift: float = 0.5) -> dict:
    """Vergleicht jüngste Prognosefehler gegen die Referenz-Fehlerverteilung und gibt einen Drift-Status.

    reference_errors / recent_errors: Fehler e = actual - forecast (gleiche Definition!) je Periode.
    Rueckgabe dict: n_ref/n_recent, bias_*, mae_*, std_*, bias_shift_abs, bias_shift_in_ref_std,
      mae_ratio, psi, ks, ks_crit_5pct, ks_exceeds_5pct, status ('stable'|'watch'|'drift'), reasons[].
    """
    ref = _clean(reference_errors)
    rec = _clean(recent_errors)
    if not ref or not rec:
        raise ValueError("Leere Referenz- oder Recent-Reihe.")

    bias_ref, bias_rec = _mean(ref), _mean(rec)
    mae_ref = _mean([abs(x) for x in ref])
    mae_rec = _mean([abs(x) for x in rec])
    std_ref = _std(ref)
    bias_shift = bias_rec - bias_ref
    bias_shift_std = abs(bias_shift) / std_ref if std_ref > 1e-12 else float("inf") if abs(bias_shift) > 1e-12 else 0.0
    mae_ratio = mae_rec / mae_ref if mae_ref > 1e-12 else float("inf") if mae_rec > 1e-12 else 1.0
    psi = population_stability_index(ref, rec, n_bins=n_bins)
    ks = ks_statistic(ref, rec)
    ks_crit = 1.358 * math.sqrt((len(ref) + len(rec)) / (len(ref) * len(rec)))

    reasons = []
    level = 0  # 0 stable, 1 watch, 2 drift

    def flag(metric, value, watch, drift, fmt):
        nonlocal level
        if value > drift:
            reasons.append(f"DRIFT {metric}={fmt.format(value)} (> {drift})")
            level = max(level, 2)
        elif value > watch:
            reasons.append(f"WATCH {metric}={fmt.format(value)} (> {watch})")
            level = max(level, 1)

    flag("psi", psi, psi_watch, psi_drift, "{:.3f}")
    flag("mae_ratio", mae_ratio, mae_ratio_watch, mae_ratio_drift, "{:.3f}")
    flag("bias_shift_in_ref_std", bias_shift_std, bias_shift_watch, bias_shift_drift, "{:.3f}")

    status = ("stable", "watch", "drift")[level]
    return {
        "n_ref": len(ref),
        "n_recent": len(rec),
        "bias_ref": round(bias_ref, 4),
        "bias_recent": round(bias_rec, 4),
        "bias_shift_abs": round(bias_shift, 4),
        "bias_shift_in_ref_std": round(bias_shift_std, 4) if math.isfinite(bias_shift_std) else None,
        "mae_ref": round(mae_ref, 4),
        "mae_recent": round(mae_rec, 4),
        "mae_ratio": round(mae_ratio, 4) if math.isfinite(mae_ratio) else None,
        "std_ref": round(std_ref, 4),
        "std_recent": round(_std(rec), 4),
        "psi": round(psi, 4),
        "ks": round(ks, 4),
        "ks_crit_5pct": round(ks_crit, 4),
        "ks_exceeds_5pct": ks > ks_crit,
        "status": status,
        "reasons": reasons or ["stable: kein Signal ueber Schwelle"],
    }


def coverage_report(lower, upper, actual, *, nominal: float = 0.8, tol: float = 0.1) -> dict:
    """Empirische Ueberdeckung eines Prognoseintervalls [lower, upper] gegen den nominalen Wert.

    Fuer P10/P90 ist nominal=0,8 (jeweils ~0,1 in jedem Schwanz). Meldet Drift, wenn die beobachtete
    Ueberdeckung um mehr als tol vom Nominal abweicht — das deutet auf eine Kalibrierungs-Drift der CQR.
    """
    L, U, A, dropped = [], [], [], 0
    for lo, up, a in zip(list(lower), list(upper), list(actual)):
        lf, uf, af = _finite(lo), _finite(up), _finite(a)
        if None in (lf, uf, af):
            dropped += 1
            continue
        L.append(lf); U.append(uf); A.append(af)
    n = len(A)
    if n == 0:
        raise ValueError("Keine finiten (lower, upper, actual)-Tripel.")
    inside = sum(1 for lo, up, a in zip(L, U, A) if lo <= a <= up)
    below = sum(1 for lo, a in zip(L, A) if a < lo)
    above = sum(1 for up, a in zip(U, A) if a > up)
    cov = inside / n
    status = "drift" if abs(cov - nominal) > tol else "stable"
    return {
        "n": n,
        "n_dropped": dropped,
        "coverage": round(cov, 4),
        "nominal": nominal,
        "coverage_gap": round(cov - nominal, 4),
        "frac_below_lower": round(below / n, 4),
        "frac_above_upper": round(above / n, 4),
        "status": status,
        "note": "Ueberdeckung < nominal => Intervalle zu eng (Modell zu selbstsicher); "
                "> nominal => zu weit. Schwanz-Anteile zeigen die Richtung.",
    }
