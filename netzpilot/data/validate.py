# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Messdaten-Plausibilisierung & Ersatzwertbildung für Lastgänge.

Prüft eingehende Messreihen auf Plausibilität (Lücken, Ausreißer, eingefrorene Zähler,
Vorzeichenfehler) und bildet Ersatzwerte — Standard im Energiedatenmanagement (EDM). Genau das schützt
JEDE nachgelagerte Prognose/Abrechnung — „garbage in, garbage out" ist der häufigste reale Fehler bei
Stadtwerke-Lastdaten. NetzPilot bringt das mit, bevor Daten in die Engine gehen.

Prüfungen (robust, transparent):
  - LÜCKEN: None / nicht-finite Werte.
  - AUSREISSER: |v − Median| > k · 1.4826 · MAD (robuste Z-Score-Schwelle; MAD≈0 -> übersprungen).
  - EINGEFROREN: Läufe identischer aufeinanderfolgender Werte der Länge >= frozen_run (Zähler steht).
  - NEGATIV / AUSSER REICHWEITE: < 0 (wenn nicht erlaubt) bzw. > max_plausible.

Ersatzwertbildung (nur für eindeutig fehlerhafte Werte — Lücke/Ausreißer/negativ/außer Reichweite;
EINGEFROREN wird nur GEMELDET, nicht automatisch ersetzt, da flache Phasen legitim sein können):
  1. SAISONAL: gleicher Slot am Nachbar-Tag (i ± period, i ± 2·period), erster saubere Wert.
  2. KURZE LÜCKE: lineare Interpolation zwischen sauberen Nachbarn, wenn Lückenlänge <= gap_interp_max.
  3. LOKAL: Median der nächsten sauberen Werte im Fenster.
  Methode wird je Ersatzwert protokolliert. Ist kein Bezug verfügbar -> als nicht-ersetzbar markiert.

Reine stdlib. Additiv (netzpilot/data/validate.py).
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


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def validate_load(values, period_per_day: int = 24, *, mad_k: float = 6.0,
                  frozen_run: int = 6, allow_negative: bool = False,
                  max_plausible=None, gap_interp_max: int = 2) -> dict:
    """Plausibilisiere einen Lastgang und bilde Ersatzwerte.

    values: Liste der Lastwerte je Periode (darf None / nicht-finite enthalten).
    Rückgabe dict:
      n, n_missing, n_outlier, n_frozen, n_negative, n_out_of_range, n_replaced, n_unreplaceable,
      quality_score (Anteil ursprünglich sauberer Werte), cleaned (Liste mit Ersatzwerten; None wo
      nicht ersetzbar), issues (Liste {index, type}), replacements (Liste {index, method, value}).
    """
    n = len(values)
    if n == 0:
        raise ValueError("Leere Reihe.")
    fin = [_finite(v) for v in values]
    finite_vals = [v for v in fin if v is not None]
    if not finite_vals:
        raise ValueError("Keine finiten Werte — Reihe nicht plausibilisierbar.")

    med = _median(finite_vals)
    mad = _median([abs(v - med) for v in finite_vals])
    sigma = 1.4826 * mad                                   # robuste Std-Schätzung
    issues = []
    bad = [False] * n          # eindeutig fehlerhaft -> Ersatzwert
    # --- Punktprüfungen ---
    for i, v in enumerate(fin):
        if v is None:
            issues.append({"index": i, "type": "missing"}); bad[i] = True; continue
        if not allow_negative and v < 0:
            issues.append({"index": i, "type": "negative"}); bad[i] = True
        if max_plausible is not None and v > float(max_plausible):
            issues.append({"index": i, "type": "out_of_range"}); bad[i] = True
        if sigma > 1e-12 and abs(v - med) > mad_k * sigma:
            issues.append({"index": i, "type": "outlier"}); bad[i] = True
    # --- Eingefrorene Läufe (nur melden) ---
    n_frozen = 0
    run_start = 0
    for i in range(1, n + 1):
        same = i < n and fin[i] is not None and fin[i - 1] is not None and fin[i] == fin[i - 1]
        if not same:
            run_len = i - run_start
            if run_len >= frozen_run and fin[run_start] is not None:
                for j in range(run_start, i):
                    issues.append({"index": j, "type": "frozen"})
                n_frozen += run_len
            run_start = i

    good = [fin[i] is not None and not bad[i] for i in range(n)]
    cleaned = [fin[i] for i in range(n)]
    replacements = []
    n_unreplaceable = 0

    def seasonal(i):
        for off in (period_per_day, 2 * period_per_day, 3 * period_per_day):
            for j in (i - off, i + off):
                if 0 <= j < n and good[j]:
                    return fin[j]
        return None

    def interp(i):
        # nächster sauberer Nachbar links/rechts; nur wenn die Lücke kurz ist
        lo = i - 1
        while lo >= 0 and not good[lo]:
            lo -= 1
        hi = i + 1
        while hi < n and not good[hi]:
            hi += 1
        if lo < 0 or hi >= n:
            return None
        if (hi - lo - 1) > gap_interp_max:
            return None
        frac = (i - lo) / (hi - lo)
        return fin[lo] + (fin[hi] - fin[lo]) * frac

    def local_median(i, window=None):
        w = window or period_per_day
        near = [fin[j] for j in range(max(0, i - w), min(n, i + w + 1)) if good[j]]
        return _median(near) if near else None

    for i in range(n):
        if not bad[i]:
            continue
        val, method = seasonal(i), "seasonal_neighbor_day"
        if val is None:
            val, method = interp(i), "linear_interpolation"
        if val is None:
            val, method = local_median(i), "local_median"
        if val is None:
            cleaned[i] = None
            n_unreplaceable += 1
        else:
            cleaned[i] = round(float(val), 4)
            replacements.append({"index": i, "method": method, "value": cleaned[i]})

    n_missing = sum(1 for it in issues if it["type"] == "missing")
    n_outlier = sum(1 for it in issues if it["type"] == "outlier")
    n_negative = sum(1 for it in issues if it["type"] == "negative")
    n_oor = sum(1 for it in issues if it["type"] == "out_of_range")
    n_clean = sum(1 for i in range(n) if good[i])
    return {
        "n": n,
        "n_missing": n_missing,
        "n_outlier": n_outlier,
        "n_frozen": n_frozen,
        "n_negative": n_negative,
        "n_out_of_range": n_oor,
        "n_replaced": len(replacements),
        "n_unreplaceable": n_unreplaceable,
        "quality_score": round(n_clean / n, 4),
        "cleaned": cleaned,
        "issues": issues,
        "replacements": replacements,
        "note": "Plausibilisierung + Ersatzwertbildung; EINGEFROREN wird nur gemeldet (flache Phasen "
                "können legitim sein), nicht automatisch ersetzt. Robuste MAD-Schwelle.",
    }
