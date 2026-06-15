# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Intraday-Update: Resttag-Prognose aus den heute bereits beobachteten Stunden nachschärfen.

Gemessen (scripts/measure_intraday.py, 84 Testtage, 3 echte Reihen, 2026-06-05): ein Level-Shift
aus den EWM-gewichteten heutigen Residuen reduziert den Resttag-MAE im Mittel um ~13 % (schwächste
Reihe +8,8 %; mittags +9–17 %, abends bis +28 %; h=3 klein positiv). Robusteste Variante und
deshalb Default: Gewichtung ewm (Halbwertszeit 3 h), shrink 0,5. Einzelne (kind, s, h)-Zellen
können negativ sein (EVDB h=9: −0,4 %) — dokumentiert, kein Verschweigen.

Mechanik (T50-Muster, eine Ebene tiefer):
    δ = shrink · Σ w_i · (ist_i − p50_i),  i = beobachtete Stunden (jüngste am stärksten gewichtet)
    Resttag: P10/P50/P90 werden GEMEINSAM um δ verschoben — Bandmitte korrigiert, Bandbreite
    unverändert (die Kalibrierung der Breite bleibt Sache der Band-Mechanik).

Leakage-frei per Konstruktion: ausschließlich Stunden < Update-Stunde DESSELBEN Tages.
Meter-Lücken: NaN-Stunden werden aus der Gewichtung entfernt; zu wenig valide Stunden → No-op.
Reine Python/numpy, additiv — Engine/Runner bleiben unberührt, bis T54 verdrahtet.
"""
from __future__ import annotations

import numpy as np

DEFAULT_SHRINK = 0.5
DEFAULT_HALF_LIFE_H = 3.0
DEFAULT_MIN_HOURS = 2


def _weights(n: int, half_life_h: float) -> np.ndarray:
    if half_life_h <= 0:
        return np.full(n, 1.0 / n)
    w = 0.5 ** ((n - 1 - np.arange(n)) / float(half_life_h))
    return w / w.sum()


def intraday_update(hours, actual_today, *, shrink: float = DEFAULT_SHRINK,
                    half_life_h: float = DEFAULT_HALF_LIFE_H,
                    min_hours: int = DEFAULT_MIN_HOURS,
                    round_digits: int | None = 1) -> dict:
    """Resttag-Update der Day-ahead-Prognose.

    hours: Stundenliste im Service-Format [{"hour", "p50"[, "p10", "p90"]}], aufsteigend.
    actual_today: beobachtete Ist-Werte (MW) der Stunden 0..h-1 (len = Update-Stunde h);
                  NaN = Messlücke (wird ignoriert).
    Rückgabe: {"applied", "update_hour", "n_hours_used", "delta_mw", "hours_rest", "method",
               "caveat", ggf. "reason"}. Eingaben werden NICHT mutiert.
    """
    if not 0.0 <= shrink <= 1.0:
        raise ValueError("shrink muss in [0,1] liegen.")
    if min_hours < 1:
        raise ValueError("min_hours muss >= 1 sein.")
    H = len(hours)
    actual = np.asarray(list(actual_today), dtype=float)
    h = int(len(actual))
    base = {"method": f"Level-Shift δ = {shrink}·EWM(ist−P50, Halbwertszeit {half_life_h} h) "
                      "auf P10/P50/P90 gemeinsam; Bandbreite unverändert",
            "caveat": ("Gemessen auf 3 echten Reihen (84 Tage): Resttag-MAE im Mittel ~13 % besser; "
                       "nicht jeder Tag gewinnt (54–69 % der Tage). Nur Stunden vor der "
                       "Update-Stunde desselben Tages — leakage-frei.")}

    def _noop(reason: str) -> dict:
        return {**base, "applied": False, "update_hour": h, "n_hours_used": 0,
                "delta_mw": 0.0, "hours_rest": [], "reason": reason}

    if h == 0:
        return _noop("keine beobachteten Stunden — Day-ahead bleibt unverändert")
    if h >= H:
        return _noop(f"Tag vollständig beobachtet (h={h} >= {H}) — nichts mehr zu prognostizieren")

    p50_obs = np.array([float(hours[i]["p50"]) for i in range(h)])
    resid = actual - p50_obs
    valid = np.isfinite(resid)
    if int(valid.sum()) < min_hours:
        return _noop(f"zu wenig valide Ist-Stunden ({int(valid.sum())} < {min_hours}) — No-op")

    w = _weights(h, half_life_h)
    w = np.where(valid, w, 0.0)
    w = w / w.sum()
    delta = float(shrink * np.sum(w * np.where(valid, resid, 0.0)))

    def _fmt(x):
        x = float(x)
        return x if round_digits is None else round(x, round_digits)

    rest = []
    for i in range(h, H):
        src = hours[i]
        upd = {"hour": int(src["hour"]), "p50": _fmt(float(src["p50"]) + delta)}
        if "p10" in src and "p90" in src:
            upd["p10"] = _fmt(float(src["p10"]) + delta)
            upd["p90"] = _fmt(float(src["p90"]) + delta)
        rest.append(upd)

    return {**base, "applied": True, "update_hour": h, "n_hours_used": int(valid.sum()),
            "delta_mw": round(delta, 4), "hours_rest": rest}
