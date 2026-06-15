#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify Intraday-Update (netzpilot/intraday.py). Exit!=0 bei Fehler."""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from netzpilot.intraday import intraday_update, _weights

N = [0]
def _safe(s):
    return str(s).encode("ascii", "replace").decode("ascii")

def check(ok, msg):
    N[0] += 1
    print(("ok  " if ok else "FAIL"), f"{N[0]:2d}:", _safe(msg))
    if not ok:
        sys.exit(1)

HOURS = [{"hour": i, "p50": 20.0 + i, "p10": 18.0 + i, "p90": 23.0 + i} for i in range(24)]

# 1) Handrechnung: h=2, half_life=3 -> w = [0.5^(1/3), 1]/norm; shrink=0.5
a = [21.0, 22.5]                                   # resid = [1.0, 1.5]
w0 = 0.5 ** (1 / 3.0); w = np.array([w0, 1.0]); w = w / w.sum()
delta_exp = 0.5 * (w[0] * 1.0 + w[1] * 1.5)
r = intraday_update(HOURS, a, round_digits=None)
check(r["applied"] and r["update_hour"] == 2 and r["n_hours_used"] == 2, "applied, h=2, 2 Stunden genutzt")
check(abs(r["delta_mw"] - round(delta_exp, 4)) < 5e-5, f"δ exakt per Hand ({delta_exp:.4f})")
check(len(r["hours_rest"]) == 22 and r["hours_rest"][0]["hour"] == 2, "Resttag = Stunden 2..23")
check(abs(r["hours_rest"][0]["p50"] - (22.0 + delta_exp)) < 1e-9, "P50 verschoben")
check(abs((r["hours_rest"][0]["p90"] - r["hours_rest"][0]["p10"]) - 5.0) < 1e-9,
      "Bandbreite UNVERÄNDERT (gemeinsamer Shift)")
check(all(h["p10"] <= h["p50"] <= h["p90"] for h in r["hours_rest"]), "Quantil-Ordnung erhalten")

# 2) Eingaben nicht mutiert
check(HOURS[2]["p50"] == 22.0 and "p10" in HOURS[2], "Original-hours unverändert")

# 3) No-ops: h=0, Tag voll, zu wenig valide
check(intraday_update(HOURS, [])["applied"] is False, "h=0 → No-op")
check(intraday_update(HOURS, [20.0] * 24)["applied"] is False, "h=24 → No-op (Tag vorbei)")
r_nan = intraday_update(HOURS, [float("nan"), float("nan"), 22.6])
check(r_nan["applied"] is False and "valide" in r_nan["reason"], "nur 1 valide < min_hours=2 → No-op")

# 4) NaN-Robustheit: Lücke wird ignoriert, Gewichte renormiert
r2 = intraday_update(HOURS, [21.0, float("nan"), 23.0, 23.5], round_digits=None)
check(r2["applied"] and r2["n_hours_used"] == 3, "NaN-Stunde ignoriert (3 von 4 genutzt)")
wfull = _weights(4, 3.0); m = np.array([1, 0, 1, 1], float)
wre = wfull * m / (wfull * m).sum()
delta2 = 0.5 * float(np.sum(wre * np.array([1.0, 0.0, 1.0, 0.5])))
check(abs(r2["delta_mw"] - round(delta2, 4)) < 5e-5, "δ mit renormierten Gewichten exakt")

# 5) Nur-P50-Stunden (Horizont-k>=2-Format) funktionieren ohne Bandfelder
H50 = [{"hour": i, "p50": 30.0} for i in range(24)]
r3 = intraday_update(H50, [31.0, 31.0])
check(r3["applied"] and "p10" not in r3["hours_rest"][0], "nur-P50-Input → nur-P50-Output")

# 6) Validierung + Determinismus + mean-Variante
try:
    intraday_update(HOURS, a, shrink=1.5); check(False, "shrink>1 nicht abgelehnt")
except ValueError:
    check(True, "shrink>1 → ValueError")
check(intraday_update(HOURS, a) == intraday_update(HOURS, a), "deterministisch")
wm = _weights(5, 0.0)
check(np.allclose(wm, 0.2), "half_life<=0 → Gleichgewichtung (mean-Variante)")

# 7) Messkonsistenz: δ-Formel identisch zur measure_intraday-Auswertung (ewm3, s=0.5)
rng = np.random.default_rng(3)
p50 = 20 + rng.normal(0, 1, 24); act = p50 + rng.normal(0, 0.5, 24)
hrs = [{"hour": i, "p50": float(p50[i])} for i in range(24)]
h = 12
wm12 = _weights(12, 3.0)
delta_ref = 0.5 * float(np.sum(wm12 * (act[:12] - p50[:12])))
rr = intraday_update(hrs, act[:12], round_digits=None)
check(abs(rr["delta_mw"] - round(delta_ref, 4)) < 5e-5, "δ identisch zur Mess-Skript-Formel (h=12)")

print(_safe(f"ALLE {N[0]} CHECKS GRUEN — Intraday-Update verifiziert."))
