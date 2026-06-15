#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify per-Horizont-Bänder (bands="per_horizon") — Kontrakt-, Paritäts- und Plausibilitätsbeweise.

Beweise:
1. T52-KONTRAKT UNVERÄNDERT: Default == bands="k1", k>=2 ohne p10/p90 (Regression-Schutz).
2. BIT-PARITÄT k=1: per_horizon ändert das Produktionsband von D+1 in keinem Bit.
3. k>=2: Band vorhanden, p10<=p50<=p90, scale>=1, Metadaten (n_cal_days>0, conf_c>=0).
4. Ungültiger bands-Wert -> ValueError; Determinismus.
5. horizon_band_backtest: Felder/Soll, Coverage in [0,100], n_days==n_test.
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from netzpilot.horizon import forecast_days, horizon_band_backtest
from netzpilot.models.robust_corrector import ShrunkCorrector

N = [0]
def check(ok, msg):
    N[0] += 1
    print(("ok  " if ok else "FAIL"), f"{N[0]:2d}:", msg)
    if not ok:
        sys.exit(1)

rng = np.random.default_rng(11)
ND, H = 140, 24
base = 20 + 5 * np.sin(np.arange(H) / 24 * 2 * np.pi)
wk = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.8])
load2d = np.array([base * wk[d % 7] * (1 + 0.05 * np.sin(d / 30)) + rng.normal(0, 0.3, H)
                   for d in range(ND)])
days = pd.date_range("2025-06-02", periods=ND, freq="D")
factory = lambda: ShrunkCorrector(10.0)

# 1) T52-Kontrakt: Default == "k1", k>=2 ohne Band
d_def = forecast_days(load2d, days, factory, horizon=3, round_digits=None)
d_k1 = forecast_days(load2d, days, factory, horizon=3, round_digits=None, bands="k1")
check(d_def["days"] == d_k1["days"] and d_def["bands_mode"] == "k1", "Default == bands='k1'")
check(all("p10" not in h for h in d_k1["days"][1]["hours"]), "k1-Modus: k=2 weiter OHNE Band (T52)")

# 2) Bit-Parität k=1 unter per_horizon
d_ph = forecast_days(load2d, days, factory, horizon=3, round_digits=None, bands="per_horizon")
check(d_ph["days"][0]["hours"] == d_k1["days"][0]["hours"],
      "per_horizon: D+1-Band BIT-identisch zum Produktionsband")

# 3) k>=2: Band + Plausibilität
for i, k in ((1, 2), (2, 3)):
    hrs = d_ph["days"][i]["hours"]
    check(all(("p10" in h and "p90" in h) for h in hrs), f"k={k}: P10/P90 vorhanden")
    check(all(h["p10"] <= h["p50"] <= h["p90"] for h in hrs), f"k={k}: Quantil-Ordnung")
    b = d_ph["days"][i]["band"]
    check(b["scale"] >= 1.0 and b["n_cal_days"] > 0 and b["conf_c"] >= 0.0,
          f"k={k}: scale>=1, Kalibriertage>0, c_k>=0 (scale={b['scale']}, n={b['n_cal_days']})")
# Breite k=3 >= Breite k=1 (scale>=1 und c_k>=0 garantieren das pro Stunde NICHT zwingend,
# aber im Mittel muss das skalierte Band mindestens das unskalierte minus c-Differenz sein —
# wir pruefen die harte Garantie: halbe Bandbreite je Stunde >= s*rq-Spanne? Stattdessen ehrlich:
w1 = np.mean([h["p90"] - h["p10"] for h in d_ph["days"][0]["hours"]])
w3 = np.mean([h["p90"] - h["p10"] for h in d_ph["days"][2]["hours"]])
check(w3 >= 0.8 * w1, f"k=3-Band nicht absurd schmaler als k=1 (w1={w1:.3f}, w3={w3:.3f})")

# 4) Validierung + Determinismus
try:
    forecast_days(load2d, days, factory, horizon=2, bands="quatsch")
    check(False, "bands='quatsch' haette ValueError geben muessen")
except ValueError:
    check(True, "ungueltiger bands-Wert -> ValueError")
d_ph2 = forecast_days(load2d, days, factory, horizon=3, round_digits=None, bands="per_horizon")
check(d_ph2 == d_ph, "per_horizon deterministisch")

# 5) Band-Backtest
bt = horizon_band_backtest(load2d, days, factory, horizon=3, n_test=14)
check(bt["soll_coverage_pct"] == 80.0 and set(bt["per_horizon"]) == {1, 2, 3}, "Backtest-Felder/Soll")
for k, v in bt["per_horizon"].items():
    check(0.0 <= v["coverage_pct"] <= 100.0 and v["n_days"] == 14 and v["mean_width_mw"] > 0,
          f"Backtest k={k}: Coverage {v['coverage_pct']}%, width {v['mean_width_mw']}, n=14")
check(bt["per_horizon"][2]["mean_scale"] >= 1.0 and bt["per_horizon"][3]["mean_scale"] >= 1.0,
      "Backtest: mittlere Breitenfaktoren >= 1")

print(f"ALLE {N[0]} CHECKS GRUEN — per-Horizont-Bänder verifiziert.")
