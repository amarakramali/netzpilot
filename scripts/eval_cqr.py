# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verifiziert CQR-Kalibrierung vs. unkonforme Baender auf echten 12-Wochen-Daten (Ridge)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from netzpilot.data.smard import load_local_json
from netzpilot.features.build import to_daily, get_holidays
from netzpilot.eval.conformal import rolling_origin_conformal, rolling_origin_cqr
from netzpilot.models.ridge_correction import RidgeCorrector

s = load_local_json("prognose_engine_v1/data/wk*.json")
load2d, days = to_daily(s)
hol = get_holidays(sorted({d.year for d in days}), "NW")
fac = lambda: RidgeCorrector(lam=10.0)
out = {}
print(f"{'Methode':30s} {'Soll':>5s} {'Coverage':>9s} {'Breite(MW)':>11s}")
for alpha, lab in [(0.2, "80%"), (0.1, "90%")]:
    _, b = rolling_origin_conformal(load2d, days, fac, alpha=alpha, cal_days=28, holiday_set=hol, online=True, per_hour=False)
    _, q = rolling_origin_cqr(load2d, days, fac, alpha=alpha, cal_days=28, holiday_set=hol, online=True, per_hour=False)
    out[f"conformal_{lab}"] = b; out[f"cqr_{lab}"] = q
    print(f"{'conformal (unkalibriert) '+lab:30s} {b['nominal_%']:>5} {b['coverage_%']:>8}% {b['mean_width_MW']:>11}")
    print(f"{'CQR (kalibriert) '+lab:30s} {q['nominal_%']:>5} {q['coverage_%']:>8}% {q['mean_width_MW']:>11}")
os.makedirs("data_cache", exist_ok=True)
json.dump(out, open("data_cache/cqr_eval.json", "w"), indent=2, ensure_ascii=False)
