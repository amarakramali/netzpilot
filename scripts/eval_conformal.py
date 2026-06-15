# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Vergleich der Unsicherheits-Intervalle (T6): v1-Residuenquantile vs. split/online-conformal,
pro-Stunde vs. gepoolt, auf echten 12-Wochen-Daten. Ehrliche Coverage/Breite."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from netzpilot.data.smard import load_local_json
from netzpilot.features.build import to_daily, get_holidays
from netzpilot.eval.backtest import rolling_origin
from netzpilot.eval.conformal import rolling_origin_conformal
from netzpilot.models.ridge_correction import RidgeCorrector

s = load_local_json("prognose_engine_v1/data/wk*.json")
load2d, days = to_daily(s)
hol = get_holidays(sorted({d.year for d in days}), "NW")
fac = lambda: RidgeCorrector(lam=10.0)

R1, sm1 = rolling_origin(load2d, days, fac, holiday_set=hol)
v1 = {"coverage_%": sm1["probabilistisch"]["Coverage_P10_P90_%"],
      "mean_width_MW": round(float(np.mean(R1["p90"] - R1["p10"])), 1), "nominal_%": 80.0}
out = {"v1_static_insample_perhour_80": v1}
print(f"v1 in-sample/per-hour      80%: cov {v1['coverage_%']:5.1f}%  width {v1['mean_width_MW']:7.1f} MW")
grid = []
for alpha, lab in [(0.2, "80"), (0.1, "90")]:
    for per_hour in (False, True):
        for online in (False, True):
            _, sm = rolling_origin_conformal(load2d, days, fac, alpha=alpha, cal_days=28,
                                             holiday_set=hol, online=online, per_hour=per_hour)
            key = f"conformal_{lab}_{'perhour' if per_hour else 'pooled'}_{'online' if online else 'split'}"
            out[key] = sm
            print(f"{key:38s}: cov {sm['coverage_%']:5.1f}%  width {sm['mean_width_MW']:7.1f} MW  (Soll {sm['nominal_%']}%)")
os.makedirs("data_cache", exist_ok=True)
json.dump(out, open("data_cache/conformal_eval.json", "w"), indent=2, ensure_ascii=False)
