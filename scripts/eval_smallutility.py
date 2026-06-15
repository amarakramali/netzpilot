# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""T9 (Proxy): Forecaster + CQR auf repraesentativer KLEIN-Stadtwerk-Last vs. nationaler Last.
Echte OPSD/BDEW-Validierung = Codex/Host (T9). Hier: verifizierbarer numpy-Test des Verhaltens bei kleiner Last."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from netzpilot.data.smard import load_local_json
from netzpilot.data.synthetic_smallutility import make_small_utility_load
from netzpilot.features.build import to_daily, get_holidays
from netzpilot.eval.backtest import rolling_origin
from netzpilot.eval.conformal import rolling_origin_cqr
from netzpilot.models.ridge_correction import RidgeCorrector

nat = load_local_json("prognose_engine_v1/data/wk*.json")
fac = lambda: RidgeCorrector(lam=10.0)

def run(series, label):
    load2d, days = to_daily(series)
    hol = get_holidays(sorted({d.year for d in days}), "NW")
    _, sm = rolling_origin(load2d, days, fac, holiday_set=hol)
    _, cq80 = rolling_origin_cqr(load2d, days, fac, alpha=0.2, cal_days=28, holiday_set=hol)
    m = sm["metriken"]["model"]
    print(f"{label:34s} MAE {m['MAE_MW']:8.1f}  MAPE {m['MAPE_%']:5.2f}%  "
          f"Skill vs snaive {m['Skill_vs_SaisonalNaiv_%']:+5.1f}%  "
          f"Cov(v1) {sm['probabilistisch']['Coverage_P10_P90_%']:.1f}%  Cov(CQR80) {cq80['coverage_%']:.1f}%")
    return {"label": label, **m, "coverage_v1_%": sm["probabilistisch"]["Coverage_P10_P90_%"],
            "coverage_cqr80_%": cq80["coverage_%"], "mittlere_last_MW": round(float(series.mean()), 1)}

print(f"{'Datensatz':34s} {'':8s}       {'':5s}        {'':5s}")
out = [run(nat, "National (Referenz, ~57 GW)")]
for seed in (0, 1, 2):
    sml = make_small_utility_load(nat, peak_mw=25.0, seed=seed)
    out.append(run(sml, f"Klein-Stadtwerk-Proxy (25 MW) s{seed}"))
os.makedirs("data_cache", exist_ok=True)
json.dump(out, open("data_cache/smallutility_eval.json", "w"), indent=2, ensure_ascii=False)
print("\nHinweis: Proxy (kein Ersatz fuer echte OPSD/BDEW-Daten -> Codex T9). Erwartung: hoehere MAPE bei kleiner Last.")
