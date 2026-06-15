"""Vergleicht Ridge vs. ShrunkCorrector auf National- und Klein-Proxy-Last (Skill vs. saisonal-naiv)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from netzpilot.data.smard import load_local_json
from netzpilot.data.synthetic_smallutility import make_small_utility_load
from netzpilot.features.build import to_daily, get_holidays
from netzpilot.eval.backtest import rolling_origin
from netzpilot.models.ridge_correction import RidgeCorrector
from netzpilot.models.robust_corrector import ShrunkCorrector

nat = load_local_json("prognose_engine_v1/data/wk*.json")
def skill(series, factory):
    load2d, days = to_daily(series)
    hol = get_holidays(sorted({d.year for d in days}), "NW")
    _, sm = rolling_origin(load2d, days, factory, holiday_set=hol)
    return sm["metriken"]["model"]["Skill_vs_SaisonalNaiv_%"], sm["metriken"]["model"]["MAPE_%"]

cases = [("National", nat)] + [(f"Klein-Proxy s{i}", make_small_utility_load(nat, 25.0, seed=i)) for i in (0,1,2)]
print(f"{'Datensatz':22s} {'Ridge Skill':>12s} {'Shrunk Skill':>13s}   (vs saisonal-naiv)")
out=[]
for label, s in cases:
    rs,_ = skill(s, lambda: RidgeCorrector(10.0))
    ss,_ = skill(s, lambda: ShrunkCorrector(10.0))
    out.append({"case":label,"ridge_skill":rs,"shrunk_skill":ss})
    print(f"{label:22s} {rs:>11.1f}% {ss:>12.1f}%")
os.makedirs("data_cache", exist_ok=True); json.dump(out, open("data_cache/robust_eval.json","w"), indent=2)
