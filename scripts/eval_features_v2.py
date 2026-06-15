"""A/B-Test zusaetzlicher Features (selbst-enthalten; aendert NICHT die geteilte build.py).
Ergebnis ist ein Vorschlag fuer Codex, falls v2 hilft. Modell: ShrunkCorrector, Backtest 28 Tage."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from netzpilot.data.smard import load_local_json
from netzpilot.data.synthetic_smallutility import make_small_utility_load
from netzpilot.features.build import to_daily, get_holidays, base, resid_target, build_features as bf1
from netzpilot.models.robust_corrector import ShrunkCorrector
from netzpilot.models.baselines import seasonal_naive
from netzpilot.eval import metrics as M

def bf2(load2d, days, d, weather2d=None, holiday_set=None):
    rowb = bf1(load2d, days, d, weather2d, holiday_set)          # [24, F]
    dow = days[d].dayofweek; wknd = 1.0 if dow >= 5 else 0.0
    extra = []
    for h in range(24):
        lag2 = load2d[d - 2, h] - load2d[d - 9, h]              # vorgestern vs. Vorwoche
        roll3 = float(np.mean([load2d[d - 1, h], load2d[d - 2, h], load2d[d - 3, h]])) - load2d[d - 7, h]
        eve = np.exp(-((h - 19) ** 2) / 8.0)
        extra.append([lag2, roll3, wknd * eve])
    return np.hstack([rowb, np.array(extra)])

def run(series, feat_fn, first=9, n_test=28):
    load2d, days = to_daily(series); hol = get_holidays(sorted({d.year for d in days}), "NW")
    ND = len(load2d)
    preds, act, sn = [], [], []
    for d in range(ND - n_test, ND):
        Xtr = np.vstack([feat_fn(load2d, days, t, None, hol) for t in range(first, d)])
        ytr = np.concatenate([resid_target(load2d, t) for t in range(first, d)])
        m = ShrunkCorrector(10.0).fit(Xtr, ytr)
        preds.append(base(load2d, d) + m.predict(feat_fn(load2d, days, d, None, hol)))
        act.append(load2d[d]); sn.append(seasonal_naive(load2d, d))
    a = np.concatenate(act); p = np.concatenate(preds); s = np.concatenate(sn)
    return M.mae(p, a), (1 - M.mae(p, a) / M.mae(s, a)) * 100

nat = load_local_json("prognose_engine_v1/data/wk*.json")
print(f"{'Datensatz':12s} {'v1 MAE':>9s} {'v1 skill':>9s} | {'v2 MAE':>9s} {'v2 skill':>9s}")
for label, series in [("National", nat)] + [(f"Proxy s{i}", make_small_utility_load(nat, 25.0, seed=i)) for i in (0, 1, 2)]:
    m1, s1 = run(series, bf1); m2, s2 = run(series, bf2)
    flag = " <- v2 besser" if m2 < m1 else ""
    print(f"{label:12s} {m1:9.1f} {s1:+8.1f}% | {m2:9.1f} {s2:+8.1f}%{flag}")
