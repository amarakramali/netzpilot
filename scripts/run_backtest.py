# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Einstiegspunkt: leakage-sicheres Rolling-Origin-Backtest auf SMARD-Last.
Beispiel (v1 reproduzieren, nur numpy/pandas):
  python scripts/run_backtest.py --data "prognose_engine_v1/data/wk*.json"
"""
import argparse, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from netzpilot.data.smard import load_local_json
from netzpilot.features.build import to_daily, get_holidays
from netzpilot.eval.backtest import rolling_origin
from netzpilot.report.report import write_report
from netzpilot.models.ridge_correction import RidgeCorrector

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prognose_engine_v1/data/wk*.json",
                    help="Glob auf SMARD-Wochendateien (oder spaeter: aus smard.fetch_series)")
    ap.add_argument("--model", choices=["ridge", "lgbm"], default="ridge")
    ap.add_argument("--region", default="NW", help="Bundesland fuer Feiertage")
    ap.add_argument("--n-test", type=int, default=28)
    ap.add_argument("--out", default="data_cache")
    a = ap.parse_args()

    series = load_local_json(a.data)
    load2d, days = to_daily(series)
    hol = get_holidays(sorted({d.year for d in days}), a.region)

    if a.model == "ridge":
        factory = lambda: RidgeCorrector(lam=10.0)
    else:
        from netzpilot.models.lgbm_quantile import LGBMQuantileCorrector
        class _Adapter:  # Median-Adapter fuer den Punkt+Residuenquantil-Backtest
            def fit(self, X, y): self.m = LGBMQuantileCorrector(alphas=(0.5,)).fit(X, y); return self
            def predict(self, X): return self.m.predict(X)[0.5]
        factory = _Adapter

    R, summary = rolling_origin(load2d, days, factory, n_test=a.n_test, holiday_set=hol)
    os.makedirs(a.out, exist_ok=True)
    write_report(summary, os.path.join(a.out, "report.md"), os.path.join(a.out, "results.json"))
    np.savez(os.path.join(a.out, "arrays.npz"), **R)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
