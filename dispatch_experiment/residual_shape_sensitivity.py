# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""
NetzPilot - Dispatch experiment, sensitivity study: what actually drives the saving?

Follow-up to dispatch_experiment.py. The single-period imbalance saving reduces to a
property of the forecast-ERROR distribution e = actual - point_forecast:
    cost(delta) = E_e[ c_short*max(e-delta,0) + c_long*max(delta-e,0) ]   (pinball)
minimized at delta* = tau-quantile(e), tau = c_short/(c_short+c_long); the point-forecast
policy uses delta = median(e). So saving% depends ONLY on (a) the cost asymmetry and
(b) the SHAPE of e - NOT on how large the errors are.

This script checks that on the real out-of-sample errors:
  - scaling the error spread x2 leaves saving% unchanged (magnitude does not matter),
  - a Gaussian error of the same std gives a different saving% (shape does matter).

Conclusion for the product: "a noisier Stadtwerk saves more" is only true if its errors
are more SKEWED / heavy-tailed, not merely larger. Confirm the shape on real customer data.

Run:
    python residual_shape_sensitivity.py
Pure numpy/pandas; reuses dispatch_experiment.py.
"""
import json
import os

import numpy as np
import pandas as pd

from dispatch_experiment import DEFAULT_DATA, load_daily, _feats, _fit, _pred

HERE = os.path.dirname(os.path.abspath(__file__))


def realized_errors(data_glob, n_test=28, lam=10.0):
    """Leakage-safe out-of-sample forecast errors e = actual - point over the test set."""
    load2d, days = load_daily(data_glob)
    nd = len(load2d)
    holid = {pd.Timestamp("2024-01-01").date()}
    first = 8
    errs = []
    for d in range(nd - n_test, nd):
        Xtr = np.vstack([_feats(load2d, days, t, holid) for t in range(first, d)])
        ytr = np.concatenate([load2d[t] - load2d[t - 7] for t in range(first, d)])
        m = _fit(Xtr, ytr, lam)
        point = load2d[d - 7] + _pred(m, _feats(load2d, days, d, holid))
        errs.append(load2d[d] - point)
    return np.concatenate(errs)


def expected_pinball(sample, delta, c_short, c_long):
    return float(np.mean(c_short * np.maximum(sample - delta, 0.0)
                         + c_long * np.maximum(delta - sample, 0.0)))


def saving_pct(sample, ratio, price=1.0):
    """Saving from nominating the cost-optimal quantile vs the median, over error dist `sample`."""
    c_long = price
    c_short = ratio * price
    tau = c_short / (c_short + c_long)
    cost_med = expected_pinball(sample, np.quantile(sample, 0.5), c_short, c_long)
    cost_tau = expected_pinball(sample, np.quantile(sample, tau), c_short, c_long)
    return round(100 * (cost_med - cost_tau) / cost_med, 2) if cost_med else 0.0


def moments(x):
    mu = x.mean()
    sd = x.std()
    z = (x - mu) / sd
    return {"std_MW": round(float(sd), 1),
            "skew": round(float((z ** 3).mean()), 3),
            "excess_kurtosis": round(float((z ** 4).mean() - 3), 3)}


def main():
    rng = np.random.default_rng(0)
    e = realized_errors(DEFAULT_DATA)
    variants = {
        "real_errors": e,
        "real_errors_x2_spread": 2.0 * e,
        "gaussian_same_std": rng.normal(e.mean(), e.std(), 200_000),
    }
    ratios = [1.5, 2.0, 3.0, 5.0]
    table = {name: {f"ratio_{r}": saving_pct(s, r) for r in ratios}
             for name, s in variants.items()}
    summary = {
        "study": "what drives the imbalance saving: error spread vs error shape",
        "real_error_moments": moments(e),
        "saving_pct_by_variant_and_asymmetry": table,
        "reading": ("real vs real_x2: identical -> saving% is INVARIANT to error magnitude. "
                    "real vs gaussian_same_std: differ -> saving% is driven by error SHAPE "
                    "(skew/tails) and the cost asymmetry, not by how noisy the portfolio is."),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(os.path.join(HERE, "sensitivity_results.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
