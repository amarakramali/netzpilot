# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""
NetzPilot - Dispatch experiment (proof, not product).

Question: does using the forecast UNCERTAINTY distribution beat using the point
forecast (P50) in euros, under asymmetric imbalance (reBAP) pricing?

Thesis (newsvendor / pinball): when under- and over-delivery are priced
asymmetrically, the cost-optimal day-ahead balancing-group nomination is the
tau-quantile of the predictive distribution, tau = c_short / (c_short + c_long),
NOT the median. This script measures the REALIZED euro saving on real SMARD load,
leakage-safe (rolling-origin; each day's residual quantiles use only earlier data).

Deliberately minimal (Simplicity First): single-period nomination, no battery,
no solver. The product step - a multi-period battery/DER dispatch MILP with SOC
coupling and the 4.2 kW section-14a dimming floor (linopy + HiGHS) - is specified
in README.md and intentionally NOT built here. Prove the principle first.

Run:
    python dispatch_experiment.py        # uses bundled v1 SMARD data
Pure numpy/pandas.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "prognose_engine_v1", "data", "wk*.json")


def load_daily(data_glob):
    """Load bundled SMARD weekly files (list of [ms, MW]) into a days x 24 array."""
    pairs = []
    for f in sorted(glob.glob(data_glob)):
        with open(f) as fh:
            pairs += json.load(fh)
    pairs.sort()
    ts = np.array([p[0] for p in pairs], dtype=np.int64)
    load = np.array([p[1] for p in pairs], float)
    assert (np.diff(ts) == 3600000).all(), "expected gap-free hourly data"
    idx = pd.to_datetime(ts, unit="ms", utc=True).tz_convert("Europe/Berlin")
    nd = len(load) // 24
    days = pd.to_datetime([idx[d * 24].date() for d in range(nd)])
    return load[: nd * 24].reshape(nd, 24), days


# --- v1 forecasting core (seasonal-naive + ridge correction), reused ---
def _feats(load2d, days, d, holid):
    dev_prev = load2d[d - 1] - load2d[d - 8]
    dev_mean = dev_prev.mean()
    trend = load2d[d - 1].mean() - load2d[d - 8].mean()
    dow = days[d].dayofweek
    wknd = 1.0 if dow >= 5 else 0.0
    h_ = 1.0 if days[d].date() in holid else 0.0
    rows = []
    for h in range(24):
        rows.append([1.0, dev_prev[h], dev_mean, trend,
                     load2d[d - 1, h] - load2d[d - 7, h],
                     np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(4 * np.pi * h / 24), np.cos(4 * np.pi * h / 24),
                     np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), wknd, h_])
    return np.array(rows)


def _fit(X, y, lam):
    mu = X[:, 1:].mean(0)
    sd = X[:, 1:].std(0)
    sd[sd == 0] = 1
    Xs = np.hstack([np.ones((len(X), 1)), (X[:, 1:] - mu) / sd])
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    A[0, 0] -= lam  # do not penalize the intercept
    return np.linalg.solve(A, Xs.T @ y), mu, sd


def _pred(m, X):
    w, mu, sd = m
    return np.hstack([np.ones((len(X), 1)), (X[:, 1:] - mu) / sd]) @ w


def imbalance_cost(actual, q, c_short, c_long):
    """c_short paid on under-nomination (actual>q), c_long on over (actual<q). MW*1h = MWh."""
    return c_short * np.maximum(actual - q, 0.0) + c_long * np.maximum(q - actual, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--n-test", type=int, default=28)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--price", type=float, default=15.0,
                    help="base imbalance price c_long [EUR/MWh] (illustrative reBAP spread)")
    ap.add_argument("--ratios", default="1,1.5,2,3,5",
                    help="asymmetry ratios c_short/c_long to sweep")
    a = ap.parse_args()

    load2d, days = load_daily(a.data)
    nd = len(load2d)
    holid = {pd.Timestamp("2024-01-01").date()}
    first = 8
    test = list(range(nd - a.n_test, nd))
    ratios = [float(x) for x in a.ratios.split(",")]

    # Per test hour: realized actual, point forecast (P50), and the leakage-safe
    # per-hour training-residual sample that defines the predictive distribution.
    samples = []  # (actual, point, residuals_for_this_hour)
    for d in test:
        Xtr = np.vstack([_feats(load2d, days, t, holid) for t in range(first, d)])
        ytr = np.concatenate([load2d[t] - load2d[t - 7] for t in range(first, d)])
        m = _fit(Xtr, ytr, a.lam)
        fitted = np.concatenate([load2d[t - 7] for t in range(first, d)]) + _pred(m, Xtr)
        res = np.concatenate([load2d[t] for t in range(first, d)]) - fitted
        point = load2d[d - 7] + _pred(m, _feats(load2d, days, d, holid))
        for h in range(24):
            samples.append((load2d[d, h], point[h], res[h::24]))

    # Deterministic (nominate the median quantile) vs stochastic (nominate the
    # cost-optimal tau-quantile). Both are anchored to the SAME point forecast, so
    # at ratio=1 (tau=0.5) they are identical by construction -> saving == 0 (sanity).
    results = []
    for r in ratios:
        c_long = a.price
        c_short = r * a.price
        tau = c_short / (c_short + c_long)
        cost_det = cost_sto = 0.0
        for actual, point, res in samples:
            q_det = point + np.quantile(res, 0.5)
            q_sto = point + np.quantile(res, tau)
            cost_det += imbalance_cost(actual, q_det, c_short, c_long)
            cost_sto += imbalance_cost(actual, q_sto, c_short, c_long)
        save = cost_det - cost_sto
        results.append({
            "ratio_cshort_over_clong": r,
            "tau": round(tau, 3),
            "cost_det_EUR": round(cost_det),
            "cost_sto_EUR": round(cost_sto),
            "saving_EUR_testperiod": round(save),
            "saving_pct": round(100 * save / cost_det, 2) if cost_det else 0.0,
        })

    summary = {
        "experiment": "balancing-group day-ahead nomination under asymmetric imbalance cost",
        "data": "SMARD Netzlast DE, hourly (real)",
        "period": f"{days[0].date()}..{days[-1].date()}",
        "test_days": len(test),
        "test_hours": len(samples),
        "base_price_c_long_EUR_per_MWh": a.price,
        "leakage_safe": "rolling-origin; per-hour residual quantiles use only days < d",
        "results": results,
        "reading": ("saving_pct = realized euro reduction from nominating the cost-optimal "
                    "quantile instead of the median. ~0 at ratio=1 (symmetric) by construction; "
                    "should grow with asymmetry if the forecast quantiles are usable."),
        "caveat": ("ABSOLUTE euro figures reflect Germany-wide balancing volume and are "
                   "illustrative only - NOT a NetzPilot revenue figure. The transferable result "
                   "is saving_pct, which is INVARIANT to error magnitude and is set by the error "
                   "distribution SHAPE (skew/tails) x cost asymmetry (see "
                   "residual_shape_sensitivity.py). Confirm shape + real reBAP asymmetry on customer data."),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(os.path.join(HERE, "dispatch_results.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
