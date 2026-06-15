"""Run T6 MAPIE EnbPI interval comparison on the T2 cache."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

from netzpilot.data.residual import residual_load
from netzpilot.eval import metrics as M
from netzpilot.features.build import build_features, frame_to_daily_local, get_holidays, resid_target, to_daily_local, base


def _read_series(path: Path, column: str) -> pd.Series:
    df = pd.read_parquet(path)
    s = df[column]
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def _target_series(cache_dir: Path, target: str) -> pd.Series:
    if target == "load":
        return _read_series(cache_dir / "smard_load_hour.parquet", "load_mw")
    if target == "residual":
        return residual_load(
            _read_series(cache_dir / "smard_load_hour.parquet", "load_mw"),
            _read_series(cache_dir / "smard_pv_quarterhour.parquet", "pv_mw"),
            _read_series(cache_dir / "smard_wind_onshore_quarterhour.parquet", "wind_onshore_mw"),
            _read_series(cache_dir / "smard_wind_offshore_quarterhour.parquet", "wind_offshore_mw"),
        )
    raise ValueError(f"Unsupported target: {target}")


def _feature_rows(load2d, days, day_range, weather2d, holidays):
    X = np.vstack([build_features(load2d, days, t, weather2d, holidays) for t in day_range])
    y = np.concatenate([resid_target(load2d, t) for t in day_range])
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data_cache/t2_2022-01-01_2024-01-01")
    ap.add_argument("--target", choices=["load", "residual"], default="load")
    ap.add_argument("--n-test", type=int, default=28)
    ap.add_argument("--region", default="NW")
    ap.add_argument("--out", default="data_cache/t6_mapie")
    ap.add_argument("--n-estimators", type=int, default=80)
    ap.add_argument("--n-resamplings", type=int, default=10)
    args = ap.parse_args()

    from lightgbm import LGBMRegressor
    from mapie.regression import TimeSeriesRegressor
    from mapie.subsample import BlockBootstrap

    cache_dir = Path(args.cache_dir)
    target = _target_series(cache_dir, args.target)
    weather = pd.read_parquet(cache_dir / "openmeteo_historical_forecast_hour.parquet")
    weather.index = pd.to_datetime(weather.index, utc=True)
    weather = weather.reindex(target.index)
    if weather.isna().any().any():
        raise ValueError("Weather forecast data is missing for target timestamps.")
    load2d, days, good_dates = to_daily_local(target)
    weather2d = frame_to_daily_local(weather, good_dates)
    hol = get_holidays(sorted({d.year for d in days}), args.region)

    first = 8
    split = len(load2d) - args.n_test
    train_days = range(first, split)
    test_days = range(split, len(load2d))
    X_train, y_train = _feature_rows(load2d, days, train_days, weather2d, hol)
    X_test, _ = _feature_rows(load2d, days, test_days, weather2d, hol)
    actual = np.concatenate([load2d[t] for t in test_days])
    baseline = np.concatenate([base(load2d, t) for t in test_days])

    estimator = LGBMRegressor(
        objective="regression",
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )
    cv = BlockBootstrap(n_resamplings=args.n_resamplings, length=24 * 7, overlapping=True, random_state=42)
    mapie = TimeSeriesRegressor(estimator=estimator, method="enbpi", cv=cv, random_state=42, agg_function="mean")
    mapie.fit(X_train[:, 1:], y_train)
    pred_resid, intervals = mapie.predict(X_test[:, 1:], confidence_level=0.8)
    intervals = np.asarray(intervals)
    if intervals.ndim == 3:
        lo_resid = intervals[:, 0, 0]
        hi_resid = intervals[:, 1, 0]
    else:
        lo_resid = intervals[:, 0]
        hi_resid = intervals[:, 1]
    pred = baseline + pred_resid
    lo = baseline + lo_resid
    hi = baseline + hi_resid

    summary = {
        "target": args.target,
        "method": "MAPIE TimeSeriesRegressor EnbPI + BlockBootstrap",
        "test_tage": args.n_test,
        "test_vorhersagen": int(len(actual)),
        "MAE_MW": round(M.mae(pred, actual), 1),
        "RMSE_MW": round(M.rmse(pred, actual), 1),
        "MAPE_%": round(M.mape(pred, actual), 2),
        "Coverage_P10_P90_%": round(M.coverage(actual, lo, hi), 1),
        "Soll_%": 80,
        "Pinball_avg": round(np.mean([M.pinball(actual, lo, 0.1), M.pinball(actual, pred, 0.5), M.pinball(actual, hi, 0.9)]), 1),
        "n_resamplings": args.n_resamplings,
        "block_length_hours": 24 * 7,
        "mapie_version": "1.4.0",
        "mapie_version_note": "Pinned MAPIE 1.4.0 exposes TimeSeriesRegressor(method='enbpi'), replacing older MapieTimeSeriesRegressor naming.",
        "weather_source": "Open-Meteo Historical Forecast API",
    }

    out = Path(args.out) / args.target
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "arrays.npz", actual=actual, model=pred, p10=lo, p90=hi)
    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
