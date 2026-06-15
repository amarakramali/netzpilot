# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Run T10 improved small-utility validation with local weather and small-load features."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_t9_small_utility import DEFAULT_URL, _download, _combined_report
from netzpilot.data.openmeteo import fetch_weather
from netzpilot.data.small_utility import aggregate_opsd_grid_import
from netzpilot.eval.backtest import rolling_origin_cqr
from netzpilot.features.build import (
    build_small_load_features,
    frame_to_daily_local,
    get_holidays,
    to_daily_local,
)
from netzpilot.models.robust_corrector import ShrunkQuantileCorrector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-url", default=DEFAULT_URL)
    ap.add_argument("--raw", default="data_cache/t9_small_utility/raw/household_data_15min_singleindex.csv")
    ap.add_argument("--out", default="data_cache/t10_small_utility")
    ap.add_argument("--scale-factor", type=float, default=1000.0)
    ap.add_argument("--min-active-columns", type=int, default=4)
    ap.add_argument("--coverages", default="0.8,0.9")
    ap.add_argument("--calibration-window-days", type=int, default=7)
    ap.add_argument("--online-eta", type=float, default=0.2)
    ap.add_argument("--n-test", type=int, default=28)
    ap.add_argument("--region", default="BW")
    ap.add_argument("--n-estimators", type=int, default=260)
    ap.add_argument("--num-leaves", type=int, default=15)
    ap.add_argument("--min-child-samples", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=0.04)
    ap.add_argument("--weather-start", default="2017-01-01")
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _download(args.source_url, raw)

    df = pd.read_csv(raw)
    series, used_cols = aggregate_opsd_grid_import(
        df, scale_factor=args.scale_factor, min_active_columns=args.min_active_columns,
    )
    start = max(pd.Timestamp(args.weather_start, tz="UTC"), series.index.min()).date().isoformat()
    end = series.index.max().date().isoformat()
    weather = fetch_weather(
        47.66,
        9.18,
        start,
        end,
        historical=True,
        cache_dir=str(out / "weather"),
        location_name="konstanz",
    ).apply(pd.to_numeric, errors="coerce").dropna(how="any")
    common = series.index.intersection(weather.index)
    series = series.loc[common]
    weather = weather.loc[common]
    series.to_frame("small_utility_load_mw").to_parquet(out / "small_utility_load_hour.parquet")

    load2d, days, good_dates = to_daily_local(series)
    weather2d = frame_to_daily_local(weather, good_dates)
    hol = get_holidays(sorted({d.year for d in days}), args.region)

    runs = []
    for coverage in [float(x.strip()) for x in args.coverages.split(",") if x.strip()]:
        def factory(alphas):
            return ShrunkQuantileCorrector(
                alphas=alphas,
                n_estimators=args.n_estimators,
                num_leaves=args.num_leaves,
                min_child_samples=args.min_child_samples,
                learning_rate=args.learning_rate,
            )

        R, summary = rolling_origin_cqr(
            load2d,
            days,
            factory,
            first=14,
            n_test=args.n_test,
            holiday_set=hol,
            weather2d=weather2d,
            calibration_window_days=args.calibration_window_days,
            interval_coverage=coverage,
            online_eta=args.online_eta,
            feature_fn=build_small_load_features,
        )
        summary["target"] = "small_utility_load_mw"
        summary["model"] = "ShrunkQuantileCorrector(LGBM) + small-load features + rolling CQR"
        summary["feature_set"] = "calendar+lags+short-term deviations+morning/evening interactions+Konstanz Historical Forecast weather"
        summary["weather_source"] = "Open-Meteo Historical Forecast API, Konstanz"
        summary["source"] = "OPSD Household Data 2020-04-15, CoSSMic Konstanz"
        summary["source_url"] = args.source_url
        summary["scale_factor"] = args.scale_factor
        summary["min_active_columns"] = args.min_active_columns
        summary["used_grid_import_columns"] = used_cols
        summary["t10_params"] = {
            "first_day_index": 14,
            "n_estimators": args.n_estimators,
            "num_leaves": args.num_leaves,
            "min_child_samples": args.min_child_samples,
            "learning_rate": args.learning_rate,
        }
        stem = f"small_utility_t10_{int(round(coverage * 100))}"
        np.savez(out / f"{stem}_arrays.npz", **R)
        with (out / f"{stem}_results.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        runs.append(summary)

    source = {
        "url": args.source_url,
        "columns": used_cols,
        "scale_factor": args.scale_factor,
        "min_active_columns": args.min_active_columns,
    }
    result = {"source": source, "runs": runs}
    with (out / "results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with (out / "report.md").open("w", encoding="utf-8") as f:
        f.write(_combined_report(runs, source))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
