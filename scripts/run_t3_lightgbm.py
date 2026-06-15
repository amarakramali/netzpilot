"""Run T3 LightGBM quantile backtest on the T2 cache."""
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

from netzpilot.eval.backtest import rolling_origin_quantile
from netzpilot.features.build import frame_to_daily_local, get_holidays, to_daily_local
from netzpilot.models.lgbm_quantile import LGBMQuantileCorrector
from netzpilot.report.report import write_report


def _read_series(path: Path, column: str) -> pd.Series:
    df = pd.read_parquet(path)
    s = df[column]
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def _read_weather(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data_cache/t2_2022-01-01_2024-01-01")
    ap.add_argument("--n-test", type=int, default=28)
    ap.add_argument("--region", default="NW")
    ap.add_argument("--out", default="data_cache/t3_lightgbm")
    ap.add_argument("--n-estimators", type=int, default=180)
    ap.add_argument("--retrain-every", type=int, default=7, help="Days between expanding-window refits")
    ap.add_argument("--calibration-tail-days", type=int, default=7)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    load = _read_series(cache_dir / "smard_load_hour.parquet", "load_mw")
    weather = _read_weather(cache_dir / "openmeteo_historical_forecast_hour.parquet")
    load2d, days, good_dates = to_daily_local(load)
    weather2d = frame_to_daily_local(weather, good_dates)
    hol = get_holidays(sorted({d.year for d in days}), args.region)

    def factory():
        return LGBMQuantileCorrector(n_estimators=args.n_estimators)

    R, summary = rolling_origin_quantile(
        load2d, days, factory, n_test=args.n_test, weather2d=weather2d,
        holiday_set=hol, retrain_every=args.retrain_every,
        calibration_tail_days=args.calibration_tail_days,
    )
    summary["model"] = "LightGBMQuantileCorrector"
    summary["feature_set"] = "calendar+lags+fourier+Open-Meteo Historical Forecast"
    summary["cache_dir"] = str(cache_dir)
    summary["weather_source"] = "Open-Meteo Historical Forecast API"
    summary["retrain_every_days"] = args.retrain_every
    summary["calibration_tail_days"] = args.calibration_tail_days
    summary["v1_reference"] = {
        "window": "bundled 2024-01-01..2024-03-24, 28-day rolling",
        "MAE_MW": 1411.4,
        "MAPE_%": 2.56,
        "Coverage_P10_P90_%": 81.5,
        "Skill_vs_SaisonalNaiv_%": 4.1,
        "note": "Reference window differs from the T2 cache final 28 local days.",
    }
    summary["note"] = "Weather uses archived forecasts, not actual/reanalysis weather."

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_report(summary, str(out / "report.md"), str(out / "results.json"))
    np.savez(out / "arrays.npz", **R)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
