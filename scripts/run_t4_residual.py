"""Run T4 residual-load backtest on the T2 cache."""
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
from netzpilot.eval.backtest import rolling_origin_quantile
from netzpilot.features.build import frame_to_daily_local, get_holidays, to_daily_local
from netzpilot.models.lgbm_quantile import LGBMQuantileCorrector
from netzpilot.report.report import write_report


def _read_series(path: Path, column: str) -> pd.Series:
    df = pd.read_parquet(path)
    s = df[column]
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def _write_parquet_replace(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    if tmp.exists():
        tmp.unlink()
    frame.to_parquet(tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data_cache/t2_2022-01-01_2024-01-01")
    ap.add_argument("--n-test", type=int, default=28)
    ap.add_argument("--region", default="NW")
    ap.add_argument("--out", default="data_cache/t4_residual")
    ap.add_argument("--n-estimators", type=int, default=180)
    ap.add_argument("--retrain-every", type=int, default=7)
    ap.add_argument("--calibration-tail-days", type=int, default=7)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    load = _read_series(cache_dir / "smard_load_hour.parquet", "load_mw")
    pv = _read_series(cache_dir / "smard_pv_quarterhour.parquet", "pv_mw")
    wind_on = _read_series(cache_dir / "smard_wind_onshore_quarterhour.parquet", "wind_onshore_mw")
    wind_off = _read_series(cache_dir / "smard_wind_offshore_quarterhour.parquet", "wind_offshore_mw")
    residual = residual_load(load, pv, wind_on, wind_off)
    weather = pd.read_parquet(cache_dir / "openmeteo_historical_forecast_hour.parquet")
    weather.index = pd.to_datetime(weather.index, utc=True)

    weather = weather.reindex(residual.index)
    if weather.isna().any().any():
        raise ValueError("Weather forecast data is missing for residual-load timestamps.")
    load2d, days, good_dates = to_daily_local(residual)
    weather2d = frame_to_daily_local(weather, good_dates)
    hol = get_holidays(sorted({d.year for d in days}), args.region)

    def factory():
        return LGBMQuantileCorrector(n_estimators=args.n_estimators)

    R, summary = rolling_origin_quantile(
        load2d, days, factory, n_test=args.n_test, weather2d=weather2d,
        holiday_set=hol, retrain_every=args.retrain_every,
        calibration_tail_days=args.calibration_tail_days,
    )
    summary["target"] = "residual_load_mw"
    summary["model"] = "LightGBMQuantileCorrector"
    summary["feature_set"] = "calendar+lags+fourier+Open-Meteo Historical Forecast"
    summary["weather_source"] = "Open-Meteo Historical Forecast API"
    summary["retrain_every_days"] = args.retrain_every
    summary["calibration_tail_days"] = args.calibration_tail_days
    summary["method_limitations"] = (
        "Residual-load MVP uses historical SMARD PV/wind generation to construct the target "
        "and forecasts residual load directly. A separate physical pvlib/windpowerlib generation "
        "forecast remains the next refinement."
    )
    summary["cache_dir"] = str(cache_dir)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_parquet_replace(residual.to_frame("residual_load_mw"), out / "residual_load_hour.parquet")
    write_report(summary, str(out / "report.md"), str(out / "results.json"))
    np.savez(out / "arrays.npz", **R)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
