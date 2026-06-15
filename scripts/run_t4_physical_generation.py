"""Run T4 physical PV/wind generation forecast plus residual-load comparison."""
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

from netzpilot.data.generation_forecast import (
    hourly_generation_frame,
    physical_generation_proxies,
    rolling_generation_bias_forecast,
)
from netzpilot.eval import metrics as M
from netzpilot.eval.backtest import rolling_origin_cqr
from netzpilot.features.build import frame_to_daily_local, get_holidays, to_daily_local
from netzpilot.models.baselines import persistence, seasonal_naive
from netzpilot.models.lgbm_quantile import LGBMQuantileCorrector


def _read_series(path: Path, column: str) -> pd.Series:
    df = pd.read_parquet(path)
    s = df[column]
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def _flat_baseline(load2d, fn, test_days):
    return np.concatenate([fn(load2d, d) for d in test_days])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data_cache/t2_2022-01-01_2024-01-01")
    ap.add_argument("--out", default="data_cache/t4_physical_generation")
    ap.add_argument("--n-test", type=int, default=28)
    ap.add_argument("--region", default="NW")
    ap.add_argument("--n-estimators", type=int, default=180)
    ap.add_argument("--calibration-window-days", type=int, default=7)
    ap.add_argument("--online-eta", type=float, default=0.2)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    load = _read_series(cache_dir / "smard_load_hour.parquet", "load_mw")
    pv = _read_series(cache_dir / "smard_pv_quarterhour.parquet", "pv_mw")
    wind_on = _read_series(cache_dir / "smard_wind_onshore_quarterhour.parquet", "wind_onshore_mw")
    wind_off = _read_series(cache_dir / "smard_wind_offshore_quarterhour.parquet", "wind_offshore_mw")
    weather = pd.read_parquet(cache_dir / "openmeteo_historical_forecast_hour.parquet")
    weather.index = pd.to_datetime(weather.index, utc=True)

    gen = hourly_generation_frame(pv, wind_on, wind_off).reindex(load.index)
    proxies = physical_generation_proxies(weather).reindex(load.index)
    if gen.isna().any().any() or proxies.isna().any().any():
        raise ValueError("Generation/proxy data is missing for load timestamps.")

    load2d, days, good_dates = to_daily_local(load)
    gen2d = frame_to_daily_local(gen[["pv_mw", "wind_onshore_mw", "wind_offshore_mw"]], good_dates)
    proxy2d = frame_to_daily_local(proxies[["pv_proxy", "wind_onshore_proxy", "wind_offshore_proxy"]], good_dates)
    weather2d = frame_to_daily_local(weather.reindex(load.index), good_dates)
    hol = get_holidays(sorted({d.year for d in days}), args.region)

    first = 8
    test_days = list(range(len(load2d) - args.n_test, len(load2d)))
    gen_fc = rolling_generation_bias_forecast(
        gen2d, proxy2d, days, first=first, n_test=args.n_test, retrain_every=7,
    )

    def factory(alphas):
        return LGBMQuantileCorrector(alphas=alphas, n_estimators=args.n_estimators)

    load_R, load_summary = rolling_origin_cqr(
        load2d, days, factory, first=first, n_test=args.n_test, weather2d=weather2d,
        holiday_set=hol, retrain_every=7,
        calibration_window_days=args.calibration_window_days,
        interval_coverage=0.8, online_eta=args.online_eta,
    )
    residual2d = load2d - gen2d.sum(axis=2)
    actual_residual = np.concatenate([residual2d[d] for d in test_days])
    residual_pred = load_R["model"] - gen_fc["pred_total"]
    persist = _flat_baseline(residual2d, persistence, test_days)
    snaive = _flat_baseline(residual2d, seasonal_naive, test_days)
    scale = float(np.mean(np.abs(residual2d[first:len(residual2d)-args.n_test] - residual2d[first-1:len(residual2d)-args.n_test-1])))
    mp, ms = M.mae(persist, actual_residual), M.mae(snaive, actual_residual)

    comp_names = ["pv_mw", "wind_onshore_mw", "wind_offshore_mw"]
    generation_metrics = {}
    for i, name in enumerate(comp_names):
        generation_metrics[name] = {
            "MAE_MW": round(M.mae(gen_fc["pred_components"][:, i], gen_fc["actual_components"][:, i]), 1),
            "RMSE_MW": round(M.rmse(gen_fc["pred_components"][:, i], gen_fc["actual_components"][:, i]), 1),
        }
    generation_metrics["generation_total_mw"] = {
        "MAE_MW": round(M.mae(gen_fc["pred_total"], gen_fc["actual_total"]), 1),
        "RMSE_MW": round(M.rmse(gen_fc["pred_total"], gen_fc["actual_total"]), 1),
    }

    summary = {
        "target": "physical_residual_load_mw",
        "method": "Load LGBM+CQR minus pvlib/windpowerlib physical generation forecast with rolling Ridge bias correction",
        "test_tage": args.n_test,
        "test_vorhersagen": int(len(actual_residual)),
        "generation_metrics": generation_metrics,
        "residual_metrics": {
            "persist": {
                "MAE_MW": round(mp, 1),
                "RMSE_MW": round(M.rmse(persist, actual_residual), 1),
                "MAPE_%": round(M.mape(persist, actual_residual), 2),
                "MASE": round(M.mase(persist, actual_residual, scale), 3),
                "Skill_vs_Persistenz_%": 0.0,
                "Skill_vs_SaisonalNaiv_%": round(M.skill(mp, ms), 1),
            },
            "snaive": {
                "MAE_MW": round(ms, 1),
                "RMSE_MW": round(M.rmse(snaive, actual_residual), 1),
                "MAPE_%": round(M.mape(snaive, actual_residual), 2),
                "MASE": round(M.mase(snaive, actual_residual, scale), 3),
                "Skill_vs_Persistenz_%": round(M.skill(ms, mp), 1),
                "Skill_vs_SaisonalNaiv_%": 0.0,
            },
            "model": {
                "MAE_MW": round(M.mae(residual_pred, actual_residual), 1),
                "RMSE_MW": round(M.rmse(residual_pred, actual_residual), 1),
                "MAPE_%": round(M.mape(residual_pred, actual_residual), 2),
                "MASE": round(M.mase(residual_pred, actual_residual, scale), 3),
                "Skill_vs_Persistenz_%": round(M.skill(M.mae(residual_pred, actual_residual), mp), 1),
                "Skill_vs_SaisonalNaiv_%": round(M.skill(M.mae(residual_pred, actual_residual), ms), 1),
            },
        },
        "load_forecast_reference": load_summary["metriken"]["model"],
        "weather_source": "Open-Meteo Historical Forecast API",
        "method_limitations": "Residual intervals are not produced here; generation uncertainty is point-only. T8 remains the calibrated residual interval path.",
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "arrays.npz",
        actual_residual=actual_residual,
        residual_model=residual_pred,
        generation_actual=gen_fc["actual_total"],
        generation_model=gen_fc["pred_total"],
        load_actual=load_R["actual"],
        load_model=load_R["model"],
    )
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
