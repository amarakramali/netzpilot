"""Run T8 rolling CQR calibration for load and residual-load targets."""
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
from netzpilot.eval.backtest import rolling_origin_cqr
from netzpilot.features.build import frame_to_daily_local, get_holidays, to_daily_local
from netzpilot.models.lgbm_quantile import LGBMQuantileCorrector
from netzpilot.report.report import build_markdown, write_report


def _read_series(path: Path, column: str) -> pd.Series:
    df = pd.read_parquet(path)
    s = df[column]
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def _read_weather(cache_dir: Path) -> pd.DataFrame:
    weather = pd.read_parquet(cache_dir / "openmeteo_historical_forecast_hour.parquet")
    weather.index = pd.to_datetime(weather.index, utc=True)
    return weather


def _target(cache_dir: Path, name: str) -> tuple[pd.Series, str]:
    load = _read_series(cache_dir / "smard_load_hour.parquet", "load_mw")
    if name == "load":
        return load, "load_mw"
    if name == "residual":
        pv = _read_series(cache_dir / "smard_pv_quarterhour.parquet", "pv_mw")
        wind_on = _read_series(cache_dir / "smard_wind_onshore_quarterhour.parquet", "wind_onshore_mw")
        wind_off = _read_series(cache_dir / "smard_wind_offshore_quarterhour.parquet", "wind_offshore_mw")
        return residual_load(load, pv, wind_on, wind_off), "residual_load_mw"
    raise ValueError(f"Unknown target: {name}")


def _parse_coverages(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _combined_report(runs: list[dict]) -> str:
    lines = ["# NetzPilot - T8 CQR calibration", ""]
    lines.extend([
        "| Target | Interval | MAE [MW] | Skill vs S-Naiv | Coverage [%] | Ziel [%] | CRPS_proxy |",
        "|---|---|---|---|---|---|---|",
    ])
    for run in runs:
        m = run["metriken"]["model"]
        p = run["probabilistisch"]
        lines.append(
            f"| {run['target']} | {p['Interval_Label']} | {m['MAE_MW']} | "
            f"{m['Skill_vs_SaisonalNaiv_%']}% | {p['Coverage_Interval_%']} | "
            f"{p['Soll_%']} | {p['CRPS_proxy']} |"
        )
    lines.extend([
        "",
        "CQR nutzt nur ein zeitlich vorgelagertes Kalibrierfenster je Zieltag; Wetter bleibt Open-Meteo Historical Forecast.",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data_cache/t2_2022-01-01_2024-01-01")
    ap.add_argument("--out", default="data_cache/t8_cqr")
    ap.add_argument("--targets", default="load,residual")
    ap.add_argument("--coverages", default="0.8,0.9")
    ap.add_argument("--calibration-window-days", type=int, default=56)
    ap.add_argument("--n-test", type=int, default=28)
    ap.add_argument("--region", default="NW")
    ap.add_argument("--n-estimators", type=int, default=180)
    ap.add_argument("--retrain-every", type=int, default=7)
    ap.add_argument("--online-eta", type=float, default=0.0)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weather_all = _read_weather(cache_dir)

    runs = []
    for target_name in [t.strip() for t in args.targets.split(",") if t.strip()]:
        series, target_label = _target(cache_dir, target_name)
        weather = weather_all.reindex(series.index)
        if weather.isna().any().any():
            raise ValueError(f"Weather forecast data is missing for {target_name} timestamps.")
        load2d, days, good_dates = to_daily_local(series)
        weather2d = frame_to_daily_local(weather, good_dates)
        hol = get_holidays(sorted({d.year for d in days}), args.region)

        for coverage in _parse_coverages(args.coverages):
            def factory(alphas):
                return LGBMQuantileCorrector(alphas=alphas, n_estimators=args.n_estimators)

            R, summary = rolling_origin_cqr(
                load2d, days, factory, n_test=args.n_test, weather2d=weather2d,
                holiday_set=hol, retrain_every=args.retrain_every,
                calibration_window_days=args.calibration_window_days,
                interval_coverage=coverage,
                online_eta=args.online_eta,
            )
            summary["target"] = target_label
            summary["model"] = "LightGBMQuantileCorrector + rolling CQR"
            summary["feature_set"] = "calendar+lags+fourier+Open-Meteo Historical Forecast"
            summary["weather_source"] = "Open-Meteo Historical Forecast API"
            summary["cache_dir"] = str(cache_dir)
            summary["retrain_every_days"] = args.retrain_every
            summary["n_estimators"] = args.n_estimators
            summary["online_eta"] = args.online_eta
            summary["method_limitations"] = (
                "CQR coverage is evaluated on a rolling time-series backtest; finite-sample "
                "exchangeability assumptions are approximate for non-stationary grid load."
            )
            stem = f"{target_name}_{int(round(coverage * 100))}"
            write_report(summary, str(out / f"{stem}_report.md"), str(out / f"{stem}_results.json"))
            np.savez(out / f"{stem}_arrays.npz", **R)
            runs.append(summary)

    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"runs": runs}, f, indent=2, ensure_ascii=False)
    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(_combined_report(runs))
    print(json.dumps({"runs": runs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
