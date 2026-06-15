"""Run T9 small-utility validation on public OPSD household data."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

from netzpilot.data.small_utility import aggregate_opsd_grid_import
from netzpilot.data.openmeteo import fetch_weather
from netzpilot.eval.backtest import rolling_origin_cqr
from netzpilot.features.build import frame_to_daily_local, get_holidays, to_daily_local
from netzpilot.models.lgbm_quantile import LGBMQuantileCorrector

DEFAULT_URL = (
    "https://data.open-power-system-data.org/household_data/2020-04-15/"
    "household_data_15min_singleindex.csv"
)


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(".tmp")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.replace(tmp, path)


def _combined_report(runs: list[dict], source: dict) -> str:
    lines = [
        "# NetzPilot - T9 Small-Utility Validation",
        "",
        "Quelle: OPSD Household Data 2020-04-15, CoSSMic Konstanz, 15-min grid_import aggregation.",
        f"URL: {source['url']}",
        f"Genutzte Grid-Import-Spalten: {len(source['columns'])}",
        f"Skalierungsfaktor: {source['scale_factor']}",
        "",
        "| Interval | MAE [MW] | MAPE [%] | Skill vs S-Naiv | Coverage [%] | Ziel [%] |",
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        m = run["metriken"]["model"]
        p = run["probabilistisch"]
        lines.append(
            f"| {p['Interval_Label']} | {m['MAE_MW']} | {m['MAPE_%']} | "
            f"{m['Skill_vs_SaisonalNaiv_%']}% | {p['Coverage_Interval_%']} | {p['Soll_%']} |"
        )
    lines.extend([
        "",
        "Einordnung: Das ist keine reale Stadtwerke-Messreihe, sondern ein aus oeffentlichen Haushalts-/Kleingewerbeprofilen aggregierter Zielmarkt-Proxy.",
        "Der Prozentfehler ist erwartbar hoeher und volatiler als bei nationaler Last.",
    ])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-url", default=DEFAULT_URL)
    ap.add_argument("--raw", default="data_cache/t9_small_utility/raw/household_data_15min_singleindex.csv")
    ap.add_argument("--out", default="data_cache/t9_small_utility")
    ap.add_argument("--scale-factor", type=float, default=1000.0)
    ap.add_argument("--min-active-columns", type=int, default=4)
    ap.add_argument("--coverages", default="0.8,0.9")
    ap.add_argument("--calibration-window-days", type=int, default=7)
    ap.add_argument("--online-eta", type=float, default=0.2)
    ap.add_argument("--n-test", type=int, default=28)
    ap.add_argument("--region", default="BW")
    ap.add_argument("--n-estimators", type=int, default=180)
    ap.add_argument("--with-weather", action="store_true")
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
    weather2d = None
    weather_source = None
    if args.with_weather:
        start = max(pd.Timestamp(args.weather_start, tz="UTC"), series.index.min()).date().isoformat()
        end = series.index.max().date().isoformat()
        weather = fetch_weather(
            47.66, 9.18, start, end, historical=True,
            cache_dir=str(out / "weather"), location_name="konstanz",
        ).apply(pd.to_numeric, errors="coerce")
        weather = weather.dropna(how="any")
        common = series.index.intersection(weather.index)
        series = series.loc[common]
        weather = weather.loc[common]
        weather_source = "Open-Meteo Historical Forecast API, Konstanz"
    series.to_frame("small_utility_load_mw").to_parquet(out / "small_utility_load_hour.parquet")

    load2d, days, _good_dates = to_daily_local(series)
    if args.with_weather:
        weather2d = frame_to_daily_local(weather, _good_dates)
    hol = get_holidays(sorted({d.year for d in days}), args.region)
    runs = []
    for coverage in [float(x.strip()) for x in args.coverages.split(",") if x.strip()]:
        def factory(alphas):
            return LGBMQuantileCorrector(alphas=alphas, n_estimators=args.n_estimators)

        R, summary = rolling_origin_cqr(
            load2d, days, factory, n_test=args.n_test, holiday_set=hol, weather2d=weather2d,
            calibration_window_days=args.calibration_window_days,
            interval_coverage=coverage, online_eta=args.online_eta,
        )
        summary["target"] = "small_utility_load_mw"
        summary["model"] = "LightGBMQuantileCorrector + rolling CQR"
        summary["feature_set"] = "calendar+lags+fourier+Konstanz Historical Forecast weather" if args.with_weather else "calendar+lags+fourier; no weather proxy in T9 public-data smoke"
        if weather_source:
            summary["weather_source"] = weather_source
        summary["source"] = "OPSD Household Data 2020-04-15, CoSSMic Konstanz"
        summary["source_url"] = args.source_url
        summary["scale_factor"] = args.scale_factor
        summary["min_active_columns"] = args.min_active_columns
        summary["used_grid_import_columns"] = used_cols
        stem = f"small_utility_{int(round(coverage * 100))}"
        np.savez(out / f"{stem}_arrays.npz", **R)
        with open(out / f"{stem}_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        runs.append(summary)

    source = {
        "url": args.source_url,
        "columns": used_cols,
        "scale_factor": args.scale_factor,
        "min_active_columns": args.min_active_columns,
    }
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump({"source": source, "runs": runs}, f, indent=2, ensure_ascii=False)
    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(_combined_report(runs, source))
    print(json.dumps({"source": source, "runs": runs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
