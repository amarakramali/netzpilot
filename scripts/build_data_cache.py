# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Build the T2 SMARD + Open-Meteo cache.

Example:
  python scripts/build_data_cache.py --start 2022-01-01 --end 2024-01-01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.data import openmeteo, smard
from netzpilot.data.integrity import validate_series


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_series(path: Path, series: pd.Series, name: str) -> None:
    series.to_frame(name).to_parquet(path)


def _validate_weather_frame(weather: pd.DataFrame, resolution: str) -> dict:
    report = validate_series(weather.iloc[:, 0], resolution)
    non_finite = {col: int(pd.to_numeric(weather[col], errors="coerce").isna().sum()) for col in weather.columns}
    if any(non_finite.values()):
        raise ValueError(f"Weather frame has non-finite values: {non_finite}")
    report["columns"] = list(weather.columns)
    report["non_finite_by_column"] = non_finite
    return report


def _write_duckdb_catalog(run_dir: Path, outputs: dict[str, str]) -> str:
    import duckdb

    db_path = run_dir / "cache.duckdb"
    with duckdb.connect(str(db_path)) as con:
        for name, path in outputs.items():
            parquet_path = Path(path).as_posix()
            safe_path = parquet_path.replace("'", "''")
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{safe_path}')")
    return str(db_path)


def build_cache(args: argparse.Namespace) -> dict:
    run_dir = Path(args.cache_dir) / f"t2_{args.start}_{args.end}"
    raw_dir = run_dir / "raw"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict] = {}
    outputs: dict[str, str] = {}

    load_qh = smard.fetch_series(
        args.start, args.end, filter_id=smard.FILTER_LOAD, region=args.region,
        resolution="quarterhour", cache_dir=str(raw_dir), force=args.force,
    )
    reports["smard_load_quarterhour"] = validate_series(load_qh, "quarterhour")
    outputs["smard_load_quarterhour"] = str(run_dir / "smard_load_quarterhour.parquet")
    _write_series(Path(outputs["smard_load_quarterhour"]), load_qh, "load_mw")

    load_hour = smard.fetch_series(
        args.start, args.end, filter_id=smard.FILTER_LOAD, region=args.region,
        resolution="hour", cache_dir=str(raw_dir), force=args.force,
    )
    reports["smard_load_hour"] = validate_series(load_hour, "hour")
    outputs["smard_load_hour"] = str(run_dir / "smard_load_hour.parquet")
    _write_series(Path(outputs["smard_load_hour"]), load_hour, "load_mw")

    if args.include_generation:
        for name in ("pv", "wind_onshore", "wind_offshore"):
            gen = smard.fetch_series(
                args.start, args.end, filter_id=smard.FILTERS[name], region=args.region,
                resolution="quarterhour", cache_dir=str(raw_dir), force=args.force,
            )
            report_name = f"smard_{name}_quarterhour"
            reports[report_name] = validate_series(gen, "quarterhour")
            outputs[report_name] = str(run_dir / f"{report_name}.parquet")
            _write_series(Path(outputs[report_name]), gen, f"{name}_mw")

    if not args.skip_weather:
        weather_start = pd.DatetimeIndex(load_qh.index).tz_convert("UTC")[0].date().isoformat()
        weather_end = pd.DatetimeIndex(load_qh.index).tz_convert("UTC")[-1].date().isoformat()
        weather_hour = openmeteo.fetch_multi(
            start=weather_start,
            end=weather_end,
            historical=True,
            cache_dir=str(raw_dir),
            force=args.force,
        )
        reports["openmeteo_historical_forecast_hour"] = _validate_weather_frame(weather_hour, "hour")
        outputs["openmeteo_historical_forecast_hour"] = str(run_dir / "openmeteo_historical_forecast_hour.parquet")
        weather_hour.to_parquet(outputs["openmeteo_historical_forecast_hour"])

        weather_aligned = openmeteo.align_to_index(weather_hour, load_qh.index)
        reports["openmeteo_historical_forecast_aligned_quarterhour"] = _validate_weather_frame(
            weather_aligned, "quarterhour"
        )
        outputs["openmeteo_historical_forecast_aligned_quarterhour"] = str(
            run_dir / "openmeteo_historical_forecast_aligned_quarterhour.parquet"
        )
        weather_aligned.to_parquet(outputs["openmeteo_historical_forecast_aligned_quarterhour"])

    outputs["duckdb_catalog"] = _write_duckdb_catalog(run_dir, outputs)

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_utc_inclusive": args.start,
        "end_utc_exclusive": args.end,
        "region": args.region,
        "smard_base_url": smard.BASE,
        "smard_filters_verified": smard.FILTERS,
        "openmeteo_source": openmeteo.HIST_FORECAST,
        "openmeteo_variables": openmeteo.DEFAULT_VARS,
        "openmeteo_locations": {k: list(v) for k, v in openmeteo.DEFAULT_LOCATIONS.items()},
        "weather_training_rule": "Historical Forecast API only; do not use reanalysis actual weather in backtests.",
        "outputs": outputs,
    }

    _write_json(run_dir / "integrity_report.json", reports)
    _write_json(run_dir / "provenance.json", provenance)
    (run_dir / "README.md").write_text(
        "\n".join([
            "# T2 Data Cache",
            "",
            f"Range: [{args.start}, {args.end}) UTC",
            f"Region: {args.region}",
            "",
            "Contents:",
            *[f"- `{Path(path).name}`" for path in outputs.values()],
            "",
            "See `integrity_report.json` and `provenance.json` for validation and source metadata.",
            "",
        ]),
        encoding="utf-8",
    )
    return {"run_dir": str(run_dir), "reports": reports, "provenance": provenance}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01", help="UTC start date/time, inclusive")
    ap.add_argument("--end", default="2024-01-01", help="UTC end date/time, exclusive")
    ap.add_argument("--region", default="DE", help="SMARD region")
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--include-generation", action="store_true", help="Also cache PV and wind generation series")
    ap.add_argument("--skip-weather", action="store_true", help="Only fetch SMARD load")
    ap.add_argument("--force", action="store_true", help="Ignore existing Parquet cache files")
    args = ap.parse_args()
    result = build_cache(args)
    print(json.dumps({"run_dir": result["run_dir"], "reports": result["reports"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
