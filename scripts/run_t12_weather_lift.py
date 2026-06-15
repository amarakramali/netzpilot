# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""T12: leakage-safe weather-lift checks on real weather-coupled load.

Implemented dataset: HEAPO (Zenodo 15056919), restricted to >=2022-07-01 and
Open-Meteo Historical Forecast weather. Pre-2022 T10/OPSD weather is rechecked
and labelled as perfect-foresight upper bound when it matches Archive weather.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_v1_significance import SEED, ci95, paired_block_bootstrap
from netzpilot.data.openmeteo import fetch_weather
from netzpilot.eval.backtest import rolling_origin
from netzpilot.eval.conformal import rolling_origin_cqr
from netzpilot.features.build import frame_to_daily_local, get_holidays
from netzpilot.models.robust_corrector import ShrunkCorrector

OUT_DIR = Path("data_cache/t12_weather_lift")
HEAPO_ZIP = OUT_DIR / "raw" / "heapo_data.zip"
HEAPO_URL = "https://zenodo.org/records/15056919/files/heapo_data.zip?download=1"
FORECAST_START = pd.Timestamp("2022-07-01", tz="UTC")
HEAPO_COORDS = {"latitude": 47.3769, "longitude": 8.5417, "label": "Zurich proxy for HEAPO"}
WEATHER_VARS = ["temperature_2m", "shortwave_radiation", "wind_speed_100m", "cloud_cover", "direct_radiation"]
N_BOOT = 10000
N_TEST = 28
KEEP_DAYS = 420
PROVENANCE_HEAPO = (
    "HEAPO Zenodo 15056919, real Swiss household smart-meter data with heat pumps; "
    "evaluation restricted to >=2022-07-01 and Open-Meteo Historical Forecast weather."
)


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(".tmp")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(path)


def series_to_daily_local(series: pd.Series, keep_days: int = KEEP_DAYS, tz: str = "Europe/Zurich"):
    s = series.sort_index()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    s = s[s.index >= FORECAST_START]
    loc = s.index.tz_convert(tz)
    df = pd.DataFrame({"v": s.to_numpy(dtype=float)}, index=loc)
    df["date"] = df.index.normalize()
    df["hour"] = df.index.hour
    groups = {}
    for date, g in df.groupby("date"):
        g = g.sort_values("hour")
        vals = g["v"].to_numpy(dtype=float)
        if len(g) == 24 and sorted(g["hour"].tolist()) == list(range(24)) and np.isfinite(vals).all():
            groups[date] = vals
    good = sorted(groups)
    if keep_days:
        good = good[-keep_days:]
    load2d = np.asarray([groups[d] for d in good], dtype=float)
    days = pd.to_datetime([d.date() for d in good])
    return load2d, days, good


def read_heapo_overview(z: zipfile.ZipFile) -> pd.DataFrame:
    with z.open("heapo_data/smart_meter_data/overview/smart_meter_data_15min_overview.csv") as f:
        ov = pd.read_csv(f, sep=";")
    ov["start"] = pd.to_datetime(ov["SMD_15min_TimeAvailable_EarliestTimestamp"], utc=True)
    ov["end"] = pd.to_datetime(ov["SMD_15min_TimeAvailable_LatestTimestamp"], utc=True)
    return ov


def heapo_hourly_series(z: zipfile.ZipFile, household_id: int, value_col: str) -> pd.Series | None:
    name = f"heapo_data/smart_meter_data/15min/{household_id}.csv"
    try:
        with z.open(name) as f:
            df = pd.read_csv(f, sep=";", usecols=["Timestamp", value_col])
    except KeyError:
        return None
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    vals = pd.to_numeric(df[value_col], errors="coerce")
    qh = pd.Series(vals.to_numpy(dtype=float), index=df["Timestamp"]).sort_index()
    hourly_kwh = qh.resample("1h").sum(min_count=4)
    return hourly_kwh / 1000.0  # kWh per hour -> average MW


def select_households(z: zipfile.ZipFile, value_col: str, max_households: int) -> list[int]:
    ov = read_heapo_overview(z)
    flag = {
        "kWh_received_HeatPump": "SMD_15min_MeasurementsAvailable_HeatPump",
        "kWh_received_Total": "SMD_15min_MeasurementsAvailable_Total",
    }[value_col]
    cand = ov[
        (ov[flag] == True)
        & (ov["start"] <= FORECAST_START)
        & (ov["end"] >= pd.Timestamp("2024-02-01", tz="UTC"))
    ].copy()
    cand = cand.sort_values(["SMD_15min_TimeAvailable_NumberDays", "Household_ID"], ascending=[False, True])
    out = []
    for hid in cand["Household_ID"].astype(int):
        s = heapo_hourly_series(z, hid, value_col)
        if s is None:
            continue
        load2d, _, _ = series_to_daily_local(s)
        if len(load2d) >= 180 and float(np.nanmean(load2d)) > 0:
            out.append(hid)
        if len(out) >= max_households:
            break
    return out


def aggregate_series(series_list: list[pd.Series]) -> pd.Series:
    frame = pd.concat(series_list, axis=1, sort=True)
    return frame.sum(axis=1, min_count=len(series_list)).dropna()


def daily_mae(R: dict, name: str) -> np.ndarray:
    return np.abs(np.asarray(R[name], float) - np.asarray(R["actual"], float)).reshape(-1, 24).mean(axis=1)


def significance_weather_vs_no(wx_R: dict, no_R: dict) -> dict:
    ae_wx = daily_mae(wx_R, "model")
    ae_no = daily_mae(no_R, "model")
    rng = np.random.default_rng(SEED)
    skill, dmae = paired_block_bootstrap(ae_wx, ae_no, rng, N_BOOT)
    point = (1.0 - ae_wx.sum() / ae_no.sum()) * 100.0
    return {
        "skill_weather_vs_no_weather_%": round(float(point), 2),
        "skill_ci95_%": ci95(skill),
        "P(weather_besser)_%": round(float(np.mean(skill > 0) * 100), 1),
        "dMAE_mean_MW": round(float(dmae.mean()), 6),
        "signifikant_5pct": bool(np.percentile(skill, 2.5) > 0.0),
    }


def metric_record(summary: dict) -> dict:
    m = summary["metriken"]["model"]
    def finite_or_none(value):
        value = float(value)
        return value if math.isfinite(value) else None
    return {
        "MAE_MW": m["MAE_MW"],
        "MAPE_%": finite_or_none(m["MAPE_%"]),
        "skill_snv_%": m["Skill_vs_SaisonalNaiv_%"],
        "skill_pers_%": m["Skill_vs_Persistenz_%"],
    }


def eval_load_series(label: str, series: pd.Series, weather: pd.DataFrame, aggregation: str, target: str) -> dict | None:
    load2d, days, good_dates = series_to_daily_local(series)
    if len(load2d) < 120:
        return None
    weather2d = frame_to_daily_local(weather, good_dates)
    hol = get_holidays(sorted({d.year for d in days}), "NW")
    no_R, no_sm = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=N_TEST, holiday_set=hol)
    wx_R, wx_sm = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=N_TEST, weather2d=weather2d, holiday_set=hol)
    _, c80 = rolling_origin_cqr(load2d, days, lambda: ShrunkCorrector(10.0), alpha=0.2, cal_days=28, n_test=N_TEST, weather2d=weather2d, holiday_set=hol)
    _, c90 = rolling_origin_cqr(load2d, days, lambda: ShrunkCorrector(10.0), alpha=0.1, cal_days=28, n_test=N_TEST, weather2d=weather2d, holiday_set=hol)
    no_m = metric_record(no_sm)
    wx_m = metric_record(wx_sm)
    return {
        "dataset": "heapo",
        "target": target,
        "aggregation": aggregation,
        "label": label,
        "n_days": int(len(load2d)),
        "n_test_days": N_TEST,
        "mean_mw": round(float(np.mean(load2d)), 6),
        "provenance": PROVENANCE_HEAPO,
        "leakage_class": "leakage_safe",
        "weather_source": "Open-Meteo Historical Forecast API",
        "weather_coords": HEAPO_COORDS,
        "no_weather": no_m,
        "weather": wx_m,
        "skill_lift_pp": round(float(wx_m["skill_snv_%"] - no_m["skill_snv_%"]), 2),
        "cqr": {
            "coverage80_%": c80["coverage_%"],
            "width80_MW": c80["mean_width_MW"],
            "coverage90_%": c90["coverage_%"],
            "width90_MW": c90["mean_width_MW"],
        },
        "significance_weather_vs_no_weather": significance_weather_vs_no(wx_R, no_R),
    }


def fetch_heapo_weather(start: str = "2022-07-01", end: str = "2024-03-01") -> pd.DataFrame:
    return fetch_weather(
        HEAPO_COORDS["latitude"],
        HEAPO_COORDS["longitude"],
        start,
        end,
        hourly=WEATHER_VARS,
        historical=True,
        cache_dir=str(OUT_DIR / "weather"),
        location_name="heapo_zurich",
        chunk_days=400,
    ).apply(pd.to_numeric, errors="coerce").dropna(how="any")


def run_heapo(args) -> list[dict]:
    download_file(HEAPO_URL, HEAPO_ZIP)
    weather = fetch_heapo_weather()
    records = []
    with zipfile.ZipFile(HEAPO_ZIP) as z:
        for target, col in [("heatpump", "kWh_received_HeatPump"), ("total", "kWh_received_Total")]:
            households = select_households(z, col, args.max_households)
            series_by_id = {hid: heapo_hourly_series(z, hid, col) for hid in households}
            for hid in households[: args.individual_count]:
                rec = eval_load_series(str(hid), series_by_id[hid], weather, "individual", target)
                if rec:
                    records.append(rec)
            for i in range(args.cluster_count):
                group = households[i * args.cluster_size : (i + 1) * args.cluster_size]
                if len(group) < args.cluster_size:
                    continue
                rec = eval_load_series(
                    "cluster_" + "_".join(map(str, group[:2])) + f"_n{len(group)}",
                    aggregate_series([series_by_id[hid] for hid in group]),
                    weather,
                    f"cluster_{len(group)}",
                    target,
                )
                if rec:
                    records.append(rec)
            feeder_group = households[: max(args.cluster_size, min(len(households), args.feeder_count))]
            if feeder_group:
                rec = eval_load_series(
                    f"feeder_n{len(feeder_group)}",
                    aggregate_series([series_by_id[hid] for hid in feeder_group]),
                    weather,
                    f"feeder_{len(feeder_group)}",
                    target,
                )
                if rec:
                    records.append(rec)
    out = OUT_DIR / "heapo_eval.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def fetch_archive_weather_konstanz(start: str, end: str, hourly: list[str]) -> pd.DataFrame:
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": 47.66,
            "longitude": 9.18,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(hourly),
            "timezone": "UTC",
        },
        timeout=60,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({k: h[k] for k in hourly}, index=pd.to_datetime(h["time"], utc=True))


def verify_t10_weather() -> dict:
    candidates = [
        Path("data_cache/t10_small_utility/weather/openmeteo_historical_forecast_konstanz_2017-01-01_2017-10-12.parquet"),
        Path("data_cache/t9_small_utility_weather/weather/openmeteo_historical_forecast_konstanz_2017-01-01_2017-10-12.parquet"),
    ]
    cache = next((p for p in candidates if p.exists()), None)
    if cache is None:
        return {"status": "missing_cache", "conclusion": "T10 weather cache not found"}
    hf = pd.read_parquet(cache)
    hf.index = pd.to_datetime(hf.index, utc=True)
    hourly = [c for c in WEATHER_VARS if c in hf.columns]
    archive = fetch_archive_weather_konstanz("2017-01-01", "2017-01-07", hourly)
    common = hf.index.intersection(archive.index)
    comparisons = {}
    for col in hourly:
        diff = (hf.loc[common, col].astype(float) - archive.loc[common, col].astype(float)).abs()
        comparisons[col] = {
            "mean_abs_diff": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
            "n": int(len(diff)),
        }
    all_identical = all(v["max_abs_diff"] == 0.0 for v in comparisons.values())
    out = {
        "status": "checked",
        "cache": str(cache),
        "period_checked": "2017-01-01..2017-01-07",
        "comparison": "T10 cached Historical-Forecast vs Open-Meteo Archive API",
        "comparisons": comparisons,
        "all_checked_variables_identical_to_archive": all_identical,
        "conclusion": (
            "T10 weather is perfect-foresight/actual-weather upper bound, not leakage-safe day-ahead forecast"
            if all_identical
            else "T10 weather differs from Archive in sampled period; inspect manually"
        ),
    }
    with (OUT_DIR / "t10_weather_reverify.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def binomial_one_sided_p(k: int, n: int) -> float:
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n))


def summarize(records: list[dict], t10_reverify: dict) -> dict:
    safe = [r for r in records if r["leakage_class"] == "leakage_safe"]
    lifts = np.array([r["skill_lift_pp"] for r in safe], dtype=float)
    sig = [r["significance_weather_vs_no_weather"]["signifikant_5pct"] for r in safe]
    pos = int(np.sum(lifts > 0)) if len(lifts) else 0
    by_target = {}
    for target in sorted({r["target"] for r in safe}):
        subset = [r for r in safe if r["target"] == target]
        tlifts = np.array([r["skill_lift_pp"] for r in subset], dtype=float)
        by_target[target] = {
            "n": len(subset),
            "median_lift_pp": round(float(np.median(tlifts)), 2),
            "positive_lift": f"{int(np.sum(tlifts > 0))}/{len(subset)}",
            "significant_weather_better": f"{sum(r['significance_weather_vs_no_weather']['signifikant_5pct'] for r in subset)}/{len(subset)}",
        }
    out = {
        "question": "Leakage-safe weather lift on real weather-coupled load",
        "leakage_safe_records": len(safe),
        "leakage_safe_median_lift_pp": round(float(np.median(lifts)), 2) if len(lifts) else None,
        "leakage_safe_positive_lift": f"{pos}/{len(safe)}" if safe else "0/0",
        "leakage_safe_sign_test_p": binomial_one_sided_p(pos, len(safe)) if safe else None,
        "leakage_safe_significant_weather_better": f"{sum(sig)}/{len(safe)}" if safe else "0/0",
        "by_target": by_target,
        "upper_bound_t10": {
            "skill_lift_reported_pp": 17.9,
            "leakage_class": "perfect_foresight_upper_bound",
            "reverify": t10_reverify,
        },
        "one_sentence": (
            f"On HEAPO real heat-pump/household load, leakage-safe local Historical-Forecast weather changes "
            f"median skill by {round(float(np.median(lifts)), 2) if len(lifts) else 'n/a'} pp across {len(safe)} "
            "aggregation records; the old T10 +17.9% is a perfect-foresight upper bound, not a leakage-safe claim."
        ),
    }
    with (OUT_DIR / "t12_summary.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-households", type=int, default=40)
    ap.add_argument("--individual-count", type=int, default=8)
    ap.add_argument("--cluster-size", type=int, default=8)
    ap.add_argument("--cluster-count", type=int, default=4)
    ap.add_argument("--feeder-count", type=int, default=40)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t10 = verify_t10_weather()
    records = run_heapo(args)
    summary = summarize(records, t10)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
