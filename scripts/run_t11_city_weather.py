# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Evaluate local Historical-Forecast weather features on synthetic city profiles.

T11 validates the pipeline on modelled training-city profiles. These files are
not measured Stadtwerke load data; all outputs carry that caveat explicitly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_v1_significance import SEED, ci95, paired_block_bootstrap
from netzpilot.data.openmeteo import fetch_weather
from netzpilot.eval.backtest import rolling_origin
from netzpilot.eval.conformal import rolling_origin_cqr
from netzpilot.features.build import frame_to_daily_local, get_holidays, to_daily_local
from netzpilot.models.robust_corrector import ShrunkCorrector

DATA_DIR = Path("netzpilot/data/training_cities")
OUT_JSONL = Path("data_cache/cities_weather_eval.jsonl")
SUMMARY_JSON = Path("data_cache/cities_weather_summary.json")
COORDS_JSON = Path("data_cache/city_coords.json")
WEATHER_DIR = Path("data_cache/city_weather")
KEEP_DAYS = 110
N_TEST = 14
N_BOOT = 10000
WEATHER_VARS = ["temperature_2m", "shortwave_radiation", "wind_speed_100m", "cloud_cover", "direct_radiation"]
PROVENANCE_CAVEAT = (
    "training_cities are modelled/synthetic profiles, not measured Stadtwerke load data; "
    "T11 validates the weather-feature pipeline, not real-world city performance."
)

CITY_QUERY_NAMES = {
    "Duesseldorf": "Düsseldorf",
    "Koeln": "Köln",
    "Luebeck": "Lübeck",
    "Moenchengladbach": "Mönchengladbach",
    "Muelheim": "Mülheim",
    "Muenchen": "München",
    "Muenster": "Münster",
    "Nuernberg": "Nürnberg",
    "Osnabrueck": "Osnabrück",
    "Saarbruecken": "Saarbrücken",
    "Frankfurt": "Frankfurt am Main",
    "Halle": "Halle (Saale)",
}


def city_query_name(city: str) -> str:
    return CITY_QUERY_NAMES.get(city, city)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_coord_cache(path: Path = COORDS_JSON) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_coord_cache(cache: dict, path: Path = COORDS_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def geocode_city(city: str, cache: dict) -> dict:
    if city in cache:
        return cache[city]
    query = city_query_name(city)
    params = {"name": query, "country": "DE", "count": 1, "language": "de", "format": "json"}
    for attempt in range(4):
        try:
            r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=params, timeout=30)
            if r.status_code == 200:
                payload = r.json()
                results = payload.get("results") or []
                if results:
                    result = results[0]
                    rec = {
                        "city": city,
                        "query": query,
                        "name": result.get("name"),
                        "admin1": result.get("admin1"),
                        "country_code": result.get("country_code"),
                        "latitude": float(result["latitude"]),
                        "longitude": float(result["longitude"]),
                        "source": "Open-Meteo Geocoding API",
                    }
                    cache[city] = rec
                    save_coord_cache(cache)
                    return rec
            last = f"HTTP {r.status_code}: {r.text[:160]}"
        except requests.RequestException as exc:
            last = repr(exc)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Geocoding failed for {city} ({query}): {last}")


def load_city_load(path: Path, keep_days: int = KEEP_DAYS):
    df = pd.read_csv(path, usecols=["timestamp", "load_mw"])
    idx = pd.to_datetime(df["timestamp"])
    series = pd.Series(df["load_mw"].astype(float).to_numpy(), index=idx).sort_index().resample("1h").mean()
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    load2d, days, good_dates = to_daily_local(series)
    return load2d[-keep_days:], days[-keep_days:], good_dates[-keep_days:], series


def load_baseline_refs(path: Path) -> dict:
    refs = {}
    if not path.exists():
        return refs
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                refs[rec["city"]] = rec
    return refs


def daily_mae(arrays: dict, name: str) -> np.ndarray:
    ae = np.abs(arrays[name] - arrays["actual"])
    return ae.reshape(-1, 24).mean(axis=1)


def sig_weather_vs_no_weather(weather_arrays: dict, no_weather_arrays: dict) -> dict:
    ae_weather = daily_mae(weather_arrays, "model")
    ae_no_weather = daily_mae(no_weather_arrays, "model")
    rng = np.random.default_rng(SEED)
    skill, dmae = paired_block_bootstrap(ae_weather, ae_no_weather, rng, N_BOOT)
    skill_point = (1.0 - ae_weather.sum() / ae_no_weather.sum()) * 100.0
    return {
        "skill_weather_vs_no_weather_%": round(float(skill_point), 2),
        "skill_ci95_%": ci95(skill),
        "P(weather_besser)_%": round(float(np.mean(skill > 0) * 100), 1),
        "dMAE_mean_MW": round(float(dmae.mean()), 3),
        "dMAE_ci95_MW": [
            round(float(np.percentile(dmae, 2.5)), 3),
            round(float(np.percentile(dmae, 97.5)), 3),
        ],
        "signifikant_5pct": bool(np.percentile(skill, 2.5) > 0.0),
    }


def binomial_one_sided_p(k: int, n: int) -> float:
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n))


def summarize_records(records: list[dict]) -> dict:
    if not records:
        return {"n_cities": 0, "provenance_caveat": PROVENANCE_CAVEAT}
    lifts_cached = np.array([r["skill_lift_vs_cached_no_weather_pp"] for r in records], dtype=float)
    lifts_same = np.array([r["skill_lift_same_protocol_pp"] for r in records], dtype=float)
    weather_skill = np.array([r["weather"]["skill_snv_%"] for r in records], dtype=float)
    cov80 = np.array([r["weather_cqr"]["coverage80_%"] for r in records], dtype=float)
    cov90 = np.array([r["weather_cqr"]["coverage90_%"] for r in records], dtype=float)
    sig = [r["significance_weather_vs_no_weather"]["signifikant_5pct"] for r in records]
    pos_cached = int(np.sum(lifts_cached > 0))
    pos_same = int(np.sum(lifts_same > 0))
    return {
        "n_cities": len(records),
        "provenance_caveat": PROVENANCE_CAVEAT,
        "weather_source": "Open-Meteo Historical Forecast API",
        "leakage_check": "Hourly weather uses historical-forecast endpoint and is converted with frame_to_daily_local; no reanalysis or actual weather is used.",
        "median_weather_skill_snv_%": round(float(np.median(weather_skill)), 2),
        "median_skill_lift_vs_cached_no_weather_pp": round(float(np.median(lifts_cached)), 2),
        "median_skill_lift_same_protocol_pp": round(float(np.median(lifts_same)), 2),
        "positive_lift_vs_cached_no_weather": f"{pos_cached}/{len(records)}",
        "positive_lift_same_protocol": f"{pos_same}/{len(records)}",
        "sign_test_p_lift_vs_cached_no_weather": binomial_one_sided_p(pos_cached, len(records)),
        "sign_test_p_lift_same_protocol": binomial_one_sided_p(pos_same, len(records)),
        "median_coverage80_%": round(float(np.median(cov80)), 2),
        "median_coverage90_%": round(float(np.median(cov90)), 2),
        "significant_weather_better_cities": f"{int(sum(sig))}/{len(records)}",
        "n_boot_per_city": N_BOOT,
        "seed": SEED,
    }


def evaluate_city(path: Path, coord_cache: dict, baseline_refs: dict, args) -> dict:
    city = path.name.replace("_Netz_Lastgang_2024.csv", "")
    load2d, days, good_dates, series = load_city_load(path, args.keep_days)
    coord = geocode_city(city, coord_cache)
    start = (series.index.min().tz_convert("UTC").date() - pd.Timedelta(days=1)).isoformat()
    end = (series.index.max().tz_convert("UTC").date() + pd.Timedelta(days=1)).isoformat()
    weather = fetch_weather(
        coord["latitude"],
        coord["longitude"],
        start,
        end,
        hourly=WEATHER_VARS,
        historical=True,
        cache_dir=str(WEATHER_DIR),
        location_name=safe_name(city),
        chunk_days=400,
    ).apply(pd.to_numeric, errors="coerce").dropna(how="any")
    weather2d_all = frame_to_daily_local(weather, good_dates)
    weather2d = weather2d_all[-args.keep_days:]
    hol = get_holidays(sorted({d.year for d in days}), "NW")

    no_R, no_summary = rolling_origin(
        load2d, days, lambda: ShrunkCorrector(10.0), n_test=args.n_test, holiday_set=hol
    )
    wx_R, wx_summary = rolling_origin(
        load2d, days, lambda: ShrunkCorrector(10.0), n_test=args.n_test, weather2d=weather2d, holiday_set=hol
    )
    _, cqr80 = rolling_origin_cqr(
        load2d, days, lambda: ShrunkCorrector(10.0), alpha=0.2, cal_days=args.cal_days,
        n_test=args.n_test, weather2d=weather2d, holiday_set=hol, online=True, per_hour=False,
    )
    _, cqr90 = rolling_origin_cqr(
        load2d, days, lambda: ShrunkCorrector(10.0), alpha=0.1, cal_days=args.cal_days,
        n_test=args.n_test, weather2d=weather2d, holiday_set=hol, online=True, per_hour=False,
    )
    no_m = no_summary["metriken"]["model"]
    wx_m = wx_summary["metriken"]["model"]
    cached = baseline_refs.get(city, {})
    cached_skill = cached.get("skill_snv_%", no_m["Skill_vs_SaisonalNaiv_%"])
    rec = {
        "city": city,
        "mean_mw": round(float(load2d.mean()), 1),
        "coords": coord,
        "provenance_caveat": PROVENANCE_CAVEAT,
        "weather_source": "Open-Meteo Historical Forecast API",
        "weather_variables": WEATHER_VARS,
        "leakage_check": "Weather endpoint is historical-forecast; daily arrays are built with frame_to_daily_local and only forecast fields are used.",
        "cached_no_weather": {
            "MAE_MW": cached.get("MAE_MW"),
            "MAPE_%": cached.get("MAPE_%"),
            "skill_snv_%": cached_skill,
            "skill_pers_%": cached.get("skill_pers_%"),
        },
        "no_weather_same_protocol": {
            "MAE_MW": no_m["MAE_MW"],
            "MAPE_%": no_m["MAPE_%"],
            "skill_snv_%": no_m["Skill_vs_SaisonalNaiv_%"],
            "skill_pers_%": no_m["Skill_vs_Persistenz_%"],
        },
        "weather": {
            "MAE_MW": wx_m["MAE_MW"],
            "MAPE_%": wx_m["MAPE_%"],
            "skill_snv_%": wx_m["Skill_vs_SaisonalNaiv_%"],
            "skill_pers_%": wx_m["Skill_vs_Persistenz_%"],
        },
        "skill_lift_vs_cached_no_weather_pp": round(float(wx_m["Skill_vs_SaisonalNaiv_%"] - cached_skill), 2),
        "skill_lift_same_protocol_pp": round(float(wx_m["Skill_vs_SaisonalNaiv_%"] - no_m["Skill_vs_SaisonalNaiv_%"]), 2),
        "weather_cqr": {
            "coverage80_%": cqr80["coverage_%"],
            "width80_MW": cqr80["mean_width_MW"],
            "coverage90_%": cqr90["coverage_%"],
            "width90_MW": cqr90["mean_width_MW"],
            "cal_days": args.cal_days,
        },
        "significance_weather_vs_no_weather": sig_weather_vs_no_weather(wx_R, no_R),
    }
    return rec


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(OUT_JSONL))
    ap.add_argument("--summary-out", default=str(SUMMARY_JSON))
    ap.add_argument("--baseline", default="data_cache/cities_all_load.jsonl")
    ap.add_argument("--keep-days", type=int, default=KEEP_DAYS)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--cal-days", type=int, default=21)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    summary_out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.force:
        out.unlink(missing_ok=True)
        summary_out.unlink(missing_ok=True)

    records = read_jsonl(out)
    done = {r["city"] for r in records}
    files = sorted(Path(args.data_dir).glob("*_Netz_Lastgang_2024.csv"))
    todo = [p for p in files if p.name.replace("_Netz_Lastgang_2024.csv", "") not in done]
    if args.limit:
        todo = todo[:args.limit]
    coord_cache = load_coord_cache()
    baseline_refs = load_baseline_refs(Path(args.baseline))

    print(f"done={len(done)} todo={len(todo)} total_files={len(files)}", flush=True)
    with out.open("a", encoding="utf-8") as f:
        for path in todo:
            rec = evaluate_city(path, coord_cache, baseline_refs, args)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            records.append(rec)
            print(
                f"{rec['city']:16s} wx_skill={rec['weather']['skill_snv_%']:+6.2f}% "
                f"lift_cached={rec['skill_lift_vs_cached_no_weather_pp']:+6.2f}pp "
                f"c80={rec['weather_cqr']['coverage80_%']:5.1f}% "
                f"sig={rec['significance_weather_vs_no_weather']['signifikant_5pct']}",
                flush=True,
            )

    summary = summarize_records(records)
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
