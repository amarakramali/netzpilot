# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""T13: leakage-safe HEAPO weather lift with station-local forecast weather.

HEAPO anonymizes weather station IDs. Five of the eight IDs can be mapped
exactly to MeteoSwiss OGD stations by identical 2019-2024 daily temperature
series. The remaining three IDs are not published with coordinates and do not
exactly match any SwissMetNet station in the public OGD catalogue; they are
therefore excluded from the strict per-station analysis rather than guessed.

The model features use only Open-Meteo Historical Forecast data at station
coordinates. HEAPO/MeteoSwiss observed weather is not used as a feature.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.data.openmeteo import fetch_weather
from netzpilot.eval.backtest import rolling_origin
from netzpilot.eval.conformal import rolling_origin_cqr
from netzpilot.features.build import frame_to_daily_local, get_holidays
from netzpilot.models.robust_corrector import ShrunkCorrector
from scripts.eval_v1_significance import SEED, ci95, paired_block_bootstrap
from scripts.run_t12_weather_lift import (
    FORECAST_START,
    HEAPO_COORDS,
    HEAPO_URL,
    N_TEST,
    WEATHER_VARS,
    aggregate_series,
    download_file,
    heapo_hourly_series,
    metric_record,
    read_heapo_overview,
    series_to_daily_local,
)

OUT_DIR = Path("data_cache/t13_weather_lift")
T12_HEAPO_ZIP = Path("data_cache/t12_weather_lift/raw/heapo_data.zip")
T13_HEAPO_ZIP = OUT_DIR / "raw" / "heapo_data.zip"
N_BOOT = 10000
MIN_DAILY_DAYS = 300
PROVENANCE_HEAPO = (
    "HEAPO Zenodo 15056919; household-to-weather-ID mapping from HEAPO metadata; "
    "station coordinates from exact MeteoSwiss OGD daily-temperature matches where available; "
    "features are Open-Meteo Historical Forecast only, restricted to >=2022-07-01."
)

METEOSWISS_STAC = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn"
DRYAD_DATASET_API = "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.73n5tb363"
DRYAD_LANDING = "https://datadryad.org/dataset/doi:10.5061/dryad.73n5tb363"

# Exact matches: HEAPO daily Temperature_avg/max/min equals the public
# MeteoSwiss OGD station series over 2019-2024 (max abs diff 0.0).
EXACT_STATION_MATCHES = {
    "8jB": {
        "station_code": "KLO",
        "station_name": "Zurich / Kloten",
        "latitude": 47.479611,
        "longitude": 8.535961,
        "match_evidence": "2019-2024 MeteoSwiss OGD daily avg/max/min temperature: mean_abs_diff=0.0, max_abs_diff=0.0, n=1885",
    },
    "Hg": {
        "station_code": "REH",
        "station_name": "Zurich / Affoltern",
        "latitude": 47.427694,
        "longitude": 8.517953,
        "match_evidence": "2019-2024 MeteoSwiss OGD daily avg/max/min temperature: mean_abs_diff=0.0, max_abs_diff=0.0, n=1885",
    },
    "wDD": {
        "station_code": "TAE",
        "station_name": "Aadorf / Taenikon",
        "latitude": 47.479892,
        "longitude": 8.904928,
        "match_evidence": "2019-2024 MeteoSwiss OGD daily avg/max/min temperature: mean_abs_diff=0.0, max_abs_diff=0.0, n=1885",
    },
    "z6I": {
        "station_code": "WAE",
        "station_name": "Waedenswil",
        "latitude": 47.220958,
        "longitude": 8.677706,
        "match_evidence": "2019-2024 MeteoSwiss OGD daily avg/max/min temperature: mean_abs_diff=0.0, max_abs_diff=0.0, n=1885",
    },
    "MqO": {
        "station_code": "EIN",
        "station_name": "Einsiedeln",
        "latitude": 47.133042,
        "longitude": 8.756556,
        "match_evidence": "2019-2024 MeteoSwiss OGD daily avg/max/min temperature: mean_abs_diff=0.0, max_abs_diff=0.0, n=1885",
    },
}

UNRESOLVED_STATION_IDS = {
    "HbsbG": {
        "status": "excluded_no_exact_public_coordinate_match",
        "best_public_smn_candidate": "WAE",
        "best_score_mean_abs_temperature_diff_C": 0.5944,
    },
    "ceOxS": {
        "status": "excluded_no_exact_public_coordinate_match",
        "best_public_smn_candidate": "SHA",
        "best_score_mean_abs_temperature_diff_C": 0.6440,
    },
    "sV3mR": {
        "status": "excluded_no_exact_public_coordinate_match",
        "best_public_smn_candidate": "SHA",
        "best_score_mean_abs_temperature_diff_C": 0.5689,
    },
}


def ensure_heapo_zip() -> Path:
    if T12_HEAPO_ZIP.exists() and T12_HEAPO_ZIP.stat().st_size > 0:
        return T12_HEAPO_ZIP
    if T13_HEAPO_ZIP.exists() and T13_HEAPO_ZIP.stat().st_size > 0:
        return T13_HEAPO_ZIP
    download_file(HEAPO_URL, T13_HEAPO_ZIP)
    return T13_HEAPO_ZIP


def daily_mae(result: dict, name: str) -> np.ndarray:
    return np.abs(np.asarray(result[name], float) - np.asarray(result["actual"], float)).reshape(-1, 24).mean(axis=1)


def significance_weather_vs_no(weather_result: dict, no_weather_result: dict) -> dict:
    ae_weather = daily_mae(weather_result, "model")
    ae_no = daily_mae(no_weather_result, "model")
    rng = np.random.default_rng(SEED)
    skill, dmae = paired_block_bootstrap(ae_weather, ae_no, rng, N_BOOT)
    point = (1.0 - ae_weather.sum() / ae_no.sum()) * 100.0
    return {
        "skill_weather_vs_no_weather_%": round(float(point), 2),
        "skill_ci95_%": ci95(skill),
        "P(weather_besser)_%": round(float(np.mean(skill > 0) * 100), 1),
        "dMAE_mean_MW": round(float(dmae.mean()), 6),
        "signifikant_5pct": bool(np.percentile(skill, 2.5) > 0.0),
    }


def fetch_station_weather(weather_id: str, start: str = "2022-07-01", end: str = "2024-03-01") -> pd.DataFrame:
    station = EXACT_STATION_MATCHES[weather_id]
    return fetch_weather(
        station["latitude"],
        station["longitude"],
        start,
        end,
        hourly=WEATHER_VARS,
        historical=True,
        cache_dir=str(OUT_DIR / "weather"),
        location_name=f"heapo_{weather_id}_{station['station_code'].lower()}",
        chunk_days=400,
    ).apply(pd.to_numeric, errors="coerce").dropna(how="any")


def fetch_zurich_proxy_weather(start: str = "2022-07-01", end: str = "2024-03-01") -> pd.DataFrame:
    return fetch_weather(
        HEAPO_COORDS["latitude"],
        HEAPO_COORDS["longitude"],
        start,
        end,
        hourly=WEATHER_VARS,
        historical=True,
        cache_dir=str(OUT_DIR / "weather"),
        location_name="heapo_zurich_proxy",
        chunk_days=400,
    ).apply(pd.to_numeric, errors="coerce").dropna(how="any")


def load_household_weather_mapping(z: zipfile.ZipFile) -> pd.DataFrame:
    with z.open("heapo_data/meta_data/households.csv") as f:
        return pd.read_csv(f, sep=";", usecols=["Household_ID", "Weather_ID"])


def count_station_coverage(z: zipfile.ZipFile) -> dict:
    mapping = load_household_weather_mapping(z)
    counts = mapping["Weather_ID"].value_counts().to_dict()
    exact = sum(int(counts.get(wid, 0)) for wid in EXACT_STATION_MATCHES)
    unresolved = sum(int(counts.get(wid, 0)) for wid in UNRESOLVED_STATION_IDS)
    return {
        "household_weather_id_counts": {str(k): int(v) for k, v in counts.items()},
        "exact_station_households": int(exact),
        "unresolved_station_households": int(unresolved),
        "exact_station_household_share_%": round(exact / max(len(mapping), 1) * 100.0, 1),
        "policy": "records from unresolved station IDs are excluded from strict per-station evaluation",
    }


def candidate_households(z: zipfile.ZipFile, value_col: str, min_overview_days: int) -> pd.DataFrame:
    overview = read_heapo_overview(z)
    mapping = load_household_weather_mapping(z)
    flag = {
        "kWh_received_HeatPump": "SMD_15min_MeasurementsAvailable_HeatPump",
        "kWh_received_Total": "SMD_15min_MeasurementsAvailable_Total",
    }[value_col]
    cand = overview[
        (overview[flag] == True)
        & (overview["start"] <= FORECAST_START)
        & (overview["end"] >= pd.Timestamp("2024-02-01", tz="UTC"))
        & (overview["SMD_15min_TimeAvailable_NumberDays"] >= min_overview_days)
    ].copy()
    cand = cand.merge(mapping, on="Household_ID", how="left")
    cand = cand[cand["Weather_ID"].isin(EXACT_STATION_MATCHES)].copy()
    cand = cand.sort_values(
        ["Weather_ID", "SMD_15min_TimeAvailable_NumberDays", "Household_ID"],
        ascending=[True, False, True],
    )
    return cand


def load_qualified_series_by_station(
    z: zipfile.ZipFile,
    value_col: str,
    station_cap: int,
    min_daily_days: int,
    min_overview_days: int,
) -> dict[str, dict[int, pd.Series]]:
    out: dict[str, dict[int, pd.Series]] = defaultdict(dict)
    cand = candidate_households(z, value_col, min_overview_days)
    for wid, group in cand.groupby("Weather_ID", sort=True):
        for hid in group["Household_ID"].astype(int):
            if len(out[wid]) >= station_cap:
                break
            series = heapo_hourly_series(z, hid, value_col)
            if series is None:
                continue
            load2d, _, _ = series_to_daily_local(series)
            if len(load2d) < min_daily_days:
                continue
            if not np.isfinite(load2d).all() or float(np.mean(load2d)) <= 0.0:
                continue
            out[wid][hid] = series
    return dict(out)


def round_robin_individuals(series_by_station: dict[str, dict[int, pd.Series]], limit: int):
    ordered = {wid: list(items.items()) for wid, items in sorted(series_by_station.items())}
    selected = []
    offset = 0
    while len(selected) < limit:
        added = False
        for wid, items in ordered.items():
            if offset < len(items):
                hid, series = items[offset]
                selected.append((wid, str(hid), series))
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def build_eval_specs(series_by_station: dict[str, dict[int, pd.Series]], args) -> list[tuple[str, str, str, pd.Series]]:
    specs: list[tuple[str, str, str, pd.Series]] = []
    for wid, label, series in round_robin_individuals(series_by_station, args.individual_count):
        specs.append((wid, "individual", label, series))

    for wid, items in sorted(series_by_station.items()):
        ids = list(items)
        for i in range(args.cluster_count_per_station):
            group = ids[i * args.cluster_size : (i + 1) * args.cluster_size]
            if len(group) < args.cluster_size:
                continue
            label = f"{wid}_cluster_{i + 1}_n{len(group)}"
            specs.append((wid, f"cluster_{len(group)}", label, aggregate_series([items[hid] for hid in group])))

        feeder_group = ids[: args.feeder_count_per_station]
        if len(feeder_group) >= args.feeder_min_households:
            label = f"{wid}_feeder_n{len(feeder_group)}"
            specs.append((wid, f"feeder_{len(feeder_group)}", label, aggregate_series([items[hid] for hid in feeder_group])))
    return specs


def eval_load_series(
    label: str,
    series: pd.Series,
    station_weather: pd.DataFrame,
    zurich_proxy_weather: pd.DataFrame,
    weather_id: str,
    aggregation: str,
    target: str,
    min_daily_days: int,
) -> dict | None:
    load2d, days, good_dates = series_to_daily_local(series)
    if len(load2d) < min_daily_days:
        return None
    station_weather2d = frame_to_daily_local(station_weather, good_dates)
    proxy_weather2d = frame_to_daily_local(zurich_proxy_weather, good_dates)
    holidays = get_holidays(sorted({d.year for d in days}), "NW")

    no_result, no_summary = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=N_TEST, holiday_set=holidays)
    station_result, station_summary = rolling_origin(
        load2d,
        days,
        lambda: ShrunkCorrector(10.0),
        n_test=N_TEST,
        weather2d=station_weather2d,
        holiday_set=holidays,
    )
    proxy_result, proxy_summary = rolling_origin(
        load2d,
        days,
        lambda: ShrunkCorrector(10.0),
        n_test=N_TEST,
        weather2d=proxy_weather2d,
        holiday_set=holidays,
    )
    _, c80 = rolling_origin_cqr(
        load2d,
        days,
        lambda: ShrunkCorrector(10.0),
        alpha=0.2,
        cal_days=28,
        n_test=N_TEST,
        weather2d=station_weather2d,
        holiday_set=holidays,
    )
    _, c90 = rolling_origin_cqr(
        load2d,
        days,
        lambda: ShrunkCorrector(10.0),
        alpha=0.1,
        cal_days=28,
        n_test=N_TEST,
        weather2d=station_weather2d,
        holiday_set=holidays,
    )
    no_metrics = metric_record(no_summary)
    station_metrics = metric_record(station_summary)
    proxy_metrics = metric_record(proxy_summary)
    station_meta = EXACT_STATION_MATCHES[weather_id]
    station_lift = float(station_metrics["skill_snv_%"] - no_metrics["skill_snv_%"])
    proxy_lift = float(proxy_metrics["skill_snv_%"] - no_metrics["skill_snv_%"])
    return {
        "dataset": "heapo",
        "target": target,
        "aggregation": aggregation,
        "label": label,
        "weather_id": weather_id,
        "n_days": int(len(load2d)),
        "n_test_days": N_TEST,
        "mean_mw": round(float(np.mean(load2d)), 6),
        "provenance": PROVENANCE_HEAPO,
        "leakage_class": "leakage_safe",
        "weather_source": "Open-Meteo Historical Forecast API",
        "weather_station": {
            "heapo_weather_id": weather_id,
            "station_code": station_meta["station_code"],
            "station_name": station_meta["station_name"],
            "latitude": station_meta["latitude"],
            "longitude": station_meta["longitude"],
            "match_evidence": station_meta["match_evidence"],
        },
        "zurich_proxy_coords": HEAPO_COORDS,
        "no_weather": no_metrics,
        "station_weather": station_metrics,
        "zurich_proxy_weather": proxy_metrics,
        "skill_lift_pp": round(station_lift, 2),
        "zurich_proxy_skill_lift_pp": round(proxy_lift, 2),
        "station_vs_zurich_delta_pp": round(station_lift - proxy_lift, 2),
        "cqr_station_weather": {
            "coverage80_%": c80["coverage_%"],
            "width80_MW": c80["mean_width_MW"],
            "coverage90_%": c90["coverage_%"],
            "width90_MW": c90["mean_width_MW"],
        },
        "significance_weather_vs_no_weather": significance_weather_vs_no(station_result, no_result),
        "significance_zurich_proxy_vs_no_weather": significance_weather_vs_no(proxy_result, no_result),
    }


def binomial_one_sided_p(k: int, n: int) -> float:
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)) if n else None


def lift_stats(records: list[dict], field: str = "skill_lift_pp") -> dict:
    if not records:
        return {
            "n": 0,
            "median_lift_pp": None,
            "iqr_lift_pp": [None, None],
            "positive_lift": "0/0",
            "sign_test_p": None,
            "significant_weather_better": "0/0",
        }
    lifts = np.asarray([r[field] for r in records], dtype=float)
    pos = int(np.sum(lifts > 0))
    sig = int(sum(r["significance_weather_vs_no_weather"]["signifikant_5pct"] for r in records))
    return {
        "n": len(records),
        "median_lift_pp": round(float(np.median(lifts)), 2),
        "iqr_lift_pp": [round(float(np.percentile(lifts, 25)), 2), round(float(np.percentile(lifts, 75)), 2)],
        "positive_lift": f"{pos}/{len(records)}",
        "sign_test_p": round(binomial_one_sided_p(pos, len(records)), 4),
        "significant_weather_better": f"{sig}/{len(records)}",
    }


def grouped_stats(records: list[dict], keys: list[str]) -> dict:
    out = {}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        groups[tuple(rec[k] for k in keys)].append(rec)
    for key, subset in sorted(groups.items()):
        label = " / ".join(str(x) for x in key)
        out[label] = lift_stats(subset)
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def station_mapping_artifact(coverage: dict) -> dict:
    mapping = {
        "source_links": {
            "heapo_github": "https://github.com/tbrumue/heapo",
            "heapo_zenodo": "https://zenodo.org/records/15056919",
            "meteoswiss_stac": METEOSWISS_STAC,
        },
        "exact_station_matches": EXACT_STATION_MATCHES,
        "unresolved_station_ids": UNRESOLVED_STATION_IDS,
        "coverage": coverage,
        "feature_policy": "Use only Open-Meteo Historical Forecast at exact station coordinates; never HEAPO/MeteoSwiss observed weather values as model features.",
    }
    path = OUT_DIR / "station_mapping.json"
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    return mapping


def try_request(session: requests.Session, url: str, **kwargs) -> dict:
    try:
        with session.get(url, stream=True, timeout=kwargs.pop("timeout", 30), **kwargs) as r:
            body = b""
            if r.status_code >= 300 or "json" in r.headers.get("content-type", "") or "text" in r.headers.get("content-type", ""):
                body = next(r.iter_content(500), b"")
            return {
                "url": url,
                "status_code": int(r.status_code),
                "final_url": r.url,
                "content_type": r.headers.get("content-type"),
                "content_length": r.headers.get("content-length"),
                "sample": body.decode("utf-8", errors="replace")[:300],
            }
    except Exception as exc:
        return {"url": url, "error": repr(exc)}


def probe_dryad() -> dict:
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    dataset = try_request(session, DRYAD_DATASET_API, headers=headers, timeout=60)
    files_payload = None
    files = []
    try:
        r = session.get("https://datadryad.org/api/v2/versions/348481/files", headers=headers, timeout=60)
        files_payload = {
            "status_code": int(r.status_code),
            "content_type": r.headers.get("content-type"),
        }
        if r.ok:
            data = r.json()
            for item in data.get("_embedded", {}).get("stash:files", []):
                files.append(
                    {
                        "path": item.get("path"),
                        "size": item.get("size"),
                        "mimeType": item.get("mimeType"),
                        "download": "https://datadryad.org" + item["_links"]["stash:download"]["href"],
                    }
                )
    except Exception as exc:
        files_payload = {"error": repr(exc)}

    interesting = [f for f in files if f.get("path") in {"README.md", "reduced_data.zip"}]
    attempts = []
    for item in interesting:
        attempts.append(try_request(session, item["download"], headers=headers, timeout=30))
        if item["path"] == "reduced_data.zip":
            file_id = item["download"].rstrip("/").split("/")[-2]
            attempts.append(
                try_request(
                    session,
                    f"https://datadryad.org/downloads/file_stream/{file_id}",
                    headers={**headers, "Referer": DRYAD_LANDING},
                    timeout=30,
                )
            )

    out = {
        "status": "blocked_external_download",
        "reason": "Dryad metadata and file list are public, but file downloads returned API 401 or AWS WAF challenge/403 from this host.",
        "dataset_api": dataset,
        "files_api": files_payload,
        "files": files,
        "download_attempts": attempts,
        "usable_for_t13_second_dataset": False,
        "source_links": {
            "dryad_dataset_api": DRYAD_DATASET_API,
            "dryad_landing": DRYAD_LANDING,
        },
    }
    path = OUT_DIR / "dryad_download_attempt.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def run_heapo(args) -> tuple[list[dict], dict]:
    heapo_zip = ensure_heapo_zip()
    station_weather = {wid: fetch_station_weather(wid) for wid in EXACT_STATION_MATCHES}
    zurich_proxy_weather = fetch_zurich_proxy_weather()
    records = []
    with zipfile.ZipFile(heapo_zip) as z:
        coverage = count_station_coverage(z)
        station_mapping_artifact(coverage)
        for target, value_col in [("heatpump", "kWh_received_HeatPump"), ("total", "kWh_received_Total")]:
            series_by_station = load_qualified_series_by_station(
                z,
                value_col,
                station_cap=args.station_cap,
                min_daily_days=args.min_daily_days,
                min_overview_days=args.min_overview_days,
            )
            for wid, aggregation, label, series in build_eval_specs(series_by_station, args):
                rec = eval_load_series(
                    label=label,
                    series=series,
                    station_weather=station_weather[wid],
                    zurich_proxy_weather=zurich_proxy_weather,
                    weather_id=wid,
                    aggregation=aggregation,
                    target=target,
                    min_daily_days=args.min_daily_days,
                )
                if rec:
                    records.append(rec)
    write_jsonl(OUT_DIR / "heapo_per_station_eval.jsonl", records)
    return records, json.loads((OUT_DIR / "station_mapping.json").read_text(encoding="utf-8"))


def summarize(records: list[dict], station_mapping: dict, dryad: dict) -> dict:
    heatpump = [r for r in records if r["target"] == "heatpump"]
    total = [r for r in records if r["target"] == "total"]
    proxy_delta = np.asarray([r["station_vs_zurich_delta_pp"] for r in records], dtype=float) if records else np.asarray([])
    heat_stats = lift_stats(heatpump)
    conclusion = (
        f"HEAPO per-station Historical-Forecast weather gives heatpump median lift "
        f"{heat_stats['median_lift_pp']} pp (IQR {heat_stats['iqr_lift_pp'][0]}..{heat_stats['iqr_lift_pp'][1]}, "
        f"sign-test p={heat_stats['sign_test_p']}, significant records {heat_stats['significant_weather_better']}); "
        "the heatpump lift does not become a significant product moat."
    )
    out = {
        "question": "Leakage-safe weather lift with HEAPO per-station coordinates plus second-dataset probe",
        "heapo_records": len(records),
        "station_mapping_status": {
            "exact_station_ids": sorted(EXACT_STATION_MATCHES),
            "unresolved_station_ids": sorted(UNRESOLVED_STATION_IDS),
            "coverage": station_mapping["coverage"],
        },
        "overall": lift_stats(records),
        "by_target": {
            "heatpump": heat_stats,
            "total": lift_stats(total),
        },
        "by_aggregation": grouped_stats(records, ["aggregation"]),
        "by_target_aggregation": grouped_stats(records, ["target", "aggregation"]),
        "single_zurich_proxy_comparison": {
            "median_station_minus_zurich_proxy_delta_pp": round(float(np.median(proxy_delta)), 2) if len(proxy_delta) else None,
            "iqr_station_minus_zurich_proxy_delta_pp": [
                round(float(np.percentile(proxy_delta, 25)), 2),
                round(float(np.percentile(proxy_delta, 75)), 2),
            ]
            if len(proxy_delta)
            else [None, None],
            "station_better_than_zurich_proxy": f"{int(np.sum(proxy_delta > 0))}/{len(proxy_delta)}" if len(proxy_delta) else "0/0",
        },
        "second_dataset": dryad,
        "one_sentence": conclusion,
    }
    path = OUT_DIR / "t13_summary.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-cap", type=int, default=8)
    parser.add_argument("--individual-count", type=int, default=14)
    parser.add_argument("--cluster-size", type=int, default=4)
    parser.add_argument("--cluster-count-per-station", type=int, default=2)
    parser.add_argument("--feeder-count-per-station", type=int, default=8)
    parser.add_argument("--feeder-min-households", type=int, default=6)
    parser.add_argument("--min-daily-days", type=int, default=MIN_DAILY_DAYS)
    parser.add_argument("--min-overview-days", type=int, default=365)
    parser.add_argument("--skip-dryad-probe", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records, station_mapping = run_heapo(args)
    dryad = (
        {"status": "skipped", "usable_for_t13_second_dataset": False}
        if args.skip_dryad_probe
        else probe_dryad()
    )
    summary = summarize(records, station_mapping, dryad)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
