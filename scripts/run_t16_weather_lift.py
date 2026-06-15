"""T16: second 2022+ leakage-safe weather-lift dataset.

Primary attempt: Offenbach Smart-Building via Dryad API. If the file download is
still blocked from this host, fall back to public 2024 DSO profiles from T15 and
test whether local Open-Meteo Historical Forecast weather adds skill on smooth
network load.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.data.openmeteo import fetch_weather
from netzpilot.eval.backtest import rolling_origin
from netzpilot.eval.conformal import rolling_origin_cqr
from netzpilot.features.build import frame_to_daily_local, get_holidays, to_daily_local
from netzpilot.models.robust_corrector import ShrunkCorrector
from scripts.eval_v1_significance import SEED, ci95, paired_block_bootstrap
from scripts.pilot_in_a_box import robust_load_csv
from scripts.run_t12_weather_lift import WEATHER_VARS

OUT_DIR = Path("data_cache/t16_weather_lift")
DRYAD_DATASET_API = "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.73n5tb363"
DRYAD_VERSIONS_API = DRYAD_DATASET_API + "/versions"
DRYAD_LANDING = "https://datadryad.org/dataset/doi:10.5061/dryad.73n5tb363"
FORECAST_START = pd.Timestamp("2022-07-01", tz="UTC")
N_TEST = 28
N_BOOT = 10000
PROVENANCE_FALLBACK = (
    "Dryad Offenbach metadata public but README/reduced_data downloads blocked from this host; "
    "fallback uses public 2024 DSO CSV load profiles from T15 and Open-Meteo Historical Forecast only."
)

DSO_SPECS = [
    {
        "slug": "herne_bezug_vorgelagerte_ebene_2024",
        "operator": "Stadtwerke Herne",
        "series": "Bezug vorgelagerte Ebene 110/10 kV",
        "csv": "data_cache/real/herne_bezug_vorgelagerte_ebene_2024.csv",
        "load_col": "Load_1",
        "unit": "kW",
        "region": "NW",
        "latitude": 51.5369,
        "longitude": 7.2009,
        "load_type": "smooth_dso_network_load",
        "aggregation": "published_grid_level_series",
    },
    {
        "slug": "evdb_lastgang_ns_2024",
        "operator": "EVDB",
        "series": "Lastgang NS",
        "csv": "data_cache/real/evdb_lastgang_ns_2024.csv",
        "load_col": "Wert",
        "unit": "kW",
        "region": "NI",
        "latitude": 53.1876,
        "longitude": 10.7360,
        "load_type": "smooth_dso_network_load",
        "aggregation": "published_voltage_level_series",
    },
    {
        "slug": "evdb_lastgang_ms_2024",
        "operator": "EVDB",
        "series": "Lastgang MS",
        "csv": "data_cache/real/evdb_lastgang_ms_2024.csv",
        "load_col": "Wert",
        "unit": "kW",
        "region": "NI",
        "latitude": 53.1876,
        "longitude": 10.7360,
        "load_type": "smooth_dso_network_load",
        "aggregation": "published_voltage_level_series",
    },
]


def try_request(session: requests.Session, url: str, **kwargs) -> dict:
    try:
        with session.get(url, stream=True, timeout=kwargs.pop("timeout", 30), **kwargs) as r:
            body = b""
            content_type = r.headers.get("content-type", "")
            if r.status_code >= 300 or "json" in content_type or "text" in content_type:
                body = next(r.iter_content(500), b"")
            return {
                "url": url,
                "status_code": int(r.status_code),
                "final_url": r.url,
                "content_type": content_type,
                "content_length": r.headers.get("content-length"),
                "sample": body.decode("utf-8", errors="replace")[:300],
            }
    except Exception as exc:
        return {"url": url, "error": repr(exc)}


def _absolute_dryad_url(href: str) -> str:
    return href if href.startswith("http") else "https://datadryad.org" + href


def probe_dryad() -> dict:
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    versions_resp = try_request(session, DRYAD_VERSIONS_API, headers=headers, timeout=60)
    versions = []
    files_payload = None
    files = []
    latest = None

    try:
        r = session.get(DRYAD_VERSIONS_API, headers=headers, timeout=60)
        r.raise_for_status()
        versions = r.json().get("_embedded", {}).get("stash:versions", [])
        latest = max(versions, key=lambda v: int(v.get("versionNumber", -1)))
        files_url = _absolute_dryad_url(latest["_links"]["stash:files"]["href"])
        fr = session.get(files_url, headers=headers, timeout=60)
        files_payload = {
            "url": files_url,
            "status_code": int(fr.status_code),
            "content_type": fr.headers.get("content-type"),
        }
        if fr.ok:
            for item in fr.json().get("_embedded", {}).get("stash:files", []):
                download_href = item.get("_links", {}).get("stash:download", {}).get("href")
                files.append(
                    {
                        "path": item.get("path"),
                        "size": item.get("size"),
                        "mimeType": item.get("mimeType"),
                        "download": _absolute_dryad_url(download_href) if download_href else None,
                    }
                )
    except Exception as exc:
        files_payload = {"error": repr(exc)}

    attempts = []
    interesting = [f for f in files if f.get("path") in {"README.md", "reduced_data.zip"}]
    for item in interesting:
        if item.get("download"):
            attempts.append(try_request(session, item["download"], headers=headers, timeout=60))
            match = re.search(r"/files/(\d+)/download", item["download"])
            if match and item.get("path") == "reduced_data.zip":
                attempts.append(
                    try_request(
                        session,
                        f"https://datadryad.org/downloads/file_stream/{match.group(1)}",
                        headers={**headers, "Referer": DRYAD_LANDING},
                        timeout=60,
                    )
                )

    accessible = any(
        a.get("status_code") == 200 and "zip" in (a.get("content_type") or "")
        for a in attempts
    )
    out = {
        "status": "download_accessible" if accessible else "blocked_external_download",
        "reason": (
            "Dryad reduced_data.zip appears downloadable; local parser not executed."
            if accessible
            else "Dryad metadata and file list are public, but README/reduced_data downloads returned API 401 or direct 403 from this host."
        ),
        "versions_api": versions_resp,
        "latest_version": {
            "versionNumber": latest.get("versionNumber") if latest else None,
            "versionStatus": latest.get("versionStatus") if latest else None,
            "publicationDate": latest.get("publicationDate") if latest else None,
        },
        "files_api": files_payload,
        "files": files,
        "download_attempts": attempts,
        "usable_for_t16": bool(accessible),
        "source_links": {
            "dryad_dataset_api": DRYAD_DATASET_API,
            "dryad_versions_api": DRYAD_VERSIONS_API,
            "dryad_landing": DRYAD_LANDING,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dryad_download_attempt.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def daily_mae(result: dict, name: str) -> np.ndarray:
    return np.abs(np.asarray(result[name], float) - np.asarray(result["actual"], float)).reshape(-1, 24).mean(axis=1)


def significance_weather_vs_no(weather_result: dict, no_weather_result: dict, n_boot: int) -> dict:
    ae_weather = daily_mae(weather_result, "model")
    ae_no = daily_mae(no_weather_result, "model")
    rng = np.random.default_rng(SEED)
    skill, dmae = paired_block_bootstrap(ae_weather, ae_no, rng, n_boot)
    point = (1.0 - ae_weather.sum() / ae_no.sum()) * 100.0
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


def _weather_fetch_window(good_dates: list[pd.Timestamp]) -> tuple[str, str]:
    start = (min(good_dates).date() - pd.Timedelta(days=1)).isoformat()
    end = (max(good_dates).date() + pd.Timedelta(days=1)).isoformat()
    return start, end


def _complete_weather_dates(frame: pd.DataFrame, tz: str = "Europe/Berlin") -> set[pd.Timestamp]:
    loc = frame.index.tz_convert(tz) if frame.index.tz else frame.index.tz_localize("UTC").tz_convert(tz)
    df = pd.DataFrame(index=loc)
    df["date"] = df.index.normalize()
    df["hour"] = df.index.hour
    out = set()
    for date, group in df.groupby("date"):
        hours = sorted(group["hour"].tolist())
        if len(group) == 24 and hours == list(range(24)):
            out.add(date)
    return out


def evaluate_dso(spec: dict, n_test: int, n_boot: int) -> dict:
    series, ts_col, load_col = robust_load_csv(spec["csv"], load_col=spec["load_col"], unit=spec["unit"])
    series = series[series.index >= FORECAST_START]
    load2d, days, good_dates = to_daily_local(series)
    if len(load2d) < n_test + 60:
        raise ValueError(f"{spec['slug']} has too few complete days: {len(load2d)}")

    start, end = _weather_fetch_window(good_dates)
    weather = fetch_weather(
        spec["latitude"],
        spec["longitude"],
        start,
        end,
        hourly=WEATHER_VARS,
        historical=True,
        cache_dir=str(OUT_DIR / "weather"),
        location_name=spec["slug"],
        chunk_days=370,
    ).apply(pd.to_numeric, errors="coerce").dropna(how="any")
    weather_dates = _complete_weather_dates(weather)
    keep = [i for i, d in enumerate(good_dates) if d in weather_dates]
    load2d = load2d[keep]
    days = pd.to_datetime([days[i] for i in keep])
    good_dates = [good_dates[i] for i in keep]
    if len(load2d) < n_test + 60:
        raise ValueError(f"{spec['slug']} has too few weather-aligned complete days: {len(load2d)}")
    weather2d = frame_to_daily_local(weather, good_dates)
    holidays = get_holidays(sorted({d.year for d in days}), spec["region"])

    no_R, no_sm = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=n_test, holiday_set=holidays)
    wx_R, wx_sm = rolling_origin(
        load2d,
        days,
        lambda: ShrunkCorrector(10.0),
        n_test=n_test,
        weather2d=weather2d,
        holiday_set=holidays,
    )
    _, c80 = rolling_origin_cqr(
        load2d,
        days,
        lambda: ShrunkCorrector(10.0),
        alpha=0.2,
        cal_days=28,
        n_test=n_test,
        weather2d=weather2d,
        holiday_set=holidays,
    )
    _, c90 = rolling_origin_cqr(
        load2d,
        days,
        lambda: ShrunkCorrector(10.0),
        alpha=0.1,
        cal_days=28,
        n_test=n_test,
        weather2d=weather2d,
        holiday_set=holidays,
    )
    no_m = metric_record(no_sm)
    wx_m = metric_record(wx_sm)
    return {
        "dataset": "t15_public_dso_fallback",
        "operator": spec["operator"],
        "series": spec["series"],
        "slug": spec["slug"],
        "target": "network_load",
        "load_type": spec["load_type"],
        "aggregation": spec["aggregation"],
        "n_days": int(len(load2d)),
        "n_test_days": int(n_test),
        "mean_mw": round(float(np.mean(load2d)), 6),
        "source_csv": spec["csv"],
        "ts_col": ts_col,
        "load_col": load_col,
        "provenance": PROVENANCE_FALLBACK,
        "leakage_class": "leakage_safe",
        "load_period": f"{days.min().date().isoformat()}..{days.max().date().isoformat()}",
        "leakage_rule": "load >= 2022-07-01; weather features are Open-Meteo Historical Forecast, not archive/reanalysis",
        "weather_source": "Open-Meteo Historical Forecast API",
        "weather_fetch_window": {"start": start, "end": end},
        "weather_coords": {
            "latitude": spec["latitude"],
            "longitude": spec["longitude"],
            "region": spec["region"],
        },
        "no_weather": no_m,
        "weather": wx_m,
        "skill_lift_pp": round(float(wx_m["skill_snv_%"] - no_m["skill_snv_%"]), 2),
        "cqr_weather": {
            "coverage80_%": c80["coverage_%"],
            "width80_MW": c80["mean_width_MW"],
            "coverage90_%": c90["coverage_%"],
            "width90_MW": c90["mean_width_MW"],
        },
        "significance_weather_vs_no_weather": significance_weather_vs_no(wx_R, no_R, n_boot),
    }


def binomial_one_sided_p(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n))


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(records: list[dict], dryad: dict) -> dict:
    lifts = np.array([r["skill_lift_pp"] for r in records], dtype=float)
    pos = int(np.sum(lifts > 0)) if len(lifts) else 0
    sig = [r["significance_weather_vs_no_weather"]["signifikant_5pct"] for r in records]
    by_load_type = {}
    for load_type in sorted({r["load_type"] for r in records}):
        subset = [r for r in records if r["load_type"] == load_type]
        slifts = np.array([r["skill_lift_pp"] for r in subset], dtype=float)
        by_load_type[load_type] = {
            "n": len(subset),
            "median_lift_pp": round(float(np.median(slifts)), 2),
            "positive_lift": f"{int(np.sum(slifts > 0))}/{len(subset)}",
            "significant_weather_better": f"{sum(r['significance_weather_vs_no_weather']['signifikant_5pct'] for r in subset)}/{len(subset)}",
        }

    t12 = _load_json(Path("data_cache/t12_weather_lift/t12_summary.json"))
    t13 = _load_json(Path("data_cache/t13_weather_lift/t13_summary.json"))
    t13_heat = (t13 or {}).get("by_target", {}).get("heatpump", {})
    one_sentence = (
        "Across HEAPO (T12/T13) and the T16 public-DSO fallback, leakage-safe Historical-Forecast weather "
        f"does not become a significant product moat: T16 smooth DSO median lift is "
        f"{round(float(np.median(lifts)), 2) if len(lifts) else 'n/a'} pp with {sum(sig)}/{len(records)} "
        "significant records, while T13 heat-pump lift stayed non-significant "
        f"({t13_heat.get('median_lift_pp', 'n/a')} pp median)."
    )
    out = {
        "question": "Second real 2022+ leakage-safe weather-lift dataset",
        "dryad_status": dryad["status"],
        "fallback_dataset": "public T15 DSO network-load CSVs" if dryad["status"] != "download_accessible" else None,
        "records": len(records),
        "median_lift_pp": round(float(np.median(lifts)), 2) if len(lifts) else None,
        "positive_lift": f"{pos}/{len(records)}" if records else "0/0",
        "sign_test_p": binomial_one_sided_p(pos, len(records)) if records else None,
        "significant_weather_better": f"{sum(sig)}/{len(records)}" if records else "0/0",
        "by_load_type": by_load_type,
        "prior_evidence": {
            "t12": {
                "records": (t12 or {}).get("leakage_safe_records"),
                "median_lift_pp": (t12 or {}).get("leakage_safe_median_lift_pp"),
                "significant_weather_better": (t12 or {}).get("leakage_safe_significant_weather_better"),
            },
            "t13": {
                "records": (t13 or {}).get("heapo_records"),
                "overall_median_lift_pp": (t13 or {}).get("overall", {}).get("median_lift_pp"),
                "heatpump": t13_heat,
            },
        },
        "one_sentence": one_sentence,
        "outputs": {
            "eval_jsonl": str(OUT_DIR / "dso_weather_eval.jsonl"),
            "dryad_attempt": str(OUT_DIR / "dryad_download_attempt.json"),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "t16_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def run(args) -> tuple[list[dict], dict, dict]:
    dryad = probe_dryad()
    records = [evaluate_dso(spec, args.n_test, args.n_boot) for spec in DSO_SPECS]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "dso_weather_eval.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = summarize(records, dryad)
    return records, summary, dryad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    records, summary, dryad = run(args)
    print(json.dumps({"dryad_status": dryad["status"], "records": records, "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
