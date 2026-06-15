#!/usr/bin/env python3
"""Normalize ENTSO-E Power Statistics national hourly load values for T30.

This uses the official public ENTSO-E Power Statistics CSV, not the tokened REST API.
It is still national aggregate TSO load and must remain in data_cache/intl, never in
the DSO corpus or pool prior.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

INTL = "data_cache/intl"
YEAR = 2024
POWER_STATS_PAGE = "https://www.entsoe.eu/data/power-stats/"
RAW_URL = "https://www.entsoe.eu/publications/data/power-stats/2024/monthly_hourly_load_values_2024.csv"
COUNTRIES = {
    "DE": "Germany",
    "NL": "Netherlands",
    "AT": "Austria",
    "CH": "Switzerland",
    "FR": "France",
}


def download_if_missing(url: str, path: str, force: bool = False) -> None:
    if os.path.exists(path) and not force:
        return
    with requests.get(url, timeout=(20, 300), stream=True) as response:
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=INTL)
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, f"monthly_hourly_load_values_{YEAR}.csv")
    download_if_missing(RAW_URL, raw_path, force=args.force_download)

    df = pd.read_csv(raw_path, sep="\t")
    source = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_page": POWER_STATS_PAGE,
        "raw_url": RAW_URL,
        "raw_path": raw_path,
        "label": "national aggregate TSO load (ENTSO-E Power Statistics), not distribution-network load",
        "entries": [],
    }
    for code, name in COUNTRIES.items():
        part = df[df["CountryCode"] == code].copy()
        if part.empty:
            raise RuntimeError(f"Country {code} not found in {raw_path}")
        part["timestamp_utc"] = pd.to_datetime(part["DateUTC"], format="%d-%m-%Y %H:%M", utc=True)
        part["load_mw"] = pd.to_numeric(part["Value"], errors="coerce")
        part["coverage_ratio"] = pd.to_numeric(part["Cov_ratio"], errors="coerce")
        part = part.sort_values("timestamp_utc")
        out = os.path.join(args.out_dir, f"entsoe_{code.lower()}_{YEAR}.csv")
        part[["timestamp_utc", "load_mw", "coverage_ratio", "CountryCode"]].to_csv(out, index=False)
        entry = {
            "country_code": code,
            "country": name,
            "year": YEAR,
            "path": out,
            "n_rows": int(len(part)),
            "expected_hourly_rows_leap_year": 8784,
            "min_coverage_ratio": float(part["coverage_ratio"].min()),
            "max_coverage_ratio": float(part["coverage_ratio"].max()),
            "source_page": POWER_STATS_PAGE,
            "raw_url": RAW_URL,
        }
        source["entries"].append(entry)
        print(f"[entsoe] {code} {name}: {entry['n_rows']} rows -> {out}")

    with open(os.path.join(args.out_dir, "entsoe_power_stats_source.json"), "w", encoding="utf-8") as f:
        json.dump(source, f, indent=2, ensure_ascii=False)

    sources_md = os.path.join(args.out_dir, "SOURCES.md")
    lines = [
        "# International National Load Sources (ENTSO-E Power Statistics)",
        "",
        f"Generated: {source['generated_utc']}",
        "",
        "All rows here are ENTSO-E national aggregate TSO load for method demonstration only.",
        "They are not distribution-network data, not city-utility data, and are never part of the DSO corpus or pool prior.",
        "",
        "| File | Country | Year | Granularity | Label | Official source |",
        "|---|---|---:|---|---|---|",
    ]
    for entry in source["entries"]:
        lines.append(
            f"| `{os.path.basename(entry['path'])}` | {entry['country']} ({entry['country_code']}) | "
            f"{entry['year']} | hourly, MW | national aggregate load (TSO) | {RAW_URL} |"
        )
    lines += [
        "",
        "Source page: https://www.entsoe.eu/data/power-stats/",
        "Raw file kept locally as `monthly_hourly_load_values_2024.csv`.",
        "France has fewer than 8784 rows in the official file; evaluation drops incomplete local days.",
    ]
    with open(sources_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
