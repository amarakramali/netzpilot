#!/usr/bin/env python3
"""Fetch public French regional load series for the real corpus.

Sources:
- Enedis conso-inf36-region: regional <=36 kVA DSO aggregate, 30 min.
- RTE/ODRE eco2mix regional consolidated/definitive: TSO regional load, 30 min.

The output CSVs are normalized to timestamp_utc/load_mw so the existing Pilot-in-a-Box
loader and benchmark suite can consume them with explicit pinned columns.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

REAL = "data_cache/real"
YEAR = 2024

ENEDIS_PAGE = "https://opendata.enedis.fr/datasets/conso-inf36-region"
ENEDIS_API = "https://opendata.enedis.fr/api/explore/v2.1/catalog/datasets/conso-inf36-region/records"
RTE_PAGE = "https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/"
RTE_API = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-regional-cons-def/records"

ENEDIS_REGIONS = [
    ("ile_de_france", "11", "Ile-de-France"),
    ("bourgogne_franche_comte", "27", "Bourgogne-Franche-Comte"),
]
RTE_REGIONS = [
    ("ile_de_france", "11", "Ile-de-France"),
]


def fetch_pages(url: str, params: dict, page_size: int = 10_000) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    urls: list[str] = []
    offset = 0
    total = None
    while total is None or offset < total:
        page_params = dict(params)
        page_params.update({"limit": page_size, "offset": offset})
        response = requests.get(url, params=page_params, timeout=90)
        response.raise_for_status()
        urls.append(response.url)
        payload = response.json()
        total = int(payload["total_count"])
        batch = payload.get("results", [])
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows, urls


def parse_timestamp(value):
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="ms", utc=True)
    return pd.to_datetime(value, utc=True)


def fetch_enedis_region(slug: str, code: str, label: str, year: int, out_dir: str) -> dict:
    where = (
        f"code_region='{code}' AND "
        "plage_de_puissance_souscrite='P0: Total <= 36 kVA' AND "
        f"horodate >= date'{year}-01-01' AND horodate < date'{year + 1}-01-01'"
    )
    params = {
        "select": "horodate,sum(total_energie_soutiree_wh) as energy_wh,count(*) as n_segments",
        "where": where,
        "group_by": "horodate",
        "order_by": "horodate",
    }
    rows, urls = fetch_pages(ENEDIS_API, params)
    if not rows:
        raise RuntimeError(f"No Enedis rows for {label} {year}")
    df = pd.DataFrame(rows).sort_values("horodate")
    df["timestamp_utc"] = df["horodate"].map(parse_timestamp)
    df["load_mw"] = df["energy_wh"].astype(float) / 500_000.0
    df["region"] = label
    df["code_region"] = code
    out = os.path.join(out_dir, f"enedis_{slug}_{year}.csv")
    df[["timestamp_utc", "load_mw", "energy_wh", "n_segments", "region", "code_region"]].to_csv(
        out, index=False)
    return {
        "kind": "enedis_conso_inf36_region",
        "region": label,
        "code_region": code,
        "year": year,
        "path": out,
        "n_rows": int(len(df)),
        "expected_rows_30min": 17_568,
        "portal_url": ENEDIS_PAGE,
        "api_urls": urls,
        "license": "Licence Ouverte / Open Licence version 2.0",
        "aggregation": "P0 Total <=36 kVA, summed over all profiles per timestamp; Wh/0.5h converted to MW.",
    }


def fetch_rte_region(slug: str, code: str, label: str, year: int, out_dir: str) -> dict:
    rows = []
    urls = []
    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"
        end = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"
        where = (
            f"code_insee_region='{code}' AND "
            f"date_heure >= date'{start}' AND date_heure < date'{end}'"
        )
        params = {
            "select": "date_heure,consommation,nature,libelle_region,code_insee_region",
            "where": where,
            "order_by": "date_heure",
        }
        month_rows, month_urls = fetch_pages(RTE_API, params, page_size=100)
        rows.extend(month_rows)
        urls.extend(month_urls)
    if not rows:
        raise RuntimeError(f"No RTE eco2mix rows for {label} {year}")
    df = pd.DataFrame(rows).sort_values("date_heure")
    df["timestamp_utc"] = pd.to_datetime(df["date_heure"], utc=True)
    df["load_mw"] = pd.to_numeric(df["consommation"], errors="coerce")
    out = os.path.join(out_dir, f"eco2mix_{slug}_{year}.csv")
    df[["timestamp_utc", "load_mw", "consommation", "nature", "libelle_region", "code_insee_region"]].to_csv(
        out, index=False)
    return {
        "kind": "rte_eco2mix_regional_cons_def",
        "region": label,
        "code_region": code,
        "year": year,
        "path": out,
        "n_rows": int(len(df)),
        "expected_rows_30min": 17_568,
        "portal_url": RTE_PAGE,
        "api_urls": urls,
        "license": "ODRE / RTE open data terms; label as TSO regional load, not DSO.",
        "aggregation": "No aggregation; consommation is regional load in MW at 30-minute resolution.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=YEAR)
    ap.add_argument("--out-dir", default=REAL)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sources = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "year": args.year,
        "entries": [],
    }
    for slug, code, label in ENEDIS_REGIONS:
        entry = fetch_enedis_region(slug, code, label, args.year, args.out_dir)
        sources["entries"].append(entry)
        print(f"[enedis] {label}: {entry['n_rows']} rows -> {entry['path']}")
    for slug, code, label in RTE_REGIONS:
        entry = fetch_rte_region(slug, code, label, args.year, args.out_dir)
        sources["entries"].append(entry)
        print(f"[eco2mix] {label}: {entry['n_rows']} rows -> {entry['path']}")

    source_path = os.path.join(args.out_dir, "fr_public_loads_source.json")
    with open(source_path, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2, ensure_ascii=False)
    print(f"source metadata -> {source_path}")


if __name__ == "__main__":
    main()
