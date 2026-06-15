"""SMARD (Bundesnetzagentur) connector. CC BY 4.0, no API key.

Weekly JSON chunks:
  Index: https://www.smard.de/app/chart_data/<filter>/<region>/index_<res>.json
  Data:  https://www.smard.de/app/chart_data/<filter>/<region>/<filter>_<region>_<res>_<ts>.json
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.smard.de/app/chart_data"
FILTER_LOAD = 410             # Realized electricity consumption (grid load), verified via SMARD chart_data.
FILTER_PV = 4068              # Realized generation: photovoltaic, verified via SMARD chart_data.
FILTER_WIND_ONSHORE = 4067    # Realized generation: wind onshore, verified via SMARD chart_data.
FILTER_WIND_OFFSHORE = 1225   # Realized generation: wind offshore, verified via SMARD chart_data.

FILTERS = {
    "load": FILTER_LOAD,
    "pv": FILTER_PV,
    "wind_onshore": FILTER_WIND_ONSHORE,
    "wind_offshore": FILTER_WIND_OFFSHORE,
}
RESOLUTIONS = {"hour", "quarterhour"}


def _get(url: str, retries: int = 4, timeout: int = 20) -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"SMARD fetch failed: {url}")


def fetch_index(filter_id: int = FILTER_LOAD, region: str = "DE", resolution: str = "hour") -> list[int]:
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Unsupported SMARD resolution: {resolution}")
    return _get(f"{BASE}/{filter_id}/{region}/index_{resolution}.json")["timestamps"]


def fetch_chunk(ts: int, filter_id: int = FILTER_LOAD, region: str = "DE", resolution: str = "hour") -> list[list]:
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Unsupported SMARD resolution: {resolution}")
    return _get(f"{BASE}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{ts}.json")["series"]


def _cache_path(cache_dir: str, filter_id: int, region: str, resolution: str, start: str, end: str) -> Path:
    safe_start = start.replace(":", "").replace("/", "-")
    safe_end = end.replace(":", "").replace("/", "-")
    return Path(cache_dir) / f"smard_{filter_id}_{region}_{resolution}_{safe_start}_{safe_end}.parquet"


def fetch_series(
    start: str,
    end: str,
    filter_id: int = FILTER_LOAD,
    region: str = "DE",
    resolution: str = "hour",
    cache_dir: str | None = "data_cache",
    force: bool = False,
) -> pd.Series:
    """Fetch [start, end) as a UTC-indexed pandas Series in MW, with Parquet cache."""
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Unsupported SMARD resolution: {resolution}")
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    if cache_dir:
        cp = _cache_path(cache_dir, filter_id, region, resolution, start, end)
        if cp.exists() and not force:
            s = pd.read_parquet(cp)["value"]
            s.index = pd.to_datetime(s.index, utc=True)
            return s
    pairs: list[list] = []
    for ts in fetch_index(filter_id, region, resolution):
        if ts < start_ms - 7 * 86400_000 or ts > end_ms:
            continue
        pairs += fetch_chunk(ts, filter_id, region, resolution)
    pairs = [p for p in pairs if start_ms <= p[0] < end_ms and p[1] is not None]
    pairs.sort()
    idx = pd.to_datetime([p[0] for p in pairs], unit="ms", utc=True)
    s = pd.Series([p[1] for p in pairs], index=idx, name="value").sort_index()
    s = s[~s.index.duplicated()]
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        s.to_frame("value").to_parquet(cp)
    return s


def load_local_json(glob_pattern: str) -> pd.Series:
    """Read bundled SMARD weekly files (list of [ms, MW]) as a UTC-indexed Series."""
    pairs = []
    for f in sorted(glob.glob(glob_pattern)):
        pairs += json.load(open(f))
    pairs.sort()
    idx = pd.to_datetime([p[0] for p in pairs], unit="ms", utc=True)
    return pd.Series([p[1] for p in pairs], index=idx, name="value")
