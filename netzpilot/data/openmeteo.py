"""Open-Meteo connector.

Use Historical Forecast for training/backtests. Do not use reanalysis/historical actual weather
there, because that leaks future information into a forecast-time experiment.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

HIST_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LIVE_FORECAST = "https://api.open-meteo.com/v1/forecast"
DEFAULT_VARS = [
    "temperature_2m",
    "shortwave_radiation",
    "direct_radiation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_speed_100m",
]

# Simple representative German points for the national-load MVP.
# T3 can replace this with population/load-weighted locations.
DEFAULT_LOCATIONS = {
    "muenster_nrw": (51.96, 7.63),
    "munich_south": (48.14, 11.58),
    "berlin_east": (52.52, 13.41),
}


def _get(url, params, retries=4, timeout=30):
    last_error = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last_error = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as exc:
            last_error = repr(exc)
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Open-Meteo fetch failed: {url} {params}; last_error={last_error}")


def _cache_path(cache_dir: str, source: str, location_name: str, start: str, end: str) -> Path:
    safe_start = start.replace(":", "").replace("/", "-")
    safe_end = end.replace(":", "").replace("/", "-")
    return Path(cache_dir) / f"openmeteo_{source}_{location_name}_{safe_start}_{safe_end}.parquet"


def _fetch_weather_http(lat: float, lon: float, start: str, end: str, hourly, historical: bool) -> pd.DataFrame:
    url = HIST_FORECAST if historical else LIVE_FORECAST
    j = _get(url, {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly),
        "start_date": start,
        "end_date": end,
        "timezone": "UTC",
    })
    h = j["hourly"]
    idx = pd.to_datetime(h["time"], utc=True)
    return pd.DataFrame({k: h[k] for k in hourly}, index=idx)


def _date_chunks(start: str, end: str, chunk_days: int):
    current = pd.Timestamp(start)
    final = pd.Timestamp(end)
    while current <= final:
        chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), final)
        yield current.date().isoformat(), chunk_end.date().isoformat()
        current = chunk_end + pd.Timedelta(days=1)


def fetch_weather(
    lat: float,
    lon: float,
    start: str,
    end: str,
    hourly=DEFAULT_VARS,
    historical: bool = True,
    cache_dir: str | None = "data_cache",
    location_name: str = "location",
    force: bool = False,
    chunk_days: int = 180,
) -> pd.DataFrame:
    """Fetch hourly weather for inclusive UTC dates [start, end] as a UTC-indexed DataFrame."""
    source = "historical_forecast" if historical else "live_forecast"
    if cache_dir:
        cp = _cache_path(cache_dir, source, location_name, start, end)
        if cp.exists() and not force:
            df = pd.read_parquet(cp)
            df.index = pd.to_datetime(df.index, utc=True)
            return df

    frames = [
        _fetch_weather_http(lat, lon, chunk_start, chunk_end, hourly, historical)
        for chunk_start, chunk_end in _date_chunks(start, end, chunk_days)
    ]
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(cp)
    return df


def fetch_multi(locations=DEFAULT_LOCATIONS, weights: dict[str, float] | None = None, **kw) -> pd.DataFrame:
    """Fetch and aggregate multiple locations into one weather DataFrame.

    The default is an unweighted MVP average over three representative German locations.
    """
    frames = []
    used_weights = []
    for name, (lat, lon) in locations.items():
        frames.append(fetch_weather(lat, lon, location_name=name, **kw))
        used_weights.append(float(weights[name]) if weights else 1.0)
    total = sum(used_weights)
    out = sum(frame * weight for frame, weight in zip(frames, used_weights)) / total
    out.attrs["locations"] = locations
    out.attrs["weights"] = weights or {name: 1.0 for name in locations}
    out.attrs["source"] = "Open-Meteo Historical Forecast" if kw.get("historical", True) else "Open-Meteo Live Forecast"
    return out


def align_to_index(weather: pd.DataFrame, target_index: pd.Index, method: str = "ffill") -> pd.DataFrame:
    """Align hourly forecast weather to the load index, typically quarter-hourly, without using actual weather."""
    idx = pd.DatetimeIndex(target_index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    w = weather.sort_index().copy()
    w.index = pd.to_datetime(w.index, utc=True)
    aligned = w.reindex(idx, method=method)
    if aligned.isna().any().any():
        missing = int(aligned.isna().any(axis=1).sum())
        raise ValueError(f"Weather alignment has {missing} missing rows; extend the weather fetch range")
    return aligned
