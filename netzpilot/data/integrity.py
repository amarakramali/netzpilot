"""Data-integrity checks for SMARD load/generation and aligned weather series."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


RESOLUTION_STEPS = {
    "hour": pd.Timedelta(hours=1),
    "quarterhour": pd.Timedelta(minutes=15),
}

NOMINAL_LOCAL_DAY_SLOTS = {
    "hour": 24,
    "quarterhour": 96,
}


@dataclass(frozen=True)
class IntegrityErrorDetail:
    kind: str
    message: str


def _utc_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _expected_local_day_slots(day, timezone: str, step: pd.Timedelta) -> int:
    start = pd.Timestamp(day).tz_localize(timezone).tz_convert("UTC")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(timezone).tz_convert("UTC")
    return int((end - start) / step)


def series_integrity_report(
    series: pd.Series,
    resolution: str,
    timezone: str = "Europe/Berlin",
    allow_boundary_partial_days: bool = True,
) -> dict:
    """Return continuity, finite-value, and local DST-day checks for one time series."""
    if resolution not in RESOLUTION_STEPS:
        raise ValueError(f"Unsupported resolution: {resolution}")
    if len(series) == 0:
        raise ValueError("Cannot validate an empty series")

    s = series.sort_index()
    idx = _utc_index(s.index)
    step = RESOLUTION_STEPS[resolution]
    duplicate_count = int(idx.duplicated().sum())
    diffs = idx.to_series().diff().dropna()
    bad_steps = diffs[diffs != step]
    gaps = []
    for ts, observed in bad_steps.items():
        previous = idx[idx.get_loc(ts) - 1]
        missing_slots = max(int(observed / step) - 1, 0)
        gaps.append({
            "from_utc": previous.isoformat(),
            "to_utc": ts.isoformat(),
            "observed_seconds": int(observed.total_seconds()),
            "missing_slots": missing_slots,
        })

    values = pd.to_numeric(s, errors="coerce")
    non_finite_count = int(values.isna().sum())

    local_idx = idx.tz_convert(timezone)
    counts = pd.Series(1, index=local_idx).groupby(local_idx.date).sum()
    boundary_dates = {local_idx[0].date(), local_idx[-1].date()}
    nominal_count = NOMINAL_LOCAL_DAY_SLOTS[resolution]
    unexpected_local_day_counts = {}
    dst_transition_days = {}
    for day, count in counts.items():
        count = int(count)
        is_boundary = day in boundary_dates
        expected_slots = _expected_local_day_slots(day, timezone, step)
        if count != expected_slots and not (allow_boundary_partial_days and is_boundary):
            unexpected_local_day_counts[day.isoformat()] = count
        if count == expected_slots and expected_slots != nominal_count:
            dst_transition_days[day.isoformat()] = count

    expected_count = int(((idx[-1] - idx[0]) / step) + 1) if len(idx) > 1 else 1
    return {
        "resolution": resolution,
        "timezone": timezone,
        "count": int(len(s)),
        "expected_count_from_bounds": expected_count,
        "start_utc": idx[0].isoformat(),
        "end_utc": idx[-1].isoformat(),
        "step_seconds": int(step.total_seconds()),
        "duplicate_count": duplicate_count,
        "gap_count": len(gaps),
        "gaps": gaps,
        "non_finite_count": non_finite_count,
        "local_day_count_min": int(counts.min()),
        "local_day_count_max": int(counts.max()),
        "dst_transition_days": dst_transition_days,
        "unexpected_local_day_counts": unexpected_local_day_counts,
    }


def validate_series(
    series: pd.Series,
    resolution: str,
    timezone: str = "Europe/Berlin",
    allow_boundary_partial_days: bool = True,
) -> dict:
    """Raise ValueError if the series has gaps, duplicates, non-finite values, or invalid local-day counts."""
    report = series_integrity_report(series, resolution, timezone, allow_boundary_partial_days)
    errors: list[IntegrityErrorDetail] = []
    if report["duplicate_count"]:
        errors.append(IntegrityErrorDetail("duplicates", f"{report['duplicate_count']} duplicated timestamps"))
    if report["gap_count"]:
        errors.append(IntegrityErrorDetail("gaps", f"{report['gap_count']} non-contiguous timestamp steps"))
    if report["non_finite_count"]:
        errors.append(IntegrityErrorDetail("values", f"{report['non_finite_count']} non-finite values"))
    if report["unexpected_local_day_counts"]:
        errors.append(IntegrityErrorDetail("dst", f"unexpected local day counts: {report['unexpected_local_day_counts']}"))
    if errors:
        detail = "; ".join(f"{e.kind}: {e.message}" for e in errors)
        raise ValueError(detail)
    return report


def dataframe_report(
    series_by_name: Mapping[str, pd.Series],
    resolution_by_name: Mapping[str, str],
    timezone: str = "Europe/Berlin",
) -> dict:
    """Build integrity reports for a named set of series."""
    return {
        name: series_integrity_report(series, resolution_by_name[name], timezone)
        for name, series in series_by_name.items()
    }
