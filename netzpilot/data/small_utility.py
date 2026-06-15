# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Small-utility load construction from public OPSD household data."""
from __future__ import annotations

import pandas as pd


def aggregate_opsd_grid_import(
    df: pd.DataFrame,
    scale_factor: float = 1000.0,
    min_active_columns: int = 4,
) -> tuple[pd.Series, list[str]]:
    """Aggregate OPSD 15-min cumulative grid-import kWh counters to hourly MW.

    OPSD household columns are cumulative kWh meter readings. Differencing yields
    interval kWh; multiplying by 4 gives average kW for the quarter-hour. Hourly
    mean kW is then scaled to a representative small-utility portfolio and
    converted to MW.
    """
    if "utc_timestamp" not in df.columns:
        raise ValueError("OPSD frame needs utc_timestamp")
    cols = [c for c in df.columns if c.startswith("DE_KN_") and c.endswith("_grid_import")]
    if not cols:
        raise ValueError("No OPSD DE_KN grid_import columns found")
    work = df[["utc_timestamp", *cols]].copy()
    work["utc_timestamp"] = pd.to_datetime(work["utc_timestamp"], utc=True)
    work = work.set_index("utc_timestamp").sort_index()
    numeric = work[cols].apply(pd.to_numeric, errors="coerce")
    usable = [c for c in cols if numeric[c].notna().sum() > 1]
    if not usable:
        raise ValueError("No grid_import column has enough finite values")
    interval_kwh = numeric[usable].diff()
    interval_kwh = interval_kwh.where(interval_kwh >= 0.0)
    active = interval_kwh.notna().sum(axis=1)
    interval_kwh = interval_kwh.where(active >= int(min_active_columns))
    qh_kw = interval_kwh.sum(axis=1, min_count=int(min_active_columns)) * 4.0
    hourly_mw = qh_kw.resample("1h", label="left", closed="left").mean() * float(scale_factor) / 1000.0
    hourly_mw.name = "small_utility_load_mw"
    return hourly_mw.dropna(), usable
