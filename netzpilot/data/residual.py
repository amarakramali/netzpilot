"""Residual-load construction from load and renewable generation series."""
from __future__ import annotations

import pandas as pd


def quarterhour_generation_to_hourly(series: pd.Series) -> pd.Series:
    """Convert quarter-hourly MW generation to hourly mean MW."""
    s = series.sort_index()
    s.index = pd.to_datetime(s.index, utc=True)
    return s.resample("1h", label="left", closed="left").mean()


def residual_load(load_mw: pd.Series, pv_mw: pd.Series, wind_onshore_mw: pd.Series, wind_offshore_mw: pd.Series) -> pd.Series:
    """Build residual load = load - PV - wind onshore - wind offshore, aligned hourly."""
    load = load_mw.sort_index()
    load.index = pd.to_datetime(load.index, utc=True)
    gen = (
        quarterhour_generation_to_hourly(pv_mw)
        + quarterhour_generation_to_hourly(wind_onshore_mw)
        + quarterhour_generation_to_hourly(wind_offshore_mw)
    )
    aligned = pd.concat([load.rename("load_mw"), gen.rename("generation_mw")], axis=1, join="inner")
    out = aligned["load_mw"] - aligned["generation_mw"]
    out.name = "residual_load_mw"
    return out
