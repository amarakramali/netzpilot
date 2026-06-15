"""Leakage-safe temporal MinT reconciliation for one quarter-hourly load series.

The hierarchy is temporal, not cross-sectional: one day equals the sum of 24
hours and also the sum of 96 quarter-hours of the same measured series. This is
the axis where the summing constraint is exact for published DSO load curves.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

from netzpilot.features.build import get_holidays
from netzpilot.forecast import forecast_next_day
from netzpilot.models.reconcile import (
    build_temporal_summing_matrix,
    coherence_error,
    reconcile,
)
from netzpilot.models.robust_corrector import ShrunkCorrector


N_QUARTERS_PER_DAY = 96
N_HOURS_PER_DAY = 24
DT_HOURS = 0.25


def _read_csv_flexible(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, sep=None, engine="python", decimal=",")
    except Exception:
        return pd.read_csv(csv_path, sep=";", decimal=",")


def _parse_timestamps(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    extracted = text.str.extract(r"(\d{1,2}\.\d{1,2}\.\d{4}\s+\d{1,2}:\d{2})")[0]
    ts = pd.to_datetime(extracted, format="%d.%m.%Y %H:%M", errors="coerce")
    if ts.notna().mean() < 0.7:
        ts = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if ts.notna().mean() < 0.7:
        raise ValueError("Timestamp-Spalte konnte nicht robust geparst werden.")
    return ts


def _numeric(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.strip()
    text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def _to_mwh_per_period(values: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
    unit_key = str(unit or "MW").strip().lower()
    if unit_key in {"w", "watt"}:
        return values / 1_000_000.0 * DT_HOURS, "W power converted to MWh per 15 min"
    if unit_key in {"kw", "kilowatt"}:
        return values / 1000.0 * DT_HOURS, "kW power converted to MWh per 15 min"
    if unit_key in {"mw", "megawatt"}:
        return values * DT_HOURS, "MW power converted to MWh per 15 min"
    if unit_key in {"kwh"}:
        return values / 1000.0, "kWh energy converted to MWh per 15 min"
    if unit_key in {"mwh"}:
        return values, "MWh energy per 15 min"
    raise ValueError(f"Unsupported unit for temporal reconciliation: {unit}")


def load_quarter_hour_energy(
    csv_path: str,
    *,
    ts_col: str = "Text",
    load_col: str = "Reihe1",
    unit: str = "kW",
) -> tuple[np.ndarray, pd.DatetimeIndex, dict]:
    """Load a quarter-hourly period-ending CSV as complete local days in MWh.

    Published German loadgang files commonly timestamp the end of each 15-minute
    interval. Grouping by timestamp minus 15 minutes keeps the 00:00 row on the
    previous day and yields 96 period-start slots for normal days. Non-96 days
    (DST or incomplete) are dropped instead of stretched.
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV not found: {csv_path}")
    df = _read_csv_flexible(csv_path)
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column not found: {ts_col}")
    if load_col not in df.columns:
        raise ValueError(f"Load column not found: {load_col}")

    ts = _parse_timestamps(df[ts_col])
    load = _numeric(df[load_col])
    ok = ts.notna() & load.notna()
    if int(ok.sum()) < N_QUARTERS_PER_DAY * 30:
        raise ValueError("Too few valid quarter-hour values for temporal reconciliation.")

    period_start = ts[ok] - pd.Timedelta(minutes=15)
    mwh, unit_note = _to_mwh_per_period(load[ok].to_numpy(dtype=float), unit)
    frame = pd.DataFrame({"value": mwh}, index=pd.DatetimeIndex(period_start))
    frame = frame.sort_index()
    frame["date"] = frame.index.normalize()
    frame["slot"] = frame.index.hour * 4 + (frame.index.minute // 15)

    groups: dict[pd.Timestamp, np.ndarray] = {}
    dropped: list[str] = []
    for date, g in frame.groupby("date", sort=True):
        slots = g["slot"].to_numpy(dtype=int)
        if len(g) == N_QUARTERS_PER_DAY and len(set(slots)) == N_QUARTERS_PER_DAY:
            arr = np.empty(N_QUARTERS_PER_DAY, dtype=float)
            arr[slots] = g["value"].to_numpy(dtype=float)
            groups[pd.Timestamp(date)] = arr
        else:
            dropped.append(str(pd.Timestamp(date).date()))

    if len(groups) < 60:
        raise ValueError(f"Too few complete 96-slot days ({len(groups)}); need at least 60.")
    dates = sorted(groups)
    q2d = np.vstack([groups[d] for d in dates])
    days = pd.DatetimeIndex([d.date() for d in dates])
    meta = {
        "csv_path": csv_path,
        "ts_col": ts_col,
        "load_col": load_col,
        "unit_in": unit,
        "unit_out": "MWh",
        "unit_note": unit_note,
        "rows": int(len(df)),
        "valid_rows": int(ok.sum()),
        "complete_days": int(len(days)),
        "dropped_non_96_days": dropped,
        "timestamp_convention": "period_end_shifted_minus_15min",
    }
    return q2d, days, meta


def aggregate_temporal_levels(q2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return hourly and daily MWh aggregates for a [days, 96] quarter matrix."""
    q = np.asarray(q2d, dtype=float)
    if q.ndim != 2 or q.shape[1] != N_QUARTERS_PER_DAY:
        raise ValueError("q2d must have shape [days, 96].")
    h2d = q.reshape(q.shape[0], N_HOURS_PER_DAY, 4).sum(axis=2)
    d2d = q.sum(axis=1, keepdims=True)
    return h2d, d2d


def _p50_array(forecast: dict) -> np.ndarray:
    return np.asarray([float(x["p50"]) for x in forecast["hours"]], dtype=float)


def _holidays_for(days: Iterable[pd.Timestamp], region: str) -> set:
    idx = pd.DatetimeIndex(days)
    years = sorted(set(int(d.year) for d in idx))
    if years:
        years = sorted(set(years + [years[-1] + 1]))
    return get_holidays(years, region)


def _default_corrector_factory():
    return ShrunkCorrector(10.0)


def _forecast_base_from_history(
    q_hist: np.ndarray,
    days_hist: pd.DatetimeIndex,
    holiday_set: set,
    *,
    corrector_factory=None,
) -> tuple[str, np.ndarray, dict]:
    if len(q_hist) < 60:
        raise ValueError(f"Need at least 60 complete history days, got {len(q_hist)}.")
    factory = corrector_factory or _default_corrector_factory
    h_hist, d_hist = aggregate_temporal_levels(q_hist)
    q_fc = forecast_next_day(q_hist, days_hist, factory, holiday_set=holiday_set, round_digits=None)
    h_fc = forecast_next_day(h_hist, days_hist, factory, holiday_set=holiday_set, round_digits=None)
    d_fc = forecast_next_day(d_hist, days_hist, factory, holiday_set=holiday_set, round_digits=None)
    if not (q_fc["date"] == h_fc["date"] == d_fc["date"]):
        raise ValueError("Temporal base forecasts target different dates.")
    q = _p50_array(q_fc)
    h = _p50_array(h_fc)
    d = _p50_array(d_fc)
    if len(q) != 96 or len(h) != 24 or len(d) != 1:
        raise ValueError("Temporal base forecast horizons must be 96, 24 and 1.")
    base = np.concatenate([d, h, q])
    levels = {"day": d, "hour": h, "quarter": q}
    return q_fc["date"], base, levels


def _forecast_base_for_target(
    q2d: np.ndarray,
    days: pd.DatetimeIndex,
    target_index: int,
    holiday_set: set,
    *,
    corrector_factory=None,
) -> tuple[str, np.ndarray, dict]:
    target_date, base, levels = _forecast_base_from_history(
        q2d[:target_index],
        pd.DatetimeIndex(days[:target_index]),
        holiday_set,
        corrector_factory=corrector_factory,
    )
    expected = str(pd.Timestamp(days[target_index]).date())
    if target_date != expected:
        raise ValueError(
            f"Forecast target mismatch: forecast_next_day={target_date}, holdout={expected}. "
            "This usually means a non-contiguous day sits directly before the target."
        )
    return target_date, base, levels


def _actual_node_vector(q_day: np.ndarray) -> np.ndarray:
    h = q_day.reshape(N_HOURS_PER_DAY, 4).sum(axis=1)
    d = np.asarray([float(q_day.sum())], dtype=float)
    return np.concatenate([d, h, q_day.astype(float)])


def _valid_target_indices(days: pd.DatetimeIndex, *, n_test: int, min_history: int) -> list[int]:
    if int(n_test) < 1:
        raise ValueError("n_test must be >= 1.")
    idx = pd.DatetimeIndex(days)
    valid = []
    for i in range(min_history, len(idx)):
        if pd.Timestamp(idx[i - 1]) + pd.Timedelta(days=1) == pd.Timestamp(idx[i]):
            valid.append(i)
    if len(valid) < n_test:
        raise ValueError(f"Only {len(valid)} valid contiguous holdout targets, need {n_test}.")
    return valid[-n_test:]


def _residual_matrix(
    q2d: np.ndarray,
    days: pd.DatetimeIndex,
    holiday_set: set,
    target_indices: list[int],
    *,
    corrector_factory=None,
) -> np.ndarray:
    residuals = []
    for i in target_indices:
        _date, base, _levels = _forecast_base_for_target(
            q2d, days, i, holiday_set, corrector_factory=corrector_factory
        )
        residuals.append(_actual_node_vector(q2d[i]) - base)
    if len(residuals) < 2:
        raise ValueError("wls_var/mint_shrink need at least two leakage-safe residual days.")
    return np.vstack(residuals).T


def _metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    err = pred - actual
    denom = np.where(np.abs(actual) > 1e-12, np.abs(actual), np.nan)
    mape = np.nanmean(np.abs(err) / denom) * 100.0
    return {
        "mae": float(np.mean(np.abs(err))),
        "mape_pct": float(mape),
    }


def _round_float(x: float, digits: int = 6) -> float:
    return round(float(x), digits)


def _round_list(values: np.ndarray, digits: int = 6) -> list[float]:
    return [_round_float(x, digits) for x in np.asarray(values, dtype=float)]


def _summarize_metrics(parts: dict[str, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]]) -> dict:
    out = {}
    for level, (base_parts, rec_parts, actual_parts) in parts.items():
        base = np.concatenate(base_parts)
        rec = np.concatenate(rec_parts)
        actual = np.concatenate(actual_parts)
        base_m = _metrics(base, actual)
        rec_m = _metrics(rec, actual)
        base_mae = base_m["mae"]
        delta = 100.0 * (base_mae - rec_m["mae"]) / base_mae if base_mae > 0 else 0.0
        out[level] = {
            "base_mae": _round_float(base_m["mae"]),
            "reconciled_mae": _round_float(rec_m["mae"]),
            "delta_pct": _round_float(delta, 3),
            "base_mape_pct": _round_float(base_m["mape_pct"], 3),
            "reconciled_mape_pct": _round_float(rec_m["mape_pct"], 3),
        }
    return out


def temporal_reconciliation_backtest(
    q2d: np.ndarray,
    days: pd.DatetimeIndex,
    *,
    region: str = "NW",
    n_test: int = 14,
    method: str = "wls_struct",
    residual_days: int = 14,
    corrector_factory=None,
    include_daily: bool = False,
) -> dict:
    """Run leakage-safe base forecasts and reconcile them on a trailing holdout."""
    q = np.asarray(q2d, dtype=float)
    if q.ndim != 2 or q.shape[1] != N_QUARTERS_PER_DAY:
        raise ValueError("q2d must have shape [days, 96].")
    if method not in {"ols", "wls_struct", "wls_var", "mint_shrink"}:
        raise ValueError(f"Unknown reconciliation method: {method}")
    if int(n_test) < 1:
        raise ValueError("n_test must be >= 1.")
    if int(residual_days) < 2 and method in {"wls_var", "mint_shrink"}:
        raise ValueError("residual_days must be >= 2 for variance-weighted reconciliation.")

    S, node_names = build_temporal_summing_matrix(N_QUARTERS_PER_DAY, [96, 4])
    days_idx = pd.DatetimeIndex(days)
    holiday_set = _holidays_for(days_idx, region)
    targets = _valid_target_indices(days_idx, n_test=int(n_test), min_history=60)

    residuals = None
    residual_source = None
    if method in {"wls_var", "mint_shrink"}:
        residual_candidates = [
            i for i in _valid_target_indices(days_idx[:targets[0]], n_test=min(residual_days, targets[0] - 60), min_history=60)
        ] if targets[0] - 60 >= 2 else []
        if len(residual_candidates) < 2:
            raise ValueError("Not enough pre-holdout leakage-safe residual days for variance weighting.")
        residuals = _residual_matrix(
            q, days_idx, holiday_set, residual_candidates[-int(residual_days):],
            corrector_factory=corrector_factory,
        )
        residual_source = (
            f"{residuals.shape[1]} pre-holdout rolling-origin residual days; "
            "no target-day or in-sample residuals"
        )

    parts = {
        "day": ([], [], []),
        "hour": ([], [], []),
        "quarter": ([], [], []),
    }
    coh_before = []
    coh_after = []
    daily_rows = []
    for i in targets:
        date, base_vec, _levels = _forecast_base_for_target(
            q, days_idx, i, holiday_set, corrector_factory=corrector_factory
        )
        rec_vec = reconcile(base_vec, S, method=method, residuals=residuals)
        actual_vec = _actual_node_vector(q[i])
        before = coherence_error(base_vec, S)
        after = coherence_error(rec_vec, S)
        coh_before.append(before)
        coh_after.append(after)

        slices = {
            "day": slice(0, 1),
            "hour": slice(1, 25),
            "quarter": slice(25, 121),
        }
        for level, sl in slices.items():
            parts[level][0].append(base_vec[sl])
            parts[level][1].append(rec_vec[sl])
            parts[level][2].append(actual_vec[sl])
        if include_daily:
            daily_rows.append({
                "date": date,
                "coherence_before": _round_float(before),
                "coherence_after": _round_float(after),
            })

    return {
        "status": "available",
        "method": method,
        "node_names": node_names,
        "n_test_days": int(len(targets)),
        "target_start": str(pd.Timestamp(days_idx[targets[0]]).date()),
        "target_end": str(pd.Timestamp(days_idx[targets[-1]]).date()),
        "history_days_used": int(len(q)),
        "dt_h": DT_HOURS,
        "unit": "MWh per node period",
        "residual_source": residual_source or "wls_struct/ols structural weights; no residuals required",
        "coherence": {
            "before_mean": _round_float(np.mean(coh_before)),
            "before_max": _round_float(np.max(coh_before)),
            "after_mean": _round_float(np.mean(coh_after)),
            "after_max": _round_float(np.max(coh_after)),
        },
        "metrics": _summarize_metrics(parts),
        "daily": daily_rows if include_daily else None,
        "caveat": (
            "Accuracy is measured honestly per level on the same leakage-safe holdout. "
            "Reconciliation guarantees temporal coherence; it does not guarantee lower MAE on every level."
        ),
    }


def temporal_reconciliation_next_day(
    q2d: np.ndarray,
    days: pd.DatetimeIndex,
    *,
    region: str = "NW",
    method: str = "wls_struct",
    residual_days: int = 14,
    corrector_factory=None,
) -> dict:
    """Return a coherent next-day P50 across day/hour/quarter levels."""
    q = np.asarray(q2d, dtype=float)
    S, node_names = build_temporal_summing_matrix(N_QUARTERS_PER_DAY, [96, 4])
    days_idx = pd.DatetimeIndex(days)
    holiday_set = _holidays_for(days_idx, region)
    residuals = None
    residual_source = "wls_struct/ols structural weights; no residuals required"
    if method in {"wls_var", "mint_shrink"}:
        valid = _valid_target_indices(days_idx, n_test=min(int(residual_days), len(days_idx) - 60), min_history=60)
        residuals = _residual_matrix(q, days_idx, holiday_set, valid, corrector_factory=corrector_factory)
        residual_source = (
            f"{residuals.shape[1]} trailing rolling-origin residual days before forecast date; "
            "no in-sample residuals"
        )
    date, base_vec, _levels = _forecast_base_from_history(
        q, days_idx, holiday_set, corrector_factory=corrector_factory
    )
    rec_vec = reconcile(base_vec, S, method=method, residuals=residuals)
    return {
        "date": date,
        "method": method,
        "node_names": node_names,
        "unit": "MWh",
        "residual_source": residual_source,
        "coherence_before": _round_float(coherence_error(base_vec, S)),
        "coherence_after": _round_float(coherence_error(rec_vec, S)),
        "base": {
            "day_mwh": _round_float(base_vec[0]),
            "hour_mwh": _round_list(base_vec[1:25]),
            "quarter_mwh": _round_list(base_vec[25:]),
        },
        "reconciled": {
            "day_mwh": _round_float(rec_vec[0]),
            "hour_mwh": _round_list(rec_vec[1:25]),
            "quarter_mwh": _round_list(rec_vec[25:]),
        },
    }


def build_temporal_reconciliation_payload(
    csv_path: str,
    *,
    ts_col: str = "Text",
    load_col: str = "Reihe1",
    unit: str = "kW",
    region: str = "NW",
    method: str = "wls_struct",
    n_test: int = 7,
    residual_days: int = 14,
) -> dict:
    """Load a real 15-minute loadgang and return demo/service payload."""
    q2d, days, meta = load_quarter_hour_energy(
        csv_path,
        ts_col=ts_col,
        load_col=load_col,
        unit=unit,
    )
    holdout = temporal_reconciliation_backtest(
        q2d,
        days,
        region=region,
        n_test=n_test,
        method=method,
        residual_days=residual_days,
    )
    forecast = temporal_reconciliation_next_day(
        q2d,
        days,
        region=region,
        method=method,
        residual_days=residual_days,
    )
    return {
        "status": "available",
        "source": "real 15-minute measurements",
        "input": meta,
        "forecast": forecast,
        "holdout": holdout,
        "summary": (
            "NetzPilot forecasts are temporally coherent: quarter-hour nominations, "
            "hourly schedules and daily energy add up exactly after reconciliation."
        ),
    }
