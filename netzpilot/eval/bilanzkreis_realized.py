# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Realized Bilanzkreis settlement on rolling-origin backtest rows."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .bilanzkreis import (
    compare_forecasts_eur,
    imbalance_premium_eur,
    savings_contrib_per_qh,
)
from .economics import expected_saving_eur
from .mc_savings import block_bootstrap_band

LOCAL_TZ = "Europe/Berlin"


def _read_qh_price_csv(path: str | Path, value_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if "Zeit" not in df.columns:
        raise ValueError(f"Preis-CSV braucht eine Zeit-Spalte: {path}")
    if value_col is None:
        value_cols = [c for c in df.columns if c != "Zeit"]
        if len(value_cols) != 1:
            raise ValueError(f"Preis-CSV braucht genau eine Wertspalte: {path}")
        value_col = value_cols[0]
    ts = pd.to_datetime(df["Zeit"], utc=True).dt.tz_convert(LOCAL_TZ)
    out = pd.DataFrame({"value": pd.to_numeric(df[value_col], errors="coerce").to_numpy()}, index=ts)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def hourly_price_frame(rebap_csv: str | Path, spot_csv: str | Path) -> tuple[pd.DataFrame, dict]:
    """Average QH reBAP and spot prices to hourly prices on the local CET/CEST axis."""
    reb = _read_qh_price_csv(rebap_csv, "reBAP_EUR_MWh").rename(columns={"value": "rebap_eur_mwh"})
    spot = _read_qh_price_csv(spot_csv, "spot_DA_EUR_MWh").rename(columns={"value": "spot_eur_mwh"})
    qh = reb.join(spot, how="inner").dropna()
    if qh.empty:
        raise ValueError("Keine gemeinsamen QH-Preise fuer reBAP/Spot.")
    hourly = qh.groupby(pd.Grouper(freq="h")).mean().dropna()
    return hourly, {
        "price_resolution_input": "quarter_hour",
        "settlement_resolution": "hour",
        "resolution_rule": "QH reBAP and Spot are averaged per local hour; MW load is settled as MWh per hour.",
        "n_qh_prices": int(len(qh)),
        "n_hourly_prices": int(len(hourly)),
    }


def official_rebap_asymmetry_count(path: str | Path) -> dict:
    """Count official QH rows where under/over covered reBAP columns differ."""
    def parse_de(raw: str) -> float:
        return float(str(raw).strip().replace(".", "").replace(",", "."))

    n = 0
    n_diff = 0
    max_abs_diff = 0.0
    with Path(path).open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if "reBAP unterdeckt" not in row or "reBAP ueberdeckt" not in row:
                raise ValueError("Offizielle reBAP-CSV braucht unterdeckt/ueberdeckt-Spalten.")
            a = parse_de(row["reBAP unterdeckt"])
            b = parse_de(row["reBAP ueberdeckt"])
            n += 1
            diff = abs(a - b)
            if diff > 1e-9:
                n_diff += 1
                max_abs_diff = max(max_abs_diff, diff)
    return {"n_qh": n, "n_asymmetric_qh": n_diff, "max_abs_diff_eur_mwh": round(max_abs_diff, 6)}


def _backtest_index(days, n_test: int) -> pd.DatetimeIndex:
    idx = []
    for day in list(days)[-n_test:]:
        start = pd.Timestamp(day)
        if start.tzinfo is None:
            start = start.tz_localize(LOCAL_TZ)
        else:
            start = start.tz_convert(LOCAL_TZ)
        for h in range(24):
            idx.append(start + pd.Timedelta(hours=h))
    return pd.DatetimeIndex(idx)


def _signed_expected(delta_mae_mw: float, rebap, spot, hours_per_year: float) -> dict:
    sign = 1.0
    dmae = float(delta_mae_mw)
    if dmae < 0:
        sign = -1.0
        dmae = abs(dmae)
    out = expected_saving_eur(dmae, rebap, spot, hours_per_year=hours_per_year)
    out["expected_eur_per_year"] = round(sign * out["expected_eur_per_year"])
    out["delta_mae_mw_signed"] = round(float(delta_mae_mw), 4)
    return out


def _scale_premium(premium: dict, scale: float) -> dict:
    out = dict(premium)
    for key in ("total_premium_eur", "bias_term_eur", "correlation_term_eur"):
        out[f"{key}_per_year"] = round(float(premium[key]) * scale, 2)
    return out


def realized_settlement_from_backtest(R: dict, days, n_test: int, rebap_csv: str | Path,
                                      spot_csv: str | Path, *, baseline_key: str = "snaive",
                                      model_key: str = "model", include_band: bool = True,
                                      seed: int = 0, n_resamples: int = 2000) -> dict:
    """Compute realized hourly settlement from a rolling-origin backtest result."""
    price_hourly, price_meta = hourly_price_frame(rebap_csv, spot_csv)
    idx = _backtest_index(days, n_test)
    if len(idx) != len(R["actual"]):
        raise ValueError(f"Backtest-Laenge ({len(R['actual'])}) passt nicht zu Index ({len(idx)}).")

    bt = pd.DataFrame({
        "actual_mwh": np.asarray(R["actual"], dtype=float),
        "baseline_mwh": np.asarray(R[baseline_key], dtype=float),
        "model_mwh": np.asarray(R[model_key], dtype=float),
    }, index=idx)
    joined = bt.join(price_hourly, how="inner").dropna()
    if joined.empty:
        raise ValueError("Keine gemeinsame Schnittmenge zwischen Backtest-Stunden und Preisstunden.")

    actual = joined["actual_mwh"].to_list()
    baseline = joined["baseline_mwh"].to_list()
    model = joined["model_mwh"].to_list()
    rebap = joined["rebap_eur_mwh"].to_list()
    spot = joined["spot_eur_mwh"].to_list()

    cmp = compare_forecasts_eur(actual, baseline, model, rebap, spot)
    pa = imbalance_premium_eur(baseline, actual, rebap, spot)
    pb = imbalance_premium_eur(model, actual, rebap, spot)
    n_periods = int(len(joined))
    n_days = n_periods / 24.0
    scale = 365.0 / n_days
    mae_a = float(np.mean(np.abs(joined["baseline_mwh"] - joined["actual_mwh"])))
    mae_b = float(np.mean(np.abs(joined["model_mwh"] - joined["actual_mwh"])))
    expected = _signed_expected(mae_a - mae_b, rebap, spot, hours_per_year=n_periods * scale)

    out = {
        "method": "rolling-origin realized settlement",
        "baseline": "seasonal_naive" if baseline_key == "snaive" else baseline_key,
        "model": "NetzPilot P50" if model_key == "model" else model_key,
        "basis": cmp["basis"],
        "resolution": "hour",
        "energy_unit": "MWh per hour (MW * 1h)",
        "price_meta": price_meta,
        "n_periods": n_periods,
        "n_days": round(n_days, 3),
        "annualization_factor": round(scale, 6),
        "savings_eur_period": cmp["savings_b_vs_a_eur"],
        "savings_eur_per_year": round(cmp["savings_b_vs_a_eur"] * scale, 2),
        "linear_expected_eur_per_year": expected["expected_eur_per_year"],
        "linear_signed_mean_spread_eur_mwh": expected["signed_mean_spread_eur_mwh"],
        "delta_mae_mw": round(mae_a - mae_b, 4),
        "mae_baseline_mw": round(mae_a, 4),
        "mae_model_mw": round(mae_b, 4),
        "premium_baseline": _scale_premium(pa, scale),
        "premium_model": _scale_premium(pb, scale),
        "savings_bias_term_eur_per_year": round((pa["bias_term_eur"] - pb["bias_term_eur"]) * scale, 2),
        "savings_correlation_term_eur_per_year": round(
            (pa["correlation_term_eur"] - pb["correlation_term_eur"]) * scale, 2
        ),
        "caveat": (
            "No Intraday trading modelled; this is settlement exposure without position smoothing, "
            "annualized from the rolling-origin evaluation window."
        ),
    }
    if include_band:
        contrib = savings_contrib_per_qh(actual, baseline, model, rebap, spot)
        scaled_contrib = [c * scale for c in contrib]
        band = block_bootstrap_band(scaled_contrib, block_len=24, seed=seed, n_resamples=n_resamples)
        band.update({
            "block": "day",
            "period": "hour",
            "annualized": True,
            "annualization_factor": round(scale, 6),
        })
        out["band"] = band
        week_band = block_bootstrap_band(scaled_contrib, block_len=24 * 7, seed=seed,
                                         n_resamples=n_resamples)
        week_band.update({
            "block": "week",
            "period": "hour",
            "annualized": True,
            "annualization_factor": round(scale, 6),
            "sensitivity": "longer blocks preserve more multi-day autocorrelation.",
        })
        out["band_week_sensitivity"] = week_band
    return out
