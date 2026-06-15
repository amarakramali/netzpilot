# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.forecast import forecast_next_day
from netzpilot.horizon import forecast_days, rolling_horizon_backtest, _fit_like_next_day
from netzpilot.models.robust_corrector import ShrunkCorrector


def _synthetic_case():
    rng = np.random.default_rng(7)
    nd, hours = 140, 24
    base = 20 + 5 * np.sin(np.arange(hours) / 24 * 2 * np.pi)
    week = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.8])
    load2d = np.array([
        base * week[d % 7] * (1 + 0.05 * np.sin(d / 30)) + rng.normal(0, 0.3, hours)
        for d in range(nd)
    ])
    days = pd.date_range("2025-06-02", periods=nd, freq="D")
    return load2d, days, lambda: ShrunkCorrector(10.0)


def test_forecast_days_k1_matches_next_day_and_horizon_shape():
    load2d, days, factory = _synthetic_case()

    fd = forecast_days(load2d, days, factory, horizon=3, round_digits=None)
    f1 = forecast_next_day(
        load2d,
        days,
        factory,
        holiday_set=None,
        calibrate=False,
        residual_feedback=False,
        round_digits=None,
    )

    assert fd["days"][0]["date"] == f1["date"]
    assert np.allclose([x["p50"] for x in fd["days"][0]["hours"]],
                       [x["p50"] for x in f1["hours"]], atol=1e-9)
    assert np.allclose([x["p10"] for x in fd["days"][0]["hours"]],
                       [x["p10"] for x in f1["hours"]], atol=1e-9)
    assert "p10" not in fd["days"][1]["hours"][0]
    assert "p90" not in fd["days"][1]["hours"][0]
    assert [d["horizon"] for d in fd["days"]] == [1, 2, 3]
    assert [d["date"] for d in fd["days"]] == [
        str((days[-1] + pd.Timedelta(days=k)).date()) for k in (1, 2, 3)
    ]


def test_forecast_days_fit_is_fixed_and_limits_are_clear():
    load2d, days, factory = _synthetic_case()

    _model, fit_end, _res = _fit_like_next_day(load2d, days, factory, 8, 28, None, None)
    assert fit_end <= load2d.shape[0]

    fd3 = forecast_days(load2d, days, factory, horizon=3, round_digits=None)
    fd1 = forecast_days(load2d, days, factory, horizon=1, round_digits=None)
    assert np.allclose([x["p50"] for x in fd1["days"][0]["hours"]],
                       [x["p50"] for x in fd3["days"][0]["hours"]], atol=1e-9)

    fd2a = forecast_days(load2d, days, factory, horizon=2, round_digits=None)
    fd2b = forecast_days(load2d, days, factory, horizon=2, round_digits=None)
    assert fd2a == fd2b

    with pytest.raises(ValueError):
        forecast_days(load2d, days, factory, horizon=8)
    with pytest.raises(ValueError):
        rolling_horizon_backtest(load2d[:50], days[:50], factory, horizon=3, n_test=42)


def test_rolling_horizon_backtest_alignment_and_skill():
    load2d, days, factory = _synthetic_case()

    bt = rolling_horizon_backtest(load2d, days, factory, horizon=3, n_test=21)
    per_k = bt["per_horizon"]

    assert set(per_k) == {1, 2, 3}
    assert all(per_k[k]["n_days"] == 21 for k in per_k)
    assert per_k[1]["mae_mw"] > 0
    assert per_k[1]["mape_pct"] > 0
    assert per_k[1]["mae_mw"] <= per_k[3]["mae_mw"] + 0.05

    k = 2
    expected = round((1.0 - per_k[k]["mae_mw"] / per_k[k]["mae_snaive_mw"]) * 100.0, 1)
    assert abs(per_k[k]["skill_vs_snaive_pct"] - expected) < 0.11


def test_forecast_days_bands_default_keeps_t52_contract():
    load2d, days, factory = _synthetic_case()

    default = forecast_days(load2d, days, factory, horizon=3, round_digits=None)
    k1 = forecast_days(load2d, days, factory, horizon=3, round_digits=None, bands="k1")

    assert default["days"] == k1["days"]
    assert default["bands_mode"] == "k1"
    assert k1["bands_mode"] == "k1"
    assert "nur P50" in k1["bands"]
    assert all("p10" not in h and "p90" not in h for d in k1["days"][1:] for h in d["hours"])

    with pytest.raises(ValueError):
        forecast_days(load2d, days, factory, horizon=3, bands="x")


def test_forecast_days_per_horizon_keeps_d1_and_adds_ordered_bands():
    load2d, days, factory = _synthetic_case()

    k1 = forecast_days(load2d, days, factory, horizon=3, round_digits=None, bands="k1")
    per_horizon = forecast_days(load2d, days, factory, horizon=3, round_digits=None,
                                bands="per_horizon")

    assert per_horizon["bands_mode"] == "per_horizon"
    assert per_horizon["days"][0]["hours"] == k1["days"][0]["hours"]
    assert "kalibriertem Band" in per_horizon["bands"]

    for day in per_horizon["days"][1:]:
        assert "band" in day
        assert day["band"]["scale"] >= 1.0
        assert day["band"]["n_cal_days"] > 0
        assert "conf_c" in day["band"]
        assert all("p10" in h and "p90" in h for h in day["hours"])
        assert all(h["p10"] <= h["p50"] <= h["p90"] for h in day["hours"])
