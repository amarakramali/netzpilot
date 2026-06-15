# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Tests fuer Online-Residuen-Feedback (T50).

S1-S6 aus scripts/verify_residual_feedback.py portiert (AR(1)-Gewinn, ρ~φ, Kausalitaet,
White-Noise-No-Harm, Warmup, Validierung). Plus Integrationstests:
forecast_next_day(residual_feedback=False) und rolling_origin(residual_feedback=False)
sind bit-identisch zum alten Verhalten; =True liefert die additiven Felder.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.models.residual_feedback import online_residual_feedback
from netzpilot.forecast import forecast_next_day
from netzpilot.eval.backtest import rolling_origin
from netzpilot.models.robust_corrector import ShrunkCorrector


# ---------- S1-S6: Synthetische Checks portiert ----------

@pytest.fixture(scope="module")
def ar1_series():
    """AR(1) Residuen mit phi=0.6 -> Online-Feedback sollte MAE senken."""
    rng = np.random.default_rng(0)
    n, H = 160, 4
    phi = 0.6
    r = np.zeros((n, H))
    for d in range(1, n):
        r[d] = phi * r[d - 1] + rng.normal(0, 1, H)
    f = np.full((n, H), 10.0)
    a = f + r
    return f, a, phi


def test_S1_ar1_corrected_mae_smaller_than_base(ar1_series):
    f, a, _ = ar1_series
    _, _, corr = online_residual_feedback(f, a, window=28, shrink=1.0, min_window=14)
    base = float(np.mean(np.abs(a[30:] - f[30:])))
    cmae = float(np.mean(np.abs(a[30:] - corr[30:])))
    assert cmae < base


def test_S2_rho_approx_phi_after_warmup(ar1_series):
    f, a, phi = ar1_series
    rho, _, _ = online_residual_feedback(f, a, window=28, shrink=1.0, min_window=14)
    assert abs(float(rho[40:].mean()) - phi) < 0.25


def test_S3_causal_no_leakage(ar1_series):
    """actual[i0] aendern -> corrected[:i0+1] unveraendert; corrected[i0+1] reagiert."""
    f, a, _ = ar1_series
    _, _, corr = online_residual_feedback(f, a, window=28, shrink=1.0)
    a2 = a.copy()
    i0 = 100
    a2[i0] += 50.0
    _, _, corr2 = online_residual_feedback(f, a2, window=28, shrink=1.0)
    assert np.allclose(corr[:i0 + 1], corr2[:i0 + 1])
    assert not np.allclose(corr[i0 + 1], corr2[i0 + 1])


def test_S4_white_noise_no_harm():
    rng = np.random.default_rng(0)
    n, H = 160, 4
    f = np.full((n, H), 10.0)
    rw = rng.normal(0, 1, (n, H))
    aw = f + rw
    rho_w, _, corr_w = online_residual_feedback(f, aw, window=28, shrink=1.0)
    assert float(rho_w[40:].mean()) < 0.2
    base = float(np.mean(np.abs(aw[30:] - f[30:])))
    corr_mae = float(np.mean(np.abs(aw[30:] - corr_w[30:])))
    assert corr_mae <= base * 1.02


def test_S5_first_min_window_days_rho_zero(ar1_series):
    f, a, _ = ar1_series
    rho, _, _ = online_residual_feedback(f, a, window=28, shrink=1.0, min_window=14)
    assert np.allclose(rho[:14], 0.0)


def test_S6_validation_errors(ar1_series):
    f, a, _ = ar1_series
    with pytest.raises(ValueError):
        online_residual_feedback(f[:, 0], a[:, 0])
    with pytest.raises(ValueError):
        online_residual_feedback(f, a[:50])
    with pytest.raises(ValueError):
        online_residual_feedback(f, a, shrink=1.5)


# ---------- Integration in forecast.py ----------

def _synth_load(n_days=70, H=24, seed=42):
    rng = np.random.default_rng(seed)
    base = 50.0 + 20.0 * np.sin(np.linspace(0, 2 * np.pi, H, endpoint=False))
    load = np.zeros((n_days, H), dtype=float)
    for d in range(n_days):
        load[d] = base + 5.0 * np.sin(2 * np.pi * d / 7.0) + rng.normal(0, 2, H)
    days = list(pd.date_range("2025-01-01", periods=n_days, freq="D"))
    return load, days


def test_T50_forecast_residual_feedback_false_bit_compatible():
    """residual_feedback=False (default) muss bit-identisch zum alten Verhalten sein."""
    load2d, days = _synth_load()
    fp_default = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0))
    fp_off = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0), residual_feedback=False)
    assert fp_default == fp_off


def test_T50_forecast_residual_feedback_true_adds_rf_field():
    """residual_feedback=True liefert das additive residual_feedback-Feld."""
    load2d, days = _synth_load()
    fp = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0), residual_feedback=True)
    assert "residual_feedback" in fp
    rf = fp["residual_feedback"]
    assert "rho" in rf and "delta_mean_MW" in rf
    assert rf["window"] == 28


# ---------- Integration in backtest.py ----------

def test_T50_backtest_residual_feedback_false_bit_compatible():
    """rolling_origin(residual_feedback=False) bleibt bit-identisch."""
    load2d, days = _synth_load(n_days=84)
    R0, sm0 = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=14)
    R1, sm1 = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=14,
                             residual_feedback=False)
    assert sm0 == sm1


def test_T50_backtest_residual_feedback_true_reports_rf_block():
    """rolling_origin(residual_feedback=True) liefert residual_feedback + metriken_naiv."""
    load2d, days = _synth_load(n_days=120)
    _, sm = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=42,
                           residual_feedback=True)
    assert "residual_feedback" in sm
    assert "metriken_naiv" in sm
    rf = sm["residual_feedback"]
    assert rf["window"] == 28
    assert "rho_mean" in rf and "delta_abs_mean_MW" in rf
    # Naive Pinball + Coverage sind reproduziert, naive MAE im Block
    assert "model_MAE_MW_naiv" in sm["metriken_naiv"]
