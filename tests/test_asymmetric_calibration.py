# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Tests fuer asymmetrische Coverage-Kalibrierung (T49).

S1-S7 aus scripts/verify_asymmetric_calibration.py portiert (deterministisch).
Plus 1 Integrationstest: backtest meldet die T49-Felder (beide Tails, s_lo/s_hi, method).
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.eval.coverage_calibration import (
    asymmetric_coverage_scale, apply_asymmetric, rolling_asymmetric_scale,
)
from netzpilot.eval.backtest import rolling_origin
from netzpilot.models.robust_corrector import ShrunkCorrector


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def rechtsschief(rng):
    """Verschobene Exponentialverteilung — mehr Masse ueber +1 als unter -1."""
    N = 20000
    r = rng.exponential(0.6, N) - 0.5
    p50 = np.zeros(N); p10 = p50 - 1.0; p90 = p50 + 1.0
    return r, p10, p50, p90


def test_S1_setup_rechtsschief(rechtsschief):
    r, p10, p50, p90 = rechtsschief
    flo = float(np.mean(r < -1) * 100)
    fhi = float(np.mean(r > 1) * 100)
    assert fhi > flo, f"erwartet rechtsschief: fhi {fhi:.2f} > flo {flo:.2f}"


def test_S2_s_hi_groesser_s_lo_bei_rechtsschief(rechtsschief):
    r, p10, p50, p90 = rechtsschief
    s_lo, s_hi = asymmetric_coverage_scale(r, p10, p50, p90, target_tail=0.1, shrink=1.0)
    assert s_hi > s_lo


def test_S3_beide_tails_naeher_10_nach_apply(rechtsschief):
    r, p10, p50, p90 = rechtsschief
    flo0 = float(np.mean(r < p10) * 100)
    fhi0 = float(np.mean(r > p90) * 100)
    s_lo, s_hi = asymmetric_coverage_scale(r, p10, p50, p90, target_tail=0.1, shrink=1.0)
    lo, hi = apply_asymmetric(p10, p50, p90, s_lo, s_hi)
    flo1 = float(np.mean(r < lo) * 100)
    fhi1 = float(np.mean(r > hi) * 100)
    assert abs(flo1 - 10) <= abs(flo0 - 10) + 0.5
    assert abs(fhi1 - 10) <= abs(fhi0 - 10) + 0.5


def test_S4_symmetrisch_s_lo_approx_s_hi(rng):
    g = rng.normal(0.0, 1.0 / 1.2815515594, 20000)
    p50 = np.zeros(20000); p10 = p50 - 1.0; p90 = p50 + 1.0
    s_lo, s_hi = asymmetric_coverage_scale(g, p10, p50, p90, target_tail=0.1, shrink=1.0)
    assert abs(s_lo - s_hi) < 0.25, f"erwartet ~gleich, bekommen {s_lo:.3f}/{s_hi:.3f}"


def test_S5_shrink_zero_gives_unit_factors(rechtsschief):
    r, p10, p50, p90 = rechtsschief
    assert asymmetric_coverage_scale(r, p10, p50, p90, target_tail=0.1, shrink=0.0) == (1.0, 1.0)


def test_S6_validation_errors(rechtsschief):
    r, p10, p50, p90 = rechtsschief
    with pytest.raises(ValueError):
        asymmetric_coverage_scale(r, p10, p50, p90, target_tail=0.6)
    with pytest.raises(ValueError):
        apply_asymmetric(p10, p50, p90, -1, 1)
    with pytest.raises(ValueError):
        asymmetric_coverage_scale([], [], [], [])


# ---------- Rolling ----------

@pytest.fixture(scope="module")
def rolling_data(rng):
    n, H = 120, 24
    a = rng.exponential(0.6, (n, H)) - 0.5
    P50 = np.zeros((n, H)); P10 = P50 - 1.0; P90 = P50 + 1.0
    return a, P10, P50, P90


def test_S7_rolling_causal_no_leakage(rolling_data):
    a, P10, P50, P90 = rolling_data
    slo, shi, _, _ = rolling_asymmetric_scale(a, P10, P50, P90, window=28, min_window=14)
    ap = a.copy()
    i0 = 80
    ap[i0] = 999.0
    slo2, shi2, _, _ = rolling_asymmetric_scale(ap, P10, P50, P90, window=28, min_window=14)
    assert np.allclose(slo[:i0 + 1], slo2[:i0 + 1])
    assert np.allclose(shi[:i0 + 1], shi2[:i0 + 1])


def test_S7_rolling_first_min_window_no_op(rolling_data):
    a, P10, P50, P90 = rolling_data
    slo, shi, _, _ = rolling_asymmetric_scale(a, P10, P50, P90, window=28, min_window=14)
    assert np.allclose(slo[:14], 1.0)
    assert np.allclose(shi[:14], 1.0)


def test_S7_rolling_picks_up_rechtsschief_after_warmup(rolling_data):
    a, P10, P50, P90 = rolling_data
    slo, shi, _, _ = rolling_asymmetric_scale(a, P10, P50, P90, window=28, min_window=14)
    assert shi[40:].mean() > slo[40:].mean()


def test_S7_rolling_shape_validation():
    a = np.zeros((20, 24)); p10 = a - 1; p50 = a; p90 = a + 1
    with pytest.raises(ValueError):
        rolling_asymmetric_scale(a[:, 0], p10[:, 0], p50[:, 0], p90[:, 0])
    with pytest.raises(ValueError):
        rolling_asymmetric_scale(a, p10[:-1], p50, p90)


# ---------- Integration in backtest._calibration_summary ----------

def _synth_load(n_days=240, H=24, seed=42):
    rng = np.random.default_rng(seed)
    base = 50.0 + 20.0 * np.sin(np.linspace(0, 2 * np.pi, H, endpoint=False))
    load = np.zeros((n_days, H), dtype=float)
    for d in range(n_days):
        load[d] = base + 5.0 * np.sin(2 * np.pi * d / 7.0) + rng.normal(0.0, 2.0, H)
    days = list(pd.date_range("2025-01-01", periods=n_days, freq="D"))
    return load, days


def test_T49_backtest_reports_asymmetric_fields():
    """Summary enthaelt T49-Felder: beide Tail-Anteile (naiv+kalibriert), s_lo/s_hi, method."""
    load2d, days = _synth_load(n_days=240)
    _, sm = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0),
                           n_test=84, calibrate=True)
    prob = sm["probabilistisch"]
    assert prob.get("coverage_scale_method") == "online-rolling-asymmetric"
    for k in ("frac_below_P10_%", "frac_above_P90_%",
              "frac_below_P10_kalibriert_%", "frac_above_P90_kalibriert_%",
              "coverage_scale_lo_used", "coverage_scale_hi_used",
              "coverage_scale_lo_median", "coverage_scale_hi_median"):
        assert prob.get(k) is not None, f"erwartetes Feld {k!r} fehlt"


def test_T49_backtest_pinball_not_worse_under_asymmetric():
    """Asymmetrische Kalibrierung darf Pinball nicht verschlechtern (vs. naiv)."""
    load2d, days = _synth_load(n_days=240)
    _, sm = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0),
                           n_test=84, calibrate=True)
    prob = sm["probabilistisch"]
    pin_naiv = float(prob["Pinball_avg"])
    pin_kal = float(prob["Pinball_avg_kalibriert"])
    assert pin_kal <= pin_naiv + 0.15, f"Pinball verschlechtert: {pin_naiv} -> {pin_kal}"
