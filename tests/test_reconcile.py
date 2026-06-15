# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import os

import numpy as np
import pytest

from netzpilot.models.reconcile import (
    _w_inv,
    build_summing_matrix,
    build_temporal_summing_matrix,
    coherence_error,
    reconcile,
)
from netzpilot.service.reconcile_temporal import (
    load_quarter_hour_energy,
    temporal_reconciliation_backtest,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_CSV = os.path.join(ROOT, "data_cache", "real", "Netzumsatz-Lastgang-2025.csv")


def test_summing_matrix_and_basic_coherence():
    S, names = build_summing_matrix(["a", "b", "c"], {"total": ["a", "b", "c"]})
    assert names == ["total", "a", "b", "c"]
    assert S.shape == (4, 3)
    base = np.array([100.0, 30.0, 30.0, 30.0])
    for method in ("ols", "wls_struct"):
        rec = reconcile(base, S, method=method)
        assert coherence_error(rec, S) < 1e-9


def test_projection_idempotent_and_preserves_coherent_input():
    S, _ = build_summing_matrix(["a", "b", "c"], {"total": ["a", "b", "c"]})
    base = np.array([100.0, 30.0, 30.0, 30.0])
    r1 = reconcile(base, S, "ols")
    r2 = reconcile(r1, S, "ols")
    assert np.allclose(r1, r2, atol=1e-9)
    coherent = np.array([45.0, 20.0, 15.0, 10.0])
    assert np.allclose(reconcile(coherent, S, "ols"), coherent, atol=1e-9)


def test_ols_hand_value():
    S, _ = build_summing_matrix(["a", "b"], {"total": ["a", "b"]})
    rec = reconcile(np.array([10.0, 4.0, 5.0]), S, "ols")
    assert np.allclose(rec, [29 / 3, 13 / 3, 16 / 3], atol=1e-6)
    assert abs(rec[0] - (rec[1] + rec[2])) < 1e-9


def test_ols_projection_never_increases_total_sse_for_coherent_truth():
    S, _ = build_summing_matrix(["a", "b", "c"], {"total": ["a", "b", "c"]})
    rng = np.random.default_rng(42)
    n_draw = 4000
    worse = 0
    sse_base = 0.0
    sse_rec = 0.0
    for _ in range(n_draw):
        bottoms = rng.uniform(5.0, 15.0, 3)
        truth = np.array([bottoms.sum(), *bottoms])
        base = truth + rng.normal(0.0, 1.0, 4)
        rec = reconcile(base, S, "ols")
        eb = float(np.sum((base - truth) ** 2))
        er = float(np.sum((rec - truth) ** 2))
        sse_base += eb
        sse_rec += er
        if er > eb + 1e-9:
            worse += 1
    assert worse == 0
    assert sse_rec < sse_base


def test_wls_struct_reduces_mean_total_sse_and_has_expected_weights():
    S, _ = build_summing_matrix(["a", "b", "c"], {"total": ["a", "b", "c"]})
    rng = np.random.default_rng(7)
    n_draw = 4000
    sse_base = 0.0
    sse_rec = 0.0
    for _ in range(n_draw):
        bottoms = rng.uniform(5.0, 15.0, 3)
        truth = np.array([bottoms.sum(), *bottoms])
        base = truth + rng.normal(0.0, 1.0, 4)
        rec = reconcile(base, S, "wls_struct")
        sse_base += float(np.sum((base - truth) ** 2))
        sse_rec += float(np.sum((rec - truth) ** 2))
    assert sse_rec < sse_base
    Winv = _w_inv("wls_struct", S, None, None)
    assert np.allclose(np.diag(Winv), [1 / 3, 1, 1, 1])


def test_variance_methods_and_horizon_matrix_are_coherent():
    S, _ = build_summing_matrix(["a", "b", "c"], {"total": ["a", "b", "c"]})
    rng = np.random.default_rng(7)
    resid = rng.normal(0, 1, (4, 200))
    for method in ("wls_var", "mint_shrink"):
        rec = reconcile(np.array([100.0, 30.0, 30.0, 30.0]), S, method=method, residuals=resid)
        assert coherence_error(rec, S) < 1e-9
    baseH = np.array([[100.0, 90.0], [30.0, 30.0], [30.0, 30.0], [30.0, 30.0]])
    recH = reconcile(baseH, S, "ols")
    assert recH.shape == (4, 2)
    assert coherence_error(recH, S) < 1e-9


def test_validation_errors():
    S, _ = build_summing_matrix(["a", "b", "c"], {"total": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        build_summing_matrix(["a"], {"t": ["x"]})
    with pytest.raises(ValueError):
        reconcile(np.array([1.0, 1.0]), build_summing_matrix(["a"], {"t": ["a"]})[0], "wls_var")
    with pytest.raises(ValueError):
        reconcile(np.array([1.0, 2.0]), S, "ols")


def test_temporal_hierarchy_reconciles_exactly():
    St, names = build_temporal_summing_matrix(12, [12, 4])
    assert St.shape == (16, 12)
    assert names[-12:] == [f"q{i}" for i in range(12)]
    rng = np.random.default_rng(3)
    base = rng.uniform(1.0, 5.0, 16)
    rec = reconcile(base, St, "ols")
    assert coherence_error(rec, St) < 1e-9
    assert abs(rec[0] - rec[-12:].sum()) < 1e-9
    with pytest.raises(ValueError):
        build_temporal_summing_matrix(12, [5])


def test_real_temporal_reconciliation_holdout_is_coherent():
    q2d, days, meta = load_quarter_hour_energy(
        REAL_CSV,
        ts_col="Text",
        load_col="Reihe1",
        unit="kW",
    )
    assert q2d.shape[1] == 96
    assert meta["complete_days"] >= 360
    result = temporal_reconciliation_backtest(q2d[-90:], days[-90:], n_test=2)
    assert result["coherence"]["before_max"] > 1e-6
    assert result["coherence"]["after_max"] < 1e-6
