# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Tests fuer feiertagsbewusste Baseline (T48).

Synthetische S1-S7 aus scripts/verify_holiday_base.py portiert (deterministisch).
Plus Integrationstests: forecast_next_day + rolling_origin sind ohne lw-Feiertag-Tag
bit-identisch zu altem Verhalten; mit lw-Feiertag aendert sich nur der erwartete Anker.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.features.holiday_base import holiday_aware_base, holiday_aware_resid_target
from netzpilot.forecast import forecast_next_day
from netzpilot.eval.backtest import rolling_origin
from netzpilot.models.robust_corrector import ShrunkCorrector


# ---------- S1-S7: Synthetische Checks portiert aus verify_holiday_base.py ----------

@pytest.fixture(scope="module")
def synth_index():
    """Zeile i hat Wert i -> Rueckgabewert verraet die gewaehlte Referenz-Zeile."""
    N = 40
    l2 = np.tile(np.arange(N).reshape(-1, 1), (1, 3)).astype(float)
    days = pd.date_range("2025-01-06", periods=N, freq="D")  # Montag-Start
    return l2, days


def test_S1_backward_compatible_without_calendar(synth_index):
    l2, _ = synth_index
    assert np.array_equal(holiday_aware_base(l2, 21), l2[14])


def test_S2_d_minus_7_holiday_picks_d_minus_14(synth_index):
    l2, days = synth_index
    d = 21
    hs = {days[d - 7].date()}
    assert holiday_aware_base(l2, d, days, hs)[0] == d - 14


def test_S3_d_minus_7_and_14_holiday_picks_d_minus_21(synth_index):
    l2, days = synth_index
    d = 21
    hs = {days[d - 7].date(), days[d - 14].date()}
    assert holiday_aware_base(l2, d, days, hs)[0] == d - 21


def test_S4_d_minus_7_not_a_holiday_no_op(synth_index):
    l2, days = synth_index
    d = 21
    hs = {days[d - 8].date()}    # anderer Tag Feiertag — d-7 selbst nicht
    assert holiday_aware_base(l2, d, days, hs)[0] == d - 7


def test_S5_reference_strictly_before_d(synth_index):
    """Leakage-Guard: Referenz IMMER < d."""
    l2, days = synth_index
    for dd in range(7, len(l2)):
        hs = {days[dd - 7].date(), days[dd - 14].date()} if dd >= 14 else {days[dd - 7].date()}
        ref = holiday_aware_base(l2, dd, days, hs)[0]
        assert ref < dd, f"Leakage bei d={dd}: ref={ref}"


def test_S6_resid_target_consistent(synth_index):
    l2, days = synth_index
    d = 21
    hs = {days[d - 7].date()}
    rt = holiday_aware_resid_target(l2, d, days, hs)
    assert np.array_equal(rt, l2[d] - holiday_aware_base(l2, d, days, hs))


def test_S7_edge_cases_no_negative_index(synth_index):
    l2, days = synth_index
    with pytest.raises(ValueError):
        holiday_aware_base(l2, 5)
    # Alle moeglichen Vorwochen-Refs Feiertag -> faellt auf kleinste gueltige (>=0) zurueck.
    hs_all = {days[i].date() for i in range(len(l2))}
    ref_small = holiday_aware_base(l2, 13, days, hs_all)[0]   # d=13: d-7=6; d-14=-1 nicht erlaubt
    assert ref_small == 6


# ---------- Integrationstests: forecast.py + backtest.py ----------

def _synth_load(n_days=70, H=24, seed=42):
    rng = np.random.default_rng(seed)
    base = 50.0 + 20.0 * np.sin(np.linspace(0, 2 * np.pi, H, endpoint=False))
    load = np.zeros((n_days, H), dtype=float)
    for d in range(n_days):
        load[d] = base + 5.0 * np.sin(2 * np.pi * d / 7.0) + rng.normal(0.0, 2.0, H)
    days = list(pd.date_range("2025-01-01", periods=n_days, freq="D"))
    return load, days


def test_T48_forecast_no_op_without_holiday_set():
    """Ohne `holiday_set` (None) bit-identisch zum frueheren Verhalten (load2d[d-7]-Anker)."""
    load2d, days = _synth_load()
    fp_none = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0))
    fp_empty = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0), holiday_set=set())
    assert fp_none == fp_empty


def test_T48_backtest_no_op_without_holiday_set():
    """rolling_origin liefert ohne holiday_set bit-identisches Summary."""
    load2d, days = _synth_load(n_days=84)
    _, sm_none = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=14)
    _, sm_empty = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=14, holiday_set=set())
    assert sm_none == sm_empty


def test_T48_forecast_changes_when_lw_day_is_holiday():
    """Wenn ND-7 (lw-Tag) ein Feiertag ist, aendert sich der Anker → p50 verschiebt sich."""
    load2d, days = _synth_load(n_days=70)
    lw_day = pd.Timestamp(days[len(load2d) - 7]).date()
    fp_off = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0), holiday_set=set())
    fp_on = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0), holiday_set={lw_day})
    p50_off = [h["p50"] for h in fp_off["hours"]]
    p50_on = [h["p50"] for h in fp_on["hours"]]
    # Mindestens 1 Stunde muss anders sein (Anker = load2d[ND-14] statt load2d[ND-7])
    assert p50_off != p50_on


def test_T48_forecast_unchanged_when_holiday_far_from_lw():
    """Wenn Feiertag NICHT der lw-Tag ist, aendert sich gar nichts (Reparatur feuert nicht)."""
    load2d, days = _synth_load(n_days=70)
    # Setze einen Tag VOR fit_end als Feiertag — er ist nicht der lw-Tag der Prognose,
    # liegt aber im Trainings-Loop. Wenn er KEINEN d-7-Match in der ganzen Backtest-Loop ist,
    # bleibt alles gleich. Wir nehmen Tag 0 (vor first=8 → kein Trainingsverwendung).
    irrelevant = pd.Timestamp(days[0]).date()
    fp_off = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0), holiday_set=set())
    fp_on = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0), holiday_set={irrelevant})
    # Tag 0 ist < first=8, also nirgendwo ein d-7 in [first..ND]; deshalb bit-identisch.
    assert fp_off == fp_on
