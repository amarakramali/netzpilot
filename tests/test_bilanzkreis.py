# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.eval.bilanzkreis import compare_forecasts_eur, imbalance_premium_eur
from netzpilot.eval.bilanzkreis_realized import official_rebap_asymmetry_count

HERNE_CSV = "data_cache/real/herne_bezug_vorgelagerte_ebene_2024.csv"
REBAP_CSV = "data_cache/real/rebap_2024.csv"
REBAP_OFFICIAL_CSV = "data_cache/real/rebap_2024_official.csv"
SPOT_CSV = "data_cache/real/spot_da_2024.csv"


def _premium(errors, spreads):
    return imbalance_premium_eur([0.0] * len(errors), errors, spreads, None)


def test_perfect_forecast_has_zero_premium():
    r = _premium([0.0, 0.0, 0.0], [50.0, -30.0, 80.0])
    assert r["total_premium_eur"] == 0.0


def test_uncorrelated_zero_mean_error_beats_linear_approximation():
    r = _premium([3.0, -3.0, 3.0, -3.0], [100.0] * 4)
    linear = r["sum_abs_error_mwh"] * r["mean_spread_eur_mwh"]
    assert abs(r["total_premium_eur"]) < 1e-9
    assert linear == 1200.0
    assert r["sum_abs_error_mwh"] == 12.0


def test_correlated_error_is_correlation_term():
    r = _premium([2.0, 2.0, -2.0, -2.0], [50.0, 50.0, -50.0, -50.0])
    assert r["total_premium_eur"] == 400.0
    assert abs(r["bias_term_eur"]) < 1e-9
    assert r["correlation_term_eur"] == 400.0
    cmp = compare_forecasts_eur(
        [2.0, 2.0, -2.0, -2.0],
        [0.0] * 4,
        [2.0, 2.0, -2.0, -2.0],
        [50.0, 50.0, -50.0, -50.0],
        None,
    )
    assert cmp["savings_b_vs_a_eur"] == 400.0


def test_savings_linearity_and_decomposition_identity():
    actual = [10.0, 12.0, 9.0, 11.0, 8.0]
    sa = [9.0, 13.0, 9.5, 10.0, 9.0]
    sb = [10.2, 11.8, 9.1, 11.0, 8.2]
    reb = [60.0, -20.0, 120.0, 5.0, -40.0]
    spot = [40.0, 35.0, 45.0, 38.0, 30.0]
    cmp = compare_forecasts_eur(actual, sa, sb, reb, spot)
    direct = sum(((actual[i] - sa[i]) - (actual[i] - sb[i])) * (reb[i] - spot[i])
                 for i in range(len(actual)))
    assert cmp["savings_b_vs_a_eur"] == round(direct, 2)

    r = _premium([1.5, -0.5, 2.0, -3.0, 0.7, 4.0], [30.0, -10.0, 90.0, 12.0, -25.0, 7.0])
    assert abs(r["total_premium_eur"] - (r["bias_term_eur"] + r["correlation_term_eur"])) < 1e-9


def test_nan_drop_and_mismatch_validation():
    r = imbalance_premium_eur([0.0, 0.0, 0.0, 0.0],
                              [2.0, float("nan"), 3.0, None],
                              [10.0, 10.0, 10.0, 10.0],
                              None)
    assert r["n"] == 2
    assert r["n_dropped"] == 2
    assert math.isfinite(r["total_premium_eur"])
    assert r["total_premium_eur"] == 50.0

    with pytest.raises(ValueError):
        imbalance_premium_eur([0.0, 0.0], [1.0], [10.0, 10.0], None)


def test_spot_none_equals_zero_spot():
    r0 = imbalance_premium_eur([0.0] * 3, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0], None)
    rz = imbalance_premium_eur([0.0] * 3, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [0.0, 0.0, 0.0])
    assert r0["total_premium_eur"] == rz["total_premium_eur"]
    assert r0["total_premium_eur"] == 140.0


@pytest.mark.skipif(
    not all(os.path.exists(p) for p in [HERNE_CSV, REBAP_CSV, REBAP_OFFICIAL_CSV, SPOT_CSV]),
    reason="Bilanzkreis-Realdaten fehlen",
)
def test_runner_realized_economics_smoke_on_real_data():
    from netzpilot.service.runner import run_forecast

    asym = official_rebap_asymmetry_count(REBAP_OFFICIAL_CSV)
    assert asym["n_qh"] == 35136
    assert asym["n_asymmetric_qh"] == 0

    r = run_forecast(
        HERNE_CSV,
        utility="HerneTest",
        unit="kW",
        ts_col="Datum+von",
        load_col="Load_1",
        rebap_csv=REBAP_CSV,
        spot_csv=SPOT_CSV,
        realized_economics=True,
    )
    e = r["economics_realized"]
    assert e is not None and e.get("status") != "not_available"
    assert e["resolution"] == "hour"
    assert e["n_periods"] == 336
    assert e["savings_eur_per_year"] != e["linear_expected_eur_per_year"]
    assert abs(e["savings_eur_per_year"]
               - (e["savings_bias_term_eur_per_year"] + e["savings_correlation_term_eur_per_year"])) < 1.0
