# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import math

import pytest

from netzpilot.grid.thermal import (
    DEFAULT_PARAMS,
    hotspot_trajectory,
    probabilistic_thermal_risk,
    relative_aging_factor,
)


def test_thermal_verify_anchors():
    rated = hotspot_trajectory([100.0] * 24, [20.0] * 24, rating_kw=100.0)
    assert abs(rated["hourly"][-1]["hotspot_c"] - 98.0) < 0.5
    assert abs(rated["equivalent_aging_factor"] - 1.0) < 0.05

    hot = hotspot_trajectory([140.0] * 24, [35.0] * 24, rating_kw=100.0)
    assert hot["max_hotspot_c"] > 120.0
    assert hot["equivalent_aging_factor"] > 20.0

    cool_low = hotspot_trajectory([90.0] * 8, [15.0] * 8, 100.0)
    cool_high = hotspot_trajectory([110.0] * 8, [15.0] * 8, 100.0)
    warm_low = hotspot_trajectory([90.0] * 8, [30.0] * 8, 100.0)
    assert cool_high["max_hotspot_c"] > cool_low["max_hotspot_c"]
    assert warm_low["max_hotspot_c"] > cool_low["max_hotspot_c"]

    assert abs(relative_aging_factor(98.0) - 1.0) < 1e-12
    assert abs(relative_aging_factor(104.0) - 2.0) < 1e-12


def test_thermal_recursion_one_step_matches_exponential():
    p = DEFAULT_PARAMS
    step = hotspot_trajectory(
        [100.0],
        [20.0],
        100.0,
        initial_top_oil_rise_c=0.0,
        initial_winding_rise_c=0.0,
    )
    h0 = step["hourly"][0]
    oil_expected = p.top_oil_rise_rated_c * (1.0 - math.exp(-1.0 / p.tau_oil_h))
    wdg_expected = p.winding_rise_rated_c * (1.0 - math.exp(-1.0 / p.tau_winding_h))
    assert abs(h0["top_oil_rise_c"] - oil_expected) < 1e-4
    assert abs(h0["winding_rise_c"] - wdg_expected) < 1e-4


def test_probabilistic_thermal_risk_and_validation():
    point = [100.0, 110.0]
    residuals = [[-10.0, 0.0, 20.0, 40.0], [-10.0, 0.0, 20.0, 40.0]]
    r = probabilistic_thermal_risk(point, residuals, 100.0, [25.0, 25.0], hotspot_limit_c=120.0)
    assert r["n_scenarios"] == 4
    assert r["max_exceedance_prob"] > 0.0
    assert r["expected_loss_of_life_h_total"] > 0.0
    assert len(r["hourly"]) == 2

    with pytest.raises(ValueError):
        hotspot_trajectory([], [], 100.0)
    with pytest.raises(ValueError):
        hotspot_trajectory([1.0], [20.0], 0.0)
    with pytest.raises(ValueError):
        hotspot_trajectory([1.0, 2.0], [20.0], 100.0)
    with pytest.raises(ValueError):
        probabilistic_thermal_risk([1.0], [[0.0]], 100.0, risk_alpha=1.0)
