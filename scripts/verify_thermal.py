#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify transformer thermal model (grid/thermal.py) -- stdlib only, no internet."""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.grid.thermal import (
    DEFAULT_PARAMS,
    hotspot_trajectory,
    probabilistic_thermal_risk,
    relative_aging_factor,
)

ok = True


def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond


# S1: rated load at 20 C -> hotspot near 98 C and aging factor near 1.
r = hotspot_trajectory([100.0] * 24, [20.0] * 24, rating_kw=100.0)
check("S1: rated load hotspot ~98 C", abs(r["hourly"][-1]["hotspot_c"] - 98.0) < 0.5)
check("S1: rated load aging factor ~1", abs(r["equivalent_aging_factor"] - 1.0) < 0.05)

# S2: overload in hot ambient -> strong aging.
hot = hotspot_trajectory([140.0] * 24, [35.0] * 24, rating_kw=100.0)
check("S2: overload+heat exceeds 120 C", hot["max_hotspot_c"] > 120.0)
check("S2: overload+heat aging >> 1", hot["equivalent_aging_factor"] > 20.0)

# S3: monotonicity in load and ambient.
cool_low = hotspot_trajectory([90.0] * 8, [15.0] * 8, 100.0)
cool_high = hotspot_trajectory([110.0] * 8, [15.0] * 8, 100.0)
warm_low = hotspot_trajectory([90.0] * 8, [30.0] * 8, 100.0)
check("S3: higher load -> higher hotspot", cool_high["max_hotspot_c"] > cool_low["max_hotspot_c"])
check("S3: higher ambient -> higher hotspot", warm_low["max_hotspot_c"] > cool_low["max_hotspot_c"])

# S4: recursion matches one-step exponential update from zero initial state.
p = DEFAULT_PARAMS
step = hotspot_trajectory(
    [100.0],
    [20.0],
    100.0,
    initial_top_oil_rise_c=0.0,
    initial_winding_rise_c=0.0,
)
oil_expected = p.top_oil_rise_rated_c * (1.0 - math.exp(-1.0 / p.tau_oil_h))
wdg_expected = p.winding_rise_rated_c * (1.0 - math.exp(-1.0 / p.tau_winding_h))
h0 = step["hourly"][0]
check("S4: top-oil one-step exponential", abs(h0["top_oil_rise_c"] - oil_expected) < 1e-4)
check("S4: winding one-step exponential", abs(h0["winding_rise_c"] - wdg_expected) < 1e-4)

# S5: probabilistic risk over residual scenarios.
point = [100.0, 110.0]
residuals = [[-10.0, 0.0, 20.0, 40.0], [-10.0, 0.0, 20.0, 40.0]]
pr = probabilistic_thermal_risk(point, residuals, 100.0, [25.0, 25.0], hotspot_limit_c=120.0)
check("S5: scenario count", pr["n_scenarios"] == 4)
check("S5: nonzero thermal risk", pr["max_exceedance_prob"] > 0.0)
check("S5: aging positive", pr["expected_loss_of_life_h_total"] > 0.0)

# S6: validation.
def raises(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


check("S6: empty horizon -> ValueError", raises(lambda: hotspot_trajectory([], [], 100.0)))
check("S6: rating<=0 -> ValueError", raises(lambda: hotspot_trajectory([1.0], [20.0], 0.0)))
check("S6: ambient length mismatch -> ValueError", raises(lambda: hotspot_trajectory([1.0, 2.0], [20.0], 100.0)))
check("S6: invalid risk alpha -> ValueError",
      raises(lambda: probabilistic_thermal_risk([1.0], [[0.0]], 100.0, risk_alpha=1.0)))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG - siehe FAIL")
sys.exit(0 if ok else 1)
