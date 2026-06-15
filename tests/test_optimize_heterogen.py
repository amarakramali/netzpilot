import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.control.optimize import optimize_setpoints, optimize_setpoints_heterogen
from netzpilot.control.schema import (
    MIN_GUARANTEED_KW,
    make_fahrplan,
    normalize_steuve_devices,
)

TOL = 1e-6


def test_heterogeneous_optimizer_no_congestion():
    devices = [{"demand_kw": 11.0}, {"demand_kw": 7.0}, {"demand_kw": 22.0}]
    r = optimize_setpoints_heterogen(devices, cap_kw=45.0)
    assert r["limits_kw"] == [11.0, 7.0, 22.0]
    assert r["total_shed_kw"] == 0.0
    assert r["feasible"] is True
    assert r["binding_floor"] is False


def test_heterogeneous_optimizer_exact_cap_and_default_floors():
    devices = [{"demand_kw": 11.0}, {"demand_kw": 7.0}, {"demand_kw": 22.0}]
    r = optimize_setpoints_heterogen(devices, cap_kw=28.0)
    assert abs(sum(r["limits_kw"]) - 28.0) < TOL
    assert abs(r["total_shed_kw"] - 12.0) < 1e-3
    assert all(l >= MIN_GUARANTEED_KW - 1e-9 for l in r["limits_kw"])
    assert all(l <= d["demand_kw"] + 1e-9 for l, d in zip(r["limits_kw"], devices))
    assert r["feasible"] is True


def test_heterogeneous_optimizer_respects_individual_floors():
    devices = [
        {"demand_kw": 11.0, "floor_kw": 4.2},
        {"demand_kw": 15.0, "floor_kw": 7.0},
        {"demand_kw": 10.0, "floor_kw": 0.0},
    ]
    r = optimize_setpoints_heterogen(devices, cap_kw=18.0)
    assert abs(sum(r["limits_kw"]) - 18.0) < TOL
    assert r["limits_kw"][0] >= 4.2 - 1e-9
    assert r["limits_kw"][1] >= 7.0 - 1e-9
    assert r["limits_kw"][2] >= -1e-9


def test_heterogeneous_optimizer_infeasible_returns_floors():
    devices = [
        {"demand_kw": 11.0, "floor_kw": 4.2},
        {"demand_kw": 15.0, "floor_kw": 7.0},
        {"demand_kw": 10.0, "floor_kw": 0.0},
    ]
    r = optimize_setpoints_heterogen(devices, cap_kw=5.0)
    assert r["feasible"] is False
    assert r["limits_kw"] == [4.2, 7.0, 0.0]


def test_heterogeneous_optimizer_weight_semantics_and_weight_below_one():
    weighted = [{"demand_kw": 20.0, "weight": 2.0}, {"demand_kw": 20.0, "weight": 1.0}]
    r = optimize_setpoints_heterogen(weighted, cap_kw=30.0)
    shed0 = 20.0 - r["limits_kw"][0]
    shed1 = 20.0 - r["limits_kw"][1]
    assert shed0 > shed1 + 1e-6
    assert abs(shed0 - 2 * shed1) < 5e-3
    assert abs(sum(r["limits_kw"]) - 30.0) < TOL

    low_weight = [{"demand_kw": 100.0, "weight": 0.3}, {"demand_kw": 50.0, "weight": 0.3}]
    r = optimize_setpoints_heterogen(low_weight, cap_kw=40.0)
    assert abs(sum(r["limits_kw"]) - 40.0) < TOL
    assert all(l >= MIN_GUARANTEED_KW - 1e-9 for l in r["limits_kw"])


def test_homogeneous_optimizer_regression():
    r = optimize_setpoints([10.0, 10.0, 4.0], cap_kw=18.0)
    assert abs(sum(r["limits_kw"]) - 18.0) < TOL
    assert r["limits_kw"] == [7.0, 7.0, 4.0]


def test_schema_normalizes_devices_and_accepts_explicit_low_floor():
    devices = normalize_steuve_devices(
        steuve_devices=[{"demand_kw": 10, "floor_kw": 0, "weight": 2, "device_id": "battery"}]
    )
    assert devices == [{"demand_kw": 10.0, "floor_kw": 0.0, "weight": 2.0, "device_id": "battery"}]
    fp = make_fahrplan(
        "DE0001234567890",
        [{"start_utc": "2026-01-01T10:00:00", "end_utc": "2026-01-01T11:00:00",
          "p_limit_kw": 0.0, "floor_kw": 0.0}],
    )
    assert fp["setpoints"][0]["p_limit_kw"] == 0.0

    with pytest.raises(ValueError):
        make_fahrplan(
            "DE0001234567890",
            [{"start_utc": "2026-01-01T10:00:00", "end_utc": "2026-01-01T11:00:00",
              "p_limit_kw": 0.0}],
        )
