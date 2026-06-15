import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _assets3():
    return [
        {"id": "A", "demand_kw": [10.0, 10.0], "floor_kw": 4.2},
        {"id": "B", "demand_kw": [10.0, 10.0], "floor_kw": 4.2},
        {"id": "C", "demand_kw": [10.0, 10.0], "floor_kw": 4.2},
    ]


def test_vpp_verify_anchors():
    from netzpilot.control.vpp_pool import pool_dispatch

    r = pool_dispatch(_assets3(), [40.0, 40.0])
    assert all(h["pool_limit_kw"] == 30.0 for h in r["hourly"])
    assert r["pool_shed_kwh"] == 0.0
    assert r["grid_safe"] is True
    assert r["all_feasible"] is True

    r = pool_dispatch(_assets3(), [24.0, 24.0])
    assert all(abs(h["pool_limit_kw"] - 24.0) < 1e-6 for h in r["hourly"])
    assert all(abs(x - 8.0) < 1e-3 for h in r["hourly"] for x in h["asset_limits_kw"])
    assert all(x >= 4.2 - 1e-9 for h in r["hourly"] for x in h["asset_limits_kw"])
    assert abs(r["pool_shed_kwh"] - 12.0) < 1e-6
    assert r["grid_safe"] is True

    r = pool_dispatch(_assets3(), [10.0, 10.0])
    assert r["all_feasible"] is False
    assert all(abs(x - 4.2) < 1e-9 for h in r["hourly"] for x in h["asset_limits_kw"])
    assert r["grid_safe"] is False

    r = pool_dispatch(_assets3(), [40.0, 40.0])
    assert all(abs(b["min_kw"] - 12.6) < 1e-6 for b in r["pool_band"])
    assert all(b["max_kw"] == 30.0 for b in r["pool_band"])

    r = pool_dispatch(_assets3(), [24.0, 24.0])
    assert all(abs(h["pool_limit_kw"] - sum(h["asset_limits_kw"])) < 1e-6 for h in r["hourly"])
    assert all(a["granted_kwh"] <= a["demand_kwh"] + 1e-9 and a["shed_kwh"] >= -1e-9
               for a in r["per_asset"])
    assert abs(r["pool_demand_kwh"] - (r["pool_granted_kwh"] + r["pool_shed_kwh"])) < 1e-6

    mix = [
        {"id": "WP", "demand_kw": [20.0], "floor_kw": 7.0, "weight": 1.0},
        {"id": "WB", "demand_kw": [20.0], "floor_kw": 4.2, "weight": 2.0},
    ]
    r = pool_dispatch(mix, [30.0])
    limits = r["hourly"][0]["asset_limits_kw"]
    assert abs(r["hourly"][0]["pool_limit_kw"] - 30.0) < 1e-6
    assert (20.0 - limits[1]) > (20.0 - limits[0]) - 1e-9
    assert limits[0] >= 7.0 - 1e-9

    with pytest.raises(ValueError):
        pool_dispatch([], [10.0])
    with pytest.raises(ValueError):
        pool_dispatch([{"demand_kw": [1.0, 2.0]}], [10.0])


REAL_CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_runner_adds_pool_dispatch_and_single_rating_truth():
    from netzpilot.service.runner import run_forecast

    assets = [
        {"id": "WP-1", "demand_kw": [10.0, 10.0], "floor_kw": 4.2},
        {"id": "WB-1", "demand_kw": [10.0, 10.0], "floor_kw": 4.2},
        {"id": "BAT-1", "demand_kw": [10.0, 10.0], "floor_kw": 4.2},
    ]
    out = run_forecast(
        REAL_CSV,
        utility="VppTest",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        rating_kw=33000.0,
        pool_assets=assets,
        pool_shared_cap_kw=[40.0, 24.0],
    )
    assert out["asset_limit"]["rating_kw"] == 33000.0
    assert out["asset_limit"]["feeds"]["congestion_threshold_mw"] == 33.0
    assert out["asset_limit"]["feeds"]["asset_rating_kw"] == 33000.0
    pool = out["pool_dispatch"]
    assert pool["n_assets"] == 3
    assert pool["grid_safe"] is True
    assert pool["hourly"][1]["pool_limit_kw"] == 24.0
    assert all(abs(x - 8.0) < 1e-3 for x in pool["hourly"][1]["asset_limits_kw"])


def test_runner_rejects_diverging_rating_truth():
    from netzpilot.service.runner import run_forecast

    with pytest.raises(ValueError):
        run_forecast(
            REAL_CSV,
            utility="VppTest",
            unit="kW",
            ts_col="Text",
            load_col="Reihe1",
            rating_kw=33000.0,
            asset_rating_kw=34000.0,
        )
