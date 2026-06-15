import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.control.redispatch import from_single_path, rolling_redispatch
from netzpilot.control.schema import MIN_GUARANTEED_KW

REAL_CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"


def test_rolling_redispatch_no_congestion_no_shed():
    demands = [11.0] * 50
    path = [3000.0 + sum(demands)] * 24
    r = rolling_redispatch(from_single_path(path), 5000.0, demands)
    assert r["intervention_hours"] == 0
    assert r["total_shed_kwh"] == 0.0
    assert all(not h["intervention"] for h in r["hourly"])


def test_rolling_redispatch_constraints_in_peak_hours():
    demands = [11.0] * 50
    base = 4600.0
    threshold = 5000.0
    peak = base + sum(demands)
    flat = base + 200.0
    r = rolling_redispatch(from_single_path([flat] * 8 + [peak] * 4 + [flat] * 12),
                           threshold, demands)
    interv = [h for h in r["hourly"] if h["intervention"]]
    assert len(interv) == 4
    assert all(base + sum(h["limits_kw"]) <= threshold + 1e-6 for h in interv)
    assert all(all(l >= MIN_GUARANTEED_KW - 1e-9 for l in h["limits_kw"]) for h in interv)


def test_rolling_redispatch_sheds_less_than_naive_when_forecast_updates():
    demands = [11.0] * 50
    threshold = 5000.0
    base = 4600.0
    flat = base + 200.0
    peak = base + sum(demands)
    actual_next = {10: peak, 11: peak, 12: flat, 13: flat, 14: flat, 15: peak}
    forecasts = []
    for t in range(24):
        forecasts.append([actual_next.get(t, flat)] + [peak] * (24 - t - 1))
    r = rolling_redispatch(forecasts, threshold, demands)
    assert r["intervention_hours"] == 3
    assert r["saved_vs_naive_kwh"] > 0
    assert r["total_shed_kwh"] <= r["naive_shed_kwh"]


def test_rolling_redispatch_accepts_heterogeneous_devices():
    devices = [
        {"demand_kw": 15.0, "floor_kw": 7.0, "weight": 0.5},
        {"demand_kw": 11.0, "floor_kw": 4.2, "weight": 1.0},
        {"demand_kw": 10.0, "floor_kw": 0.0, "weight": 2.0},
    ]
    base = 1000.0
    path = [base + sum(d["demand_kw"] for d in devices)] * 24
    r = rolling_redispatch(from_single_path(path), base + 18.0, steuve_devices=devices)
    first = next(h for h in r["hourly"] if h["intervention"])
    assert r["heterogeneous"] is True
    assert abs(sum(first["limits_kw"]) - first["cap_kw"]) < 1e-6
    assert first["limits_kw"][0] >= 7.0 - 1e-9
    assert first["limits_kw"][1] >= 4.2 - 1e-9
    assert first["limits_kw"][2] >= -1e-9


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_runner_adds_static_day_ahead_redispatch_field():
    from netzpilot.service.runner import run_forecast

    r = run_forecast(
        REAL_CSV,
        utility="RedispatchTest",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        congestion_threshold_mw=33.0,
        steuve_malo="DE0001234567890",
        steuve_demands_kw=[1000.0, 800.0, 600.0],
        rolling_redispatch=True,
    )
    rd = r["redispatch"]
    assert rd is not None
    assert rd["forecast_basis"] == "day_ahead_p50_static"
    assert rd["basis"] == "load"
    assert rd["total_shed_kwh"] <= rd["naive_shed_kwh"] + 1e-9
    assert rd["saved_vs_naive_kwh"] >= -1e-9
    assert rd["intervention_hours"] > 0
    for h in [x for x in rd["hourly"] if x["intervention"]]:
        assert all(l >= MIN_GUARANTEED_KW - 1e-9 for l in h["limits_kw"])
        if h["feasible"]:
            assert sum(h["limits_kw"]) <= h["cap_kw"] + 1e-6
