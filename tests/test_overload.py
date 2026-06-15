import os

import pytest

from netzpilot.grid.overload import _exceedance_prob, hosting_capacity_kw, overload_forecast

REAL_CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"


def test_overload_verify_anchors():
    r = overload_forecast([100.0] * 4, [[-10.0, -5.0, 0.0, 5.0, 10.0]] * 4, rating_kw=200.0)
    assert all(h["exceedance_prob"] == 0.0 for h in r["hourly"])
    assert r["expected_overload_kwh_total"] == 0.0
    assert r["hours_at_risk"] == 0

    r = overload_forecast([100.0] * 4, [[-5.0, 0.0, 5.0]] * 4, rating_kw=80.0)
    assert all(h["exceedance_prob"] == 1.0 for h in r["hourly"])
    assert abs(r["hourly"][0]["expected_overload_kwh"] - 20.0) < 1e-9

    r = overload_forecast([100.0], [[-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]], rating_kw=110.0)
    h0 = r["hourly"][0]
    assert abs(h0["exceedance_prob"] - 2.0 / 7.0) < 1e-3
    assert abs(h0["expected_overload_kwh"] - 15.0 / 7.0) < 1e-3
    assert abs(h0["p90_load_kw"] - 117.0) < 1e-6

    pt = [100.0, 100.0]
    residuals = [[-20.0, -10.0, 0.0, 10.0, 20.0]] * 2
    hc = hosting_capacity_kw(pt, residuals, rating_kw=150.0, risk_alpha=0.2)
    cap = hc["hosting_capacity_kw"]
    assert max(_exceedance_prob(100.0, sorted(residuals[0]), 150.0, extra_kw=cap) for _ in range(2)) <= 0.2
    assert abs(cap - 40.0) < 0.5

    caps = [hosting_capacity_kw(pt, residuals, 150.0, risk_alpha=a)["hosting_capacity_kw"] for a in (0.05, 0.2, 0.4)]
    assert caps[0] <= caps[1] <= caps[2]

    hc = hosting_capacity_kw([100.0], [[-5.0, 0.0, 5.0]], rating_kw=90.0, risk_alpha=0.05)
    assert hc["already_at_risk"] is True
    assert hc["hosting_capacity_kw"] == 0.0


def test_overload_validation_errors():
    with pytest.raises(ValueError):
        overload_forecast([], [], 100.0)
    with pytest.raises(ValueError):
        overload_forecast([100.0], [[0.0]], 0.0)
    with pytest.raises(ValueError):
        overload_forecast([100.0, 100.0], [[0.0]], 100.0)


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_runner_adds_overload_and_hosting_capacity_on_real_data():
    from netzpilot.service.runner import run_forecast

    out = run_forecast(
        REAL_CSV,
        utility="OverloadReal",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        congestion_threshold_mw=33.0,
        asset_rating_kw=33_000.0,
        overload_risk_alpha=0.05,
    )
    overload = out["overload"]
    hosting = out["hosting_capacity"]
    assert overload is not None
    assert hosting is not None
    assert len(overload["hourly"]) == 24
    assert overload["basis"] == "load"
    assert overload["limit_consistency"]["consistent"] is True
    assert overload["rating_kw"] == 33_000.0
    assert 0.0 <= overload["max_exceedance_prob"] <= 1.0
    assert hosting["asset_rating_kw"] == 33_000.0
    assert "kein Netzlastfluss" in overload["caveat"]
