import os

import pytest

from netzpilot.control.risk import (
    _quantile,
    cvar,
    expected_value,
    imbalance_costs,
    risk_averse_nomination,
)
from netzpilot.service.dispatch_plan import build_dispatch_plan

REAL_CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"


def test_cvar_verify_anchors():
    residuals = [float(i - 50) for i in range(101)]
    point = 100.0

    r0 = risk_averse_nomination(point, residuals, c_short=2.0, c_long=1.0, beta=0.0)
    tau_q = point + _quantile(sorted(residuals), 2.0 / 3.0)
    assert abs(r0["nomination_kw"] - tau_q) < 0.5
    assert abs(r0["tau_equiv"] - 2.0 / 3.0) < 1e-3

    costs = imbalance_costs(100.0, point, residuals, 2.0, 1.0)
    assert cvar(costs, 0.95) >= expected_value(costs) - 1e-9

    skew = [float(i - 10) for i in range(21)] + [60.0, 70.0, 80.0, 90.0, 100.0]
    nominations, cvars, expected = [], [], []
    for beta in (0.0, 0.3, 0.6, 0.9, 1.0):
        r = risk_averse_nomination(point, skew, c_short=3.0, c_long=1.0, beta=beta, alpha=0.95)
        nominations.append(r["nomination_kw"])
        cvars.append(r["cvar_eur"])
        expected.append(r["expected_cost_eur"])
    assert all(nominations[i] <= nominations[i + 1] + 1e-6 for i in range(len(nominations) - 1))
    assert nominations[-1] > nominations[0]
    assert all(cvars[i] >= cvars[i + 1] - 1e-6 for i in range(len(cvars) - 1))
    assert expected[-1] >= expected[0]
    assert abs(cvar([0.0, 0.0, 0.0, 0.0, 10.0], 0.8) - 10.0) < 1e-6


def test_dispatch_plan_risk_beta_is_additive_and_changes_nomination():
    base = [80.0, 80.0]
    residuals = [[-10.0, 0.0, 60.0, 70.0, 80.0]] * 2
    kwargs = dict(
        steuve_energy_kwh=0.0,
        steuve_p_max_kw=10.0,
        grid_fee_eur_per_kwh=[0.0, 0.0],
        c_short=3.0,
        c_long=1.0,
    )
    plain = build_dispatch_plan(base, residuals, 200.0, **kwargs)
    risk = build_dispatch_plan(base, residuals, 200.0, risk_beta=0.8, risk_alpha=0.95, **kwargs)
    assert "risk_averse" not in plain
    assert risk["risk_averse"]["enabled"] is True
    assert risk["hourly"][0]["nomination_kw"] != plain["hourly"][0]["nomination_kw"]
    assert "newsvendor_nomination_kw" in risk["hourly"][0]
    assert risk["risk_averse"]["risk_cvar_delta_vs_newsvendor_eur"] <= 0.0


def test_cvar_validation():
    with pytest.raises(ValueError):
        risk_averse_nomination(100.0, [], 2.0, 1.0)
    with pytest.raises(ValueError):
        risk_averse_nomination(100.0, [0.0], 0.0, 1.0)
    with pytest.raises(ValueError):
        risk_averse_nomination(100.0, [0.0], 2.0, 1.0, beta=1.5)
    with pytest.raises(ValueError):
        cvar([1.0], 1.0)


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_runner_dispatch_risk_beta_on_real_data():
    from netzpilot.service.runner import run_forecast

    out = run_forecast(
        REAL_CSV,
        utility="CvarReal",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        congestion_threshold_mw=33.0,
        steuve_malo="DE0001234567890",
        steuve_demands_kw=[1000.0],
        rolling_redispatch=True,
        dispatch_plan_enabled=True,
        dispatch_steuve_energy_kwh=20.0,
        dispatch_steuve_p_max_kw=1000.0,
        dispatch_c_short=0.20,
        dispatch_c_long=0.10,
        dispatch_risk_beta=0.6,
        dispatch_risk_alpha=0.95,
    )
    risk = out["dispatch_plan"]["risk_averse"]
    assert risk["enabled"] is True
    assert risk["beta"] == 0.6
    assert "cvar_imbalance_risk_eur" in risk
