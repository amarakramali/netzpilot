import os

import pytest

from netzpilot.control.dispatch import _quantile, cost_optimal_nomination, plan_day
from netzpilot.service.dispatch_plan import build_dispatch_plan

REAL_CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"


def _case():
    horizon = 24
    residuals = [-8.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0]
    base = [80.0] * horizon
    base[12] = 100.0
    return base, [list(residuals) for _ in range(horizon)], residuals, [0.10] * horizon, 100.0


def test_dispatch_grid_safety_composition_and_budget():
    base, residuals_by_hour, residuals, fee, threshold = _case()
    r = plan_day(
        base,
        residuals_by_hour,
        threshold,
        steuve_energy_kwh=200.0,
        steuve_p_max_kw=20.0,
        grid_fee_eur_per_kwh=fee,
        c_short=2.0,
        c_long=1.0,
    )
    assert max(h["total_point_kw"] for h in r["hourly"]) <= threshold + 1e-6
    assert r["grid_safe"] is True
    assert r["hourly"][12]["cap_kw"] == 0.0
    assert r["hourly"][12]["steuve_kw"] == 0.0

    for h in r["hourly"]:
        hour = h["hour"]
        assert abs(h["total_point_kw"] - round(base[hour] + h["steuve_kw"], 4)) < 1e-6
        expected = h["total_point_kw"] + _quantile(sorted(residuals), 2.0 / 3.0)
        assert abs(h["nomination_kw"] - round(expected, 4)) <= 1e-3

    total_steuve = sum(h["steuve_kw"] for h in r["hourly"])
    assert abs(total_steuve - 200.0) < 1e-3
    assert r["feasible"] is True


def test_dispatch_infeasibility_and_newsvendor_savings():
    base, residuals_by_hour, _residuals, fee, threshold = _case()
    r = plan_day(base, residuals_by_hour, threshold, 600.0, 20.0, fee, c_short=2.0, c_long=1.0)
    assert r["feasible"] is False
    assert abs(r["shortfall_kwh"] - 140.0) < 1e-3
    assert r["grid_safe"] is True

    r_sym = plan_day(base, residuals_by_hour, threshold, 200.0, 20.0, fee, c_short=1.0, c_long=1.0)
    assert abs(r_sym["newsvendor_saving_eur"]) < 1e-9
    assert abs(r_sym["exp_imbalance_tau_eur"] - r_sym["exp_imbalance_p50_eur"]) < 1e-9

    r_asym = plan_day(base, residuals_by_hour, threshold, 200.0, 20.0, fee, c_short=2.0, c_long=1.0)
    assert r_asym["newsvendor_saving_eur"] > 0.0
    assert r_asym["exp_imbalance_tau_eur"] <= r_asym["exp_imbalance_p50_eur"] + 1e-9

    savings = []
    for ratio in (1.0, 2.0, 3.0, 5.0):
        rr = plan_day(base, residuals_by_hour, threshold, 200.0, 20.0, fee, c_short=ratio, c_long=1.0)
        savings.append(rr["newsvendor_saving_eur"])
    assert abs(savings[0]) < 1e-9
    assert all(savings[i] <= savings[i + 1] + 1e-9 for i in range(len(savings) - 1))
    assert savings[-1] > savings[0]


def test_dispatch_helper_and_validation():
    residuals = [-8.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0]
    nom, tau = cost_optimal_nomination(100.0, residuals, c_short=2.0, c_long=1.0)
    assert abs(tau - 2.0 / 3.0) < 1e-9
    assert abs(nom - (100.0 + _quantile(sorted(residuals), 2.0 / 3.0))) < 1e-9

    base = [80.0, 95.0]
    residuals_by_hour = [list(residuals), list(residuals)]
    redispatch = {"hourly": [{"hour": 0, "cap_kw": 20.0}, {"hour": 1, "cap_kw": 5.0}]}
    r = build_dispatch_plan(
        base,
        residuals_by_hour,
        100.0,
        steuve_energy_kwh=15.0,
        steuve_p_max_kw=20.0,
        grid_fee_eur_per_kwh=[0.2, 0.1],
        c_short=2.0,
        c_long=1.0,
        redispatch=redispatch,
    )
    assert r["grid_safe"] is True
    assert r["redispatch_cap_consistency"]["consistent"] is True
    assert r["hourly"][1]["cap_kw"] == 5.0

    with pytest.raises(ValueError):
        cost_optimal_nomination(100.0, residuals, 0.0, 1.0)
    with pytest.raises(ValueError):
        plan_day(base, residuals_by_hour[:1], 100.0, 10.0, 20.0, [0.1, 0.1], 2.0, 1.0)
    with pytest.raises(ValueError):
        cost_optimal_nomination(100.0, [], 2.0, 1.0)


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_runner_adds_dispatch_plan_on_real_data():
    from netzpilot.service.runner import run_forecast

    # Schwelle RELATIV zur aktuellen Prognose ableiten statt hart kodieren: eine fixe MW-Zahl
    # kodiert sonst den Prognosestand ihres Entstehungstags ein (Lehre 2026-06-04: die fixe 33.0
    # passte zur VOR-T48-Prognose, deren Neujahrs-Anker faelschlich der niedrige 25.12. war; die
    # feiertagsbewusste Basis hebt P50 ehrlich an -> 33 MW waere mit 1 MW steuVE unhaltbar,
    # grid_safe=False waere die KORREKTE Antwort, aber kein Testfehler). Peak-0.5 MW garantiert:
    # Engpass existiert (P90 >= P50 > Schwelle) UND 1 MW steuVE kann die Ueberschreitung halten.
    base_run = run_forecast(REAL_CSV, utility="DispatchTest", unit="kW",
                            ts_col="Text", load_col="Reihe1")
    peak_p50_mw = max(h["p50"] for h in base_run["forecast"])
    threshold_mw = round(peak_p50_mw - 0.5, 3)

    fee = [0.05] * 6 + [0.12] * 11 + [0.28] * 5 + [0.05] * 2
    r = run_forecast(
        REAL_CSV,
        utility="DispatchTest",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        congestion_threshold_mw=threshold_mw,
        steuve_malo="DE0001234567890",
        steuve_demands_kw=[1000.0],
        rolling_redispatch=True,
        grid_fee_eur_per_kwh=fee,
        dispatch_plan_enabled=True,
        dispatch_steuve_energy_kwh=40.0,
        dispatch_steuve_p_max_kw=1000.0,
        dispatch_c_short=0.20,
        dispatch_c_long=0.10,
    )
    dp = r["dispatch_plan"]
    assert dp is not None
    assert len(dp["hourly"]) == 24
    assert dp["grid_safe"] is True
    assert dp["feasible"] is True
    assert dp["newsvendor_saving_eur"] >= -1e-9
    assert dp["redispatch_cap_consistency"]["checked"] is True
    assert dp["redispatch_cap_consistency"]["consistent"] is True
