# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import os
import random

import pytest

from netzpilot.control.tariff import optimize_grid_fee_schedule
from netzpilot.service.tariff_schedule import build_tariff_schedule, normalize_available

REAL_CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"


def test_tariff_verify_hand_cases():
    r = optimize_grid_fee_schedule([0.10, 0.05, 0.20, 0.05], 15.0, 10.0)
    assert abs(r["scheduled_kwh"] - 15.0) < 1e-9 and r["feasible"]
    assert abs(r["total_cost_eur"] - 0.75) < 1e-9
    assert r["schedule_kwh"][0] == 0.0 and r["schedule_kwh"][2] == 0.0
    assert abs(r["saving_eur"] - 0.50) < 1e-9

    r = optimize_grid_fee_schedule([0.15] * 24, 33.0, 11.0)
    assert abs(r["total_cost_eur"] - 4.95) < 1e-9
    assert abs(r["saving_eur"]) < 1e-9

    r = optimize_grid_fee_schedule([0.05, 0.20], 8.0, 10.0, cap_kw=[0.0, 10.0])
    assert r["schedule_kwh"][0] == 0.0
    assert abs(r["schedule_kwh"][1] - 8.0) < 1e-9 and r["feasible"]
    assert abs(r["total_cost_eur"] - 1.60) < 1e-9

    r = optimize_grid_fee_schedule(
        [0.01, 0.10, 0.20, 0.01],
        5.0,
        10.0,
        available=[False, True, True, False],
    )
    assert r["schedule_kwh"][0] == 0.0 and r["schedule_kwh"][3] == 0.0
    assert abs(r["scheduled_kwh"] - 5.0) < 1e-9 and r["feasible"]

    r = optimize_grid_fee_schedule([0.1, 0.1], 15.0, 5.0)
    assert r["feasible"] is False
    assert abs(r["scheduled_kwh"] - 10.0) < 1e-9
    assert abs(r["shortfall_kwh"] - 5.0) < 1e-9


def test_tariff_random_constraints_and_optimality_certificate():
    rng = random.Random(7)
    for _ in range(500):
        n = rng.randint(4, 24)
        fee = [round(rng.uniform(0.02, 0.40), 3) for _ in range(n)]
        pmax = rng.uniform(3, 15)
        caps = [rng.uniform(0, pmax) for _ in range(n)]
        avail = [rng.random() > 0.2 for _ in range(n)]
        ceil = [(min(pmax, caps[t]) if avail[t] else 0.0) for t in range(n)]
        emax = sum(ceil)
        energy = rng.uniform(0, emax) if emax > 0 else 0.0
        r = optimize_grid_fee_schedule(fee, energy, pmax, cap_kw=caps, available=avail)
        for t in range(n):
            assert r["schedule_kwh"][t] <= ceil[t] + 1e-6
        assert abs(r["scheduled_kwh"] - energy) <= 1e-6

    rng = random.Random(11)
    for _ in range(500):
        n = rng.randint(4, 24)
        fee = [round(rng.uniform(0.02, 0.40), 3) for _ in range(n)]
        pmax = rng.uniform(3, 15)
        caps = [rng.uniform(0, pmax) for _ in range(n)]
        ceil = [min(pmax, caps[t]) for t in range(n)]
        energy = rng.uniform(0, sum(ceil))
        r = optimize_grid_fee_schedule(fee, energy, pmax, cap_kw=caps)
        schedule = r["schedule_kwh"]
        for t in range(n):
            if schedule[t] > 1e-9:
                for s in range(n):
                    assert not (fee[s] < fee[t] - 1e-9 and schedule[s] < ceil[s] - 1e-6)


def test_tariff_greedy_dominates_random_feasible_schedules_and_validation():
    rng = random.Random(13)
    for _ in range(300):
        n = rng.randint(4, 12)
        fee = [round(rng.uniform(0.02, 0.40), 3) for _ in range(n)]
        pmax = rng.uniform(5, 12)
        energy = rng.uniform(0, pmax * n)
        r = optimize_grid_fee_schedule(fee, energy, pmax)
        greedy_cost = r["total_cost_eur"]
        for _ in range(5):
            alloc = [0.0] * n
            rem = energy
            for t in sorted(range(n), key=lambda _i: rng.random()):
                take = min(pmax, rem)
                alloc[t] = take
                rem -= take
                if rem <= 1e-12:
                    break
            random_cost = sum(fee[t] * alloc[t] for t in range(n))
            assert greedy_cost <= random_cost + 1e-6

    with pytest.raises(ValueError):
        optimize_grid_fee_schedule([], 5.0, 10.0)
    with pytest.raises(ValueError):
        optimize_grid_fee_schedule([0.1], -1.0, 10.0)
    with pytest.raises(ValueError):
        optimize_grid_fee_schedule([0.1, 0.2], 5.0, 10.0, cap_kw=[1.0])


def test_service_tariff_uses_redispatch_caps_and_window():
    assert normalize_available(24, start_hour=18, end_hour=6)[18] is True
    assert normalize_available(24, start_hour=18, end_hour=6)[5] is True
    assert normalize_available(24, start_hour=18, end_hour=6)[12] is False

    redispatch = {"hourly": [{"hour": 0, "cap_kw": 0.0}, {"hour": 1, "cap_kw": 10.0}]}
    r = build_tariff_schedule([0.05, 0.20], 8.0, 10.0, redispatch=redispatch)
    assert r["cap_source"] == "redispatch"
    assert r["binding_cap_hours"] == [0]
    assert r["schedule_kwh"][0] == 0.0
    assert abs(r["schedule_kwh"][1] - 8.0) < 1e-9
    assert abs(r["total_cost_eur"] - 1.60) < 1e-9


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_runner_adds_tariff_schedule_field_on_real_data():
    from netzpilot.service.runner import run_forecast

    fee = [0.08] * 6 + [0.16] * 11 + [0.28] * 5 + [0.08] * 2
    r = run_forecast(
        REAL_CSV,
        utility="TariffTest",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        congestion_threshold_mw=33.0,
        steuve_malo="DE0001234567890",
        steuve_demands_kw=[11.0],
        rolling_redispatch=True,
        grid_fee_eur_per_kwh=fee,
        tariff_energy_kwh=40.0,
        tariff_p_max_kw=11.0,
        tariff_available_start_hour=18,
        tariff_available_end_hour=6,
    )
    ts = r["tariff_schedule"]
    assert ts is not None
    assert ts["feasible"] is True
    assert abs(ts["scheduled_kwh"] - 40.0) < 1e-6
    assert ts["saving_eur"] >= -1e-9
    assert ts["cap_source"].startswith("redispatch")
    for hour, kwh in enumerate(ts["schedule_kwh"]):
        assert kwh <= (ts["cap_kw"][hour] if ts["cap_kw"] else ts["p_max_kw"]) + 1e-6
        if not ts["available"][hour]:
            assert kwh == 0.0
