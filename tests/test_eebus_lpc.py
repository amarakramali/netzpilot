# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sp(p_limit, floor, start, end):
    return {"p_limit_kw": p_limit, "floor_kw": floor, "start_utc": start, "end_utc": end}


def test_lpc_mapping_verify_anchors():
    from netzpilot.control.eebus_lpc import EebusLpcAdapter, fahrplan_to_lpc
    from netzpilot.control.schema import make_fahrplan

    t0 = "2026-01-01T12:00:00+00:00"
    t1 = "2026-01-01T13:00:00+00:00"
    t2 = "2026-01-01T15:00:00+00:00"

    lpc = fahrplan_to_lpc(make_fahrplan("MALO123456", [_sp(10.0, 4.2, t0, t1)]))
    limit = lpc["limits"][0]
    assert limit["consumption_limit_w"] == 10000.0
    assert limit["failsafe_value_w"] == 4200.0
    assert limit["duration_s"] == 3600.0
    assert limit["is_limit_active"] is True
    assert lpc["n_limits"] == 1
    assert "LPC" in lpc["use_case"]
    assert lpc["transport"] == "stub_external"

    lpc = fahrplan_to_lpc(make_fahrplan("MALO123456", [_sp(7.5, 4.2, t0, t1)]))
    assert lpc["limits"][0]["consumption_limit_w"] == 7500.0

    fp = make_fahrplan("MALO123456", [_sp(20.0, 7.0, t0, t1), _sp(12.0, 4.2, t1, t2)])
    lpc = fahrplan_to_lpc(fp)
    assert lpc["limits"][0]["failsafe_value_w"] == 7000.0
    assert lpc["failsafe_value_w"] == 4200.0
    assert lpc["n_limits"] == 2

    bad_fp = {"malo": "MALO123456", "schedule_id": "x", "setpoints": [_sp(3.0, 4.2, t0, t1)]}
    with pytest.raises(ValueError):
        fahrplan_to_lpc(bad_fp)

    lpc = fahrplan_to_lpc(make_fahrplan("MALO123456", [_sp(10.0, 4.2, t0, t2)]))
    assert lpc["limits"][0]["duration_s"] == 10800.0

    fp = make_fahrplan("MALO123456", [_sp(12.0, 4.2, t1, t2), _sp(10.0, 4.2, t0, t1)])
    lpc = fahrplan_to_lpc(fp)
    assert lpc["limits"][0]["start_utc"] == t0
    assert lpc["limits"][1]["start_utc"] == t1

    ack = EebusLpcAdapter().submit(make_fahrplan("MALO123456", [_sp(10.0, 4.2, t0, t1)]))
    assert ack["status"] == "MAPPED"
    assert ack["lpc"]["n_limits"] == 1
    assert ack["lpc"]["transport"] == "stub_external"


REAL_CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_runner_eebus_lpc_adapter_exposes_payload():
    from netzpilot.service.runner import run_forecast

    out = run_forecast(
        REAL_CSV,
        utility="LpcTest",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        congestion_threshold_mw=33.0,
        steuve_malo="DE0001234567890",
        submit_to_aemt=True,
        aemt_adapter="eebus_lpc",
    )
    assert out["fahrplan"] is not None
    assert out["aemt_ack"]["status"] == "MAPPED"
    assert out["fahrplan_lpc"] is not None
    assert out["fahrplan_lpc"]["n_limits"] == len(out["fahrplan"]["setpoints"])
    assert out["fahrplan_lpc"]["limits"][0]["consumption_limit_w"] >= 4200.0
    assert out["fahrplan_lpc"]["transport"] == "stub_external"


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_runner_aemt_adapter_default_stays_mock():
    from netzpilot.service.runner import run_forecast

    out = run_forecast(
        REAL_CSV,
        utility="LpcTest",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        congestion_threshold_mw=33.0,
        steuve_malo="DE0001234567890",
        submit_to_aemt=True,
    )
    assert out["aemt_ack"]["status"] == "ACCEPTED"
    assert out["fahrplan_lpc"] is None
