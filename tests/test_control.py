# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import pytest
from netzpilot.control.schema import make_fahrplan, validate_fahrplan, active_limit_kw, MIN_GUARANTEED_KW
from netzpilot.control.aemt_mock import AEMTMock
from netzpilot.control.hems_sim import Hems

DAY = "2024-01-15"
def _sp(p): return [{"start_utc": f"{DAY}T17:00:00+00:00", "end_utc": f"{DAY}T20:00:00+00:00", "p_limit_kw": p}]

def test_floor_enforced():
    with pytest.raises(ValueError):
        make_fahrplan("DE0001234567890", _sp(2.0))   # unter 4,2 kW -> abgelehnt

def test_active_limit_window():
    fp = make_fahrplan("DE0001234567890", _sp(4.2))
    assert active_limit_kw(fp, f"{DAY}T18:00:00+00:00") == 4.2
    assert active_limit_kw(fp, f"{DAY}T09:00:00+00:00") is None

def test_end_to_end_loop_and_rejection():
    aemt = AEMTMock().start()
    try:
        import json
        from urllib.request import urlopen, Request
        from urllib.error import HTTPError
        fp = make_fahrplan("DE0001234567890", _sp(4.2))
        req = Request(aemt.base_url + "/fahrplan", data=json.dumps(fp).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=5) as r:
            assert r.status == 202
        hems = Hems(aemt.base_url, "DE0001234567890", 11.0)
        assert hems.applied_power_kw(f"{DAY}T18:00:00+00:00")["applied_kw"] == 4.2   # gedrosselt
        assert hems.applied_power_kw(f"{DAY}T09:00:00+00:00")["applied_kw"] == 11.0  # frei
        # illegaler Fahrplan wird mit 422 abgelehnt
        bad = {**fp, "setpoints": _sp(2.0)}
        rejected = False
        try:
            req2 = Request(aemt.base_url + "/fahrplan", data=json.dumps(bad).encode(),
                           headers={"Content-Type": "application/json"}, method="POST")
            urlopen(req2, timeout=5)
        except HTTPError as e:
            rejected = e.code == 422
        assert rejected
    finally:
        aemt.stop()
