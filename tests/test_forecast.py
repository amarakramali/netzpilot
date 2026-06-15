# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

from netzpilot.data.smard import load_local_json
from netzpilot.features.build import to_daily, get_holidays
from netzpilot.forecast import forecast_next_day
from netzpilot.models.robust_corrector import ShrunkCorrector
def test_forecast_next_day_shape_and_monotonic():
    s = load_local_json("prognose_engine_v1/data/wk*.json")
    load2d, days = to_daily(s); hol = get_holidays(sorted({d.year for d in days}), "NW")
    fp = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0), holiday_set=hol)
    assert len(fp["hours"]) == 24
    for h in fp["hours"]:
        assert h["p10"] <= h["p50"] <= h["p90"]
