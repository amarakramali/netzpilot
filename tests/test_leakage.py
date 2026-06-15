# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import numpy as np
from netzpilot.data.smard import load_local_json
from netzpilot.features.build import to_daily, build_features, get_holidays
def test_no_future_leakage():
    """Features fuer Tag d duerfen sich nicht aendern, wenn Last ab Tag d verfaelscht wird."""
    s = load_local_json("prognose_engine_v1/data/wk*.json")
    load2d, days = to_daily(s)
    hol = get_holidays(sorted({d.year for d in days}), "NW")
    d = 40
    X1 = build_features(load2d, days, d, None, hol)
    load2d2 = load2d.copy(); load2d2[d:] = -999.0
    X2 = build_features(load2d2, days, d, None, hol)
    assert np.allclose(X1, X2), "Leakage: Feature nutzt Daten ab dem Zieltag!"
