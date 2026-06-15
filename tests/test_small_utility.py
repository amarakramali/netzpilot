# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import pandas as pd

from netzpilot.data.small_utility import aggregate_opsd_grid_import


def test_aggregate_opsd_grid_import_converts_quarterhour_kwh_to_hourly_mw():
    idx = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "utc_timestamp": idx,
        "DE_KN_residential1_grid_import": [i * 0.25 for i in range(8)],
        "DE_KN_residential2_grid_import": [i * 0.50 for i in range(8)],
    })
    series, cols = aggregate_opsd_grid_import(df, scale_factor=1000.0, min_active_columns=2)
    assert cols == ["DE_KN_residential1_grid_import", "DE_KN_residential2_grid_import"]
    assert list(series) == [3.0, 3.0]
