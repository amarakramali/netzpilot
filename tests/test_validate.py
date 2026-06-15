# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import math
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from netzpilot.data.validate import validate_load

REAL_CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"


def test_validate_verify_anchors():
    day = [50.0 + 10.0 * (h % 6) for h in range(24)]
    base = day * 3

    r = validate_load(list(base))
    assert r["quality_score"] == 1.0
    assert r["n_missing"] == 0
    assert r["n_outlier"] == 0
    assert r["n_frozen"] == 0

    v = list(base)
    v[30] = None
    r = validate_load(v)
    assert r["n_missing"] == 1
    assert r["n_replaced"] >= 1
    assert abs(r["cleaned"][30] - base[6]) < 1e-6
    assert any(x["index"] == 30 and x["method"] == "seasonal_neighbor_day" for x in r["replacements"])

    v = list(base)
    v[40] = 100000.0
    r = validate_load(v)
    assert r["n_outlier"] == 1
    assert abs(r["cleaned"][40] - base[16]) < 1e-6

    v = list(base)
    v[10] = -5.0
    assert validate_load(v)["n_negative"] == 1
    assert validate_load(v, allow_negative=True)["n_negative"] == 0

    v = list(base)
    for i in range(12, 19):
        v[i] = 77.0
    r = validate_load(v, frozen_run=6)
    assert r["n_frozen"] >= 7
    assert abs(r["cleaned"][15] - 77.0) < 1e-9


def test_runner_uses_cleaned_values_for_input_gate():
    from netzpilot.service.runner import run_forecast

    idx = pd.date_range("2025-01-01", periods=24 * 75, freq="h")
    values_kw = []
    for i in range(len(idx)):
        hour = i % 24
        values_kw.append(12000.0 + 1500.0 * math.sin(hour / 24.0 * 2.0 * math.pi))
    values_kw[30] = -5000.0
    values_kw[120] = 9_999_999.0
    missing_i = 250

    df = pd.DataFrame({"Text": idx.strftime("%Y-%m-%d %H:%M:%S"), "Reihe1": values_kw})
    df = df.drop(index=missing_i)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "defective_load.csv"
        df.to_csv(path, index=False)

        out = run_forecast(
            str(path),
            utility="ValidateSynthetic",
            unit="kW",
            ts_col="Text",
            load_col="Reihe1",
            validate_input=True,
            validate_max_plausible=50.0,
        )
    validation = out["input_validation"]
    assert validation["enabled"] is True
    assert validation["cleaned_values_used"] is True
    assert validation["original_preserved"] is True
    assert validation["n_missing"] >= 1
    assert validation["n_negative"] >= 1
    assert validation["n_out_of_range"] >= 1
    assert validation["n_replaced"] >= 3
    assert out["n_days_history"] >= 70
    assert len(out["forecast"]) == 24


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_runner_validation_smoke_on_real_data():
    from netzpilot.service.runner import run_forecast

    out = run_forecast(
        REAL_CSV,
        utility="ValidateReal",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        validate_input=True,
    )
    validation = out["input_validation"]
    assert validation["enabled"] is True
    assert 0.0 <= validation["quality_score"] <= 1.0
    assert "cleaned" not in validation
    assert validation["original_preserved"] is True
