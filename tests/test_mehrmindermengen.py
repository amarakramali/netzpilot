# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mehrmindermengen_verify_anchors():
    from netzpilot.eval.mehrmindermengen import compare_forecasts_mmm, mehr_mindermengen

    price = 60.0

    report = mehr_mindermengen([10.0] * 4, [10.0] * 4, price)
    assert report["mehrmenge_mwh"] == 0
    assert report["mindermenge_mwh"] == 0
    assert report["netto_mwh"] == 0
    assert report["abs_volumen_mwh"] == 0

    report = mehr_mindermengen([10.0] * 4, [12.0] * 4, price)
    assert report["mehrmenge_mwh"] == 8.0
    assert report["mindermenge_mwh"] == 0.0
    assert report["netto_mwh"] == 8.0
    assert abs(report["netto_eur"] - 480.0) < 1e-6

    report = mehr_mindermengen([10.0, 10.0, 10.0, 10.0], [12.0, 12.0, 8.0, 8.0], price)
    assert report["mehrmenge_mwh"] == 4.0
    assert report["mindermenge_mwh"] == 4.0
    assert report["netto_mwh"] == 0.0
    assert report["abs_volumen_mwh"] == 8.0
    assert report["mehrmenge_eur"] == 240.0
    assert report["mindermenge_eur"] == 240.0
    assert report["netto_eur"] == 0.0

    report = mehr_mindermengen([5.0, 8.0, 3.0, 10.0, 7.0], [6.0, 4.0, 9.0, 2.0, 7.0], price)
    assert abs(report["netto_mwh"] - (report["mehrmenge_mwh"] - report["mindermenge_mwh"])) < 1e-9
    assert abs(report["abs_volumen_mwh"] - (report["mehrmenge_mwh"] + report["mindermenge_mwh"])) < 1e-9

    actual = [10.0, 12.0, 9.0, 11.0]
    cmp = compare_forecasts_mmm(actual, [8.0, 15.0, 12.0, 8.0], list(actual), price)
    assert cmp["abs_volumen_b_mwh"] == 0.0
    assert abs(cmp["abs_volumen_reduktion_mwh"] - 11.0) < 1e-6

    report = mehr_mindermengen([10.0] * 4, [12.0] * 4, price, dt_h=0.25)
    assert abs(report["mehrmenge_mwh"] - 2.0) < 1e-9

    report = mehr_mindermengen([10.0, 10.0, 10.0], [12.0, float("nan"), None], price)
    assert report["n"] == 1
    assert report["n_dropped"] == 2

    with pytest.raises(ValueError):
        mehr_mindermengen([1.0, 2.0], [1.0], price)
    with pytest.raises(ValueError):
        mehr_mindermengen([], [], price)


REAL_CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_runner_mmm_uses_realized_backtest_series():
    from netzpilot.service.runner import run_forecast

    out = run_forecast(
        REAL_CSV,
        utility="MmmTest",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        mmm_price_eur_mwh=60.0,
    )
    mmm = out["mmm"]
    assert mmm["status"] == "available"
    assert mmm["mmm_price_eur_mwh"] == 60.0
    assert mmm["report_snaive"]["n"] == mmm["report_netzpilot"]["n"]
    assert mmm["report_netzpilot"]["n"] > 0
    assert mmm["abs_volumen_reduktion_mwh"] > 0
    assert mmm["abs_volumen_reduktion_at_price_eur"] > 0
    assert "EDM" in mmm["caveat"]
