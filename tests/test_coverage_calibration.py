# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Tests fuer Coverage-Kalibrierung (T45).

Synthetische S1-S6 aus scripts/verify_coverage_calibration.py portiert (deterministisch).
Plus Anbindungstests: forecast_next_day + rolling_origin sind bei calibrate=False bit-genau
wie vorher; bei calibrate=True kommen die kalibrierten Zusatzfelder hinzu, p50 bleibt
unangetastet, Monotonie p10<=p50<=p90 erhalten.
S7 ist ein leakage-sicherer Reihen-Check auf echten DSO-Daten (skipif Manifest fehlt).
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.eval.coverage_calibration import (
    coverage_scale, apply_scale, _coverage, rolling_coverage_scale,
)
from netzpilot.forecast import forecast_next_day
from netzpilot.eval.backtest import rolling_origin, DEFAULT_CAL_VAL_RECENT
from netzpilot.models.robust_corrector import ShrunkCorrector


# ---------- S1-S6: Synthetische Checks (deterministisch) ----------

@pytest.fixture(scope="module")
def gauss80():
    """Determ. N(0, sigma) so dass P(|.|<=1) = 0.80 -> Coverage monoton in s."""
    rng = np.random.default_rng(0)
    N = 20000
    sig = 1.0 / 1.2815515594
    return rng.normal(0.0, sig, N), np.zeros(N)


def test_S1_gut_kalibriert_s_approx_1(gauss80):
    a, p50 = gauss80
    s = coverage_scale(a, p50 - 1, p50, p50 + 1, target=0.8, shrink=1.0)
    assert abs(s - 1.0) < 0.12, f"erwartet s~1, bekommen {s:.3f}"


def test_S2_ueberdeckt_s_unter_1(gauss80):
    a, p50 = gauss80
    s = coverage_scale(a, p50 - 2, p50, p50 + 2, target=0.8, shrink=1.0)
    assert s < 0.85 and abs(s - 0.5) < 0.15


def test_S3_unterdeckt_s_ueber_1(gauss80):
    a, p50 = gauss80
    s = coverage_scale(a, p50 - 0.5, p50, p50 + 0.5, target=0.8, shrink=1.0)
    assert s > 1.3 and abs(s - 2.0) < 0.3


def test_S4_shrinkage_formel(gauss80):
    a, p50 = gauss80
    s_full = coverage_scale(a, p50 - 2, p50, p50 + 2, target=0.8, shrink=1.0)
    s_half = coverage_scale(a, p50 - 2, p50, p50 + 2, target=0.8, shrink=0.5)
    s_zero = coverage_scale(a, p50 - 2, p50, p50 + 2, target=0.8, shrink=0.0)
    assert abs(s_zero - 1.0) < 1e-9
    assert abs(s_half - (1 + 0.5 * (s_full - 1))) < 1e-9


def test_S5_apply_scale_monotonie_und_breite(gauss80):
    _, p50 = gauss80
    lo1, hi1 = apply_scale(p50 - 1, p50, p50 + 1, 1.0)
    lo2, hi2 = apply_scale(p50 - 1, p50, p50 + 1, 2.0)
    los, his = apply_scale(p50 - 1, p50, p50 + 1, 0.5)
    assert float(np.mean(hi2 - lo2)) > float(np.mean(hi1 - lo1)) > float(np.mean(his - los))
    assert bool(np.all(lo1 <= p50) and np.all(p50 <= hi1))


def test_S6_validation_errors():
    with pytest.raises(ValueError):
        coverage_scale([1, 2], [0], [0], [0])
    with pytest.raises(ValueError):
        coverage_scale([], [], [], [])
    with pytest.raises(ValueError):
        coverage_scale([1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], target=1.5)
    with pytest.raises(ValueError):
        apply_scale([0.0], [0.0], [0.0], -1.0)


# ---------- Anbindungstests: forecast_next_day ----------

def _synth_load(n_days=70, H=24, seed=42):
    rng = np.random.default_rng(seed)
    base = 50.0 + 20.0 * np.sin(np.linspace(0, 2 * np.pi, H, endpoint=False))
    load = np.zeros((n_days, H), dtype=float)
    for d in range(n_days):
        weekday_amp = 5.0 * np.sin(2 * np.pi * d / 7.0)
        load[d] = base + weekday_amp + rng.normal(0.0, 2.0, H)
    days = list(pd.date_range("2025-01-01", periods=n_days, freq="D"))
    return load, days


def test_forecast_calibrate_false_is_bit_compatible():
    load2d, days = _synth_load()
    # Aufruf ohne Parameter (alter Code-Pfad) vs. expliziter calibrate=False
    fp_legacy = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0))
    fp_false = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0), calibrate=False)
    assert fp_legacy == fp_false, "calibrate=False muss bit-genau dem alten Verhalten entsprechen"


def test_forecast_calibrate_true_keeps_p50_and_monotonie():
    load2d, days = _synth_load()
    fp_off = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0), calibrate=False)
    fp_on = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0), calibrate=True)
    # Additive Felder vorhanden bei True, nicht bei False
    assert "coverage_scale_used" in fp_on and "coverage_calibrated" in fp_on
    assert "coverage_scale_used" not in fp_off and "coverage_calibrated" not in fp_off
    assert fp_on["coverage_calibrated"] is True
    assert isinstance(fp_on["coverage_scale_used"], float)
    # p50 bit-identisch zwischen off und on; Monotonie haelt
    for h_off, h_on in zip(fp_off["hours"], fp_on["hours"]):
        assert h_off["p50"] == h_on["p50"], "Kalibrierung darf p50 nicht aendern"
        assert h_on["p10"] <= h_on["p50"] <= h_on["p90"]


# ---------- Anbindungstests: rolling_origin ----------

def test_backtest_calibrate_false_is_bit_compatible():
    load2d, days = _synth_load(n_days=84)
    _, sm_legacy = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=14)
    _, sm_false = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=14, calibrate=False)
    assert sm_legacy == sm_false


def test_backtest_calibrate_true_adds_kalibrierte_coverage():
    load2d, days = _synth_load(n_days=84)
    _, sm_off = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=14, calibrate=False)
    _, sm_on = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=14, calibrate=True)
    # Naiver Wert unveraendert (additiv)
    assert sm_on["probabilistisch"]["Coverage_P10_P90_%"] == sm_off["probabilistisch"]["Coverage_P10_P90_%"]
    assert sm_on["probabilistisch"]["Pinball_avg"] == sm_off["probabilistisch"]["Pinball_avg"]
    # Neue Felder
    assert "Coverage_P10_P90_kalibriert_%" in sm_on["probabilistisch"]
    assert "Pinball_avg_kalibriert" in sm_on["probabilistisch"]
    assert "coverage_scale_used" in sm_on["probabilistisch"]


def test_T46_calibration_validation_window_is_recent_and_strictly_before_test():
    """T46-Leakage-Guard: bei langem Holdout-Fenster (n_test=84) muss das Validierungsfenster
    auf RECENT-Default (DEFAULT_CAL_VAL_RECENT=28) liegen — KUERZER als n_test, unmittelbar
    davor. Verhindert Saison-Mismatch (T45-Audit-Befund)."""
    # Genug Tage fuer n_test=84 + Validierungsfenster + Mindest-Training
    load2d, days = _synth_load(n_days=240)
    n_test = 84
    _, sm = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0),
                           n_test=n_test, calibrate=True)
    prob = sm["probabilistisch"]
    # Validierungsfenster muss explizit ausgewiesen + STRIKT < n_test sein
    assert prob.get("kalibrier_val_tage") is not None, "Validierungsfenster muss berichtet werden."
    val_tage = int(prob["kalibrier_val_tage"])
    assert val_tage == DEFAULT_CAL_VAL_RECENT, (
        f"T46-Default: erwartet {DEFAULT_CAL_VAL_RECENT}, bekommen {val_tage}")
    assert val_tage < n_test, (
        f"T46-Audit: Validierungsfenster ({val_tage}) muss KUERZER sein als n_test ({n_test}).")
    # Kalibrierte Coverage muss berechnet worden sein
    assert prob.get("Coverage_P10_P90_kalibriert_%") is not None
    assert prob.get("coverage_scale_used") is not None


def test_T46_calibration_respects_explicit_cal_val_days_override():
    """Wenn cal_val_days explizit gesetzt ist, ueberstimmt es den RECENT-Default."""
    load2d, days = _synth_load(n_days=240)
    _, sm = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0),
                           n_test=84, calibrate=True, cal_val_days=14)
    assert int(sm["probabilistisch"]["kalibrier_val_tage"]) == 14


# ---------- T47: rolling_coverage_scale (Online-rollend) ----------

def _rolling_synth(n=60, H=24, seed=7):
    """Tagesweise Baender + Actuals fuer einen synthetischen Backtest [n, H]."""
    rng = np.random.default_rng(seed)
    p50 = np.zeros((n, H))
    p10 = p50 - 1.0
    p90 = p50 + 1.0
    sig = 1.0 / 1.2815515594   # so dass P(|.|<=1) = 0.8
    actual = rng.normal(0.0, sig, (n, H))
    return actual, p10, p50, p90


def test_T47_rolling_first_min_window_days_have_s_equals_one():
    """In den ersten `min_window` Tagen liegt keine ausreichende Vergangenheit vor → s=1."""
    actual, p10, p50, p90 = _rolling_synth(n=40)
    s, lo, hi = rolling_coverage_scale(actual, p10, p50, p90, window=14, min_window=10)
    assert np.all(s[:10] == 1.0)
    # lo/hi unveraendert in der no-op-Phase
    assert np.allclose(lo[:10], p10[:10])
    assert np.allclose(hi[:10], p90[:10])


def test_T47_rolling_strict_causality_no_leakage():
    """Verfaelschen von actual[i0] darf s_arr[:i0+1] NICHT aendern (KAUSALITAET / Leakage-Guard)."""
    actual, p10, p50, p90 = _rolling_synth(n=60)
    s_ref, _, _ = rolling_coverage_scale(actual, p10, p50, p90, window=20, min_window=10)
    actual_perturbed = actual.copy()
    i0 = 30
    actual_perturbed[i0] += 1000.0    # massive Stoerung an Tag i0
    s_pert, _, _ = rolling_coverage_scale(actual_perturbed, p10, p50, p90, window=20, min_window=10)
    # s_arr bis einschliesslich i0 darf sich NICHT aendern (i0 selbst wird aus [i0-20, i0) bestimmt
    # — ohne actual[i0]); Werte ab i0+1 duerfen sich aendern, weil actual[i0] dann im Fenster liegt.
    assert np.array_equal(s_ref[:i0 + 1], s_pert[:i0 + 1])
    assert not np.array_equal(s_ref[i0 + 1:], s_pert[i0 + 1:])


def test_T47_rolling_shape_validation():
    """Shape-Mismatch zwischen actual/p10/p50/p90 muss als ValueError fehlschlagen."""
    actual, p10, p50, p90 = _rolling_synth(n=20)
    with pytest.raises(ValueError):
        rolling_coverage_scale(actual, p10[:-1], p50, p90, window=10)
    with pytest.raises(ValueError):
        rolling_coverage_scale(actual.ravel(), p10.ravel(), p50.ravel(), p90.ravel(), window=10)


def test_T47_backtest_uses_online_rolling_method():
    """Benchmark-Verdrahtung weist eine `online-rolling`-Methode aus + mittleres s.
    T49 stellte den Default auf `online-rolling-asymmetric` um."""
    load2d, days = _synth_load(n_days=240)
    _, sm = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0),
                           n_test=84, calibrate=True)
    prob = sm["probabilistisch"]
    method = prob.get("coverage_scale_method", "")
    assert method.startswith("online-rolling"), f"unerwartete Methode {method!r}"
    assert prob.get("coverage_scale_used") is not None
    assert prob.get("coverage_scale_median") is not None
    assert prob.get("kalibrier_window_tage") == DEFAULT_CAL_VAL_RECENT


def test_T47_backtest_pinball_not_worse_under_rolling_calibration():
    """Auf synthetischen Daten darf Online-Rolling den Pinball nicht schlechter machen."""
    load2d, days = _synth_load(n_days=240)
    _, sm = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0),
                           n_test=84, calibrate=True)
    prob = sm["probabilistisch"]
    pin_naiv = float(prob["Pinball_avg"])
    pin_kal = float(prob["Pinball_avg_kalibriert"])
    # Toleranz fuer Rundung auf 1 Dezimalstelle
    assert pin_kal <= pin_naiv + 0.15, f"Pinball wurde schlechter: {pin_naiv} -> {pin_kal}"


# ---------- S7: echte Reihen (leakage-sicher, optional) ----------

def test_real_series_calibration_reduces_coverage_error_if_files_present():
    """Tunen auf Vergangenheit, anwenden auf Holdout — mean|cov-80| muss sinken."""
    try:
        from scripts.dataset_manifest import MANIFEST as DM
        from scripts.pilot_in_a_box import robust_load_csv
        from netzpilot.features.build import to_daily_local, get_holidays
    except Exception as e:
        pytest.skip(f"Manifest/Loader nicht verfuegbar: {e}")
    idx = {m["key"]: m for m in DM}
    keys = ["bitterfeld_ms_2024", "neuruppin_ns_2022", "hilden_netzumsatz_2025"]
    if not all(k in idx and os.path.exists(idx[k]["csv"]) for k in keys):
        pytest.skip("Echte DSO-CSVs fehlen.")
    dev_naiv, dev_cal = [], []
    fac = lambda: ShrunkCorrector(10.0)
    for key in keys:
        e = idx[key]
        hourly = robust_load_csv(e["csv"], ts_col=e["ts"], load_col=e["col"], unit=e["unit"],
                                 return_meta=True)[0]
        l2, days, _ = to_daily_local(hourly)
        hol = get_holidays(sorted({d.year for d in days}), "NW")
        NT = 28
        ND = len(l2)
        if ND < 2 * NT + 10:
            pytest.skip(f"Reihe {key} zu kurz fuer leakage-sicheren Test.")
        Rt, _ = rolling_origin(l2, days, fac, n_test=NT, holiday_set=hol)
        Rv, _ = rolling_origin(l2[:ND - NT], days[:ND - NT], fac, n_test=NT, holiday_set=hol)
        s = coverage_scale(Rv["actual"], Rv["p10"], Rv["model"], Rv["p90"], target=0.8, shrink=0.5)
        lo, hi = apply_scale(Rt["p10"], Rt["model"], Rt["p90"], s)
        c_naiv = _coverage(Rt["actual"], Rt["p10"], Rt["p90"]) * 100
        c_cal = _coverage(Rt["actual"], lo, hi) * 100
        dev_naiv.append(abs(c_naiv - 80))
        dev_cal.append(abs(c_cal - 80))
    assert float(np.mean(dev_cal)) < float(np.mean(dev_naiv)), \
        f"Kalibrierung sollte mean|cov-80| senken: naiv {np.mean(dev_naiv):.2f} -> kal {np.mean(dev_cal):.2f}"
