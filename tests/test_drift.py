import math
import os
import random
from pathlib import Path

import pytest

from netzpilot.eval.drift import (
    coverage_report,
    drift_report,
    ks_statistic,
    population_stability_index,
)
from netzpilot.service.drift_monitor import build_drift_payload

HERNE_CSV = "data_cache/real/herne_bezug_vorgelagerte_ebene_2024.csv"


def _gauss(n=5000):
    rng = random.Random(20260602)
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def test_drift_verify_anchors():
    gauss = _gauss()
    r = drift_report(gauss, list(gauss))
    assert abs(r["psi"]) < 1e-9
    assert abs(r["ks"]) < 1e-9
    assert abs(r["mae_ratio"] - 1.0) < 1e-9
    assert abs(r["bias_shift_abs"]) < 1e-9
    assert r["status"] == "stable"

    ref = [float(i) for i in range(1000)]
    recent = [50.0] * 150
    for c in (150, 250, 350, 450, 550, 650, 750, 850):
        recent += [float(c)] * 100
    recent += [950.0] * 50
    assert abs(population_stability_index(ref, recent, n_bins=10) - 0.054931) < 5e-3

    psi_small = population_stability_index(gauss, [x + 0.5 for x in gauss])
    psi_big = population_stability_index(gauss, [x + 1.5 for x in gauss])
    assert psi_small >= 0 and psi_big >= 0
    assert psi_big > psi_small

    unif = [i / 1000.0 for i in range(1000)]
    assert abs(ks_statistic(unif, [x + 0.5 for x in unif]) - 0.5) < 0.02
    assert abs(ks_statistic(unif, list(unif))) < 1e-9
    assert ks_statistic(gauss, [x + 1.5 for x in gauss]) > ks_statistic(gauss, [x + 0.5 for x in gauss])


def test_drift_status_and_coverage_cases():
    rng = random.Random(20260602)
    gauss = _gauss()

    r = drift_report(gauss, [x + 1.5 for x in gauss])
    assert abs(r["bias_shift_abs"] - 1.5) < 0.05
    assert abs(r["bias_shift_in_ref_std"] - 1.5) < 0.1
    assert r["status"] == "drift"
    assert any("bias" in s for s in r["reasons"])

    r = drift_report(gauss, [x * 2.0 for x in gauss])
    assert abs(r["mae_ratio"] - 2.0) < 0.1
    assert r["status"] == "drift"
    assert any("mae_ratio" in s for s in r["reasons"])

    r = drift_report(gauss, [x + 0.35 for x in gauss])
    assert r["status"] == "watch"
    assert r["psi"] <= 0.25

    gz = [rng.gauss(0.0, 1.0) for _ in range(8000)]
    q80 = 1.2816
    cov = coverage_report([-q80] * len(gz), [q80] * len(gz), gz, nominal=0.8, tol=0.1)
    assert 0.77 <= cov["coverage"] <= 0.83
    assert cov["status"] == "stable"

    q50 = 0.6745
    cov = coverage_report([-q50] * len(gz), [q50] * len(gz), gz, nominal=0.8, tol=0.1)
    assert 0.46 <= cov["coverage"] <= 0.54
    assert cov["status"] == "drift"
    assert abs(cov["frac_below_lower"] - cov["frac_above_upper"]) < 0.05


def test_drift_nan_drop_and_validation():
    rng = random.Random(20260602)
    gauss = _gauss()
    r = drift_report(gauss, [1.0, 2.0, float("nan"), 3.0, None] + [rng.gauss(0, 1) for _ in range(50)])
    assert r["n_recent"] == 53
    assert math.isfinite(r["psi"]) and math.isfinite(r["ks"])

    with pytest.raises(ValueError):
        drift_report(gauss, [])
    with pytest.raises(ValueError):
        coverage_report([], [], [])


def test_drift_monitor_persists_reference_and_recent(tmp_path):
    rng = random.Random(20260602)
    reference_days = 28
    recent_days = 14
    n = (reference_days + recent_days) * 24
    model = [10.0] * n
    errors = [rng.gauss(0.0, 1.0) for _ in range(reference_days * 24)]
    errors += [rng.gauss(0.0, 1.0) * 1.8 + 2.0 for _ in range(recent_days * 24)]
    actual = [m + e for m, e in zip(model, errors)]
    R = {
        "model": model,
        "actual": actual,
        "p10": [m - 1.3 for m in model],
        "p90": [m + 1.3 for m in model],
    }
    out = build_drift_payload(
        R,
        utility="UnitTestSW",
        base_dir=str(tmp_path),
        reference_days=reference_days,
        recent_days=recent_days,
        min_recent_days=7,
    )
    assert out["status"] == "drift"
    assert out["needs_recalibration"] is True
    assert Path(out["reference"]["path"]).exists()
    assert Path(out["recent"]["path"]).exists()
    assert (tmp_path / "UnitTestSW" / "latest_reference.json").exists()
    assert (tmp_path / "UnitTestSW" / "latest_recent.json").exists()


@pytest.mark.skipif(not os.path.exists(HERNE_CSV), reason="Herne-Realdaten fehlen")
def test_runner_drift_monitoring_smoke_on_real_data(tmp_path):
    from netzpilot.service.runner import run_forecast

    out = run_forecast(
        HERNE_CSV,
        utility="HerneDriftTest",
        unit="kW",
        ts_col="Datum+von",
        load_col="Load_1",
        drift_monitoring=True,
        drift_store_dir=str(tmp_path),
    )
    drift = out["drift"]
    assert drift is not None
    assert drift["status"] in {"stable", "watch", "drift", "insufficient_data", "not_available"}
    assert isinstance(drift["needs_recalibration"], bool)
    assert drift["action"] == "warn_only_no_auto_retraining"
    if drift["status"] not in {"insufficient_data", "not_available"}:
        assert Path(drift["reference"]["path"]).exists()
        assert Path(drift["recent"]["path"]).exists()
        assert drift["coverage"] is None or "coverage" in drift["coverage"]
