import math
import os
import random
import statistics
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.eval.bilanzkreis import compare_forecasts_eur, savings_contrib_per_qh
from netzpilot.eval.mc_savings import block_bootstrap_band


def test_block_bootstrap_is_deterministic_and_observed_is_sum():
    contrib = [math.sin(i) * 10 + (i % 7 - 3) for i in range(96 * 30)]
    r1 = block_bootstrap_band(contrib, seed=123, n_resamples=800)
    r2 = block_bootstrap_band(contrib, seed=123, n_resamples=800)
    r3 = block_bootstrap_band(contrib, seed=999, n_resamples=800)
    assert r1 == r2
    assert r3["observed_total_eur"] == r1["observed_total_eur"]
    assert r1["observed_total_eur"] == round(sum(contrib), 2)


def test_constant_and_zero_boundaries():
    const = [2.0] * (96 * 10)
    r = block_bootstrap_band(const, block_len=96, n_resamples=300)
    assert r["p5_eur"] == r["p95_eur"] == r["observed_total_eur"]
    assert r["std_eur"] == 0.0
    assert r["prob_positive"] == 1.0

    z = block_bootstrap_band([0.0] * (96 * 5), n_resamples=200)
    assert z["observed_total_eur"] == 0.0
    assert z["p95_eur"] == 0.0
    assert z["std_eur"] == 0.0
    assert z["prob_positive"] == 0.0


def test_bootstrap_mean_and_std_are_reasonable():
    contrib = [math.sin(i) * 10 + (i % 7 - 3) for i in range(96 * 30)]
    n_resamples = 3000
    r = block_bootstrap_band(contrib, seed=7, n_resamples=n_resamples)
    se = r["std_eur"] / math.sqrt(n_resamples)
    assert abs(r["mean_eur"] - r["observed_total_eur"]) < 4 * se + 1e-9

    rng = random.Random(1)
    vals = [rng.gauss(0.5, 3.0) for _ in range(300)]
    r = block_bootstrap_band(vals, block_len=1, n_resamples=5000, seed=3)
    expected_std = math.sqrt(len(vals)) * statistics.pstdev(vals)
    rel = abs(r["std_eur"] - expected_std) / expected_std
    assert rel < 0.10
    assert r["n_blocks"] == len(vals)


def test_prob_positive_symmetric_case_and_partial_block():
    rng = random.Random(2)
    sym = []
    for _ in range(100):
        v = abs(rng.gauss(0, 5)) + 0.5
        sym += [v / 96] * 96
        sym += [-v / 96] * 96
    r = block_bootstrap_band(sym, block_len=96, n_resamples=2500, seed=5)
    assert abs(r["observed_total_eur"]) < 1e-9
    assert 0.4 <= r["prob_positive"] <= 0.6

    part = [1.0] * (96 * 4 + 37)
    r = block_bootstrap_band(part, block_len=96, n_resamples=100)
    assert r["n_blocks"] == 5
    assert r["observed_total_eur"] == round(sum(part), 2)


def test_band_binds_to_compare_forecasts_savings():
    actual = [10.0 + math.sin(i / 5) for i in range(96 * 12)]
    sa = [a + math.cos(i / 4) * 0.8 for i, a in enumerate(actual)]
    sb = [a + math.cos(i / 4) * 0.2 for i, a in enumerate(actual)]
    reb = [50.0 + 80 * math.sin(i / 9) for i in range(len(actual))]
    spot = [40.0 + 5 * math.cos(i / 11) for i in range(len(actual))]
    contrib = savings_contrib_per_qh(actual, sa, sb, reb, spot)
    cmp = compare_forecasts_eur(actual, sa, sb, reb, spot)
    band = block_bootstrap_band(contrib, n_resamples=300)
    assert abs(band["observed_total_eur"] - cmp["savings_b_vs_a_eur"]) < 0.02


def test_validation_errors():
    with pytest.raises(ValueError):
        block_bootstrap_band([])
    with pytest.raises(ValueError):
        block_bootstrap_band([1.0], block_len=0)
    with pytest.raises(ValueError):
        block_bootstrap_band([1.0], n_resamples=0)
