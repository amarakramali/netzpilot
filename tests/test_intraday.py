import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.intraday import intraday_update, _weights


HOURS = [{"hour": i, "p50": 20.0 + i, "p10": 18.0 + i, "p90": 23.0 + i}
         for i in range(24)]


def test_intraday_hand_calculation_and_resttag_shift():
    actuals = [21.0, 22.5]
    w0 = 0.5 ** (1 / 3.0)
    w = np.array([w0, 1.0])
    w = w / w.sum()
    delta_exp = 0.5 * (w[0] * 1.0 + w[1] * 1.5)

    r = intraday_update(HOURS, actuals, round_digits=None)

    assert r["applied"] is True
    assert r["update_hour"] == 2
    assert r["n_hours_used"] == 2
    assert abs(r["delta_mw"] - round(delta_exp, 4)) < 5e-5
    assert len(r["hours_rest"]) == 22
    assert r["hours_rest"][0]["hour"] == 2
    assert abs(r["hours_rest"][0]["p50"] - (22.0 + delta_exp)) < 1e-9
    assert abs((r["hours_rest"][0]["p90"] - r["hours_rest"][0]["p10"]) - 5.0) < 1e-9
    assert all(h["p10"] <= h["p50"] <= h["p90"] for h in r["hours_rest"])


def test_intraday_inputs_are_immutable_and_noops_are_explained():
    assert HOURS[2]["p50"] == 22.0 and "p10" in HOURS[2]

    assert intraday_update(HOURS, [])["applied"] is False
    assert intraday_update(HOURS, [20.0] * 24)["applied"] is False
    r_nan = intraday_update(HOURS, [float("nan"), float("nan"), 22.6])
    assert r_nan["applied"] is False
    assert "valide" in r_nan["reason"]
    assert HOURS[2]["p50"] == 22.0 and "p10" in HOURS[2]


def test_intraday_nan_gaps_are_renormalized():
    r = intraday_update(HOURS, [21.0, float("nan"), 23.0, 23.5], round_digits=None)

    assert r["applied"] is True
    assert r["n_hours_used"] == 3
    wfull = _weights(4, 3.0)
    mask = np.array([1, 0, 1, 1], float)
    wre = wfull * mask / (wfull * mask).sum()
    delta = 0.5 * float(np.sum(wre * np.array([1.0, 0.0, 1.0, 0.5])))
    assert abs(r["delta_mw"] - round(delta, 4)) < 5e-5


def test_intraday_p50_only_validation_determinism_and_measure_formula():
    h50 = [{"hour": i, "p50": 30.0} for i in range(24)]
    r = intraday_update(h50, [31.0, 31.0])
    assert r["applied"] is True
    assert "p10" not in r["hours_rest"][0]

    with pytest.raises(ValueError):
        intraday_update(HOURS, [21.0, 22.5], shrink=1.5)
    assert intraday_update(HOURS, [21.0, 22.5]) == intraday_update(HOURS, [21.0, 22.5])
    assert np.allclose(_weights(5, 0.0), 0.2)

    rng = np.random.default_rng(3)
    p50 = 20 + rng.normal(0, 1, 24)
    actual = p50 + rng.normal(0, 0.5, 24)
    hours = [{"hour": i, "p50": float(p50[i])} for i in range(24)]
    w12 = _weights(12, 3.0)
    delta_ref = 0.5 * float(np.sum(w12 * (actual[:12] - p50[:12])))
    rr = intraday_update(hours, actual[:12], round_digits=None)
    assert abs(rr["delta_mw"] - round(delta_ref, 4)) < 5e-5
