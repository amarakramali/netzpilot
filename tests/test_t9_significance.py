# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Regression tests for the T9 small-utility significance result."""
import json
import math
from pathlib import Path

from scripts.eval_t9_significance import N_BOOT, SEED, summarize_npz

_ROOT = Path(__file__).resolve().parents[1]
_RESULT = _ROOT / "data_cache" / "t9_significance" / "results.json"


def _committed():
    if not _RESULT.exists():
        return None
    with _RESULT.open(encoding="utf-8") as f:
        return json.load(f)


def test_t9_significance_reproduces_committed():
    committed = _committed()
    if committed is None:
        return
    assert committed["seed"] == SEED and committed["n_boot"] == N_BOOT

    for run in committed["runs"]:
        reproduced = summarize_npz(run["label"], _ROOT / run["arrays"], N_BOOT, SEED)
        assert reproduced["daily_mae_MW"] == run["daily_mae_MW"]
        for key in ("model_vs_snaive", "model_vs_persist"):
            cb = run["bootstrap"][key]
            rb = reproduced["bootstrap"][key]
            for metric in (
                "skill_point_%",
                "skill_ci95_%",
                "P(model_besser)_%",
                "dMAE_mean_MW",
                "dMAE_ci95_MW",
                "tage_modell_gewinnt_%",
                "signifikant_5pct",
            ):
                assert cb[metric] == rb[metric], (run["label"], key, metric, cb[metric], rb[metric])


def test_t9_significance_is_internally_consistent():
    committed = _committed()
    if committed is None:
        return

    def leaves(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from leaves(v)
        elif isinstance(o, list):
            for v in o:
                yield from leaves(v)
        elif isinstance(o, (int, float)):
            yield o

    assert all(math.isfinite(x) for x in leaves(committed))
    for run in committed["runs"]:
        assert run["n_test_days"] == 28
        for boot in run["bootstrap"].values():
            lo, hi = boot["skill_ci95_%"]
            assert boot["signifikant_5pct"] == (lo > 0.0)
            assert 0.0 <= boot["P(model_besser)_%"] <= 100.0
            dlo, dhi = boot["dMAE_ci95_MW"]
            assert dlo <= boot["dMAE_mean_MW"] <= dhi

    no_weather = committed["runs"][0]["bootstrap"]
    assert no_weather["model_vs_snaive"]["signifikant_5pct"] is False
    assert no_weather["model_vs_persist"]["signifikant_5pct"] is True
