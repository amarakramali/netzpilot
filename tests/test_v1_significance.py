# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Regression-/Konsistenztest fuer die v1-Signifikanzanalyse.

Sichert das Codex-Ergebnis `data_cache/v1_significance/results.json` ab:
1. Reproduziert Punkt- und Bootstrap-Kennzahlen EXAKT aus dem dep-freien
   v1-Backtest (gleicher Seed/Methode) -> Golden-Lock gegen stille Drift.
2. Prueft interne Konsistenz (keine NaN; Skill aus MAE rekonstruierbar; MASE-
   Nenner konsistent; CI-vs-Signifikanzflagge konsistent; P in [0, 100]).
3. Unit-Test des wiederverwendbaren `paired_block_bootstrap` auf kontrollierten
   Faellen (identische Fehler -> Skill 0, nicht signifikant; Modell strikt
   besser -> P=100 %, signifikant). Dieses Werkzeug ist die Grundlage fuer den
   noch offenen T9/T10-Signifikanztest auf Klein-Last.

Nur numpy/stdlib. Leakage-frei (identisches Backtest-Protokoll wie v1).
"""
import importlib.util
import json
import math
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULT = os.path.join(_ROOT, "data_cache", "v1_significance", "results.json")
_REPRO_CACHE = {}


def _load_sig_module():
    spec = importlib.util.spec_from_file_location(
        "eval_v1_significance",
        os.path.join(_ROOT, "scripts", "eval_v1_significance.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _committed():
    with open(_RESULT, encoding="utf-8") as f:
        return json.load(f)


def _reproduce():
    """Rerun the exact dep-free v1 significance pipeline once; cache it."""
    if _REPRO_CACHE:
        return _REPRO_CACHE["out"]
    ev = _load_sig_module()
    from netzpilot.data.smard import load_local_json
    from netzpilot.features.build import to_daily, get_holidays
    from netzpilot.eval.backtest import rolling_origin
    from netzpilot.models.ridge_correction import RidgeCorrector

    series = load_local_json("prognose_engine_v1/data/wk*.json")
    load2d, days = to_daily(series)
    hol = get_holidays(sorted({d.year for d in days}), "NW")
    R, summary = rolling_origin(
        load2d, days, lambda: RidgeCorrector(lam=10.0), n_test=28, holiday_set=hol
    )
    ae = {m: ev.daily_mae(R, m) for m in ["model", "snaive", "persist"]}
    rng = np.random.default_rng(ev.SEED)
    boot = {}
    for ref in ["snaive", "persist"]:
        skill, dmae = ev.paired_block_bootstrap(ae["model"], ae[ref], rng, ev.N_BOOT)
        sp = (1.0 - ae["model"].sum() / ae[ref].sum()) * 100.0
        boot["model_vs_" + ref] = {
            "skill_point_%": round(float(sp), 2),
            "skill_ci95_%": ev.ci95(skill),
            "P(model_besser)_%": round(float(np.mean(skill > 0) * 100), 1),
            "dMAE_mean_MW": round(float(dmae.mean()), 1),
            "signifikant_5pct": bool(np.percentile(skill, 2.5) > 0),
        }
    _REPRO_CACHE["out"] = {"summary": summary, "bootstrap": boot}
    return _REPRO_CACHE["out"]


def test_v1_significance_reproduces_committed():
    """Golden lock: committed Codex JSON reproduces exactly from the dep-free run."""
    rep, c = _reproduce(), _committed()
    assert abs(rep["summary"]["metriken"]["model"]["MAE_MW"] - 1411.4) < 0.5
    for ref in ["snaive", "persist"]:
        cb, rb = c["bootstrap"]["model_vs_" + ref], rep["bootstrap"]["model_vs_" + ref]
        for key in ("skill_point_%", "skill_ci95_%", "P(model_besser)_%",
                    "dMAE_mean_MW", "signifikant_5pct"):
            assert cb[key] == rb[key], (ref, key, cb[key], rb[key])


def test_committed_result_is_internally_consistent():
    c = _committed()
    assert c["seed"] == 20260530 and c["n_test_days"] == 28 and c["n_boot"] == 10000

    def _leaves(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from _leaves(v)
        elif isinstance(o, list):
            for v in o:
                yield from _leaves(v)
        elif isinstance(o, (int, float)):
            yield o

    assert all(math.isfinite(x) for x in _leaves(c)), "non-finite value in result"

    pm = c["punkt_metriken"]
    mae = {k: pm[k]["MAE_MW"] for k in ("persist", "snaive", "model")}

    def skill(a, b):
        return round((b - a) / b * 100, 1)

    assert pm["model"]["Skill_vs_SaisonalNaiv_%"] == skill(mae["model"], mae["snaive"])
    assert pm["model"]["Skill_vs_Persistenz_%"] == skill(mae["model"], mae["persist"])
    assert pm["snaive"]["Skill_vs_Persistenz_%"] == skill(mae["snaive"], mae["persist"])

    den = [pm[k]["MAE_MW"] / pm[k]["MASE"] for k in ("persist", "snaive", "model")]
    assert max(den) - min(den) < 2.0, den  # single MASE scaling across models

    for b in c["bootstrap"].values():
        lo, hi = b["skill_ci95_%"]
        assert b["signifikant_5pct"] == (not (lo <= 0 <= hi))
        assert 0.0 <= b["P(model_besser)_%"] <= 100.0
        dlo, dhi = b["dMAE_ci95_MW"]
        assert dlo <= b["dMAE_mean_MW"] <= dhi


def test_paired_block_bootstrap_identical_errors():
    """Identical model/ref errors -> zero skill, never 'better', not significant."""
    ev = _load_sig_module()
    rng = np.random.default_rng(0)
    ae = np.array([10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 7.0])
    skill, dmae = ev.paired_block_bootstrap(ae, ae.copy(), rng, 2000)
    assert np.allclose(skill, 0.0) and np.allclose(dmae, 0.0)
    assert float(np.mean(skill > 0) * 100) == 0.0
    assert bool(np.percentile(skill, 2.5) > 0) is False


def test_paired_block_bootstrap_strictly_better():
    """Model better on every day -> P(better)=100 %, significant, positive skill."""
    ev = _load_sig_module()
    rng = np.random.default_rng(1)
    ref = np.array([10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 7.0])
    model = ref - 2.0  # strictly smaller error each day
    skill, dmae = ev.paired_block_bootstrap(model, ref, rng, 2000)
    assert float(np.mean(skill > 0) * 100) == 100.0
    assert bool(np.percentile(skill, 2.5) > 0) is True
    assert dmae.mean() > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        fn()
        ok += 1
        print("PASS", fn.__name__)
    print("v1_significance tests: %d passed" % ok)
