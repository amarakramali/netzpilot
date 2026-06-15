"""Signifikanz/Unsicherheit der v1-Punktprognose auf dem 12-Wochen-Fenster.

Kontext (07:30-Lauf, Fallback): Die geplante T4-Residuallast-Mini-Demo war in der
Sandbox NICHT ausfuehrbar -- SMARD ist per WebFetch nicht erreichbar (Provenance-
Restriktion), und die bereits gecachte Erzeugung liegt nur als Parquet vor
(ohne pyarrow/duckdb in der Sandbox nicht lesbar). Statt etwas Unverifiziertes zu
behaupten, quantifiziere ich hier ehrlich die UNSICHERHEIT der bereits selbst
verifizierten v1-Kennzahl (MAE 1411,4; Skill +4,1 % vs. Saisonal-Naiv).

Frage: Ist der v1-Vorsprung gegen Saisonal-Naiv auf 12 Wochen statistisch von 0 zu
unterscheiden? Methode: Paired Block-Bootstrap (Block = ganzer Tag, respektiert die
Intraday-Korrelation) ueber die 28 Test-Tage. Erwartung laut Notizen: auf 12 Wochen
duenn; der grosse Sprung kommt erst mit T3-LightGBM+Wetter auf 2 Jahren (+55,8 %).

Nur numpy/stdlib. Leakage-frei: identischer Rolling-Origin-Backtest wie v1, keine
Aenderung am Evaluationsprotokoll. Reproduzierbar (fester Seed).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from netzpilot.data.smard import load_local_json
from netzpilot.features.build import to_daily, get_holidays
from netzpilot.eval.backtest import rolling_origin
from netzpilot.models.ridge_correction import RidgeCorrector

SEED = 20260530
N_BOOT = 10000
N_TEST = 28


def daily_mae(R, name):
    """Tageweise MAE [n_test] aus den flachen [n_test*24]-Arrays des Backtests."""
    ae = np.abs(R[name] - R["actual"])
    return ae.reshape(-1, 24).mean(axis=1)


def paired_block_bootstrap(ae_model, ae_ref, rng, n_boot):
    """Paired Block-Bootstrap ueber ganze Tage.

    Resamplet Tagesindizes mit Zuruecklegen; Modell und Referenz teilen denselben
    Index (paired), damit gemeinsame schwere/leichte Tage die Differenz nicht
    kuenstlich aufblaehen. Skill% = (1 - sum AE_model / sum AE_ref) * 100.
    """
    n = len(ae_model)
    idx = rng.integers(0, n, size=(n_boot, n))
    sm = ae_model[idx].sum(axis=1)
    sr = ae_ref[idx].sum(axis=1)
    skill = (1.0 - sm / sr) * 100.0
    dmae = (ae_ref[idx] - ae_model[idx]).mean(axis=1)  # MW/Tag (Ref - Modell)
    return skill, dmae


def ci95(x):
    return [round(float(np.percentile(x, 2.5)), 2), round(float(np.percentile(x, 97.5)), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prognose_engine_v1/data/wk*.json")
    ap.add_argument("--region", default="NW")
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="data_cache/v1_significance")
    a = ap.parse_args()

    series = load_local_json(a.data)
    load2d, days = to_daily(series)
    hol = get_holidays(sorted({d.year for d in days}), a.region)
    R, summary = rolling_origin(
        load2d, days, lambda: RidgeCorrector(lam=10.0), n_test=a.n_test, holiday_set=hol
    )

    ae = {m: daily_mae(R, m) for m in ["model", "snaive", "persist"]}
    rng = np.random.default_rng(SEED)

    out = {
        "frage": "Ist der v1-Skill vs. Saisonal-Naiv auf 12 Wochen von 0 unterscheidbar?",
        "fenster": "12 Wochen (2024-01-01..2024-03-25), v1-Last (SMARD 410)",
        "n_test_days": a.n_test,
        "n_boot": a.n_boot,
        "seed": SEED,
        "methode": "paired block-bootstrap, Block = ganzer Tag (24 h)",
        "punkt_metriken": summary["metriken"],
        "bootstrap": {},
    }

    for ref in ["snaive", "persist"]:
        skill, dmae = paired_block_bootstrap(ae["model"], ae[ref], rng, a.n_boot)
        skill_point = (1.0 - ae["model"].sum() / ae[ref].sum()) * 100.0
        out["bootstrap"][f"model_vs_{ref}"] = {
            "skill_point_%": round(float(skill_point), 2),
            "skill_ci95_%": ci95(skill),
            "P(model_besser)_%": round(float(np.mean(skill > 0) * 100), 1),
            "dMAE_mean_MW": round(float(dmae.mean()), 1),
            "dMAE_ci95_MW": [round(float(np.percentile(dmae, 2.5)), 1),
                             round(float(np.percentile(dmae, 97.5)), 1)],
            "tage_modell_gewinnt_%": round(float(np.mean(ae["model"] < ae[ref]) * 100), 1),
            "signifikant_5pct": bool(np.percentile(skill, 2.5) > 0),
        }

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
