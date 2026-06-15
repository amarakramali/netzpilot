# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Blind-Challenge: leakage-sicherer Sofort-Beweis auf FREMDEN Daten.

Der Termin-Moment: jemand bringt seinen eigenen Lastgang mit, und NetzPilot rechnet vor seinen
Augen den identischen, leakage-sicheren Rolling-Origin-Backtest wie im Benchmark-Board — gegen
Persistenz UND Saisonal-Naiv, mit paired Block-Bootstrap (Block = ganzer Tag) und Signifikanz.
Keine Folie, kein Vertrauensvorschuss: die eigene Datei ist der Beweis.

Bewusst KEINE neue Mathematik: wiederverwendet werden exakt die verifizierten Bausteine der
Benchmark-Suite (daily_mae, paired_block_bootstrap) und der Engine (rolling_origin mit
feiertagsbewusster Basis, ShrunkCorrector). Fester Seed -> reproduzierbar.

Ehrlichkeits-Regeln (identisch zum Board):
- MAPE nur, wenn sinnvoll (nicht-signierte Reihe, Mittel deutlich > 0, MAPE endlich/<60 %).
- Signifikanz = CI95-Untergrenze des Skills > 0 (paired Block-Bootstrap, Block = Tag).
- Zu wenig Historie -> klarer Fehler statt weicher Zahl (min. first+30 Fit-Tage vor dem Testfenster).
"""
from __future__ import annotations

import math

import numpy as np

from netzpilot.eval.backtest import rolling_origin
from netzpilot.features.build import get_holidays, to_daily_local
from netzpilot.models.robust_corrector import ShrunkCorrector
from scripts.benchmark_suite import SEED, daily_mae, paired_block_bootstrap, pct
from scripts.pilot_in_a_box import robust_load_csv

_MIN_FIT_DAYS = 38   # first=8 + ~30 Tage echtes Fit-Fenster vor dem ersten Testtag
_MIN_TEST = 28


def _boot_block(ae_model, ae_ref, rng, n_boot):
    res = paired_block_bootstrap(ae_model, ae_ref, rng, n_boot)
    if res is None:
        return None
    skill, _dmae, _n = res
    skill_point = (1.0 - float(np.nansum(ae_model)) / float(np.nansum(ae_ref))) * 100.0
    lo, hi = pct(skill, 2.5), pct(skill, 97.5)
    return {
        "skill_pct": round(skill_point, 1),
        "ci95": [round(lo, 1), round(hi, 1)],
        "significant_5pct": bool(lo > 0),
        "p_model_better_pct": round(float(np.nanmean(skill > 0) * 100.0), 1),
        "days_won_pct": round(float(np.mean(ae_model < ae_ref) * 100.0), 1),
    }


def run_challenge(csv_path: str, *, ts_col=None, load_col=None, unit: str = "MW",
                  region: str = "NW", n_test: int = 84, n_boot: int = 4000) -> dict:
    """Backtest + Signifikanz auf einer (fremden) Lastgang-Datei. Wirft ValueError bei zu wenig Historie."""
    hourly, used_ts, used_col, meta = robust_load_csv(
        csv_path, ts_col=ts_col, load_col=load_col, unit=unit, return_meta=True)
    load2d, days, _ = to_daily_local(hourly)
    n_days = int(len(load2d))

    n_test_eff = int(min(int(n_test), n_days - _MIN_FIT_DAYS))
    if n_test_eff < _MIN_TEST:
        raise ValueError(
            f"Zu wenig Historie für eine belastbare Challenge: {n_days} vollständige Tage; "
            f"mindestens ~{_MIN_FIT_DAYS + _MIN_TEST} nötig (Fit-Vorlauf + {_MIN_TEST} Testtage)."
        )

    hol = get_holidays(sorted({d.year for d in days}), region)
    R, summary = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0),
                                n_test=n_test_eff, holiday_set=hol)

    ae = {m: daily_mae(R, m) for m in ("model", "snaive", "persist")}
    rng = np.random.default_rng(SEED)
    vs_snaive = _boot_block(ae["model"], ae["snaive"], rng, int(n_boot))
    vs_persist = _boot_block(ae["model"], ae["persist"], rng, int(n_boot))

    m = summary["metriken"]["model"]
    mean_load = float(np.nanmean(load2d))
    mape_raw = m["MAPE_%"]
    mape_meaningless = bool(abs(mean_load) < 1.0 or not math.isfinite(mape_raw) or mape_raw > 60.0)

    return {
        "source_file": str(csv_path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
        "ts_col": used_ts, "load_col": used_col, "unit_in": unit,
        "n_days_history": n_days,
        "n_test": n_test_eff, "n_test_requested": int(n_test), "n_boot": int(n_boot),
        "seed": SEED,
        "mean_load_mw": round(mean_load, 3),
        "mae_model_mw": m["MAE_MW"],
        "mae_snaive_mw": summary["metriken"]["snaive"]["MAE_MW"],
        "mae_persist_mw": summary["metriken"]["persist"]["MAE_MW"],
        "mape_pct": None if mape_meaningless else mape_raw,
        "mape_note": ("MAPE bei signierten/niedrigen Reihen bedeutungslos — Skill/MAE führen"
                      if mape_meaningless else None),
        "coverage_p10_p90_pct": summary["probabilistisch"]["Coverage_P10_P90_%"],
        "vs_snaive": vs_snaive,
        "vs_persist": vs_persist,
        "daily_mae_model": [round(float(x), 4) for x in ae["model"]],
        "daily_mae_snaive": [round(float(x), 4) for x in ae["snaive"]],
        "method": ("leakage-sicherer Rolling-Origin-Backtest (expanding window, feiertagsbewusste "
                   "Basis, ShrunkCorrector) + paired Block-Bootstrap (Block = ganzer Tag)"),
        "caveats": [
            "Jede Prognose nutzt ausschließlich Daten VOR dem Zieltag (kein Leakage).",
            "Signifikanz: CI95-Untergrenze des Skills > 0; Bootstrap respektiert Intraday-Korrelation.",
            "Saisonal-Naiv = Branchenpraxis (Vorwoche, gleicher Wochentag); Persistenz = Vortag.",
            "Ergebnis gilt für DIESE Reihe und diesen Zeitraum — kein Versprechen für andere Netze.",
        ],
    }
