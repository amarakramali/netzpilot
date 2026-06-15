#!/usr/bin/env python3
"""Modell-Shootout: Champion vs. Varianten auf echten Daten — ehrlich auf Signifikanz der DIFFERENZ.

Frage: Ist eine Modellvariante WIRKLICH besser als der aktuelle Champion (ShrunkCorrector +
build_features) — oder ist der Unterschied nur Rauschen? Antwort via paired Block-Bootstrap der
tageweisen MAE-DIFFERENZ (Champion - Variante). Block = ganzer Tag.

Nur so darf man Modelle vergleichen: nicht "Variante hat MAPE 4,1 statt 4,3" (kann Zufall sein),
sondern "die Differenz ist auf dem 5%-Niveau von 0 verschieden (oder eben nicht)".

Kandidaten:
  champion   = ShrunkCorrector(10) + build_features            (aktueller Stand)
  ensemble   = EnsembleCorrector(3,10,30) + build_features      (Varianzreduktion)
  smallfeat  = ShrunkCorrector(10) + build_small_load_features  (mehr Lags fuer volatile Kleinlast)
  ens_small  = EnsembleCorrector + build_small_load_features    (beides)

Nur numpy/pandas/stdlib. Fester Seed. Aufruf:
  python scripts/model_shootout.py --only hilden_netzumsatz evdb_ns
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from netzpilot.features.build import (get_holidays, to_daily_local,
                                      build_features, build_small_load_features)
from netzpilot.eval.backtest import rolling_origin
from netzpilot.models.robust_corrector import ShrunkCorrector
from netzpilot.models.ensemble_corrector import EnsembleCorrector
from scripts.pilot_in_a_box import robust_load_csv
from scripts.dataset_manifest import MANIFEST

SEED = 20260601
N_TEST = 28
N_BOOT = 4000


def candidates():
    return {
        "champion":  (lambda: ShrunkCorrector(10.0), build_features),
        "ensemble":  (lambda: EnsembleCorrector((3.0, 10.0, 30.0)), build_features),
        "smallfeat": (lambda: ShrunkCorrector(10.0), build_small_load_features),
        "ens_small": (lambda: EnsembleCorrector((3.0, 10.0, 30.0)), build_small_load_features),
    }


def daily_mae(R):
    ae = np.abs(R["model"] - R["actual"]).reshape(-1, 24)
    return np.nanmean(ae, axis=1)


def run_variant(load2d, days, hol, factory, feature_fn, n_test, first=14):
    # first=14: build_small_load_features braucht >=14 Vortage; alle Varianten gleich -> paired & fair.
    R, summary = rolling_origin(load2d, days, factory, first=first, n_test=n_test,
                                holiday_set=hol, feature_fn=feature_fn)
    return daily_mae(R), summary["metriken"]["model"]


def paired_diff_bootstrap(ae_champ, ae_var, rng, n_boot):
    """Bootstrap der tageweisen MAE-Differenz (champ - var). >0 => Variante besser."""
    mask = np.isfinite(ae_champ) & np.isfinite(ae_var)
    ac, av = ae_champ[mask], ae_var[mask]
    n = len(ac)
    idx = rng.integers(0, n, size=(n_boot, n))
    diff = (ac[idx] - av[idx]).mean(axis=1)   # MW/Tag, positiv = Variante besser
    rel = (1.0 - av[idx].sum(axis=1) / ac[idx].sum(axis=1)) * 100.0  # % Verbesserung ggü Champion
    return diff, rel, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--region", default="NW")
    ap.add_argument("--out", default="data_cache/benchmark")
    a = ap.parse_args()

    entries = MANIFEST if not a.only else [e for e in MANIFEST if e["key"] in a.only]
    cands = candidates()
    all_rows = []
    for e in entries:
        if not os.path.exists(e["csv"]):
            print(f"[SKIP] {e['name']}: CSV fehlt")
            continue
        print(f"[run ] {e['name']} …")
        hourly, _ts, _lc, _m = robust_load_csv(e["csv"], ts_col=e["ts"], load_col=e["col"],
                                               unit=e["unit"], return_meta=True)
        load2d, days, _ = to_daily_local(hourly)
        if len(load2d) < a.n_test + 30:
            print(f"[SKIP] {e['name']}: zu wenig Tage")
            continue
        hol = get_holidays(sorted({d.year for d in days}), a.region)

        ae = {}
        mae_mw = {}
        for name, (factory, ffn) in cands.items():
            ae[name], m = run_variant(load2d, days, hol, factory, ffn, a.n_test)
            mae_mw[name] = m["MAE_MW"]

        rng = np.random.default_rng(SEED)
        row = {"dataset": e["name"], "mae_champion_MW": mae_mw["champion"], "variants": {}}
        for name in cands:
            if name == "champion":
                continue
            diff, rel, n = paired_diff_bootstrap(ae["champion"], ae[name], rng, a.n_boot)
            lo, hi = float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))
            row["variants"][name] = {
                "mae_MW": mae_mw[name],
                "rel_improvement_%": round(float((1.0 - ae[name].sum() / ae["champion"].sum()) * 100), 2),
                "rel_ci95_%": [round(lo, 2), round(hi, 2)],
                "P_besser_%": round(float(np.mean(diff > 0) * 100), 1),
                "signifikant_besser_5pct": bool(lo > 0),
                "signifikant_schlechter_5pct": bool(hi < 0),
            }
        all_rows.append(row)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "n_test": a.n_test, "n_boot": a.n_boot,
        "frage": "Schlaegt eine Variante den Champion (ShrunkCorrector+build_features) signifikant?",
        "rows": all_rows,
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "model_shootout.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Konsolentabelle
    print("\n=== Modell-Shootout (rel. Verbesserung ggü. Champion, CI95) ===")
    for r in all_rows:
        print(f"\n{r['dataset']} (Champion MAE {r['mae_champion_MW']} MW):")
        for name, v in r["variants"].items():
            flag = "✓besser" if v["signifikant_besser_5pct"] else ("✗schlechter" if v["signifikant_schlechter_5pct"] else "○neutral")
            print(f"  {name:10s}: {v['rel_improvement_%']:+5.2f}% "
                  f"[{v['rel_ci95_%'][0]:+.1f},{v['rel_ci95_%'][1]:+.1f}]  {flag}")
    print(f"\n-> {a.out}/model_shootout.json")


if __name__ == "__main__":
    main()
