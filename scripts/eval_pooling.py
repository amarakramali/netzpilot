#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Beweis: Multi-Mandanten-Pooling hilft Haeusern mit WENIG Historie.

Leave-one-utility-out auf den (modellierten) Stadt-Profilen:
  1) Pool-Prior aus N-1 Staedten lernen (fit_pool_prior).
  2) Ziel-Stadt mit KUENSTLICH BESCHNITTENER Historie (z.B. nur 14/21/28 Tage) prognostizieren —
     einmal ISOLIERT (RidgeCorrector, nur eigene Daten) und einmal GEPOOLT (PooledCorrector).
  3) Vergleiche den Day-ahead-MAE auf einem festen Testfenster.

Erwartung: bei wenig Historie schlaegt GEPOOLT das isolierte Modell (borrowing strength); bei viel
Historie gleichen sich beide an. Das ist der Datennetzwerkeffekt — und der Beleg, dass er real wirkt.

EHRLICH: Die Stadt-Profile sind MODELLIERT (synthetisch), siehe Memory. Dieses Experiment zeigt, dass
der Pooling-MECHANISMUS funktioniert; die Staerke des Effekts auf ECHTEN Stadtwerken kann abweichen und
waechst erst mit echter Kundenbasis. Reine numpy/pandas/stdlib. Fester Seed.

Aufruf:  python scripts/eval_pooling.py --hist-days 14 21 28
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from netzpilot.features.build import build_features, base, resid_target, get_holidays
from netzpilot.models.ridge_correction import RidgeCorrector
from netzpilot.models.pooled_corrector import fit_pool_prior, PooledCorrector

CITIES_DIR = "netzpilot/data/training_cities"
FIRST = 8


def load_city(path):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"])
    s = pd.Series(df["load_mw"].to_numpy(float), index=ts).sort_index()
    n = len(s) - (len(s) % 24)
    load2d = s.values[:n].reshape(-1, 24)
    days = pd.to_datetime([s.index[d * 24].date() for d in range(len(load2d))])
    return load2d, days


def normalize(load2d, scale):
    """Auf eine Referenz-Last skalieren, damit Korrektur-Muster ueber Haeuser vergleichbar werden.
    Pooling MUSS im last-normalisierten Raum passieren — sonst mittelt man inkommensurable Niveaus
    (Berlin 45 MW vs. Kleinstadt 5 MW) und der Prior schadet. Das ist der methodische Kernpunkt."""
    return load2d / scale


def train_matrix(load2d, days, hol, lo, hi):
    X = np.vstack([build_features(load2d, days, t, None, hol) for t in range(lo, hi)])
    y = np.concatenate([resid_target(load2d, t) for t in range(lo, hi)])
    return X, y


def day_mae(load2d, days, hol, model, test_days):
    errs = []
    for d in test_days:
        yhat = base(load2d, d) + model.predict(build_features(load2d, days, d, None, hol))
        errs.append(np.mean(np.abs(yhat - load2d[d])))
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist-days", type=int, nargs="*", default=[14, 21, 28])
    ap.add_argument("--n-test", type=int, default=14)
    ap.add_argument("--max-cities", type=int, default=20, help="Pool-Groesse begrenzen (Sandbox-Budget)")
    ap.add_argument("--targets", type=int, default=6, help="wie viele Ziel-Staedte mitteln")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(CITIES_DIR, "*.csv")))[:a.max_cities]
    if len(files) < 4:
        raise SystemExit(f"Zu wenig Stadt-Profile in {CITIES_DIR}.")
    cities = {os.path.basename(f).split("_")[0]: load_city(f) for f in files}
    names = list(cities)
    hol = get_holidays([2024], "NW")

    print(f"Pool aus {len(names)} Staedten · {a.n_test} Testtage · Ziele: {names[:a.targets]}\n")
    print(f"{'Historie':>9} | {'isoliert MAE':>13} | {'gepoolt MAE':>12} | {'Verbesserung':>12} | besser?")
    print("-" * 70)

    results = {}
    for hist in a.hist_days:
        iso_list, pool_list = [], []
        for tgt in names[:a.targets]:
            load2d_raw, days = cities[tgt]
            ND = len(load2d_raw)
            test_days = list(range(ND - a.n_test, ND))
            train_hi = ND - a.n_test
            # Normalisierung je Haus auf mittlere Last des EIGENEN Trainingsfensters (leakage-sicher).
            tgt_scale = float(np.mean(load2d_raw[max(FIRST, train_hi - hist):train_hi]))
            load2d = normalize(load2d_raw, tgt_scale)

            # Pool-Prior aus allen ANDEREN Staedten — jede auf IHRE eigene mittlere Last normalisiert.
            others = []
            for o in names:
                if o == tgt:
                    continue
                l2_raw, dy = cities[o]
                l2 = normalize(l2_raw, float(np.mean(l2_raw[FIRST:len(l2_raw) - a.n_test])))
                others.append(train_matrix(l2, dy, hol, FIRST, len(l2) - a.n_test))
            w_pool, _sd = fit_pool_prior(others)

            train_lo = max(FIRST, train_hi - hist)
            Xtr, ytr = train_matrix(load2d, days, hol, train_lo, train_hi)

            iso = RidgeCorrector(10.0).fit(Xtr, ytr)
            pooled = PooledCorrector(w_pool, lam=10.0, tau_days=30.0).fit(Xtr, ytr)
            # MAE im ECHTEN MW-Raum (zurueckskalieren), damit Vergleich fair & interpretierbar.
            iso_list.append(day_mae(load2d, days, hol, iso, test_days) * tgt_scale)
            pool_list.append(day_mae(load2d, days, hol, pooled, test_days) * tgt_scale)

        iso_mae, pool_mae = float(np.mean(iso_list)), float(np.mean(pool_list))
        impr = (1 - pool_mae / iso_mae) * 100
        results[hist] = impr
        print(f"{hist:>7}d | {iso_mae:>13.3f} | {pool_mae:>12.3f} | {impr:>+11.1f}% | "
              f"{'JA' if impr > 0 else 'nein'}")

    print()
    helped = [h for h, i in results.items() if i > 0]
    if helped and min(results, key=results.get) == max(a.hist_days):
        print("BEFUND: Pooling hilft — am staerksten bei WENIG Historie, "
              "verschwindet mit mehr eigenen Daten (erwartetes Partial-Pooling-Verhalten).")
    elif helped:
        print(f"BEFUND: Pooling hilft bei {helped} Tagen Historie (borrowing strength bestaetigt).")
    else:
        print("BEFUND: Pooling half hier nicht — Pool-Prior passt nicht zu den Ziel-Staedten.")
    print("EHRLICH: Stadt-Profile sind modelliert; echte Effektstaerke erst mit Kundenbasis.")


if __name__ == "__main__":
    main()
