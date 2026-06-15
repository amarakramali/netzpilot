#!/usr/bin/env python3
"""Pool-Prior aus mehreren Lastgang-Reihen bauen und speichern (produktiver Multi-Mandanten-Effekt).

Speist den persistierten Pool-Prior (data_cache/pool/pool_prior.json), den der Dienst nutzt, wenn ein
neues Stadtwerk wenig Historie hat. Je mehr (echte) Reihen, desto besser der Prior für neue Kunden.

Quellen:
  --cities          alle modellierten Stadt-Profile (Methoden-Demo; synthetisch, siehe Notizen)
  --csv A B C ...    konkrete reale DSO-Lastgänge (Format wie pilot_in_a_box; je --col/--unit global)

Beispiel:
  python scripts/build_pool_prior.py --cities --max 40
  python scripts/build_pool_prior.py --csv data_cache/real/evdb_lastgang_ns_2024.csv --col Wert --unit kW
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from netzpilot.features.build import to_daily_local
from netzpilot.models.pool_prior import build_prior_from_series, save_prior, DEFAULT_PATH
from scripts.pilot_in_a_box import robust_load_csv

CITIES_DIR = "netzpilot/data/training_cities"


def _city_series(path):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"])
    s = pd.Series(df["load_mw"].to_numpy(float), index=ts).sort_index()
    n = len(s) - (len(s) % 24)
    load2d = s.values[:n].reshape(-1, 24)
    days = pd.to_datetime([s.index[d * 24].date() for d in range(len(load2d))])
    return load2d, days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", action="store_true", help="modellierte Stadt-Profile nutzen")
    ap.add_argument("--max", type=int, default=40, help="max. Anzahl Stadt-Profile")
    ap.add_argument("--csv", nargs="*", default=[], help="reale DSO-CSV-Pfade")
    ap.add_argument("--col", default=None, help="Lastspalte für --csv")
    ap.add_argument("--unit", default="MW")
    ap.add_argument("--corpus-index", default=None,
                    help="data_cache/real/corpus_index.json nutzen (explizite Spalten je Reihe)")
    ap.add_argument("--out", default=DEFAULT_PATH)
    a = ap.parse_args()

    series = []
    if a.cities:
        for f in sorted(glob.glob(os.path.join(CITIES_DIR, "*.csv")))[:a.max]:
            series.append(_city_series(f))
    for path in a.csv:
        hourly, _ts, _lc, _m = robust_load_csv(path, load_col=a.col, unit=a.unit, return_meta=True)
        load2d, days, _ = to_daily_local(hourly)
        series.append((load2d, days))
    if a.corpus_index:
        with open(a.corpus_index, encoding="utf-8") as f:
            corpus = json.load(f)
        for e in corpus.get("entries", []):
            if not e.get("include_in_pool", False):
                continue
            hourly, _ts, _lc, _m = robust_load_csv(
                e["path"], ts_col=e.get("ts"), load_col=e["col"], unit=e.get("unit", "MW"), return_meta=True)
            load2d, days, _ = to_daily_local(hourly)
            series.append((load2d, days))

    if len(series) < 2:
        raise SystemExit("Mind. 2 Reihen nötig (--cities, mehrere --csv oder --corpus-index).")

    prior = build_prior_from_series(series)
    path = save_prior(prior, a.out)
    print(f"Pool-Prior gespeichert: {path}")
    print(f"  Häuser: {prior['n_houses']} · Features: {prior['n_features']} · Raum: {prior['space']}")
    print("  Der Dienst nutzt ihn automatisch für Mandanten mit < 60 Tagen Historie (>= 21 Tage).")


if __name__ == "__main__":
    main()
