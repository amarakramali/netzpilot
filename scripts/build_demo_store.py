#!/usr/bin/env python3
"""Befüllt den Service-Store mit Ergebnis-JSONs für ALLE kuratierten echten Korpus-Reihen.

Zweck: Das Cockpit braucht Ergebnis-JSONs (Upload / „Latest laden"). Bisher existierten nur
die Läufe eines Mandanten. Dieses Skript erzeugt je kuratierter ECHTER Reihe (corpus_index.json,
T28b-kuratiert: include_in_benchmark & network_representative & nicht mape_meaningless) einen
leakage-sicheren run_forecast und speichert ihn wie der Dienst selbst (ForecastStore →
data_cache/service_store/<Mandant>/<Datum>.json + latest.json).

Ehrlichkeit:
- Nur Defaults, KEINE erfundenen Ratings/steuVE/€-Preise je Reihe.
- forecast_store_path je Reihe gesetzt → Track-Record-Kette startet (1 pending, 0 realisiert — ehrlich).
- FR-Reihen laufen mit region="NW": deutsche Feiertage als Näherung (betrifft nur die
  deterministische Feiertags-Basis, keine berichtete Kennzahl).

Resumable: existiert <Mandant>/latest.json bereits, wird die Reihe übersprungen (--force überschreibt).
Aufruf:  python scripts/build_demo_store.py [--start 0] [--limit 999] [--force]
"""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.service.runner import run_forecast
from netzpilot.service.store import ForecastStore

REGION_BY_PREFIX = [
    ("hilden", "NW"), ("herne", "NW"), ("evdb", "NI"), ("neuruppin", "BB"),
    ("bitterfeld", "ST"), ("waren", "MV"), ("ten_", "TH"), ("neusw", "MV"),
    ("passau", "BY"), ("enedis", "NW"), ("eco2mix", "NW"),  # FR: DE-Feiertage als Näherung
]


def region_for(key: str) -> str:
    for p, r in REGION_BY_PREFIX:
        if key.startswith(p):
            return r
    return "NW"


def curated_entries(index_path: str = "data_cache/real/corpus_index.json") -> list[dict]:
    j = json.load(open(index_path, encoding="utf-8"))
    items = j.get("series") or j.get("entries") or []
    return [it for it in items
            if it.get("include_in_benchmark")
            and not it.get("mape_meaningless")
            and it.get("network_representative", True)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=999)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    entries = curated_entries()
    store = ForecastStore()
    todo = entries[args.start:args.start + args.limit]
    print(f"kuratierte Reihen: {len(entries)} | dieser Lauf: {len(todo)} (ab Index {args.start})")
    n_ok = n_skip = n_err = 0
    for i, it in enumerate(todo, start=args.start):
        key, name = it["key"], it["name"]
        udir = store._utility_dir(name)
        if not args.force and os.path.exists(os.path.join(udir, "latest.json")):
            n_skip += 1
            print(f"[{i:2d}] SKIP  {key} (latest.json existiert)")
            continue
        t0 = time.time()
        try:
            out = run_forecast(
                it["csv"], utility=name, region=region_for(key),
                unit=it.get("unit", "MW"), ts_col=it.get("ts"), load_col=it.get("col"),
                drift_monitoring=True,
                forecast_store_path=f"data_cache/forecast_store/{key}.jsonl",
            )
            fc = out.get("forecast") or []
            assert len(fc) == 24 and all(
                f["p10"] <= f["p50"] <= f["p90"] for f in fc), "Quantil-Ordnung/24h verletzt"
            store.save(name, out)
            n_ok += 1
            peak = max(f["p50"] for f in fc)
            print(f"[{i:2d}] OK    {key:34s} {out.get('forecast_date')} peak={peak:8.2f} MW  {time.time()-t0:5.1f}s")
        except Exception as e:  # noqa: BLE001 — Batch soll weiterlaufen, Fehler ehrlich listen
            n_err += 1
            print(f"[{i:2d}] FEHLER {key}: {type(e).__name__}: {e}")
    print(f"fertig: ok={n_ok} skip={n_skip} fehler={n_err}")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
