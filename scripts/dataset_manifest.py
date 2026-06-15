# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Manifest der echten DSO-Datensaetze mit EXPLIZIT gepinnten Spalten/Ebenen.

T28 promotes ``data_cache/real/corpus_index.json`` to the machine-readable source of truth.
If that registry is missing, this module falls back to the earlier small benchmark set so old
commands still run.
"""
from __future__ import annotations

import json
import os

REAL = "data_cache/real"
CORPUS_INDEX = f"{REAL}/corpus_index.json"

LEGACY_MANIFEST = [
    {"key": "hilden_netzumsatz", "name": "Stadtwerke Hilden - Netzumsatz",
     "csv": f"{REAL}/Netzumsatz-Lastgang-2025.csv", "ts": "Text", "col": "Reihe1", "unit": "kW"},
    {"key": "evdb_ns", "name": "EVDB - Lastgang NS",
     "csv": f"{REAL}/evdb_lastgang_ns_2024.csv", "ts": None, "col": "Wert", "unit": "kW"},
    {"key": "evdb_ms", "name": "EVDB - Lastgang MS",
     "csv": f"{REAL}/evdb_lastgang_ms_2024.csv", "ts": None, "col": "Wert", "unit": "kW"},
    {"key": "herne_110_10kv", "name": "Stadtwerke Herne - Bezug 110/10 kV",
     "csv": f"{REAL}/herne_bezug_vorgelagerte_ebene_2024.csv", "ts": None, "col": "Load_1", "unit": "kW"},
    {"key": "bitterfeld_ns", "name": "NG Bitterfeld-Wolfen - NS",
     "csv": f"{REAL}/bitterfeld_jhl_ns_2024.csv", "ts": None, "col": "Wert", "unit": "kW"},
    {"key": "bitterfeld_msns", "name": "NG Bitterfeld-Wolfen - MS/NS",
     "csv": f"{REAL}/bitterfeld_jhl_msns_2024.csv", "ts": None, "col": "Wert", "unit": "kW"},
]


def load_manifest(path: str = CORPUS_INDEX):
    if not os.path.exists(path):
        return list(LEGACY_MANIFEST)
    with open(path, encoding="utf-8") as f:
        corpus = json.load(f)
    out = []
    for e in corpus.get("entries", []):
        if not e.get("include_in_benchmark", False):
            continue
        out.append({
            "key": e["key"],
            "name": e["name"],
            "csv": e["path"],
            "ts": e.get("ts"),
            "col": e["col"],
            "unit": e.get("unit", "MW"),
            "level": e.get("level"),
            "country": e.get("country", "DE"),
            "network_kind": e.get("network_kind", "dso_real"),
            "signed": e.get("signed", False),
            "mape_meaningless": e.get("mape_meaningless", False),
            "mape_note": e.get("mape_note"),
            "value_hash": e.get("value_hash"),
            "independent_network": e.get("independent_network", True),
            "network_representative": e.get("network_representative"),
            "redundant_of": e.get("redundant_of"),
            "corr": e.get("corr"),
            "source_url": e.get("source_url"),
        })
    return out


def load_corpus_summary(path: str = CORPUS_INDEX):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        corpus = json.load(f)
    keys = [
        "generated_utc",
        "n_entries_total",
        "n_series",
        "n_independent_networks",
        "n_correlation_redundant",
        "n_pool_series",
        "country_counts",
        "network_kind_counts",
        "n_signifikant_vs_snaive_5pct",
        "independence_method",
        "pool_correlation_policy",
    ]
    return {k: corpus[k] for k in keys if k in corpus}


MANIFEST = load_manifest()
CORPUS_SUMMARY = load_corpus_summary()
