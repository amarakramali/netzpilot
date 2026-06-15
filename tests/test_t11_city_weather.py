# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Checks for T11 city-weather pipeline artifacts."""
import json
import math
from pathlib import Path

from scripts.run_t11_city_weather import PROVENANCE_CAVEAT, city_query_name, summarize_records

_ROOT = Path(__file__).resolve().parents[1]
_JSONL = _ROOT / "data_cache" / "cities_weather_eval.jsonl"
_SUMMARY = _ROOT / "data_cache" / "cities_weather_summary.json"


def _records():
    if not _JSONL.exists():
        return []
    with _JSONL.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_city_query_name_umlauts():
    assert city_query_name("Muenster") == "Münster"
    assert city_query_name("Koeln") == "Köln"
    assert city_query_name("Muelheim") == "Mülheim"
    assert city_query_name("Frankfurt") == "Frankfurt am Main"


def test_t11_summary_matches_jsonl_if_present():
    records = _records()
    if not records:
        return
    with _SUMMARY.open(encoding="utf-8") as f:
        committed = json.load(f)
    reproduced = summarize_records(records)
    assert committed == reproduced
    assert committed["n_cities"] == 50
    assert committed["provenance_caveat"] == PROVENANCE_CAVEAT
    assert "Historical Forecast" in committed["weather_source"]
    assert "no reanalysis" in committed["leakage_check"]


def test_t11_records_are_complete_if_present():
    records = _records()
    if not records:
        return
    assert len(records) == 50
    for rec in records:
        assert rec["provenance_caveat"] == PROVENANCE_CAVEAT
        assert rec["coords"]["source"] == "Open-Meteo Geocoding API"
        assert rec["weather_source"] == "Open-Meteo Historical Forecast API"
        assert rec["weather_cqr"]["coverage80_%"] > 0
        assert rec["weather_cqr"]["coverage90_%"] > 0
        sig = rec["significance_weather_vs_no_weather"]
        lo, hi = sig["skill_ci95_%"]
        assert sig["signifikant_5pct"] == (lo > 0.0)
        assert 0.0 <= sig["P(weather_besser)_%"] <= 100.0
        assert all(math.isfinite(v) for v in [
            rec["weather"]["skill_snv_%"],
            rec["skill_lift_vs_cached_no_weather_pp"],
            rec["skill_lift_same_protocol_pp"],
        ])
