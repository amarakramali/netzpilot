# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Checks for T13 leakage-safe per-station weather-lift artifacts."""
import json
import math
from pathlib import Path

from scripts.run_t13_weather_lift import EXACT_STATION_MATCHES, UNRESOLVED_STATION_IDS

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "data_cache" / "t13_weather_lift"
_EVAL = _OUT / "heapo_per_station_eval.jsonl"
_SUMMARY = _OUT / "t13_summary.json"
_MAPPING = _OUT / "station_mapping.json"
_DRYAD = _OUT / "dryad_download_attempt.json"


def _records():
    if not _EVAL.exists():
        return []
    with _EVAL.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_t13_station_mapping_documents_exact_and_unresolved_ids():
    if not _MAPPING.exists():
        return
    with _MAPPING.open(encoding="utf-8") as f:
        mapping = json.load(f)

    assert set(mapping["exact_station_matches"]) == set(EXACT_STATION_MATCHES)
    assert set(mapping["unresolved_station_ids"]) == set(UNRESOLVED_STATION_IDS)
    assert mapping["coverage"]["exact_station_household_share_%"] >= 90.0
    assert "Open-Meteo Historical Forecast" in mapping["feature_policy"]
    for station in mapping["exact_station_matches"].values():
        assert "max_abs_diff=0.0" in station["match_evidence"]


def test_t13_eval_artifacts_are_leakage_safe_if_present():
    records = _records()
    if not records:
        return
    with _SUMMARY.open(encoding="utf-8") as f:
        summary = json.load(f)

    assert len(records) == summary["heapo_records"]
    assert len(records) >= 30
    assert {r["target"] for r in records} == {"heatpump", "total"}
    assert {"individual", "cluster_4", "feeder_8"}.issubset({r["aggregation"] for r in records})
    assert all(r["leakage_class"] == "leakage_safe" for r in records)
    assert all("Historical Forecast" in r["weather_source"] for r in records)
    assert all(r["weather_id"] in EXACT_STATION_MATCHES for r in records)
    assert not any(r["weather_id"] in UNRESOLVED_STATION_IDS for r in records)

    for rec in records:
        for section in ("no_weather", "station_weather", "zurich_proxy_weather"):
            for value in rec[section].values():
                if value is not None:
                    assert math.isfinite(float(value))
        sig = rec["significance_weather_vs_no_weather"]
        lo, _ = sig["skill_ci95_%"]
        assert sig["signifikant_5pct"] == (lo > 0.0)
        assert rec["cqr_station_weather"]["coverage80_%"] >= 0.0
        assert rec["cqr_station_weather"]["coverage90_%"] >= 0.0

    heat = summary["by_target"]["heatpump"]
    assert heat["median_lift_pp"] > 0.0
    assert heat["sign_test_p"] > 0.05
    assert "does not become a significant product moat" in summary["one_sentence"]


def test_t13_dryad_probe_records_download_blocker_if_present():
    if not _DRYAD.exists():
        return
    with _DRYAD.open(encoding="utf-8") as f:
        dryad = json.load(f)

    assert dryad["files_api"]["status_code"] == 200
    assert any(f["path"] == "reduced_data.zip" for f in dryad["files"])
    assert dryad["usable_for_t13_second_dataset"] is False
    statuses = {a.get("status_code") for a in dryad["download_attempts"]}
    assert 401 in statuses or 403 in statuses
