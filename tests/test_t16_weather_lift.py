# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Checks for T16 second real 2022+ weather-lift artifacts."""
import json
import math
from pathlib import Path

from scripts.run_t16_weather_lift import DSO_SPECS

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "data_cache" / "t16_weather_lift"
_EVAL = _OUT / "dso_weather_eval.jsonl"
_SUMMARY = _OUT / "t16_summary.json"
_DRYAD = _OUT / "dryad_download_attempt.json"


def _records():
    if not _EVAL.exists():
        return []
    with _EVAL.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_t16_dryad_probe_documents_blocker_if_present():
    if not _DRYAD.exists():
        return
    with _DRYAD.open(encoding="utf-8") as f:
        dryad = json.load(f)

    assert dryad["latest_version"]["versionNumber"] == 12
    assert any(item["path"] == "reduced_data.zip" for item in dryad["files"])
    assert dryad["usable_for_t16"] is False
    statuses = {a.get("status_code") for a in dryad["download_attempts"]}
    assert 401 in statuses or 403 in statuses


def test_t16_dso_eval_artifacts_are_leakage_safe_if_present():
    records = _records()
    if not records:
        return
    with _SUMMARY.open(encoding="utf-8") as f:
        summary = json.load(f)

    assert len(records) == summary["records"]
    assert len(records) == len(DSO_SPECS)
    assert summary["fallback_dataset"] == "public T15 DSO network-load CSVs"
    assert all(r["leakage_class"] == "leakage_safe" for r in records)
    assert all("Historical Forecast" in r["weather_source"] for r in records)
    assert all(r["load_period"].startswith("2024-") for r in records)
    assert {r["slug"] for r in records} == {spec["slug"] for spec in DSO_SPECS}

    for rec in records:
        for section in ("no_weather", "weather"):
            for value in rec[section].values():
                if value is not None:
                    assert math.isfinite(float(value))
        sig = rec["significance_weather_vs_no_weather"]
        lo, _ = sig["skill_ci95_%"]
        assert sig["signifikant_5pct"] == (lo > 0.0)
        assert rec["cqr_weather"]["coverage80_%"] >= 0.0
        assert rec["cqr_weather"]["coverage90_%"] >= 0.0

    assert "does not become a significant product moat" in summary["one_sentence"]
