"""Checks for T12 leakage-safe weather-lift artifacts."""
import json
import math
from pathlib import Path

from scripts.run_t12_weather_lift import FORECAST_START, HEAPO_ZIP, series_to_daily_local

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "data_cache" / "t12_weather_lift"
_HEAPO = _OUT / "heapo_eval.jsonl"
_SUMMARY = _OUT / "t12_summary.json"
_T10 = _OUT / "t10_weather_reverify.json"


def _records():
    if not _HEAPO.exists():
        return []
    with _HEAPO.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_series_to_daily_local_enforces_forecast_window():
    import pandas as pd

    idx = pd.date_range("2022-06-25", "2022-07-10", freq="1h", tz="UTC", inclusive="left")
    s = pd.Series(1.0, index=idx)
    _, days, _ = series_to_daily_local(s, keep_days=100)
    assert pd.Timestamp(days.min()).date() >= FORECAST_START.date()


def test_t12_artifacts_are_complete_if_present():
    records = _records()
    if not records:
        return
    with _SUMMARY.open(encoding="utf-8") as f:
        summary = json.load(f)
    with _T10.open(encoding="utf-8") as f:
        t10 = json.load(f)

    assert len(records) == summary["leakage_safe_records"] == 26
    assert {r["target"] for r in records} == {"heatpump", "total"}
    assert all(r["leakage_class"] == "leakage_safe" for r in records)
    assert all("Historical Forecast" in r["weather_source"] for r in records)
    assert t10["all_checked_variables_identical_to_archive"] is True
    assert summary["upper_bound_t10"]["leakage_class"] == "perfect_foresight_upper_bound"
    assert "perfect-foresight" in summary["one_sentence"]

    for rec in records:
        for section in ("no_weather", "weather"):
            for key, value in rec[section].items():
                if value is not None:
                    assert math.isfinite(float(value)), (rec["label"], section, key, value)
        sig = rec["significance_weather_vs_no_weather"]
        lo, _ = sig["skill_ci95_%"]
        assert sig["signifikant_5pct"] == (lo > 0.0)


def test_heapo_zip_is_cached_if_artifacts_exist():
    if _HEAPO.exists():
        assert HEAPO_ZIP.exists()
        assert HEAPO_ZIP.stat().st_size > 400_000_000
