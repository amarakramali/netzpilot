from netzpilot.data.smard import load_local_json
from netzpilot.data.integrity import series_integrity_report, validate_series
from netzpilot.data.openmeteo import align_to_index
from netzpilot.data.residual import residual_load
from netzpilot.features.build import to_daily
import pandas as pd

def test_bundled_continuity():
    s = load_local_json("prognose_engine_v1/data/wk*.json")
    assert len(s) % 24 == 0 and len(s) >= 168 * 12
    load2d, days = to_daily(s)   # wirft, wenn Stundenraster nicht lueckenlos
    assert load2d.shape[1] == 24

def test_quarterhour_dst_spring_is_valid():
    idx = pd.date_range("2024-03-30 23:00", "2024-04-01 21:45", freq="15min", tz="UTC")
    s = pd.Series(range(len(idx)), index=idx)
    report = validate_series(s, "quarterhour")
    assert report["dst_transition_days"]["2024-03-31"] == 92
    assert report["unexpected_local_day_counts"] == {}

def test_quarterhour_gap_is_reported():
    idx = pd.date_range("2024-01-01 00:00", periods=8, freq="15min", tz="UTC").delete(3)
    s = pd.Series(range(len(idx)), index=idx)
    report = series_integrity_report(s, "quarterhour")
    assert report["gap_count"] == 1
    assert report["gaps"][0]["missing_slots"] == 1

def test_weather_alignment_to_quarterhour_index():
    weather_idx = pd.date_range("2024-01-01 00:00", periods=3, freq="1h", tz="UTC")
    weather = pd.DataFrame({"temperature_2m": [1.0, 2.0, 3.0]}, index=weather_idx)
    load_idx = pd.date_range("2024-01-01 00:00", periods=8, freq="15min", tz="UTC")
    aligned = align_to_index(weather, load_idx)
    assert list(aligned["temperature_2m"].iloc[:4]) == [1.0, 1.0, 1.0, 1.0]
    assert aligned["temperature_2m"].iloc[4] == 2.0

def test_residual_load_aligns_generation_to_hourly_mean():
    load_idx = pd.date_range("2024-01-01 00:00", periods=2, freq="1h", tz="UTC")
    qh_idx = pd.date_range("2024-01-01 00:00", periods=8, freq="15min", tz="UTC")
    load = pd.Series([100.0, 120.0], index=load_idx)
    pv = pd.Series([4.0] * 8, index=qh_idx)
    wind_on = pd.Series([3.0] * 8, index=qh_idx)
    wind_off = pd.Series([2.0] * 8, index=qh_idx)
    residual = residual_load(load, pv, wind_on, wind_off)
    assert list(residual) == [91.0, 111.0]
