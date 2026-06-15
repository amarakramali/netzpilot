import numpy as np, pandas as pd
from netzpilot.features.build import build_small_load_features, to_daily_local, frame_to_daily_local

def _utc_hours(start, end):
    return pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")

def test_to_daily_local_drops_dst_day():
    # Spring-DST in Europe/Berlin: 2024-03-31 hat nur 23 Lokalstunden -> muss verworfen werden
    idx = _utc_hours("2024-03-29 00:00", "2024-04-03 00:00")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    load2d, days, good = to_daily_local(s)
    assert load2d.shape[1] == 24
    assert pd.Timestamp("2024-03-31") not in pd.DatetimeIndex(days)   # DST-Tag verworfen
    assert pd.Timestamp("2024-03-30") in pd.DatetimeIndex(days)       # Nachbartage bleiben

def test_frame_to_daily_local_alignment():
    idx = _utc_hours("2024-01-01 00:00", "2024-01-05 00:00")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    load2d, days, good = to_daily_local(s)
    w = pd.DataFrame({"temperature_2m": np.arange(len(idx), dtype=float),
                      "wind_speed_100m": np.arange(len(idx), dtype=float) * 0.5}, index=idx)
    w2d = frame_to_daily_local(w, good)
    assert w2d.shape == (len(days), 24, 2)

def test_small_load_features_do_not_use_target_load():
    days = pd.date_range("2024-01-01", periods=20, freq="D")
    load2d = np.arange(20 * 24, dtype=float).reshape(20, 24)
    weather2d = np.ones((20, 24, 2), dtype=float)
    d = 15
    x1 = build_small_load_features(load2d, days, d, weather2d, holiday_set=set())
    changed = load2d.copy()
    changed[d:] += 100000.0
    x2 = build_small_load_features(changed, days, d, weather2d, holiday_set=set())
    assert x1.shape[0] == 24
    assert x1.shape[1] > 13
    assert np.allclose(x1, x2)

def test_small_load_features_require_enough_history():
    days = pd.date_range("2024-01-01", periods=20, freq="D")
    load2d = np.arange(20 * 24, dtype=float).reshape(20, 24)
    try:
        build_small_load_features(load2d, days, 13)
    except ValueError as exc:
        assert "14 prior days" in str(exc)
    else:
        raise AssertionError("expected ValueError")
