import numpy as np
import pandas as pd

from netzpilot.data.generation_forecast import physical_generation_proxies, rolling_generation_bias_forecast


def test_physical_generation_proxies_are_non_negative():
    idx = pd.date_range("2024-06-01", periods=24, freq="1h", tz="UTC")
    weather = pd.DataFrame({
        "temperature_2m": [20.0] * 24,
        "shortwave_radiation": [0.0] * 6 + [500.0] * 12 + [0.0] * 6,
        "direct_radiation": [0.0] * 6 + [350.0] * 12 + [0.0] * 6,
        "cloud_cover": [20.0] * 24,
        "wind_speed_10m": [4.0] * 24,
        "wind_speed_100m": [9.0] * 24,
    }, index=idx)
    proxies = physical_generation_proxies(weather)
    assert list(proxies.columns) == ["pv_proxy", "wind_onshore_proxy", "wind_offshore_proxy"]
    assert np.isfinite(proxies.to_numpy()).all()
    assert (proxies >= 0.0).all().all()


def test_rolling_generation_bias_forecast_shapes():
    days = pd.date_range("2024-01-01", periods=20, freq="D")
    proxy2d = np.ones((20, 24, 3), dtype=float)
    gen2d = np.ones((20, 24, 3), dtype=float) * 5.0
    out = rolling_generation_bias_forecast(gen2d, proxy2d, days, first=8, n_test=3)
    assert out["pred_components"].shape == (72, 3)
    assert out["actual_total"].shape == (72,)
