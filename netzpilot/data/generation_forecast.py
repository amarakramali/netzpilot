# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Physical renewable-generation proxies and leakage-safe bias correction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def quarterhour_to_hourly_mean(series: pd.Series) -> pd.Series:
    s = series.sort_index()
    s.index = pd.to_datetime(s.index, utc=True)
    return s.resample("1h", label="left", closed="left").mean()


def hourly_generation_frame(pv_qh: pd.Series, wind_on_qh: pd.Series, wind_off_qh: pd.Series) -> pd.DataFrame:
    """Convert SMARD quarter-hour generation series to hourly MW components."""
    out = pd.DataFrame({
        "pv_mw": quarterhour_to_hourly_mean(pv_qh),
        "wind_onshore_mw": quarterhour_to_hourly_mean(wind_on_qh),
        "wind_offshore_mw": quarterhour_to_hourly_mean(wind_off_qh),
    }).dropna()
    out["generation_mw"] = out.sum(axis=1)
    return out


def _pv_proxy(weather: pd.DataFrame, latitude: float, longitude: float) -> np.ndarray:
    w = weather.sort_index()
    ghi = np.maximum(w["shortwave_radiation"].to_numpy(float), 0.0)
    try:
        import pvlib
    except ModuleNotFoundError:
        temp = w.get("temperature_2m", pd.Series(15.0, index=w.index)).to_numpy(float)
        temp_factor = np.clip(1.0 - 0.003 * (temp - 25.0), 0.75, 1.1)
        return np.clip((ghi / 1000.0) * temp_factor, 0.0, None)

    times = pd.DatetimeIndex(w.index)
    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    direct = np.maximum(w.get("direct_radiation", pd.Series(0.0, index=w.index)).to_numpy(float), 0.0)
    zenith = solpos["apparent_zenith"].to_numpy(float)
    cosz = np.maximum(np.cos(np.deg2rad(zenith)), 0.0)
    dni = np.divide(direct, np.maximum(cosz, 0.08), out=np.zeros_like(direct), where=cosz > 0)
    dhi = np.maximum(ghi - dni * cosz, 0.0)
    poa_raw = pvlib.irradiance.get_total_irradiance(
        surface_tilt=30,
        surface_azimuth=180,
        solar_zenith=zenith,
        solar_azimuth=solpos["azimuth"].to_numpy(float),
        dni=dni,
        ghi=ghi,
        dhi=dhi,
    )["poa_global"]
    poa = np.asarray(poa_raw, dtype=float)
    temp = w.get("temperature_2m", pd.Series(15.0, index=w.index)).to_numpy(float)
    wind = w.get("wind_speed_10m", pd.Series(1.0, index=w.index)).to_numpy(float)
    cell_temp = pvlib.temperature.faiman(np.maximum(poa, 0.0), temp, wind_speed=np.maximum(wind, 0.1))
    dc = pvlib.pvsystem.pvwatts_dc(np.maximum(poa, 0.0), cell_temp, pdc0=1.0, gamma_pdc=-0.003)
    return np.clip(np.asarray(dc, dtype=float), 0.0, None)


def _wind_proxy(speed: np.ndarray, offshore: bool = False) -> np.ndarray:
    speeds = np.asarray(speed, dtype=float)
    if offshore:
        curve_speeds = np.array([0, 3, 5, 8, 11, 14, 25, 26], dtype=float)
        curve_power = np.array([0, 0, 0.08, 0.35, 0.75, 1.0, 1.0, 0], dtype=float)
    else:
        curve_speeds = np.array([0, 3, 5, 8, 12, 25, 26], dtype=float)
        curve_power = np.array([0, 0, 0.10, 0.45, 1.0, 1.0, 0], dtype=float)
    try:
        from windpowerlib.power_output import power_curve
    except ModuleNotFoundError:
        return np.clip(np.interp(speeds, curve_speeds, curve_power), 0.0, None)

    out = power_curve(speeds, curve_speeds, curve_power)
    return np.clip(np.asarray(out, dtype=float), 0.0, None)


def physical_generation_proxies(
    weather: pd.DataFrame,
    latitude: float = 51.16,
    longitude: float = 10.45,
) -> pd.DataFrame:
    """Build dimensionless PV/wind physical proxies from forecast weather."""
    w = weather.sort_index()
    out = pd.DataFrame(index=w.index)
    out["pv_proxy"] = _pv_proxy(w, latitude, longitude)
    speed100 = w["wind_speed_100m"].to_numpy(float)
    out["wind_onshore_proxy"] = _wind_proxy(speed100, offshore=False)
    out["wind_offshore_proxy"] = _wind_proxy(speed100 * 1.08, offshore=True)
    return out


def generation_from_weather_csv(
    weather_csv: str,
    *,
    pv_capacity_mw: float = 0.0,
    wind_capacity_mw: float = 0.0,
    latitude: float = 51.16,
    longitude: float = 10.45,
    ts_col: str | None = None,
) -> pd.Series:
    """Stündliche Erzeugungs-Reihe (MW) aus einer Wetter-CSV + installierten Kapazitäten.

    Brücke zwischen der physikalischen Proxy-Engine (physical_generation_proxies) und dem
    Residuallast-Pfad des Dienstes: ein Stadtwerk nennt seine installierte PV-/Wind-Leistung,
    NetzPilot rechnet aus dem Wetter-Forecast die erwartete Einspeisung. Der Proxy ist
    dimensionslos (0..~1 ≈ Auslastung), Skalierung über die genannten Kapazitäten.

    Wetter-CSV-Spalten (wie Open-Meteo, siehe openmeteo.DEFAULT_VARS): mind. shortwave_radiation
    + wind_speed_100m; temperature_2m/direct_radiation/wind_speed_10m verbessern den PV-Proxy.
    Zeitspalte: ts_col oder die erste als datetime parsebare Spalte. Gibt UTC-stündliche MW zurück.

    HINWEIS: Reine Markt-/Physik-Näherung. Belastbar wird die Erzeugung erst mit der realen
    Einspeise-Zeitreihe des Stadtwerks (Pilot). Live-Wetter-Beschaffung läuft über data.openmeteo
    (Internet, Host) — diese Funktion arbeitet auf einer bereits vorliegenden Wetter-CSV.
    """
    df = pd.read_csv(weather_csv)
    if ts_col is None:
        for c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
            if parsed.notna().mean() > 0.7:
                ts_col = c
                break
        if ts_col is None:
            raise ValueError("Keine Zeitspalte in der Wetter-CSV erkannt — ts_col angeben.")
    idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    w = df.drop(columns=[ts_col]).set_index(idx).sort_index()
    w = w[~w.index.isna()]
    if "shortwave_radiation" not in w or "wind_speed_100m" not in w:
        raise ValueError("Wetter-CSV braucht mind. shortwave_radiation und wind_speed_100m.")
    proxies = physical_generation_proxies(w, latitude=latitude, longitude=longitude)
    gen = (pv_capacity_mw * proxies["pv_proxy"]
           + wind_capacity_mw * proxies["wind_onshore_proxy"])
    gen.name = "generation_mw"
    return gen.dropna()


def generation_feature_matrix(proxy2d: np.ndarray, days: pd.DatetimeIndex, d: int) -> np.ndarray:
    """Hourly features for generation-bias correction on day d."""
    rows = []
    dow = days[d].dayofweek
    doy = days[d].dayofyear
    for h in range(24):
        pv, won, woff = proxy2d[d, h]
        rows.append([
            1.0,
            pv, pv ** 2,
            won, won ** 2,
            woff, woff ** 2,
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 366), np.cos(2 * np.pi * doy / 366),
        ])
    return np.asarray(rows, dtype=float)


def rolling_generation_bias_forecast(
    gen2d: np.ndarray,
    proxy2d: np.ndarray,
    days: pd.DatetimeIndex,
    first: int = 8,
    n_test: int = 28,
    retrain_every: int = 7,
) -> dict[str, np.ndarray]:
    """Forecast PV/wind generation using physical proxies plus rolling Ridge bias correction."""
    from sklearn.linear_model import Ridge

    ND = len(gen2d)
    test_days = list(range(ND - n_test, ND))
    pred_rows, actual_rows = [], []
    models = None
    last_fit = None
    for d in test_days:
        if models is None or (d - last_fit) >= retrain_every:
            Xtr = np.vstack([generation_feature_matrix(proxy2d, days, t) for t in range(first, d)])
            ytr = np.vstack([gen2d[t] for t in range(first, d)]).reshape(-1, gen2d.shape[2])
            models = [Ridge(alpha=1.0).fit(Xtr[:, 1:], ytr[:, j]) for j in range(gen2d.shape[2])]
            last_fit = d
        Xd = generation_feature_matrix(proxy2d, days, d)
        pred = np.column_stack([m.predict(Xd[:, 1:]) for m in models])
        pred_rows.append(np.clip(pred, 0.0, None))
        actual_rows.append(gen2d[d])
    pred_arr = np.vstack(pred_rows)
    actual_arr = np.vstack(actual_rows)
    return {
        "pred_components": pred_arr,
        "actual_components": actual_arr,
        "pred_total": pred_arr.sum(axis=1),
        "actual_total": actual_arr.sum(axis=1),
    }
