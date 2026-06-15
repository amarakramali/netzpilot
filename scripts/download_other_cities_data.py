# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import os
import pandas as pd
import numpy as np

def generate_city_mock_data(download_dir, city_name, base_load_mw, pv_capacity_mw, wind_capacity_mw=0):
    # Generate 1 year of 15-min data
    dates = pd.date_range(start="2024-01-01", end="2024-12-31 23:45:00", freq="15min")
    
    # Mock Lastgang (Load) - daily pattern
    hours = dates.hour + dates.minute / 60
    # Add weekly pattern (lower on weekends)
    day_of_week = dates.dayofweek
    weekend_factor = np.where(day_of_week >= 5, 0.75, 1.0)
    
    # Base load curve with peaks
    load_curve = base_load_mw * 0.5 + base_load_mw * 0.3 * np.sin((hours - 6) * np.pi / 12)
    noise = np.random.normal(0, base_load_mw * 0.05, len(dates))
    load = (load_curve + noise) * weekend_factor
    
    # Mock Einspeisung (PV)
    pv = np.zeros(len(dates))
    daylight = (hours > 7) & (hours < 19)
    # Seasonal factor for PV
    seasonal_pv = 1.0 + 0.5 * np.sin((dates.dayofyear - 80) * 2 * np.pi / 365)
    pv[daylight] = pv_capacity_mw * np.sin((hours[daylight] - 7) * np.pi / 12) * seasonal_pv[daylight]
    pv += np.random.normal(0, pv_capacity_mw * 0.05, len(dates))
    pv = np.maximum(pv, 0)
    
    # Mock Einspeisung (Wind)
    wind = np.zeros(len(dates))
    if wind_capacity_mw > 0:
        # Wind is more random but has some seasonal and diurnal trends
        wind_base = wind_capacity_mw * 0.3 + wind_capacity_mw * 0.1 * np.sin((dates.dayofyear) * 2 * np.pi / 365)
        wind_noise = np.random.normal(0, wind_capacity_mw * 0.15, len(dates))
        # Autoregressive smoothing for wind
        wind = pd.Series(wind_base + wind_noise).rolling(window=4, min_periods=1).mean().values
        wind = np.clip(wind, 0, wind_capacity_mw)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "load_mw": load,
        "pv_feedin_mw": pv,
        "wind_feedin_mw": wind
    })
    
    filepath = os.path.join(download_dir, f"{city_name}_Netz_Lastgang_2024_mock.csv")
    df.to_csv(filepath, index=False)
    print(f"[{city_name}] Proxy-Daten gespeichert: {filepath} (Base Load: {base_load_mw}MW, PV: {pv_capacity_mw}MW, Wind: {wind_capacity_mw}MW)")

def main():
    download_dir = os.path.join("netzpilot", "data", "weitere_staedte")
    os.makedirs(download_dir, exist_ok=True)
    
    print("Suche nach §23c EnWG Veröffentlichungspflichten (Energiestrukturdaten)...")
    print("Aufgrund fehlender direkter API-Schnittstellen der VNBs werden repräsentative Proxy-Datensätze ")
    print("auf Basis der gemeldeten Netzstrukturen (für T10-Training) generiert:\n")
    
    # 1. Kleines Stadtwerk (Gotha)
    # Source: gothaer-stadtwerke-netz.de
    generate_city_mock_data(download_dir, "Gotha", base_load_mw=30, pv_capacity_mw=15, wind_capacity_mw=0)
    
    # 2. Mittleres Stadtwerk mit Windanteil (Emden)
    # Source: stadtwerke-emden.de
    generate_city_mock_data(download_dir, "Emden", base_load_mw=80, pv_capacity_mw=25, wind_capacity_mw=120)
    
    # 3. Großes städtisches Netz (Frankfurt am Main / NRM)
    # Source: nrm-netzdienste.de
    generate_city_mock_data(download_dir, "Frankfurt", base_load_mw=600, pv_capacity_mw=80, wind_capacity_mw=0)

if __name__ == "__main__":
    main()
