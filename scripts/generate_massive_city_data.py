# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import os
import pandas as pd
import numpy as np

def generate_city_mock_data(download_dir, city_name, base_load_mw, pv_capacity_mw, wind_capacity_mw=0, year=2024):
    dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:45:00", freq="15min")
    
    hours = dates.hour + dates.minute / 60
    day_of_week = dates.dayofweek
    weekend_factor = np.where(day_of_week >= 5, 0.75, 1.0)
    
    # Adding some random variations per city to make them distinct
    city_random_shift = np.random.uniform(-1, 1)
    
    load_curve = base_load_mw * 0.5 + base_load_mw * 0.3 * np.sin((hours - 6 + city_random_shift) * np.pi / 12)
    noise = np.random.normal(0, base_load_mw * 0.05, len(dates))
    load = (load_curve + noise) * weekend_factor
    
    pv = np.zeros(len(dates))
    daylight = (hours > 7) & (hours < 19)
    seasonal_pv = 1.0 + 0.5 * np.sin((dates.dayofyear - 80) * 2 * np.pi / 365)
    pv[daylight] = pv_capacity_mw * np.sin((hours[daylight] - 7) * np.pi / 12) * seasonal_pv[daylight]
    pv += np.random.normal(0, pv_capacity_mw * 0.05, len(dates))
    pv = np.maximum(pv, 0)
    
    wind = np.zeros(len(dates))
    if wind_capacity_mw > 0:
        wind_base = wind_capacity_mw * 0.3 + wind_capacity_mw * 0.1 * np.sin((dates.dayofyear) * 2 * np.pi / 365)
        wind_noise = np.random.normal(0, wind_capacity_mw * 0.15, len(dates))
        wind_series = pd.Series(wind_base + wind_noise).rolling(window=4, min_periods=1).mean().values
        wind = np.clip(wind_series, 0, wind_capacity_mw)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "load_mw": np.round(load, 2),
        "pv_feedin_mw": np.round(pv, 2),
        "wind_feedin_mw": np.round(wind, 2)
    })
    
    filepath = os.path.join(download_dir, f"{city_name}_Netz_Lastgang_{year}.csv")
    df.to_csv(filepath, index=False)

def main():
    download_dir = os.path.join("netzpilot", "data", "training_cities")
    os.makedirs(download_dir, exist_ok=True)
    
    # 50 Cities: format is (Name, BaseLoad MW, PV MW, Wind MW)
    cities = [
        ("Berlin", 2500, 150, 50),
        ("Hamburg", 1500, 100, 300),
        ("Muenchen", 1400, 200, 20),
        ("Koeln", 1000, 80, 40),
        ("Frankfurt", 800, 60, 10),
        ("Stuttgart", 700, 100, 20),
        ("Duesseldorf", 650, 50, 30),
        ("Leipzig", 600, 80, 100),
        ("Dortmund", 580, 50, 40),
        ("Essen", 550, 40, 20),
        ("Bremen", 500, 60, 250),
        ("Dresden", 500, 70, 40),
        ("Hannover", 500, 60, 80),
        ("Nuernberg", 450, 80, 30),
        ("Duisburg", 400, 30, 20),
        ("Bochum", 350, 30, 20),
        ("Wuppertal", 350, 40, 30),
        ("Bielefeld", 320, 50, 20),
        ("Bonn", 300, 40, 10),
        ("Muenster", 300, 60, 80),
        ("Karlsruhe", 300, 50, 20),
        ("Mannheim", 280, 40, 10),
        ("Augsburg", 280, 60, 20),
        ("Wiesbaden", 270, 30, 10),
        ("Gelsenkirchen", 250, 20, 10),
        ("Moenchengladbach", 250, 30, 20),
        ("Braunschweig", 240, 40, 40),
        ("Chemnitz", 240, 30, 20),
        ("Kiel", 230, 20, 150),
        ("Aachen", 230, 40, 60),
        ("Halle", 220, 40, 50),
        ("Magdeburg", 220, 30, 60),
        ("Freiburg", 210, 80, 10),
        ("Krefeld", 210, 20, 10),
        ("Luebeck", 200, 30, 60),
        ("Oberhausen", 200, 20, 10),
        ("Erfurt", 200, 30, 40),
        ("Mainz", 200, 30, 10),
        ("Rostock", 190, 20, 180),
        ("Kassel", 190, 30, 40),
        ("Hagen", 180, 20, 10),
        ("Hamm", 170, 30, 30),
        ("Saarbruecken", 170, 20, 10),
        ("Muelheim", 160, 20, 10),
        ("Potsdam", 160, 30, 50),
        ("Ludwigshafen", 150, 20, 10),
        ("Oldenburg", 150, 30, 80),
        ("Leverkusen", 150, 20, 10),
        ("Osnabrueck", 150, 40, 60),
        ("Solingen", 150, 20, 10)
    ]
    
    print(f"Generiere Daten für {len(cities)} Städte für das Jahr 2024...")
    for city, load, pv, wind in cities:
        generate_city_mock_data(download_dir, city, load, pv, wind, 2024)
        
    print(f"Erfolgreich {len(cities)} Datensätze im Ordner {download_dir} erstellt!")

if __name__ == "__main__":
    main()
