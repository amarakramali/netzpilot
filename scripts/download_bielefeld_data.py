import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def main():
    base_url = "https://www.bielefelder-netz.de/stromnetz/netzzugang/energiestrukturdaten"
    download_dir = os.path.join("netzpilot", "data", "bielefeld_netz")
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Scraping URL: {base_url}")
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching page: {e}")
        # Trying a broader page if the specific one fails
        base_url = "https://www.bielefelder-netz.de/"
        print(f"Trying base URL: {base_url} and looking for CSVs...")
        try:
            response = requests.get(base_url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Error fetching base page: {e}")
            return

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)
    
    csv_links = []
    for link in links:
        href = link['href']
        if href.endswith('.csv') or href.endswith('.xlsx') or 'csv' in href.lower():
            full_url = urljoin(base_url, href)
            if full_url not in csv_links:
                csv_links.append(full_url)
                
    if not csv_links:
        print("No CSV links found directly on this page.")
        # Create a mock CSV dataset for the prototype if we can't scrape it immediately due to complex navigation
        print("Generating mock Bielefeld 15-min proxy data for the prototype so development can continue...")
        generate_mock_data(download_dir)
        return
        
    for url in csv_links:
        filename = url.split('/')[-1]
        filepath = os.path.join(download_dir, filename)
        print(f"Downloading {filename}...")
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(r.content)
            print(f"Saved to {filepath}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

def generate_mock_data(download_dir):
    import pandas as pd
    import numpy as np
    
    # Generate 1 year of 15-min data
    dates = pd.date_range(start="2024-01-01", end="2024-12-31 23:45:00", freq="15min")
    
    # Mock Lastgang (Load) - daily pattern
    hours = dates.hour + dates.minute / 60
    base_load = 50 + 20 * np.sin((hours - 6) * np.pi / 12)  # Peak around 12-18
    noise = np.random.normal(0, 2, len(dates))
    load = base_load + noise
    
    # Mock Einspeisung (PV)
    pv = np.zeros(len(dates))
    daylight = (hours > 7) & (hours < 19)
    pv[daylight] = 30 * np.sin((hours[daylight] - 7) * np.pi / 12) + np.random.normal(0, 1, np.sum(daylight))
    pv = np.maximum(pv, 0)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "load_mw": load,
        "pv_feedin_mw": pv
    })
    
    filepath = os.path.join(download_dir, "Bielefelder_Netz_Lastgang_2024_mock.csv")
    df.to_csv(filepath, index=False)
    print(f"Mock data saved to {filepath}")

if __name__ == "__main__":
    main()
