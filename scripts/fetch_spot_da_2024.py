"""Hole Day-Ahead-Spotpreise DE-LU 2024 (offizielle SMARD/EPEX-Daten via Energy-Charts API).

Schreibt:
- data_cache/real/spot_da_2024_raw.json  (Rohantwort, Provenienz)
- data_cache/real/spot_da_2024.csv       (QH-aligned an reBAP-2024 Zeitachse, EUR/MWh)
- data_cache/real/spot_da_2024_source.json (reproduzierbare Quelle)

Energy-Charts (Fraunhofer ISE) gibt SMARD/Bundesnetzagentur-Daten via stabiler API zurueck
(License: CC BY 4.0 from Bundesnetzagentur | SMARD.de). Stundenaufloesung Day-Ahead-Auktion;
auf reBAP-QH durch konstante Wiederholung der Stundenpreise (Auktion clearet hourly).
"""
from __future__ import annotations
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

API_URL = "https://api.energy-charts.info/price"
PARAMS = {"bzn": "DE-LU", "start": "2024-01-01", "end": "2024-12-31"}
ROOT = Path(__file__).resolve().parent.parent
REBAP_CSV = ROOT / "data_cache" / "real" / "rebap_2024.csv"
OUT_RAW = ROOT / "data_cache" / "real" / "spot_da_2024_raw.json"
OUT_CSV = ROOT / "data_cache" / "real" / "spot_da_2024.csv"
OUT_PROV = ROOT / "data_cache" / "real" / "spot_da_2024_source.json"


def fetch() -> dict:
    r = requests.get(API_URL, params=PARAMS, timeout=60)
    r.raise_for_status()
    return r.json()


def _hour_floor_utc(ts_iso: str) -> int:
    dt = datetime.fromisoformat(ts_iso).astimezone(timezone.utc)
    return int(dt.replace(minute=0, second=0, microsecond=0).timestamp())


def main() -> None:
    print(f"GET {API_URL} {PARAMS}")
    payload = fetch()
    OUT_RAW.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    secs = payload["unix_seconds"]
    prices = payload["price"]
    unit = payload.get("unit", "")
    license_info = payload.get("license_info", "")
    if unit.replace(" ", "").lower() != "eur/mwh":
        raise SystemExit(f"Unerwartete Einheit der Spot-API: {unit!r}")
    if len(secs) != len(prices):
        raise SystemExit(f"unix_seconds ({len(secs)}) != price ({len(prices)})")
    spot_by_hour = {int(s): float(p) for s, p in zip(secs, prices) if p is not None}
    print(f"  -> {len(spot_by_hour)} Spot-Stundenpreise (erwartet ~8784 fuer Schaltjahr 2024)")

    # reBAP-QH-Achse lesen und Spot-Preis je QH auf Stunden-Bucket joinen.
    rows_out = []
    missing = 0
    with REBAP_CSV.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        if header[:1] != ["Zeit"]:
            raise SystemExit(f"Unerwartete reBAP-Kopfzeile: {header!r}")
        for row in reader:
            ts = row[0]
            key = _hour_floor_utc(ts)
            spot = spot_by_hour.get(key)
            if spot is None:
                missing += 1
                rows_out.append((ts, ""))
            else:
                rows_out.append((ts, f"{spot:.2f}"))

    if missing:
        print(f"  WARN: {missing} QH ohne Spot-Match (Stunde fehlt in API-Response)")
    if not rows_out:
        raise SystemExit("Keine Zeilen zum Schreiben — reBAP-CSV leer?")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Zeit", "spot_DA_EUR_MWh"])
        w.writerows(rows_out)
    print(f"  -> {OUT_CSV} ({len(rows_out)} QH)")

    OUT_PROV.write_text(json.dumps({
        "source_page": "https://api.energy-charts.info/",
        "api_url": API_URL,
        "params": PARAMS,
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "license_info": license_info,
        "upstream_attribution": "Bundesnetzagentur | SMARD.de (CC BY 4.0)",
        "period": "2024-01-01/2024-12-31 inclusive",
        "unit": unit,
        "n_hours_api": len(spot_by_hour),
        "n_qh_aligned": len(rows_out),
        "n_qh_missing_spot": missing,
        "qh_axis_source": str(REBAP_CSV.relative_to(ROOT)),
        "output_csv": str(OUT_CSV.relative_to(ROOT)),
        "raw_json": str(OUT_RAW.relative_to(ROOT)),
        "method": "hourly Day-Ahead Auktion auf QH ausgedehnt via UTC-Stunden-Bucket der reBAP-QH-Achse",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {OUT_PROV}")


if __name__ == "__main__":
    main()
