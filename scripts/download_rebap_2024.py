# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Download and normalize official reBAP quarter-hour prices from netztransparenz.de."""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urljoin

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.data.rebap import load_rebap  # noqa: E402
from netzpilot.eval.economics import rebap_spread_stats  # noqa: E402


PAGE_URL = "https://www.netztransparenz.de/Regelenergie/Ausgleichsenergiepreis/reBAP"
OUT = Path("data_cache/real/rebap_2024.csv")
RAW_OUT = Path("data_cache/real/rebap_2024_official.csv")
META_OUT = Path("data_cache/real/rebap_2024_source.json")


def _extract_download_config(page_html: str) -> dict:
    decoder = json.JSONDecoder()
    marker = "downloadHandlerConfig = "
    pos = 0
    while True:
        start = page_html.find(marker, pos)
        if start < 0:
            break
        obj_start = page_html.find("{", start)
        config, end = decoder.raw_decode(page_html[obj_start:])
        title = config.get("Settings", {}).get("Title", "")
        route = config.get("Settings", {}).get("WebApiRoute")
        if title.lower().startswith("rebap unterdeckt") and route:
            return config
        pos = obj_start + end
    raise RuntimeError("Kein reBAP-Download-Configblock auf der offiziellen Seite gefunden.")


def _encoded_export_url(config: dict, year: int) -> str:
    request = {
        "LocalFrom": f"{year}-01-01",
        "LocalTo": f"{year + 1}-01-01",
        "ResultTimeZone": "cet",
        "Settings": config["Settings"],
    }
    payload = json.dumps(request, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded = base64.b64encode(payload).decode("ascii")
    return urljoin(PAGE_URL, config["CsvDownloadApiRoute"]) + "?request=" + quote(encoded)


def _parse_de_float(raw: str) -> float:
    return float(str(raw).strip().replace(".", "").replace(",", "."))


def _normalize_official_csv(raw_text: str) -> tuple[str, dict]:
    rows = list(csv.DictReader(io.StringIO(raw_text), delimiter=";"))
    out = io.StringIO()
    writer = csv.writer(out, delimiter=";", lineterminator="\n")
    writer.writerow(["Zeit", "reBAP_EUR_MWh"])
    under_vals = []
    over_diffs = 0
    for row in rows:
        date_s = (row.get("\ufeffDatum") or row.get("Datum") or "").strip()
        from_s = (row.get("von") or "").strip()
        zone = (row.get("Zeitzone") or "").strip().upper()
        under_raw = row.get("reBAP unterdeckt")
        over_raw = row.get("reBAP ueberdeckt") or row.get("reBAP überdeckt")
        if not date_s or not from_s or under_raw is None:
            continue
        dt = datetime.strptime(date_s + " " + from_s, "%d.%m.%Y %H:%M")
        offset = "+02:00" if zone == "CEST" else "+01:00"
        under = _parse_de_float(under_raw)
        if over_raw is not None and abs(_parse_de_float(over_raw) - under) > 1e-9:
            over_diffs += 1
        under_vals.append(under)
        writer.writerow([dt.strftime("%Y-%m-%dT%H:%M:%S") + offset, f"{under:.2f}"])
    return out.getvalue(), {"raw_rows": len(rows), "asymmetric_quarters": over_diffs}


def main() -> None:
    year = 2024
    OUT.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    page = session.get(PAGE_URL, timeout=60)
    page.raise_for_status()
    config = _extract_download_config(page.text)
    export_url = _encoded_export_url(config, year)
    resp = session.get(export_url, headers={"Referer": PAGE_URL}, timeout=180)
    resp.raise_for_status()
    raw_text = resp.content.decode("utf-8-sig")
    RAW_OUT.write_text(raw_text, encoding="utf-8")

    normalized, raw_meta = _normalize_official_csv(raw_text)
    OUT.write_text(normalized, encoding="utf-8")

    prices = load_rebap(OUT)
    stats = rebap_spread_stats(prices)
    meta = {
        "source_page": PAGE_URL,
        "export_url": export_url,
        "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "period": f"{year}-01-01/{year + 1}-01-01",
        "unit": "EUR/MWh",
        "output_csv": str(OUT),
        "raw_official_csv": str(RAW_OUT),
        "official_route": config["Settings"].get("WebApiRoute"),
        "raw_rows": raw_meta["raw_rows"],
        "normalized_rows": len(prices),
        "asymmetric_quarters": raw_meta["asymmetric_quarters"],
        "spread_stats": stats,
    }
    META_OUT.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
