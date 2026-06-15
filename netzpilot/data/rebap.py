"""Loader for public reBAP quarter-hour prices from netztransparenz.de."""
from __future__ import annotations

import csv
import math
from pathlib import Path


def _parse_float_de(raw) -> float:
    if raw is None:
        return float("nan")
    s = str(raw).strip().replace("\ufeff", "")
    if not s:
        return float("nan")
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    return float(s)


def _detect_separator(line: str) -> str:
    counts = {sep: line.count(sep) for sep in (";", "\t", ",")}
    return max(counts, key=counts.get)


def _read_text(path: str | Path) -> str:
    p = Path(path)
    data = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _header_index(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        low = line.lower()
        if "rebap" in low and ("zeit" in low or "datum" in low):
            return i
    raise ValueError("Keine reBAP-Kopfzeile gefunden.")


def _value_column(fieldnames: list[str]) -> str:
    normalized = {name.lower().strip(): name for name in fieldnames}
    for key in ("rebap_eur_mwh", "rebap eur/mwh", "rebap €/mwh", "rebap"):
        if key in normalized:
            return normalized[key]
    for name in fieldnames:
        low = name.lower()
        if "rebap" in low and "unterdeckt" in low:
            return name
    for name in fieldnames:
        if "rebap" in name.lower():
            return name
    raise ValueError("Keine reBAP-Wertspalte gefunden.")


def load_rebap(path: str | Path) -> list[float]:
    """Return finite reBAP prices in EUR/MWh from normalized or official CSV exports.

    Supported inputs:
    - normalized T19 cache: ``Zeit;reBAP_EUR_MWh``
    - official netztransparenz export:
      ``Datum;Zeitzone;von;bis;Einheit;reBAP unterdeckt;reBAP ueberdeckt``
    """
    text = _read_text(path)
    lines = [line for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    idx = _header_index(lines)
    sep = _detect_separator(lines[idx])
    reader = csv.DictReader(lines[idx:], delimiter=sep)
    if not reader.fieldnames:
        raise ValueError("reBAP-CSV enthaelt keine Spaltennamen.")
    col = _value_column([name.strip().replace("\ufeff", "") for name in reader.fieldnames])
    vals: list[float] = []
    for row in reader:
        v = _parse_float_de(row.get(col))
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        raise ValueError("Keine endlichen reBAP-Werte gefunden.")
    return vals
