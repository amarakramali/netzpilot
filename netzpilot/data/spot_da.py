# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Loader fuer Day-Ahead-Spotpreise (DE-LU), QH-aligned an die reBAP-Achse.

Spot-CSV-Format (von scripts/fetch_spot_da_2024.py): ``Zeit;spot_DA_EUR_MWh`` mit
denselben Zeitstempeln wie die normalisierte reBAP-CSV (gleiche QH-Achse).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path


def load_spot_da(path: str | Path) -> list[float]:
    """Lies QH-aligned Spot-Reihe; gibt eine Liste mit float-Werten (NaN fuer Luecken)."""
    p = Path(path)
    out: list[float] = []
    with p.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        if not header or "zeit" not in header[0].strip().lower():
            raise ValueError(f"Unerwartete Spot-CSV-Kopfzeile: {header!r}")
        for row in reader:
            if len(row) < 2 or not row[1].strip():
                out.append(float("nan"))
                continue
            try:
                out.append(float(row[1]))
            except ValueError:
                out.append(float("nan"))
    if not out:
        raise ValueError("Spot-CSV leer.")
    return out


def _load_rebap_raw(path: str | Path) -> list[float]:
    """Lies reBAP-Reihe ohne NaN zu droppen — fuer QH-aligned Pairing mit Spot."""
    p = Path(path)
    out: list[float] = []
    with p.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        if not header or "zeit" not in header[0].strip().lower():
            raise ValueError(f"Unerwartete reBAP-CSV-Kopfzeile: {header!r}")
        for row in reader:
            if len(row) < 2 or not row[1].strip():
                out.append(float("nan"))
                continue
            try:
                out.append(float(row[1]))
            except ValueError:
                out.append(float("nan"))
    return out


def load_rebap_spot_pairs(rebap_csv: str | Path, spot_csv: str | Path) -> tuple[list[float], list[float]]:
    """Lies reBAP + Spot auf gemeinsamer QH-Achse, dropp NaN-Paare. Gibt (rebap, spot) gleicher Laenge.

    Spread-Funktionen in netzpilot.eval.economics erwarten gleich lange Reihen; das ist die
    saubere Standardkombination, wenn beide CSVs auf der gleichen reBAP-QH-Achse vorliegen
    (siehe scripts/fetch_spot_da_2024.py).
    """
    rebap = _load_rebap_raw(rebap_csv)
    spot = load_spot_da(spot_csv)
    if len(rebap) != len(spot):
        raise ValueError(
            f"reBAP ({len(rebap)} QH) und Spot ({len(spot)} QH) verschieden lang — "
            "Spot-CSV muss auf die reBAP-Achse aligned sein."
        )
    a, b = [], []
    for r, s in zip(rebap, spot):
        if math.isfinite(r) and math.isfinite(s):
            a.append(r)
            b.append(s)
    if not a:
        raise ValueError("Keine endlichen reBAP/Spot-Paare gefunden.")
    return a, b
