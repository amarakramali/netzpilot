# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Paragraph-14a fahrplan schema and validation."""
from __future__ import annotations

from datetime import datetime

MIN_GUARANTEED_KW = 4.2


def normalize_steuve_devices(steuve_demands_kw=None, steuve_devices=None) -> list[dict]:
    """Normalize steuVE inputs to device dicts with demand, floor and weight.

    Existing callers pass only ``steuve_demands_kw`` as a list of floats. New heterogeneous
    callers may pass ``steuve_devices`` entries with optional ``floor_kw`` and ``weight``.
    ``weight`` is the dimming participation weight: higher weight sheds more, lower weight
    protects a device more.
    """
    if steuve_devices is None:
        if steuve_demands_kw is None:
            return []
        raw_devices = [{"demand_kw": x} for x in steuve_demands_kw]
    else:
        raw_devices = steuve_devices

    devices: list[dict] = []
    for i, raw in enumerate(raw_devices):
        if isinstance(raw, dict):
            demand = raw.get("demand_kw")
            if demand is None:
                raise ValueError(f"steuve_devices[{i}].demand_kw fehlt")
            floor = raw.get("floor_kw", MIN_GUARANTEED_KW)
            weight = raw.get("weight", 1.0)
            device_id = raw.get("device_id")
        else:
            demand = raw
            floor = MIN_GUARANTEED_KW
            weight = 1.0
            device_id = None

        demand_f = float(demand)
        floor_f = float(floor)
        weight_f = float(weight)
        if demand_f < 0:
            raise ValueError(f"steuve_devices[{i}].demand_kw muss >= 0 sein")
        if floor_f < 0:
            raise ValueError(f"steuve_devices[{i}].floor_kw muss >= 0 sein")
        if weight_f <= 0:
            raise ValueError(f"steuve_devices[{i}].weight muss > 0 sein")

        dev = {"demand_kw": demand_f, "floor_kw": floor_f, "weight": weight_f}
        if device_id is not None:
            dev["device_id"] = str(device_id)
        devices.append(dev)
    return devices


def steuve_demands_from_devices(devices: list[dict]) -> list[float]:
    return [float(d["demand_kw"]) for d in devices]


def devices_are_heterogeneous(devices: list[dict]) -> bool:
    """True if any device uses non-default floor/weight metadata."""
    return any(
        abs(float(d.get("floor_kw", MIN_GUARANTEED_KW)) - MIN_GUARANTEED_KW) > 1e-9
        or abs(float(d.get("weight", 1.0)) - 1.0) > 1e-9
        for d in devices
    )


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def make_fahrplan(malo: str, setpoints: list[dict], reason: str = "forecast_congestion",
                  issued_by: str = "NetzPilot", schedule_id: str | None = None) -> dict:
    fp = {
        "schedule_id": schedule_id or f"np-{int(datetime.now().timestamp())}",
        "issued_by": issued_by,
        "malo": malo,
        "reason": reason,
        "created_utc": datetime.now().astimezone().isoformat(),
        "setpoints": setpoints,
    }
    validate_fahrplan(fp)
    return fp


def validate_fahrplan(fp: dict) -> None:
    if not isinstance(fp.get("malo"), str) or len(fp["malo"]) < 6:
        raise ValueError("malo (Markt-Lokations-ID) fehlt/ungueltig")
    if not fp.get("setpoints"):
        raise ValueError("setpoints fehlen")
    for sp in fp["setpoints"]:
        p = float(sp["p_limit_kw"])
        floor_kw = float(sp.get("floor_kw", MIN_GUARANTEED_KW))
        if floor_kw < 0:
            raise ValueError("floor_kw muss >= 0 sein")
        if p < floor_kw - 1e-9:
            raise ValueError(f"p_limit_kw={p} unterschreitet Mindestleistung {floor_kw} kW")
        if _parse(sp["end_utc"]) <= _parse(sp["start_utc"]):
            raise ValueError("setpoint end_utc <= start_utc")


def active_limit_kw(fp: dict, ts_iso: str) -> float | None:
    """Restrictive active power limit at ts, or None if no setpoint is active."""
    ts = _parse(ts_iso)
    active = [float(sp["p_limit_kw"]) for sp in fp["setpoints"]
              if _parse(sp["start_utc"]) <= ts < _parse(sp["end_utc"])]
    return min(active) if active else None
