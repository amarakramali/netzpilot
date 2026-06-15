# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Service glue for Paragraph-14a Module-3 grid-fee schedules.

The verified optimizer lives in netzpilot.control.tariff. This module only
normalizes service inputs and, when available, reuses redispatch caps as the
single network-safety source.
"""
from __future__ import annotations

import math

from netzpilot.control.tariff import optimize_grid_fee_schedule


def _finite_float(value, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric.")
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def normalize_available(n: int, available=None, start_hour=None, end_hour=None) -> list[bool] | None:
    """Return an availability mask.

    A window 18 -> 6 means hours 18..23 and 0..5 are available; end is exclusive.
    """
    if available is not None:
        values = list(available)
        if len(values) != n:
            raise ValueError(f"tariff_available length {len(values)} != horizon {n}.")
        return [bool(v) for v in values]
    if start_hour is None and end_hour is None:
        return None
    if start_hour is None or end_hour is None:
        raise ValueError("tariff availability window needs both start and end hour.")
    start = int(start_hour)
    end = int(end_hour)
    if not (0 <= start < n and 0 <= end < n):
        raise ValueError(f"availability hours must be within 0..{n - 1}.")
    if start == end:
        return [True] * n
    if start < end:
        return [start <= h < end for h in range(n)]
    return [h >= start or h < end for h in range(n)]


def _cap_from_redispatch(redispatch: dict | None, n: int, p_max_kw: float):
    if not redispatch:
        return None, "none"
    cap = [float(p_max_kw)] * n
    hourly = redispatch.get("hourly") or []
    by_hour = {int(h.get("hour")): h for h in hourly if "hour" in h}
    used = False
    for h in range(n):
        row = by_hour.get(h)
        if row is None:
            continue
        value = row.get("cap_kw")
        if value is not None:
            cap[h] = max(0.0, _finite_float(value, "redispatch cap_kw"))
            used = True
    return (cap, "redispatch") if used else (cap, "redispatch_no_binding_caps")


def build_tariff_schedule(fee_eur_per_kwh, energy_kwh, p_max_kw, *,
                          redispatch: dict | None = None,
                          available=None, available_start_hour=None,
                          available_end_hour=None, dt_h: float = 1.0) -> dict:
    """Build the additive service payload for a Module-3 grid-fee schedule."""
    fee = [_finite_float(x, "grid_fee_eur_per_kwh") for x in list(fee_eur_per_kwh or [])]
    if not fee:
        raise ValueError("grid_fee_eur_per_kwh is empty.")
    energy = _finite_float(energy_kwh, "tariff_energy_kwh")
    p_max = _finite_float(p_max_kw, "tariff_p_max_kw")
    dt = _finite_float(dt_h, "tariff_dt_h")
    avail = normalize_available(
        len(fee),
        available=available,
        start_hour=available_start_hour,
        end_hour=available_end_hour,
    )
    cap, cap_source = _cap_from_redispatch(redispatch, len(fee), p_max)
    result = optimize_grid_fee_schedule(
        fee,
        energy,
        p_max,
        cap_kw=cap,
        available=avail,
        dt_h=dt,
    )
    binding_cap_hours = []
    if cap is not None:
        binding_cap_hours = [
            h for h, c in enumerate(cap)
            if c < p_max - 1e-9 and (avail is None or avail[h])
        ]
    result.update({
        "fee_eur_per_kwh": [round(x, 6) for x in fee],
        "energy_kwh": round(energy, 6),
        "p_max_kw": round(p_max, 6),
        "dt_h": round(dt, 6),
        "available": avail,
        "cap_kw": None if cap is None else [round(x, 6) for x in cap],
        "cap_source": cap_source,
        "binding_cap_hours": binding_cap_hours,
        "method": "control.tariff.optimize_grid_fee_schedule",
        "baseline": "naive immediate placement on the same grid-fee profile",
        "caveat": (
            "Optimizes only the grid-fee component for a given Module-3 profile. "
            "Network caps dominate cost optimization; no thermal dynamics or comfort model."
        ),
    })
    return result
