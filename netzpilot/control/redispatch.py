"""Rolling Paragraph-14a redispatch.

The module uses existing control optimization only. It can run the old homogeneous
path with ``steuve_demands_kw`` or the heterogeneous path with per-device
``floor_kw`` and ``weight`` metadata.
"""
from __future__ import annotations

from .optimize import optimize_setpoints, optimize_setpoints_heterogen, naive_shed_kw
from .schema import (
    MIN_GUARANTEED_KW,
    devices_are_heterogeneous,
    normalize_steuve_devices,
    steuve_demands_from_devices,
)


def _naive_shed_devices_kw(devices: list[dict], floor_kw: float) -> float:
    if devices_are_heterogeneous(devices):
        return round(sum(
            max(0.0, float(d["demand_kw"]) - float(d.get("floor_kw", MIN_GUARANTEED_KW)))
            for d in devices
        ), 3)
    return naive_shed_kw(steuve_demands_from_devices(devices), floor_kw=floor_kw)


def _optimize_devices(devices: list[dict], cap_kw: float, floor_kw: float) -> tuple[dict, str]:
    if devices_are_heterogeneous(devices):
        return optimize_setpoints_heterogen(devices, cap_kw), "heterogeneous weighted shedding"
    return (
        optimize_setpoints(steuve_demands_from_devices(devices), cap_kw, floor_kw=floor_kw),
        "fair-min water-filling",
    )


def _rounded_limits_for_cap(limits, devices: list[dict], cap_kw: float, floor_kw: float) -> list[float]:
    rounded = [round(float(x), 3) for x in limits]
    rounded_cap = round(max(0.0, cap_kw), 3)
    over = round(sum(rounded) - rounded_cap, 6)
    if over <= 0:
        return rounded

    floors = [
        float(d.get("floor_kw", floor_kw if not devices_are_heterogeneous(devices) else MIN_GUARANTEED_KW))
        for d in devices
    ]
    for i in sorted(range(len(rounded)), key=lambda j: rounded[j] - floors[j], reverse=True):
        room = max(0.0, rounded[i] - floors[i])
        take = min(over, room)
        if take > 0:
            rounded[i] = round(rounded[i] - take, 3)
            over = round(over - take, 6)
        if over <= 0:
            break
    return rounded


def rolling_redispatch(load_forecasts, threshold_kw, steuve_demands_kw=None, *,
                       steuve_devices=None, floor_kw: float = MIN_GUARANTEED_KW,
                       horizon: int = 4):
    """Run rolling redispatch over a 24-hour day.

    ``load_forecasts[t]`` is the forecast path known at decision hour ``t`` for hours
    ``t..23``. For the static day-ahead approximation use ``from_single_path``.
    """
    devices = normalize_steuve_devices(steuve_demands_kw, steuve_devices)
    demands = steuve_demands_from_devices(devices)
    base_demand = float(sum(demands))
    hourly = []
    total_shed = 0.0
    naive_shed_total = 0.0
    n_interv = 0
    heterogeneous = devices_are_heterogeneous(devices)

    for t, path in enumerate(load_forecasts):
        if not len(path):
            continue
        fc_next = float(path[0])
        base_load = fc_next - base_demand
        cap = threshold_kw - base_load
        if fc_next <= threshold_kw or cap >= base_demand:
            limits = list(demands)
            shed = 0.0
            overload = max(0.0, fc_next - threshold_kw)
            feasible = True
            binding_floor = False
            method = "none"
            interv = False
        else:
            opt, method = _optimize_devices(devices, max(0.0, cap), floor_kw)
            limits = opt["limits_kw"]
            shed = opt["total_shed_kw"]
            overload = round(fc_next - threshold_kw, 3)
            feasible = opt["feasible"]
            binding_floor = opt["binding_floor"]
            interv = True
            n_interv += 1
            naive_shed_total += _naive_shed_devices_kw(devices, floor_kw=floor_kw)
        total_shed += shed
        rounded_cap = round(max(0.0, cap), 3) if interv else None
        rounded_limits = (
            _rounded_limits_for_cap(limits, devices, max(0.0, cap), floor_kw)
            if interv and feasible else [round(x, 3) for x in limits]
        )
        hourly.append({
            "hour": t,
            "forecast_kw": round(fc_next, 3),
            "overload_kw": round(overload, 3),
            "cap_kw": rounded_cap,
            "limits_kw": rounded_limits,
            "shed_kw": round(shed, 3),
            "feasible": feasible,
            "binding_floor": binding_floor,
            "intervention": interv,
            "optimization_method": method,
        })

    return {
        "hourly": hourly,
        "total_shed_kwh": round(total_shed, 3),
        "naive_shed_kwh": round(naive_shed_total, 3),
        "intervention_hours": n_interv,
        "saved_vs_naive_kwh": round(naive_shed_total - total_shed, 3),
        "horizon": horizon,
        "heterogeneous": heterogeneous,
        # Bedarfs-Baseline je steuVE (konstant ueber den Tag): erlaubt nachgelagert die EHRLICHE,
        # bedarfsnormalisierte Diskriminierungsfreiheits-Pruefung (Kappung shed_i relativ zum
        # Bedarf demand_i) statt nur die Gleichheit der gewaehrten Grenzen zu messen.
        "device_demands_kw": [round(float(x), 3) for x in demands],
    }


def from_single_path(load24_kw):
    """Build rolling forecast paths from one static 24-hour path."""
    p = [float(x) for x in load24_kw]
    return [p[t:] for t in range(len(p))]
