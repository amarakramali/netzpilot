"""Transformer thermal risk model for a single asset.

IEC 60076-7 / IEEE C57.91 style transformer checks are about hot-spot temperature
and insulation aging, not only "load > rating". This module is deliberately small:
single asset, standard parameters, no topology or load-flow model.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class ThermalParams:
    """Documented default parameters for an oil-immersed distribution transformer."""

    top_oil_rise_rated_c: float = 55.0
    winding_rise_rated_c: float = 23.0
    loss_ratio_r: float = 5.0
    top_oil_exponent_x: float = 0.8
    winding_exponent_y: float = 1.6
    tau_oil_h: float = 3.0
    tau_winding_h: float = 0.25


DEFAULT_PARAMS = ThermalParams()


def _as_list(value, n: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * n
    out = [float(x) for x in value]
    if len(out) != n:
        raise ValueError(f"{name} length {len(out)} != horizon {n}.")
    return out


def _clean(xs) -> list[float]:
    out = []
    for x in xs:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    p = (len(sorted_vals) - 1) * max(0.0, min(1.0, q))
    lo = int(p)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (p - lo)


def relative_aging_factor(theta_hotspot_c: float) -> float:
    """IEC mineral-oil aging acceleration: V = 2^((theta_h - 98 C) / 6)."""
    return 2.0 ** ((float(theta_hotspot_c) - 98.0) / 6.0)


def _ultimate_rises(load_pu: float, params: ThermalParams) -> tuple[float, float]:
    k = max(0.0, float(load_pu))
    oil = params.top_oil_rise_rated_c * (
        ((k * k * params.loss_ratio_r + 1.0) / (params.loss_ratio_r + 1.0))
        ** params.top_oil_exponent_x
    )
    winding = params.winding_rise_rated_c * (k ** params.winding_exponent_y)
    return oil, winding


def hotspot_trajectory(load_kw, ambient_c, rating_kw: float, *,
                       params: ThermalParams = DEFAULT_PARAMS,
                       dt_h: float = 1.0,
                       initial_top_oil_rise_c=None,
                       initial_winding_rise_c=None) -> dict:
    """Run a deterministic transformer hot-spot trajectory.

    load_kw and ambient_c are hourly-ish horizon arrays. The default initial state is the
    steady-state rise of the first period, which keeps rated load at 20 C near 98 C from hour 0.
    """
    load = [float(x) for x in load_kw]
    H = len(load)
    if H == 0:
        raise ValueError("empty thermal horizon.")
    if rating_kw <= 0:
        raise ValueError("rating_kw must be > 0.")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0.")
    if params.tau_oil_h <= 0 or params.tau_winding_h <= 0:
        raise ValueError("thermal time constants must be > 0.")
    ambient = _as_list(ambient_c, H, "ambient_c")

    first_oil_u, first_wdg_u = _ultimate_rises(load[0] / float(rating_kw), params)
    oil = first_oil_u if initial_top_oil_rise_c is None else float(initial_top_oil_rise_c)
    wdg = first_wdg_u if initial_winding_rise_c is None else float(initial_winding_rise_c)
    oil_gain = 1.0 - math.exp(-dt_h / params.tau_oil_h)
    wdg_gain = 1.0 - math.exp(-dt_h / params.tau_winding_h)

    hourly = []
    loss_total = 0.0
    max_hotspot = -float("inf")
    for h in range(H):
        k = max(0.0, load[h] / float(rating_kw))
        oil_u, wdg_u = _ultimate_rises(k, params)
        oil += (oil_u - oil) * oil_gain
        wdg += (wdg_u - wdg) * wdg_gain
        hotspot = ambient[h] + oil + wdg
        aging = relative_aging_factor(hotspot)
        loss_h = aging * dt_h
        loss_total += loss_h
        max_hotspot = max(max_hotspot, hotspot)
        hourly.append({
            "hour": h,
            "load_kw": round(load[h], 3),
            "loading_pu": round(k, 4),
            "ambient_c": round(ambient[h], 3),
            "top_oil_rise_c": round(oil, 4),
            "winding_rise_c": round(wdg, 4),
            "hotspot_c": round(hotspot, 4),
            "aging_factor": round(aging, 6),
            "loss_of_life_h": round(loss_h, 6),
        })
    return {
        "hourly": hourly,
        "rating_kw": float(rating_kw),
        "dt_h": float(dt_h),
        "max_hotspot_c": round(max_hotspot, 4),
        "loss_of_life_h_total": round(loss_total, 6),
        "equivalent_aging_factor": round(loss_total / (H * dt_h), 6),
        "params": asdict(params),
        "note": (
            "Single-transformer thermal approximation with standard parameters; "
            "not a load-flow or topology model."
        ),
    }


def probabilistic_thermal_risk(point_kw, residuals_kw, rating_kw: float, ambient_c=20.0, *,
                               params: ThermalParams = DEFAULT_PARAMS,
                               hotspot_limit_c: float = 120.0,
                               risk_alpha: float = 0.05,
                               dt_h: float = 1.0,
                               max_scenarios: int | None = None) -> dict:
    """Evaluate hot-spot and aging over empirical forecast-error scenarios."""
    point = [float(x) for x in point_kw]
    H = len(point)
    if H == 0:
        raise ValueError("empty thermal horizon.")
    if len(residuals_kw) != H:
        raise ValueError(f"residuals_kw length {len(residuals_kw)} != horizon {H}.")
    if rating_kw <= 0:
        raise ValueError("rating_kw must be > 0.")
    if not 0.0 <= risk_alpha < 1.0:
        raise ValueError("risk_alpha must be in [0,1).")
    ambient = _as_list(ambient_c, H, "ambient_c")
    residuals = [_clean(xs) for xs in residuals_kw]
    if any(not xs for xs in residuals):
        raise ValueError("each hour needs finite residual samples.")
    n = min(len(xs) for xs in residuals)
    if max_scenarios is not None:
        n = min(n, int(max_scenarios))
    if n <= 0:
        raise ValueError("no thermal scenarios available.")

    hotspot_by_hour = [[] for _ in range(H)]
    aging_by_hour = [[] for _ in range(H)]
    total_losses = []
    max_hotspots = []
    for i in range(n):
        path = [max(0.0, point[h] + residuals[h][i]) for h in range(H)]
        traj = hotspot_trajectory(
            path,
            ambient,
            rating_kw,
            params=params,
            dt_h=dt_h,
        )
        total_losses.append(float(traj["loss_of_life_h_total"]))
        max_hotspots.append(float(traj["max_hotspot_c"]))
        for h, row in enumerate(traj["hourly"]):
            hotspot_by_hour[h].append(float(row["hotspot_c"]))
            aging_by_hour[h].append(float(row["loss_of_life_h"]))

    hourly = []
    max_prob = 0.0
    peak_hour = None
    for h in range(H):
        hs = sorted(hotspot_by_hour[h])
        ag = aging_by_hour[h]
        p_exc = sum(1 for x in hs if x > hotspot_limit_c) / len(hs)
        if p_exc > max_prob:
            max_prob = p_exc
            peak_hour = h
        hourly.append({
            "hour": h,
            "exceedance_prob": round(p_exc, 4),
            "p50_hotspot_c": round(_quantile(hs, 0.5), 3),
            "p90_hotspot_c": round(_quantile(hs, 0.9), 3),
            "expected_loss_of_life_h": round(sum(ag) / len(ag), 6),
            "at_risk": p_exc > risk_alpha,
        })

    losses_sorted = sorted(total_losses)
    hotspots_sorted = sorted(max_hotspots)
    return {
        "hourly": hourly,
        "rating_kw": float(rating_kw),
        "hotspot_limit_c": float(hotspot_limit_c),
        "risk_alpha": float(risk_alpha),
        "n_scenarios": int(n),
        "hours_at_risk": sum(1 for h in hourly if h["at_risk"]),
        "max_exceedance_prob": round(max_prob, 4),
        "peak_risk_hour": peak_hour,
        "expected_loss_of_life_h_total": round(sum(total_losses) / len(total_losses), 6),
        "p50_loss_of_life_h_total": round(_quantile(losses_sorted, 0.5), 6),
        "p90_loss_of_life_h_total": round(_quantile(losses_sorted, 0.9), 6),
        "p50_max_hotspot_c": round(_quantile(hotspots_sorted, 0.5), 3),
        "p90_max_hotspot_c": round(_quantile(hotspots_sorted, 0.9), 3),
        "equivalent_aging_factor_expected": round(
            (sum(total_losses) / len(total_losses)) / (H * dt_h), 6
        ),
        "params": asdict(params),
        "note": (
            "Probabilistic single-transformer hot-spot and aging risk from empirical "
            "forecast residuals. Standard parameters; use pilot transformer data for production."
        ),
    }
