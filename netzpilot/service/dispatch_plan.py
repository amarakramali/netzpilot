"""Service glue for Paragraph-14a quantile dispatch plans."""
from __future__ import annotations

from netzpilot.control.dispatch import plan_day
from netzpilot.control.risk import cvar, imbalance_costs, risk_averse_nomination


def _redispatch_caps(redispatch: dict | None) -> dict[int, float]:
    if not redispatch:
        return {}
    out = {}
    for row in redispatch.get("hourly") or []:
        if row.get("cap_kw") is not None:
            out[int(row["hour"])] = float(row["cap_kw"])
    return out


def build_dispatch_plan(base_point_kw, base_residuals_kw, threshold_kw, *,
                        steuve_energy_kwh: float,
                        steuve_p_max_kw: float,
                        grid_fee_eur_per_kwh=None,
                        c_short: float = 0.20,
                        c_long: float = 0.10,
                        risk_beta: float = 0.0,
                        risk_alpha: float = 0.95,
                        redispatch: dict | None = None,
                        metadata: dict | None = None) -> dict:
    """Build additive dispatch_plan payload from verified control.dispatch.plan_day."""
    base = [float(x) for x in base_point_kw]
    if not base:
        raise ValueError("base_point_kw is empty.")
    fee = [0.0] * len(base) if grid_fee_eur_per_kwh is None else [float(x) for x in grid_fee_eur_per_kwh]
    result = plan_day(
        base,
        base_residuals_kw,
        float(threshold_kw),
        float(steuve_energy_kwh),
        float(steuve_p_max_kw),
        fee,
        float(c_short),
        float(c_long),
    )

    beta = float(risk_beta)
    alpha = float(risk_alpha)
    if beta < 0.0 or beta > 1.0:
        raise ValueError("risk_beta must be in [0,1].")
    if not 0.0 < alpha < 1.0:
        raise ValueError("risk_alpha must be in (0,1).")
    if beta > 0.0:
        exp_risk = cvar_risk = obj_risk = 0.0
        cvar_newsvendor = 0.0
        for h, row in enumerate(result["hourly"]):
            total_point = float(row["total_point_kw"])
            residuals = [float(x) for x in base_residuals_kw[h]]
            newsvendor_nom = float(row["nomination_kw"])
            risk = risk_averse_nomination(
                total_point,
                residuals,
                float(c_short),
                float(c_long),
                beta=beta,
                alpha=alpha,
            )
            news_costs = imbalance_costs(
                newsvendor_nom,
                total_point,
                residuals,
                float(c_short),
                float(c_long),
            )
            row["newsvendor_nomination_kw"] = round(newsvendor_nom, 4)
            row["nomination_kw"] = risk["nomination_kw"]
            row["risk_expected_cost_eur"] = risk["expected_cost_eur"]
            row["risk_cvar_eur"] = risk["cvar_eur"]
            row["risk_objective_eur"] = risk["objective_eur"]
            row["newsvendor_cvar_eur"] = round(cvar(news_costs, alpha), 4)
            exp_risk += risk["expected_cost_eur"]
            cvar_risk += risk["cvar_eur"]
            obj_risk += risk["objective_eur"]
            cvar_newsvendor += cvar(news_costs, alpha)
        result["risk_averse"] = {
            "enabled": True,
            "beta": beta,
            "alpha": alpha,
            "selected_nomination": "cvar_risk_averse",
            "exp_imbalance_risk_eur": round(exp_risk, 4),
            "cvar_imbalance_risk_eur": round(cvar_risk, 4),
            "objective_risk_eur": round(obj_risk, 4),
            "cvar_imbalance_newsvendor_eur": round(cvar_newsvendor, 4),
            "risk_expected_delta_vs_newsvendor_eur": round(exp_risk - result["exp_imbalance_tau_eur"], 4),
            "risk_cvar_delta_vs_newsvendor_eur": round(cvar_risk - cvar_newsvendor, 4),
            "risk_averse_saving_vs_p50_eur": round(result["exp_imbalance_p50_eur"] - exp_risk, 4),
            "basis": (
                "Selected nomination minimizes (1-beta)*E[cost] + beta*CVaR_alpha[cost]. "
                "Default beta=0 keeps the original Newsvendor path."
            ),
        }

    rd_caps = _redispatch_caps(redispatch)
    mismatches = []
    for hour, cap in rd_caps.items():
        if hour >= len(result["hourly"]):
            continue
        dispatch_cap = float(result["hourly"][hour]["cap_kw"])
        if abs(dispatch_cap - cap) > 1e-3:
            mismatches.append({
                "hour": hour,
                "dispatch_cap_kw": round(dispatch_cap, 3),
                "redispatch_cap_kw": round(cap, 3),
            })

    result.update({
        "cap_source": "threshold_and_base_point",
        "redispatch_cap_consistency": {
            "checked": bool(rd_caps),
            "consistent": len(mismatches) == 0,
            "mismatches": mismatches,
        },
        "residual_sample_counts": [len(list(xs)) for xs in base_residuals_kw],
        "grid_fee_source": "flat_zero" if grid_fee_eur_per_kwh is None else "provided",
        "metadata": metadata or {},
        "caveat": (
            "Deterministic steuVE placement against the point base load; forecast uncertainty is used "
            "for the Newsvendor nomination. Imbalance coefficients must be calibrated on pilot data."
        ),
    })
    return result
