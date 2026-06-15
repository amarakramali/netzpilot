# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Risk-averse Bilanzkreis-Nominierung über CVaR (Querschnitt-Recherche-Fund).

Unsere Newsvendor-Nominierung (control/dispatch.cost_optimal_nomination) minimiert den ERWARTUNGSWERT
der Ausgleichsenergiekosten -> das τ-Quantil. In seriöser Energie-Optimierung (Unit Commitment,
BRP-/Wind-Vermarktung) ist aber **Conditional Value-at-Risk (CVaR)** der Standard, um das TAIL-Risiko
zu steuern — die teuren Extrem-Viertelstunden (reBAP-Spitzen), die unsere Downside treiben. CVaR ist
linear/billig (Rockafellar-Uryasev), kein Solver nötig.

Dieses Modul macht aus dem „Downside-Schutz"-Narrativ einen EINSTELLBAREN Hebel: minimiere
    (1 − β) · E[Kosten(q)]  +  β · CVaR_α[Kosten(q)]
über die Nominierung q. β=0 reproduziert exakt die erwartungswert-optimale Newsvendor-Nominierung
(τ-Quantil); β>0 nominiert in Richtung der teuren Schwanzseite konservativer und senkt die
Tail-Kosten — auf Kosten eines minimal höheren Erwartungswerts (der klassische Risk/Return-Tradeoff).

Kostenmodell je Szenario (Residuen-Stichprobe): actual = point + residual;
    Kosten = c_short · max(actual − q, 0) + c_long · max(q − actual, 0)   [EUR], skaliert mit dt_h.
Die Zielfunktion ist konvex in q (Summe konvexer Pinball-Terme; CVaR ist konvex & monoton) -> exaktes
1D-Minimum via Ternärsuche.

Additiv, separat von dispatch.py (reine stdlib). Ehrlich: Vorteil setzt brauchbar kalibrierte
Prognose-Quantile + asymmetrische Bepreisung voraus; absolute € bleiben illustrativ bis Pilot.
"""
from __future__ import annotations


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    p = (len(sorted_vals) - 1) * max(0.0, min(1.0, q))
    lo, hi = int(p), min(int(p) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (p - lo)


def imbalance_costs(q, point_kw, residuals_kw, c_short, c_long, dt_h: float = 1.0) -> list:
    """Ausgleichsenergie-Kosten je Residuen-Szenario für eine Nominierung q [EUR]."""
    out = []
    for r in residuals_kw:
        actual = point_kw + float(r)
        short = max(actual - q, 0.0)        # Unterdeckung (zu wenig nominiert)
        long = max(q - actual, 0.0)         # Überdeckung
        out.append((c_short * short + c_long * long) * dt_h)
    return out


def cvar(costs, alpha: float) -> float:
    """Conditional Value-at-Risk auf Niveau alpha (Rockafellar-Uryasev-Schätzer).

    CVaR_alpha = VaR_alpha + (1/((1−alpha)·n)) · Σ max(cost_i − VaR_alpha, 0).
    Mittlere Kosten der schlimmsten (1−alpha)-Fraktion. alpha in (0,1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha muss in (0,1) liegen.")
    c = list(costs)
    n = len(c)
    if n == 0:
        raise ValueError("Keine Kosten-Szenarien.")
    var = _quantile(sorted(c), alpha)
    excess = sum(max(x - var, 0.0) for x in c)
    return var + excess / ((1.0 - alpha) * n)


def expected_value(costs) -> float:
    c = list(costs)
    return sum(c) / len(c) if c else float("nan")


def risk_averse_nomination(point_kw, residuals_kw, c_short, c_long, *,
                           beta: float = 0.5, alpha: float = 0.95, dt_h: float = 1.0) -> dict:
    """Risk-averse Nominierung: minimiere (1−β)·E[Kosten] + β·CVaR_α[Kosten] über q.

    beta=0  -> erwartungswert-optimal (= Newsvendor-τ-Quantil).
    beta=1  -> rein CVaR-optimal (minimiert das Schwanzrisiko).
    Rückgabe dict: nomination_kw, beta, alpha, tau_equiv (=c_short/(c_short+c_long)),
      expected_cost_eur, cvar_eur, objective_eur.
    """
    if c_short <= 0 or c_long <= 0:
        raise ValueError("c_short und c_long müssen > 0 sein.")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta muss in [0,1] liegen.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha muss in (0,1) liegen.")
    res = [float(r) for r in residuals_kw]
    if not res:
        raise ValueError("residuals_kw ist leer.")
    pt = float(point_kw)
    actuals = [pt + r for r in res]
    lo, hi = min(actuals), max(actuals)

    def objective(q):
        costs = imbalance_costs(q, pt, res, c_short, c_long, dt_h)
        return (1.0 - beta) * expected_value(costs) + beta * cvar(costs, alpha)

    if hi - lo < 1e-12:                      # degeneriert (alle Residuen gleich)
        q_star = lo
    else:
        for _ in range(100):                 # Ternärsuche auf konvexer Zielfunktion
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            if objective(m1) < objective(m2):
                hi = m2
            else:
                lo = m1
        q_star = 0.5 * (lo + hi)

    costs = imbalance_costs(q_star, pt, res, c_short, c_long, dt_h)
    return {
        "nomination_kw": round(q_star, 4),
        "beta": beta,
        "alpha": alpha,
        "tau_equiv": round(c_short / (c_short + c_long), 4),
        "expected_cost_eur": round(expected_value(costs), 4),
        "cvar_eur": round(cvar(costs, alpha), 4),
        "objective_eur": round((1.0 - beta) * expected_value(costs) + beta * cvar(costs, alpha), 4),
        "basis": "min (1-β)·E + β·CVaR_α; β=0 = Newsvendor-Erwartungswert, β>0 = Tail-Schutz. "
                 "Setzt kalibrierte Quantile + asymmetrische Bepreisung voraus; € illustrativ bis Pilot.",
    }
