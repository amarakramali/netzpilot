"""Probabilistische Asset-Überlast-Prognose & Hosting-Capacity.

Sagt, WANN ein Netzasset überlastet wird und WIE VIEL zusätzliche Last/Erzeugung es verträgt — aus der
PROBABILISTISCHEN Prognose, also nicht nur „Punktlast > Grenze", sondern die WAHRSCHEINLICHKEIT einer
Überlast je Stunde + die erwartete Überlast-Energie. Das ist genau der Auslöser für die §14a-Steuerung
(control/redispatch) und der ehrliche, kleine-Stadtwerke-taugliche Ansatz ohne vollen Netzlastfluss.

EHRLICHER SCOPE (CLAUDE.md): EINZELASSET (ein Trafo / ein Strang / ein Netzknoten) gegen seine
Bemessungsleistung. KEIN Multi-Bus-Lastfluss, KEINE GIS-Topologie, keine Spannungs-/Strom-Berechnung im
Netz. Eingang ist die (Residuallast-)Prognoseverteilung AN diesem Asset. Damit ehrlich kommunizierbar:
„Überlastwahrscheinlichkeit dieses Anschlusspunkts", nicht „Netzberechnung".

Reine stdlib. Additiv (neues Paket netzpilot/grid).
"""
from __future__ import annotations


def _clean(xs) -> list:
    out = []
    for x in xs:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if v == v and v not in (float("inf"), float("-inf")):   # finite (NaN != NaN)
            out.append(v)
    return out


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    p = (len(sorted_vals) - 1) * max(0.0, min(1.0, q))
    lo, hi = int(p), min(int(p) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (p - lo)


def _exceedance_prob(point_kw, res_sorted, rating_kw, extra_kw=0.0):
    """P(point + residual + extra > rating) aus der empirischen Residuenverteilung."""
    n = len(res_sorted)
    if n == 0:
        return float("nan")
    thr = rating_kw - extra_kw - point_kw          # residual muss > thr für Überlast
    # Anzahl Residuen strikt > thr (lineare Suche; Reihen sind kurz)
    cnt = sum(1 for r in res_sorted if r > thr)
    return cnt / n


def overload_forecast(point_kw, residuals_kw, rating_kw, dt_h: float = 1.0,
                      risk_alpha: float = 0.05) -> dict:
    """Überlast-Prognose für EIN Netzasset über den Horizont.

    point_kw[H]:        Punktprognose (P50) der Last am Asset je Periode [kW].
    residuals_kw[H][·]: je Periode eine Stichprobe der Prognosefehler (leakage-sicher).
    rating_kw:          Bemessungs-/Engpassgrenze des Assets [kW].
    risk_alpha:         Risikoschwelle; Stunden mit Überlast-Wkt > alpha gelten als "at risk".

    Rückgabe dict: hourly (exceedance_prob, expected_overload_kwh, p50_load_kw, p90_load_kw, at_risk),
    Summen (hours_at_risk, max_exceedance_prob, peak_risk_hour, expected_overload_kwh_total,
    prob_any_overload_indep).
    """
    H = len(point_kw)
    if H == 0:
        raise ValueError("Leerer Horizont.")
    if len(residuals_kw) != H:
        raise ValueError(f"residuals_kw-Länge {len(residuals_kw)} != Horizont {H}.")
    if rating_kw <= 0:
        raise ValueError("rating_kw muss > 0 sein.")

    hourly = []
    total_overload = 0.0
    max_p = 0.0
    peak_h = None
    prod_no_overload = 1.0
    for h in range(H):
        pt = float(point_kw[h])
        res = sorted(_clean(residuals_kw[h]))
        if not res:
            raise ValueError(f"Stunde {h}: keine finiten Residuen.")
        n = len(res)
        p_exc = _exceedance_prob(pt, res, rating_kw)
        exp_over_kw = sum(max(pt + r - rating_kw, 0.0) for r in res) / n
        exp_over_kwh = exp_over_kw * dt_h
        total_overload += exp_over_kwh
        at_risk = p_exc > risk_alpha
        if p_exc > max_p:
            max_p, peak_h = p_exc, h
        prod_no_overload *= (1.0 - p_exc)
        hourly.append({
            "hour": h,
            "exceedance_prob": round(p_exc, 4),
            "expected_overload_kwh": round(exp_over_kwh, 4),
            "p50_load_kw": round(pt + _quantile(res, 0.5), 3),
            "p90_load_kw": round(pt + _quantile(res, 0.9), 3),
            "at_risk": at_risk,
        })

    return {
        "hourly": hourly,
        "rating_kw": float(rating_kw),
        "risk_alpha": risk_alpha,
        "hours_at_risk": sum(1 for h in hourly if h["at_risk"]),
        "max_exceedance_prob": round(max_p, 4),
        "peak_risk_hour": peak_h,
        "expected_overload_kwh_total": round(total_overload, 4),
        "prob_any_overload_indep": round(1.0 - prod_no_overload, 4),
        "note": "Einzelasset-Überlastwahrscheinlichkeit aus der Prognoseverteilung; KEIN Netzlastfluss. "
                "prob_any_overload_indep nimmt Stunden-Unabhängigkeit an (obere Näherung).",
    }


def hosting_capacity_kw(point_kw, residuals_kw, rating_kw, risk_alpha: float = 0.05) -> dict:
    """Maximale zusätzliche (koinzidente, konstante) Last, die das Asset noch verträgt.

    Liefert das größte ΔP >= 0, sodass die Überlast-Wahrscheinlichkeit in JEDER Periode <= risk_alpha
    bleibt: für alle h gilt P(point_h + residual + ΔP > rating) <= alpha. Das ist die probabilistische
    „Kapazitätsampel" — wie viel Wärmepumpen-/Wallbox-Last
    sich noch anschließen lässt, bevor das Risiko die Schwelle reißt.

    Ehrlich: konservativ (neue Last als koinzident/konstant angenommen); kein Netzlastfluss.
    Rückgabe dict: hosting_capacity_kw, binding_hour, risk_alpha, already_at_risk.
    """
    H = len(point_kw)
    if H == 0:
        raise ValueError("Leerer Horizont.")
    if len(residuals_kw) != H:
        raise ValueError(f"residuals_kw-Länge {len(residuals_kw)} != Horizont {H}.")
    if rating_kw <= 0:
        raise ValueError("rating_kw muss > 0 sein.")
    res_sorted = [sorted(_clean(residuals_kw[h])) for h in range(H)]
    if any(not r for r in res_sorted):
        raise ValueError("Mindestens eine Periode ohne finite Residuen.")

    def max_exceedance(extra):
        return max(_exceedance_prob(float(point_kw[h]), res_sorted[h], rating_kw, extra_kw=extra)
                   for h in range(H))

    # ΔP=0 schon über Risiko -> keine Reserve.
    if max_exceedance(0.0) > risk_alpha + 1e-12:
        return {"hosting_capacity_kw": 0.0, "binding_hour": None,
                "risk_alpha": risk_alpha, "already_at_risk": True}

    # Bisektion: feasible(ΔP) = max_exceedance(ΔP) <= alpha ist monoton fallend in ΔP.
    lo, hi = 0.0, float(rating_kw)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if max_exceedance(mid) <= risk_alpha + 1e-12:
            lo = mid
        else:
            hi = mid
    cap = lo
    # bindende Stunde = die mit der höchsten Überlast-Wkt bei ΔP=cap
    binding = max(range(H), key=lambda h: _exceedance_prob(float(point_kw[h]), res_sorted[h],
                                                           rating_kw, extra_kw=cap))
    return {
        "hosting_capacity_kw": round(cap, 3),
        "binding_hour": binding,
        "risk_alpha": risk_alpha,
        "already_at_risk": False,
    }
