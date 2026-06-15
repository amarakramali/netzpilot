"""Mehrperiodiger §14a-bewusster Quantil-Dispatch — der Moat-Baustein.

Vereint, was reine Prognose-Anbieter und reine Grid-IT je *für sich* nicht haben: aus der
Prognose-UNSICHERHEIT einen §14a-konformen, kostenoptimalen Tagesfahrplan. Das Modul KOMPONIERT die
bereits verifizierten Bausteine und fügt die kostenoptimale Bilanzkreis-Nominierung (Newsvendor) hinzu:

  1. §14a-Engpass-Cap je Periode: cap_h = max(0, Schwelle − Grundlast_h), damit Grundlast + steuerbare
     Last die Netzgrenze nie überschreitet (A.1/A.2-Prinzip; hier aggregiert).
  2. Kostenoptimale Platzierung des flexiblen steuVE-Energiebudgets in die günstigsten
     Netzentgelt-Stunden INNERHALB der Caps — via control.tariff.optimize_grid_fee_schedule (verifiziert).
  3. Kostenoptimale Bilanzkreis-Nominierung je Periode = τ-Quantil der Last-Prognoseverteilung,
     τ = c_short/(c_short+c_long) (Newsvendor/Pinball-Lehrsatz). Punktprognose-getriebener Dispatch
     nominiert P50 und lässt bei Kostenasymmetrie systematisch Geld liegen.

Die Integration ist der Mehrwert: die steuVE-Platzierung (Schritt 2) verschiebt die prognostizierte
Gesamtlast und damit die Nominierung (Schritt 3) — beide im selben Tagesplan, §14a-sicher.

Ehrliche Grenzen (CLAUDE.md):
  - Deterministische steuVE-Platzierung gegen die Punkt-Grundlast (Standard-Day-ahead-Planung); die
    Last-Unsicherheit geht NUR in die Nominierung ein (dort gehört sie hin). Kein stochastisches
    Mehrstufen-Recourse, kein Solver, keine Batterie-SOC-Kopplung — das ist die bewusst einfache v1.
  - Nominierungs-Vorteil setzt KALIBRIERTE Prognose-Quantile voraus; die absolute €-Wirkung hängt an
    der realen reBAP-Asymmetrie + Fehler-Form (Pilot). Übertragbar ist die Mechanik, nicht eine feste Zahl.
  - Aggregierte steuVE-Flotte; die faire Aufteilung je Gerät innerhalb einer Engpassstunde ist Sache von
    control.optimize (A.2) und hier nicht dupliziert.

Reine stdlib (+ control.tariff). Additiv: ändert nichts an bestehenden Modulen.
"""
from __future__ import annotations

from .tariff import optimize_grid_fee_schedule


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    p = (len(sorted_vals) - 1) * max(0.0, min(1.0, q))
    lo, hi = int(p), min(int(p) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (p - lo)


def cost_optimal_nomination(point_kw: float, residuals_kw, c_short: float, c_long: float):
    """Kostenoptimale Bilanzkreis-Nominierung für eine Periode (Newsvendor).

    Bei asymmetrischer Ausgleichsenergie-Bepreisung (c_short je MWh Unterdeckung, c_long je MWh
    Überdeckung) minimiert das τ-Quantil der prädiktiven Verteilung die erwarteten Imbalance-Kosten,
    τ = c_short/(c_short+c_long). residuals_kw = Stichprobe der Prognosefehler (leakage-sicher, nur
    Vergangenheit). Rückgabe: (nominierung_kw, tau).
    """
    if c_short <= 0 or c_long <= 0:
        raise ValueError("c_short und c_long müssen > 0 sein.")
    res = sorted(float(r) for r in residuals_kw)
    if not res:
        raise ValueError("residuals_kw ist leer.")
    tau = c_short / (c_short + c_long)
    return float(point_kw) + _quantile(res, tau), tau


def _expected_imbalance_eur(nomination_kw, point_kw, residuals_kw, c_short, c_long, dt_h):
    """Erwartete Ausgleichsenergie-Kosten einer Nominierung über die Residuen-Stichprobe [EUR].
    actual = point + residual; Energie = kW*dt_h = MWh wenn kW in MW. Pinball-Kosten."""
    res = [float(r) for r in residuals_kw]
    n = len(res)
    tot = 0.0
    for r in res:
        actual = point_kw + r
        short = max(actual - nomination_kw, 0.0)   # Unterdeckung (zu wenig nominiert)
        long = max(nomination_kw - actual, 0.0)    # Überdeckung
        tot += (c_short * short + c_long * long) * dt_h
    return tot / n if n else 0.0


def plan_day(base_point_kw, base_residuals_kw, threshold_kw,
             steuve_energy_kwh, steuve_p_max_kw, grid_fee_eur_per_kwh,
             c_short, c_long, dt_h: float = 1.0) -> dict:
    """Erzeuge den §14a-konformen, kostenoptimalen Tagesfahrplan aus der Prognoseverteilung.

    base_point_kw[H]:        Punktprognose (P50) der NICHT-steuerbaren Grundlast je Periode [kW/MW].
    base_residuals_kw[H][·]: je Periode eine Stichprobe der Grundlast-Prognosefehler (leakage-sicher).
    threshold_kw:            Netzkapazität/Engpassschwelle [kW/MW] (Grundlast + steuVE darf nicht drüber).
    steuve_energy_kwh:       über den Tag zu deckender steuVE-Energiebedarf [kWh/MWh je dt].
    steuve_p_max_kw:         max. steuerbare Leistung [kW/MW].
    grid_fee_eur_per_kwh[H]: zeitvariables Netzentgelt (§14a Modul 3) je Periode.
    c_short, c_long:         asymmetrische Ausgleichsenergie-Preise [EUR/MWh].

    Rückgabe dict: hourly (cap_kw, steuve_kw, total_point_kw, nomination_kw, fee_eur), Summen
    (grid_fee_cost_eur, exp_imbalance_tau_eur, exp_imbalance_p50_eur, newsvendor_saving_eur), tau,
    feasible (steuVE-Budget deckbar), shortfall_kwh, grid_safe (Netzgrenze überall gehalten).
    """
    H = len(base_point_kw)
    if not (len(base_residuals_kw) == len(grid_fee_eur_per_kwh) == H):
        raise ValueError("base_point_kw, base_residuals_kw, grid_fee müssen gleich lang sein.")
    if H == 0:
        raise ValueError("Leerer Horizont.")
    base = [float(x) for x in base_point_kw]

    # 1. §14a-Engpass-Cap je Periode für die steuerbare Last.
    caps = [max(0.0, float(threshold_kw) - base[h]) for h in range(H)]

    # 2. Flexibles steuVE-Budget kostenoptimal in die günstigsten Stunden innerhalb der Caps legen.
    sched = optimize_grid_fee_schedule(grid_fee_eur_per_kwh, steuve_energy_kwh, steuve_p_max_kw,
                                       cap_kw=caps, dt_h=dt_h)
    steuve_kw = sched["power_kw"]

    # 3. Gesamtlast + kostenoptimale Nominierung je Periode (Newsvendor auf der Grundlast-Unsicherheit).
    tau = c_short / (c_short + c_long)
    hourly = []
    grid_fee_cost = sched["total_cost_eur"]
    exp_tau = exp_p50 = 0.0
    grid_safe = True
    for h in range(H):
        total_point = base[h] + steuve_kw[h]
        if total_point > float(threshold_kw) + 1e-6:
            grid_safe = False
        nom, _ = cost_optimal_nomination(total_point, base_residuals_kw[h], c_short, c_long)
        nom_p50, _ = cost_optimal_nomination(total_point, base_residuals_kw[h], 1.0, 1.0)  # τ=0,5
        exp_tau += _expected_imbalance_eur(nom, total_point, base_residuals_kw[h], c_short, c_long, dt_h)
        exp_p50 += _expected_imbalance_eur(nom_p50, total_point, base_residuals_kw[h], c_short, c_long, dt_h)
        hourly.append({
            "hour": h,
            "cap_kw": round(caps[h], 4),
            "steuve_kw": round(steuve_kw[h], 4),
            "total_point_kw": round(total_point, 4),
            "nomination_kw": round(nom, 4),
            "fee_eur_per_kwh": float(grid_fee_eur_per_kwh[h]),
        })

    return {
        "hourly": hourly,
        "tau": round(tau, 4),
        "grid_fee_cost_eur": round(grid_fee_cost, 4),
        "exp_imbalance_tau_eur": round(exp_tau, 4),
        "exp_imbalance_p50_eur": round(exp_p50, 4),
        "newsvendor_saving_eur": round(exp_p50 - exp_tau, 4),
        "feasible": sched["feasible"],
        "shortfall_kwh": sched["shortfall_kwh"],
        "grid_safe": grid_safe,
        "basis": "§14a-konformer Quantil-Dispatch: steuVE-Platzierung innerhalb Engpass-Caps + "
                 "Newsvendor-Nominierung (τ-Quantil). Deterministische Platzierung, Unsicherheit nur "
                 "in der Nominierung; kalibrierte Quantile + reale reBAP-Asymmetrie nötig (Pilot).",
    }
