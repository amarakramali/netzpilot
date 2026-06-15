"""Mehr-/Mindermengen-Report (EDM-Reconciliation).

Mehr-/Mindermengen (MMM) sind die Differenz zwischen der einem Lieferanten/Bilanzkreis ZUGEORDNETEN
Energie (per Prognose/SLP-Allokation/Fahrplan) und der TATSÄCHLICH gemessenen Energie über einen
Abrechnungszeitraum. Sie werden zum (regulierten) Mehr-/Mindermengenpreis abgerechnet. Das ist die
EDM-Reconciliation-Sicht — komplementär zur Viertelstunden-Ausgleichsenergie (eval/bilanzkreis.py, am
reBAP): MMM aggregiert die GROSS-Volumina (Mehrmenge UND Mindermenge getrennt) zu EINEM Preis, wie es ein
Stadtwerk für die Lieferanten-/Netz-Reconciliation braucht.

Definition (transparent):
    e_t = actual_t − forecast_t                      [MWh je Periode]
    Mehrmenge   = Σ max(e_t, 0)   (Ist über Prognose → zu wenig zugeordnet)
    Mindermenge = Σ max(−e_t, 0)  (Ist unter Prognose → zu viel zugeordnet)
    Nettomenge  = Σ e_t = Mehrmenge − Mindermenge
    €(netto)    = Nettomenge · MMM-Preis   (Reconciliation-Betrag)
Eine bessere Prognose senkt das ABSOLUTE MMM-Volumen (Mehrmenge+Mindermenge) → weniger Reconciliation-
Exposure.

Ehrliche Grenzen (CLAUDE.md): der MMM-Preis ist reguliert und wird als EINGABE übergeben (nicht erfunden).
MMM ist die EDM-/Abrechnungssicht des Prognosefehlers, NICHT die Ausgleichsenergie (die am reBAP je QH
läuft, siehe bilanzkreis.py). Reine stdlib, additiv.
"""
from __future__ import annotations

import math


def _clean_pair(forecast, actual):
    f, a = list(forecast), list(actual)
    if len(f) != len(a):
        raise ValueError(f"forecast ({len(f)}) und actual ({len(a)}) verschieden lang.")
    if not f:
        raise ValueError("Leere Reihen.")
    fo, ac, dropped = [], [], 0
    for fv, av in zip(f, a):
        try:
            ff, aa = float(fv), float(av)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if math.isfinite(ff) and math.isfinite(aa):
            fo.append(ff); ac.append(aa)
        else:
            dropped += 1
    if not fo:
        raise ValueError("Keine finiten (forecast, actual)-Paare.")
    return fo, ac, dropped


def mehr_mindermengen(forecast_mwh, actual_mwh, mmm_price_eur_mwh, dt_h: float = 1.0) -> dict:
    """Mehr-/Mindermengen-Report aus Prognose vs. Ist.

    forecast_mwh / actual_mwh: zugeordnete bzw. gemessene Energie je Periode (gleich getaktet) [MWh je dt].
    mmm_price_eur_mwh:          regulierter Mehr-/Mindermengenpreis [EUR/MWh] (Eingabe).
    dt_h:                       Periodenlänge in Stunden (falls Werte als Leistung MW vorliegen: MWh=MW·dt_h).

    Rückgabe dict: n, n_dropped, mehrmenge_mwh, mindermenge_mwh, netto_mwh, abs_volumen_mwh,
      mmm_price_eur_mwh, netto_eur, mehrmenge_eur, mindermenge_eur.
    """
    fo, ac, dropped = _clean_pair(forecast_mwh, actual_mwh)
    price = float(mmm_price_eur_mwh)
    mehr = sum(max(a - f, 0.0) for f, a in zip(fo, ac)) * dt_h
    minder = sum(max(f - a, 0.0) for f, a in zip(fo, ac)) * dt_h
    netto = sum(a - f for f, a in zip(fo, ac)) * dt_h
    return {
        "n": len(fo),
        "n_dropped": dropped,
        "mehrmenge_mwh": round(mehr, 4),
        "mindermenge_mwh": round(minder, 4),
        "netto_mwh": round(netto, 4),
        "abs_volumen_mwh": round(mehr + minder, 4),
        "mmm_price_eur_mwh": round(price, 4),
        "netto_eur": round(netto * price, 2),
        "mehrmenge_eur": round(mehr * price, 2),
        "mindermenge_eur": round(minder * price, 2),
        "basis": "EDM-Reconciliation (Mehr-/Mindermenge) am regulierten MMM-Preis; komplementär zur "
                 "QH-Ausgleichsenergie (bilanzkreis.py, reBAP). MMM-Preis ist Eingabe, nicht erfunden.",
    }


def compare_forecasts_mmm(actual_mwh, forecast_a_mwh, forecast_b_mwh,
                          mmm_price_eur_mwh, dt_h: float = 1.0) -> dict:
    """Vergleich zweier Prognosen über das MMM-Volumen: bessere Prognose senkt das absolute Volumen.

    Rückgabe: abs_volumen_a/b_mwh, abs_volumen_reduktion_mwh (A−B), netto_a/b_mwh, sowie die Einzelreports.
    """
    ra = mehr_mindermengen(forecast_a_mwh, actual_mwh, mmm_price_eur_mwh, dt_h)
    rb = mehr_mindermengen(forecast_b_mwh, actual_mwh, mmm_price_eur_mwh, dt_h)
    return {
        "abs_volumen_a_mwh": ra["abs_volumen_mwh"],
        "abs_volumen_b_mwh": rb["abs_volumen_mwh"],
        "abs_volumen_reduktion_mwh": round(ra["abs_volumen_mwh"] - rb["abs_volumen_mwh"], 4),
        "netto_a_mwh": ra["netto_mwh"],
        "netto_b_mwh": rb["netto_mwh"],
        "report_a": ra,
        "report_b": rb,
    }
