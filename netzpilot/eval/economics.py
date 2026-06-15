"""Oekonomische Uebersetzung: Prognosefehler -> Ausgleichsenergiekosten (transparent & ehrlich).

Saubere Leistungs-/Energie-Formel:
    Einsparung [EUR/Jahr] = dMAE_MW * Stunden/Jahr * reBAP_Spread [EUR/MWh]
- dMAE_MW: Reduktion der MITTLEREN absoluten Bilanzkreisabweichung (MW, Durchschnittsleistung)
  AUF PORTFOLIO-EBENE des Stadtwerks (NICHT national!).
- reBAP_Spread: effektiver Mehrpreis der Ausgleichsenergie ggue. Intraday [EUR/MWh].
  Da MW * h = MWh, ergibt MW * h * EUR/MWh = EUR.

EINHEITEN-HINWEIS (geprueft): Das Build-Briefing-Beispiel rechnete mit einer MWh-pro-Viertelstunde-
Konvention (dMAE * 35040 QH * Spread) und kam so auf hoehere Werte. Dieses Modul nutzt die
physikalisch saubere Leistungsformel (dMAE in MW * 8760 h). Bei der Pilot-Auswertung EINE
Konvention festlegen. (35040 QH * 0,25 h = 8760 h -> identisch, wenn dMAE konsistent in MW steht.)

VORBEHALTE (kein garantierter Linearertrag):
- reBAP-Vorzeichen ist nicht steuerbar; Erwartungswert langfristig nahe null Aufschlag, in
  Stressphasen aber massiv -> Nutzen ist primaer DOWNSIDE-Schutz.
- Pooling beim Direktvermarkter teilt das Risiko.
- Verteidigbarste Basis: die REALEN reBAP-Kosten des konkreten Stadtwerks (UENB-Abrechnung).
reBAP-Niveaus (Quellen, Stand 2024/25): Q1/2024 nur ~1,7 EUR/MWh ueber Spot; hoch volatil,
Einzel-Viertelstunden >1000 EUR/MWh moeglich; 2021 ~100, Krise 2022 ~158 (Jahresmittel).
"""
from __future__ import annotations
import math

HOURS_PER_YEAR = 8760.0
QH_PER_YEAR = 35040  # 4 * 8760
# rev: + echte reBAP-Spread-Statistik (rebap_spread_stats / saving_from_real_rebap)


def ausgleichsenergie_saving_eur(delta_mae_mw: float, rebap_spread_eur_mwh: float,
                                 hours_per_year: float = HOURS_PER_YEAR) -> float:
    if delta_mae_mw < 0:
        raise ValueError("delta_mae_mw muss >= 0 sein (es ist eine Fehler-REDUKTION).")
    if rebap_spread_eur_mwh < 0:
        raise ValueError("reBAP-Spread als Betrag (>= 0) angeben.")
    return float(delta_mae_mw) * float(hours_per_year) * float(rebap_spread_eur_mwh)


def labor_saving_eur(hours_per_week: float, rate_eur_per_hour: float, weeks: float = 52.0) -> float:
    return float(hours_per_week) * float(rate_eur_per_hour) * float(weeks)


def scenarios(delta_mae_mw: float, spreads=(5.0, 15.0, 30.0),
              hours_per_year: float = HOURS_PER_YEAR) -> dict:
    return {f"spread_{int(s)}_eur_mwh": round(ausgleichsenergie_saving_eur(delta_mae_mw, s, hours_per_year))
            for s in spreads}


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    p = (len(sorted_vals) - 1) * max(0.0, min(1.0, q))
    lo, hi = int(p), min(int(p) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (p - lo)


def rebap_spread_stats(prices_eur_mwh) -> dict:
    """Empirische |reBAP|-Spread-Statistik aus einer echten reBAP-Viertelstundenreihe.

    Eingabe: Liste/Array der reBAP-Preise [EUR/MWh] (regelzonenübergreifend, netztransparenz.de).
    Der für die Ausgleichsenergie wirksame *Spread* ist der Betrag |reBAP| (Kostenrisiko in beide
    Richtungen). Gibt Median + robustes Band (P25/P75) zurück — die verteidigbare Basis statt einer
    geratenen Einzelzahl. Nicht-finite Werte werden verworfen (Audit-Regel: nie über NaN mitteln).
    """
    vals = []
    for p in prices_eur_mwh:
        if p is None:
            continue
        try:
            v = float(p)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            vals.append(abs(v))
    vals.sort()
    n = len(vals)
    if n == 0:
        raise ValueError("Keine endlichen reBAP-Preise übergeben.")
    mean = sum(vals) / n
    return {
        "n": n,
        "mean_abs_spread_eur_mwh": round(mean, 2),
        "median_abs_spread_eur_mwh": round(_quantile(vals, 0.5), 2),
        "p25_eur_mwh": round(_quantile(vals, 0.25), 2),
        "p75_eur_mwh": round(_quantile(vals, 0.75), 2),
        "p95_eur_mwh": round(_quantile(vals, 0.95), 2),
    }


def saving_from_real_rebap(delta_mae_mw: float, prices_eur_mwh,
                           hours_per_year: float = HOURS_PER_YEAR) -> dict:
    """Belastbare €/Jahr-Einsparung aus ECHTER reBAP-Reihe statt Annahme.

    Liefert eine Punktzahl (Median-Spread) plus konservatives/optimistisches Band (P25/P75) und die
    zugrunde liegende Spread-Statistik — damit jede €-Aussage rückführbar und ehrlich gebandet ist.
    """
    st = rebap_spread_stats(prices_eur_mwh)
    point = ausgleichsenergie_saving_eur(delta_mae_mw, st["median_abs_spread_eur_mwh"], hours_per_year)
    lo = ausgleichsenergie_saving_eur(delta_mae_mw, st["p25_eur_mwh"], hours_per_year)
    hi = ausgleichsenergie_saving_eur(delta_mae_mw, st["p75_eur_mwh"], hours_per_year)
    return {
        "delta_mae_mw": round(float(delta_mae_mw), 4),
        "eur_per_year_point_median": round(point),
        "eur_per_year_p25": round(lo),
        "eur_per_year_p75": round(hi),
        "rebap_spread_stats": st,
        "basis": "echte reBAP-Viertelstundenreihe (|Spread|); Punkt=Median, Band=P25–P75",
    }


def spread_over_spot_stats(rebap_eur_mwh, spot_eur_mwh) -> dict:
    """Empirische |reBAP − Spot|-Aufschlagsstatistik — der KORREKTE Einsparungs-Hebel.

    Grund (siehe Notizen/_rebap_spread_caveat.md): die durch bessere Prognose vermiedene Abweichungs-MWh
    haette am Spot/Intraday ohnehin Geld gekostet. Gespart wird nur der AUFSCHLAG reBAP - Spot, nicht der
    absolute reBAP-Preis. |reBAP| absolut ueberschaetzt den Nutzen typ. um das 3-5-fache.

    rebap_eur_mwh, spot_eur_mwh: gleich lange, zeitlich gejointe Viertelstundenreihen [EUR/MWh].
    Nicht-finite Paare werden verworfen (Audit-Regel). Gibt Median + P25/P75/P95 des |Aufschlags| zurueck.
    """
    a, b = list(rebap_eur_mwh), list(spot_eur_mwh)
    if len(a) != len(b):
        raise ValueError(f"reBAP- und Spot-Reihe verschieden lang ({len(a)} vs {len(b)}).")
    spreads = []
    for r, s in zip(a, b):
        if r is None or s is None:
            continue
        try:
            rv, sv = float(r), float(s)
        except (TypeError, ValueError):
            continue
        if math.isfinite(rv) and math.isfinite(sv):
            spreads.append(abs(rv - sv))
    spreads.sort()
    n = len(spreads)
    if n == 0:
        raise ValueError("Keine endlichen reBAP/Spot-Paare uebergeben.")
    return {
        "n": n,
        "mean_abs_spread_over_spot_eur_mwh": round(sum(spreads) / n, 2),
        "median_abs_spread_over_spot_eur_mwh": round(_quantile(spreads, 0.5), 2),
        "p25_eur_mwh": round(_quantile(spreads, 0.25), 2),
        "p75_eur_mwh": round(_quantile(spreads, 0.75), 2),
        "p95_eur_mwh": round(_quantile(spreads, 0.95), 2),
    }


def expected_saving_eur(delta_mae_mw: float, rebap_eur_mwh, spot_eur_mwh,
                        hours_per_year: float = HOURS_PER_YEAR) -> dict:
    """ERWARTETE €/Jahr-Einsparung über den SIGNIERTEN Mittel-Aufschlag mean(reBAP − Spot).

    Das ist die ehrlichste *Erwartungswert*-Zahl: im Mittel zahlt ein Bilanzkreis pro MWh Abweichung
    den Aufschlag reBAP−Spot zu viel (2024 ~7 €/MWh). |reBAP−Spot| (spread_over_spot_stats) misst dagegen
    die VOLATILITÄT/Downside-Bandbreite und überschätzt den Erwartungswert um ein Vielfaches.

    WICHTIG: Auch dies ist nur eine Markt-Näherung. Der wahre Wert hängt von der Korrelation zwischen dem
    Abweichungs-Vorzeichen des Stadtwerks und dem System-NRV ab → belastbar erst mit der realen
    Abweichungs-/Bilanzkreis-Zeitreihe des konkreten Stadtwerks.
    """
    a, b = list(rebap_eur_mwh), list(spot_eur_mwh)
    if len(a) != len(b):
        raise ValueError(f"reBAP- und Spot-Reihe verschieden lang ({len(a)} vs {len(b)}).")
    diffs = []
    for r, s in zip(a, b):
        if r is None or s is None:
            continue
        try:
            rv, sv = float(r), float(s)
        except (TypeError, ValueError):
            continue
        if math.isfinite(rv) and math.isfinite(sv):
            diffs.append(rv - sv)
    if not diffs:
        raise ValueError("Keine endlichen reBAP/Spot-Paare uebergeben.")
    signed_mean = sum(diffs) / len(diffs)
    return {
        "delta_mae_mw": round(float(delta_mae_mw), 4),
        "expected_eur_per_year": round(ausgleichsenergie_saving_eur(delta_mae_mw, abs(signed_mean), hours_per_year)),
        "signed_mean_spread_eur_mwh": round(signed_mean, 2),
        "n": len(diffs),
        "basis": "ERWARTUNGSWERT: signierter Mittel-Aufschlag mean(reBAP-Spot); ehrlichste €-Punktzahl. "
                 "Belastbar erst mit realer Abweichungs-Zeitreihe des Stadtwerks.",
    }


def saving_from_rebap_spot(delta_mae_mw: float, rebap_eur_mwh, spot_eur_mwh,
                           hours_per_year: float = HOURS_PER_YEAR) -> dict:
    """BELASTBARE €/Jahr-Einsparung aus dem Aufschlag |reBAP - Spot| (Headline-Zahl).

    Bevorzugt gegenueber saving_from_real_rebap (absolutes |reBAP|), das nur ein ueberschaetzter
    oberer Rand ist. Punkt = Median-Aufschlag, Band = P25-P75. Downside-Schutz, kein garantierter Ertrag.
    """
    st = spread_over_spot_stats(rebap_eur_mwh, spot_eur_mwh)
    point = ausgleichsenergie_saving_eur(delta_mae_mw, st["median_abs_spread_over_spot_eur_mwh"], hours_per_year)
    lo = ausgleichsenergie_saving_eur(delta_mae_mw, st["p25_eur_mwh"], hours_per_year)
    hi = ausgleichsenergie_saving_eur(delta_mae_mw, st["p75_eur_mwh"], hours_per_year)
    return {
        "delta_mae_mw": round(float(delta_mae_mw), 4),
        "eur_per_year_point_median": round(point),
        "eur_per_year_p25": round(lo),
        "eur_per_year_p75": round(hi),
        "spread_over_spot_stats": st,
        "basis": "Aufschlag |reBAP - Spot| (korrekter Hebel); Punkt=Median, Band=P25-P75; Downside-Schutz",
    }


if __name__ == "__main__":
    # Illustratives Beispiel: kleines Stadtwerk, 5 MW EE-Portfolio.
    # Annahme: MAE 8% -> 5% der Nennleistung => 0,40 MW -> 0,25 MW, also dMAE = 0,15 MW.
    dmae = 0.15
    print("Beispiel: 5 MW EE-Portfolio, dMAE = 0,15 MW (MAE 8%->5%)")
    for s, eur in scenarios(dmae).items():
        print(f"  reBAP-Spread {s:>20s}: {eur:>8,} EUR/Jahr".replace(",", "."))
    print(f"  Automatisierte Arbeit (6 h/Woche x 65 EUR/h): {round(labor_saving_eur(6,65)):>8,} EUR/Jahr".replace(",", "."))
    print("  -> Groessenordnung, kein Angebot. reBAP-Nutzen = Risikoschutz, nicht garantierter Ertrag.")
