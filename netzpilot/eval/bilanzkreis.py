"""Bilanzkreis-Ausgleichsenergie-Simulator — realistische Viertelstunden-Abrechnung (Achse B.1).

Ersetzt die LINEARE Naeherung (economics.expected_saving_eur: dMAE * Stunden * Mittel-Spread) durch
die ECHTE Summe ueber Viertelstunden: jede QH wird mit ihrem EIGENEN reBAP-(minus-Spot)-Preis
abgerechnet. Damit faengt das Modell die KORRELATION zwischen dem Vorzeichen des Prognosefehlers und
dem Preis ein — genau der Effekt, den der Erwartungswert (Produkt der Marginalverteilungen) verfehlt
und den economics.py selbst als offene Luecke benennt ("belastbar erst mit der realen
Abweichungs-Zeitreihe ... haengt von der Korrelation ... ab").

Kostenmodell (transparent, ein Vorzeichen, explizit):
    Beschaffung s_t MWh am Spot + Restabweichung e_t = actual_t - s_t am reBAP =>
        Gesamtkosten_t = actual_t * reBAP_t  -  s_t * (reBAP_t - spot_t).
    Der durch die Prognose VERMEIDBARE Teil (ggue. perfekter Beschaffung, e=0) ist
        Praemie_t = e_t * (reBAP_t - spot_t),   e_t = actual_t - scheduled_t   [MWh je QH].
    Einsparung(B ggue. A) = Praemie(A) - Praemie(B) = sum (e_A - e_B) * (reBAP - spot).
    (Herleitung: Gesamtkosten_A - Gesamtkosten_B = (s_B - s_A)(reBAP-spot) = (e_A - e_B)(reBAP-spot),
     da actual fix ist und s = actual - e.)

Zerlegung (warum die lineare Naeherung daneben liegen kann):
    Praemie = sum e_t * spread_t = N * mean(e) * mean(spread)  +  N * Cov(e, spread)
            = BIAS-Term (systematische Verzerrung x Mittel-Spread) + KORRELATIONS-Term.
    Eine unverzerrte, preis-UNkorrelierte Prognose hat Praemie ~ 0 (Ueber-/Unterdeckung heben sich
    auf) — die lineare |Fehler|*Mittel-Spread-Rechnung sieht das nicht und ueberschaetzt den Nutzen.

Konventionen & Vorbehalte (ehrlich, CLAUDE.md):
  - e_t > 0 = Unterdeckung ("short", mehr verbraucht als beschafft); bei reBAP > Spot kostet das Geld.
  - reBAP-Vorzeichen ist NICHT steuerbar; der Erwartungswert der Praemie ist bei unverzerrtem,
    unkorreliertem Fehler nahe null -> realer Nutzen = Downside-Schutz + Vermeidung preis-korrelierter
    bzw. systematisch verzerrter Fehler, KEIN garantierter Linearertrag.
  - KEIN Intraday-Handel modelliert (ein BK koennte die Position vor Lieferung glattstellen) -> dies
    ist die Ausgleichsenergie-Exposure OHNE Intraday-Korrektur, also ein OBERER Rand.
  - spot=None => Spot als 0 behandelt => absolute reBAP-Abrechnung (sum e_t * reBAP_t), das ist die
    tatsaechliche UENB-Rechnung; der reBAP-Spot-Modus ist der vermeidbare Hebel ggue. perfekter Beschaffung.
  - Belastbar erst mit der REALEN Viertelstunden-Last + Beschaffungs-Fahrplan des konkreten Stadtwerks.

Reine stdlib (wie economics.py) — kein numpy/pandas noetig, ueberall importierbar. Additiv: aendert
nichts an economics.py.
"""
from __future__ import annotations

import math


def _finite(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _aligned_errors_spreads(scheduled_mwh, actual_mwh, rebap_eur_mwh, spot_eur_mwh=None):
    """Joine die Reihen QH-weise, verwerfe jede QH mit einem nicht-finiten Wert (Audit-Regel:
    nie ueber NaN abrechnen). Gibt (errors, spreads, n_dropped) zurueck, beide gleich lang.
    spot=None -> Spread = reBAP (Spot als 0)."""
    sched, act, reb = list(scheduled_mwh), list(actual_mwh), list(rebap_eur_mwh)
    if not (len(sched) == len(act) == len(reb)):
        raise ValueError(f"Reihen verschieden lang: scheduled={len(sched)}, actual={len(act)}, "
                         f"rebap={len(reb)}.")
    if spot_eur_mwh is None:
        spot = [0.0] * len(sched)
    else:
        spot = list(spot_eur_mwh)
        if len(spot) != len(sched):
            raise ValueError(f"Spot-Reihe verschieden lang ({len(spot)} vs {len(sched)}).")
    errors, spreads, dropped = [], [], 0
    for s, a, r, p in zip(sched, act, reb, spot):
        sf, af, rf, pf = _finite(s), _finite(a), _finite(r), _finite(p)
        if None in (sf, af, rf, pf):
            dropped += 1
            continue
        errors.append(af - sf)            # e_t = actual - scheduled  [MWh/QH]
        spreads.append(rf - pf)           # reBAP - Spot  [EUR/MWh]
    return errors, spreads, dropped


def imbalance_premium_eur(scheduled_mwh, actual_mwh, rebap_eur_mwh, spot_eur_mwh=None) -> dict:
    """Vermeidbare Ausgleichsenergie-Praemie [EUR] ueber alle Viertelstunden.

    scheduled_mwh, actual_mwh: beschaffte bzw. tatsaechliche Energie je QH [MWh] (gleich getaktet).
    rebap_eur_mwh:             reBAP je QH [EUR/MWh] (regelzonenuebergreifend, netztransparenz.de).
    spot_eur_mwh:              Day-ahead/Intraday-Spot je QH [EUR/MWh]; None -> 0 (absolute reBAP-Abr.).

    Rueckgabe (alles aus denselben gejointen, NaN-bereinigten QH):
      total_premium_eur      sum e_t*(reBAP-Spot)_t   (signiert; >0 = Mehrkosten ggue. perfekt)
      n, n_dropped           verwendete / verworfene QH
      sum_signed_error_mwh   sum e_t          (Bias-Indikator)
      sum_abs_error_mwh      sum |e_t|        (Volumen-Indikator, fuer die lineare Naeherung)
      mean_spread_eur_mwh    mean (reBAP-Spot)
      bias_term_eur          N*mean(e)*mean(spread)
      correlation_term_eur   total - bias_term = N*Cov(e, spread)
    Identitaet: total_premium_eur == bias_term_eur + correlation_term_eur (bis auf Rundung).
    """
    errors, spreads, dropped = _aligned_errors_spreads(scheduled_mwh, actual_mwh, rebap_eur_mwh,
                                                        spot_eur_mwh)
    n = len(errors)
    if n == 0:
        raise ValueError("Keine finiten Viertelstunden nach Bereinigung.")
    total = sum(e * s for e, s in zip(errors, spreads))
    sum_e = sum(errors)
    sum_abs_e = sum(abs(e) for e in errors)
    sum_s = sum(spreads)
    mean_e, mean_s = sum_e / n, sum_s / n
    bias_term = n * mean_e * mean_s
    corr_term = total - bias_term
    return {
        "total_premium_eur": round(total, 2),
        "n": n,
        "n_dropped": dropped,
        "sum_signed_error_mwh": round(sum_e, 4),
        "sum_abs_error_mwh": round(sum_abs_e, 4),
        "mean_spread_eur_mwh": round(mean_s, 4),
        "bias_term_eur": round(bias_term, 2),
        "correlation_term_eur": round(corr_term, 2),
    }


def compare_forecasts_eur(actual_mwh, scheduled_a_mwh, scheduled_b_mwh,
                          rebap_eur_mwh, spot_eur_mwh=None) -> dict:
    """Realisierte Einsparung von Prognose B ggue. A: Praemie(A) - Praemie(B).

    Bei festem actual und gleicher Preisreihe ist das exakt sum (e_A - e_B)*(reBAP-Spot). Positiv =>
    B ist guenstiger (weniger / besser platzierte Abweichung). Beide Prognosen werden ueber DIESELBE
    NaN-Bereinigung gerechnet, damit die Differenz fair ist.
    """
    pa = imbalance_premium_eur(scheduled_a_mwh, actual_mwh, rebap_eur_mwh, spot_eur_mwh)
    pb = imbalance_premium_eur(scheduled_b_mwh, actual_mwh, rebap_eur_mwh, spot_eur_mwh)
    return {
        "savings_b_vs_a_eur": round(pa["total_premium_eur"] - pb["total_premium_eur"], 2),
        "premium_a_eur": pa["total_premium_eur"],
        "premium_b_eur": pb["total_premium_eur"],
        "n_a": pa["n"], "n_b": pb["n"],
        "basis": "echte QH-Abrechnung e_t*(reBAP-Spot)_t; Einsparung = Praemie(A)-Praemie(B). "
                 "Kein Intraday-Handel modelliert (oberer Rand der Exposure).",
    }


def savings_contrib_per_qh(actual_mwh, scheduled_a_mwh, scheduled_b_mwh,
                           rebap_eur_mwh, spot_eur_mwh=None) -> list:
    """Per-Viertelstunde-Beitrag zur Einsparung B ggue. A [EUR/QH] — Basis fuer den Block-Bootstrap (B.2).

    Beitrag_t = (e_A,t - e_B,t)*(reBAP-Spot)_t. Da e = actual - sched, kuerzt sich actual analytisch
    weg: Beitrag_t = (sched_b,t - sched_a,t)*(reBAP-Spot)_t. actual wird dennoch mit-gejoint, damit die
    NaN-Bereinigung mit imbalance_premium_eur/compare_forecasts_eur konsistent ist (gleiche QH fallen
    weg). Summe der Liste == compare_forecasts_eur(...)['savings_b_vs_a_eur'] (roh, vor Rundung), solange
    die Fahrplaene keine NaN enthalten.

    Rueckgabe: Liste der QH-Beitraege (nicht-finite QH in irgendeiner Reihe verworfen).
    """
    a, sa, sb, r = list(actual_mwh), list(scheduled_a_mwh), list(scheduled_b_mwh), list(rebap_eur_mwh)
    n = len(a)
    if not (len(sa) == len(sb) == len(r) == n):
        raise ValueError(f"Reihen verschieden lang: actual={n}, sched_a={len(sa)}, "
                         f"sched_b={len(sb)}, rebap={len(r)}.")
    sp = [0.0] * n if spot_eur_mwh is None else list(spot_eur_mwh)
    if len(sp) != n:
        raise ValueError(f"Spot-Reihe verschieden lang ({len(sp)} vs {n}).")
    out = []
    for av, sav, sbv, rv, pv in zip(a, sa, sb, r, sp):
        af, saf, sbf, rf, pf = _finite(av), _finite(sav), _finite(sbv), _finite(rv), _finite(pv)
        if None in (af, saf, sbf, rf, pf):
            continue
        out.append((sbf - saf) * (rf - pf))      # (e_A - e_B)*(reBAP-Spot)
    return out
