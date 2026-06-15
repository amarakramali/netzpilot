"""Monte-Carlo-€-Band der Jahres-Einsparung (Achse B.2) — Tages-Block-Bootstrap auf B.1.

B.1 (bilanzkreis) liefert EINE Jahres-Einsparung fuer EINE realisierte Preis-/Fehler-Historie.
B.2 quantifiziert die UNSICHERHEIT dieser Zahl: per Block-Bootstrap (Block = Tag) wird die Verteilung
der Jahres-Einsparung erzeugt -> Band (P5..P95) + Wahrscheinlichkeit, dass die Prognose im Jahr
ueberhaupt Geld spart (prob_positive).

Warum Block-Bootstrap mit Tagesbloecken:
  Der Einspar-Effekt von B.1 lebt von der KORRELATION zwischen Prognosefehler und Preis INNERHALB
  eines Tages. Ein naiver QH-weiser Bootstrap wuerde diese Struktur zerreissen. Der Tages-Block
  resamplet ganze Tage (96 QH) mit Zuruecklegen und erhaelt die Intraday-Struktur. (Entspricht der
  bestehenden Projekt-Methodik "paired block-bootstrap, Block=Tag".)

Effizient & exakt: pro Block wird der Beitrag zur Jahressumme EINMAL aufsummiert (c_b = Summe der
QH-Beitraege im Block). Eine Bootstrap-Replikation = Summe von B mit Zuruecklegen gezogenen
Block-Beitraegen. E[Replikat] = B*mean(c) = Summe -> unverzerrt fuer die Jahressumme.

Ehrliche Vorbehalte (CLAUDE.md):
  - Misst die Sampling-Unsicherheit INNERHALB des beobachteten Preis-Regimes (z.B. 2024). Extrapoliert
    NICHT auf ein anderes Regime — eine 2022-Preiskrise steckt nicht in 2024-Daten. Das Band ist
    "Unsicherheit gegeben dieses Jahr", KEINE Zukunftsprognose.
  - Tagesbloecke erhalten Intraday-, brechen aber Mehr-Tages-Autokorrelation (Wetterlagen ueber Tage)
    -> block_len ist ein Knopf (z.B. 7 Tage) fuer eine Sensitivitaet.
  - Erbt alle B.1-Vorbehalte (kein Intraday-Handel = oberer Rand; reBAP-Vorzeichen nicht steuerbar,
    Erwartungswert bei unverzerrtem, unkorreliertem Fehler ~0).

Reine stdlib (random, seedbar) — kein numpy noetig.
"""
from __future__ import annotations

import random

QH_PER_DAY = 96     # Viertelstunden pro Tag (Standard-Blocklaenge)


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    p = (len(sorted_vals) - 1) * max(0.0, min(1.0, q))
    lo, hi = int(p), min(int(p) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (p - lo)


def block_bootstrap_band(contrib_per_period, block_len: int = QH_PER_DAY,
                         n_resamples: int = 2000, seed: int = 0) -> dict:
    """Block-Bootstrap-Band der Jahres-Einsparung aus per-QH-Einspar-Beitraegen.

    contrib_per_period: Liste der per-QH (oder per-Periode) Einspar-Beitraege [EUR]
                        (z.B. aus bilanzkreis.savings_contrib_per_qh).
    block_len:          Perioden pro Block (96 = Tag bei QH-Daten; 96*7 = Woche).
    n_resamples:        Anzahl Bootstrap-Replikationen.
    seed:               RNG-Seed (Reproduzierbarkeit).

    Rueckgabe dict:
      observed_total_eur   beobachtete Jahres-Einsparung == Summe der Beitraege (Punktschaetzung),
      mean_eur             Bootstrap-Mittel (~ observed, da unverzerrt),
      p5/p25/p50/p75/p95_eur   Perzentile der Jahres-Einsparung,
      std_eur              Bootstrap-Standardabweichung,
      prob_positive        Anteil der Replikationen mit Einsparung > 0,
      n_periods, n_blocks, block_len, n_resamples, seed.
    """
    vals = [float(x) for x in contrib_per_period]
    n = len(vals)
    if n == 0:
        raise ValueError("Keine Beitraege uebergeben.")
    if block_len < 1:
        raise ValueError("block_len muss >= 1 sein.")
    if n_resamples < 1:
        raise ValueError("n_resamples muss >= 1 sein.")

    # In Bloecke schneiden (letzter Block ggf. kuerzer) und je Block den Summen-Beitrag bilden.
    blocks = [sum(vals[i:i + block_len]) for i in range(0, n, block_len)]
    B = len(blocks)
    observed = sum(blocks)                       # == sum(vals)

    rng = random.Random(seed)
    totals = []
    for _ in range(n_resamples):
        totals.append(sum(rng.choices(blocks, k=B)))   # B Bloecke mit Zuruecklegen
    totals.sort()
    m = len(totals)
    mean = sum(totals) / m
    var = sum((t - mean) ** 2 for t in totals) / m
    std = var ** 0.5
    prob_pos = sum(1 for t in totals if t > 0) / m

    return {
        "observed_total_eur": round(observed, 2),
        "mean_eur": round(mean, 2),
        "p5_eur": round(_quantile(totals, 0.05), 2),
        "p25_eur": round(_quantile(totals, 0.25), 2),
        "p50_eur": round(_quantile(totals, 0.50), 2),
        "p75_eur": round(_quantile(totals, 0.75), 2),
        "p95_eur": round(_quantile(totals, 0.95), 2),
        "std_eur": round(std, 2),
        "prob_positive": round(prob_pos, 4),
        "n_periods": n,
        "n_blocks": B,
        "block_len": block_len,
        "n_resamples": n_resamples,
        "seed": seed,
        "basis": "Tages-Block-Bootstrap der Jahres-Einsparung; Band = Sampling-Unsicherheit INNERHALB "
                 "des beobachteten Preis-Regimes, keine Regime-Extrapolation.",
    }
