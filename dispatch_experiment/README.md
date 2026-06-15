# Dispatch-Experiment — beweist der Unsicherheits-Hebel echten €-Wert?

**Zweck:** kleinster *verifizierbarer* Beleg für die Moat-These
*„NetzPilot übersetzt Prognose-Unsicherheit in eine günstigere Entscheidung"*.
Kein Produkt — ein Experiment, das eine Zahl liefert.

## These (Newsvendor / Pinball)

Bei **asymmetrischer** Ausgleichsenergie-Bepreisung (Unterspeisung kostet anders als
Überspeisung) ist die kostenoptimale Day-ahead-Bilanzkreis-Nominierung **nicht der
Median (P50)**, sondern das **τ-Quantil** der Prognoseverteilung mit

```
τ = c_short / (c_short + c_long)
```

Das ist die Minimalstelle des Pinball-Loss (Lehrsatz). Ein punktprognose-getriebener
Dispatch nominiert P50 und lässt damit bei Kostenasymmetrie systematisch Geld liegen.

## Aufbau

- **Daten:** echte SMARD-Netzlast DE, stündlich (v1-Daten, leakage-sicher wiederverwendet).
- **Prognose:** v1-Kern (Saisonal-Naiv + Ridge-Korrektur) → P50 + stundenbedingte
  Residuen-Stichprobe als prädiktive Verteilung.
- **Validierung:** Rolling-Origin; die Residuen-Quantile jedes Testtags nutzen **nur**
  frühere Tage (kein Leakage — gleiche Regel wie v1).
- **Vergleich:** deterministisch (nominiere Median) vs. stochastisch (nominiere
  τ-Quantil) über mehrere Asymmetrie-Verhältnisse `c_short/c_long`.
- **Metrik:** realisierte Imbalance-Kosten in € auf den *tatsächlichen* Lasten.

## Erfolgskriterium

- Δ€ ≈ 0 bei Verhältnis 1 (symmetrisch) — Sanity-Check (per Konstruktion).
- Δ€ > 0 und **wachsend** mit der Asymmetrie → These bestätigt.
- Falls nicht: die Prognose-Quantile sind nicht gut genug kalibriert, um den Vorteil
  zu monetarisieren — ein ehrliches, wertvolles Negativ-Ergebnis.

## Lauf

```
python dispatch_experiment.py
```

Schreibt `dispatch_results.json`. Reines numpy/pandas.

## Vom Experiment zum Produkt (bewusst NICHT hier gebaut)

Der Beweis braucht nur die Ein-Perioden-Nominierung. Das **Produkt** ist die
Mehr-Perioden-Dispatch-Engine:

- Batterie / §14a-Lasten / BHKW über einen Tag, **SOC-Kopplung**,
  **4,2-kW-§14a-Floor als harte Nebenbedingung**.
- Zweistufige stochastische Recourse; Szenarien aus den Quantilen
  (Inverse-CDF → Gauß-Copula für zeitliche Korrelation → k-means-Reduktion auf 5–10).
- Stack: `linopy`/`cvxpy` + **HiGHS** (Open-Source-Solver), kein Gurobi/CPLEX.
- Befund der Recherche: *der Großteil des Gewinns kommt aus guter Prognose + Recourse,
  nicht aus exotischer Optimierung* (EMSx-Benchmark, arXiv:2304.14808) → einfache Sache
  gut bauen.
- Zurückgestellt (research-grade): decision-focused Training (Forecaster auf €-Fehler
  statt RMSE), DRO/chance-constrained SMPC.

## Ehrliche Vorbehalte

- Die %-Einsparung ist **invariant gegen die Fehler-*Größe*** (Streuung ×2 ändert nichts —
  belegt in `residual_shape_sensitivity.py`) und wird allein von der **Form** der
  Fehlerverteilung (Schiefe/Tails) × Kostenasymmetrie bestimmt. „Verrauschter ⇒ mehr
  Ersparnis" ist damit **falsch**. Ein kleines Stadtwerk spart nur dann mehr, wenn seine
  Prognosefehler stärker schief/tail-lastig sind als die nationale Last — plausibel bei
  spitzen Einzellasten, aber nur an echten Kundendaten zu belegen (kann auch geringer sein).
- Das reBAP-Vorzeichen ist nicht steuerbar; der Nutzen ist v. a. **Downside-Schutz** in
  Stressphasen, kein garantierter Linearertrag (siehe Build-Briefing §7).
- Absolute € skalieren linear mit dem unterstellten Preis; belastbar wird die Zahl erst
  mit den realen reBAP-Kosten eines Pilotkunden.
