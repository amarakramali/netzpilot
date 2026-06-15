# NetzPilot — Methodik & Validierung

*Stand: 2. Juni 2026. Dieses Dokument beschreibt, was NetzPilot tut, wie es geprüft wurde und — ebenso
wichtig — was (noch) NICHT belegt ist. Alle Zahlen stammen aus reproduzierbaren Läufen im Repository;
die Quelle ist je Abschnitt genannt. Es werden keine Werte behauptet, die nicht aus einem Artefakt
nachvollziehbar sind.*

---

## 1. Kurzfassung

NetzPilot ist eine Day-ahead-Prognose- und §14a-Koordinationssoftware für kleine Stadtwerke und
Verteilnetzbetreiber. Sie sagt die Verteilnetz-Last bzw. Residuallast des nächsten Tages stündlich mit
Unsicherheitsband (P10/P50/P90) voraus und leitet daraus zwei operative Funktionen ab: eine faire,
minimale §14a-Steuerung steuerbarer Verbrauchseinrichtungen bei Netzengpässen und eine realistische
Abschätzung der Bilanzkreis-/Ausgleichsenergie-Exposure.

Belegt ist heute: Die Prognose schlägt auf **46 echten, öffentlichen Netz-/Regionallastgängen** auf dem
statistisch sauberen 84-Tage-Fenster in **44 von 46 Reihen signifikant** die naive Branchenpraxis
(Saisonal-Naiv; mittlerer Skill +23,7 %; die zwei Ausnahmen sind Quasi-Duplikate der Herne-Niederspannung
mit +3,1 % bei P(Modell besser)=92 % — ehrliche Grenzfälle, kein Gegenbeweis; gegen Persistenz gewinnt
die große Mehrzahl). Die
§14a-Optimierung reduziert die Abregelung gegenüber pauschalem Dimmen erheblich und hält dabei
nachweislich Netzgrenze und gesetzliche Mindestleistung ein. Der €-Nutzen aus vermiedener
Ausgleichsenergie ist **reihen- und regimeabhängig** — bei manchen Netzen robust positiv, bei anderen
nahe null — und wird ehrlich mit Unsicherheitsband statt als feste Zahl ausgewiesen. Der theoriegestützte,
robuste Hebel ist die **Newsvendor-Logik** (§9): bei asymmetrischer Ausgleichsenergie-Bepreisung senkt die
Nominierung des kostenoptimalen τ-Quantils statt des Medians die Kosten nachweisbar, wachsend mit der
Asymmetrie (leakage-sicher belegt; prozentual übertragbar, absolute € nur illustrativ).

Nicht belegt ist: ein Live-Pilot mit der echten, kundeneigenen RLM-Gesamtlast und echter
reBAP-/Bilanzkreis-Abrechnung. Bis dahin sind alle Lastgänge öffentliche DSO-Reihen, die als Proxy für
die Bilanzkreis-Entnahme dienen. Das ist die einzige echte Validierungslücke — und sie ist bewusst so
benannt. Das Instrument, sie zu schließen, läuft bereits: jede ausgegebene Prognose wird hash-verkettet
**zum Ausgabezeitpunkt** gespeichert und gegen eintreffende Ist-Werte abgerechnet (Live-Track-Record, §12).

---

## 2. Problemstellung & Geltungsbereich

Kleine Stadtwerke stehen unter wachsendem Druck: §14a EnWG verpflichtet Netzbetreiber, steuerbare
Verbrauchseinrichtungen (Wärmepumpen, Wallboxen, Speicher) netzdienlich zu steuern, ohne sie unzulässig
zu benachteiligen; gleichzeitig verursachen Prognosefehler im Bilanzkreis Ausgleichsenergiekosten. Große
Tools (KISTERS, Seven2one) adressieren das, sind für kleine EVU aber teuer und schwer.

NetzPilot zielt auf die day-ahead-Ebene: 24×1 h Prognose der Last/Residuallast mit P10/P50/P90, daraus
ein §14a-Fahrplan und eine €-Einordnung. Der Geltungsbereich ist bewusst eng gehalten — keine
Intraday-Handelsoptimierung, keine Behauptung, „genauer als kommerzielle Profitools" zu sein (ein
solcher Direktvergleich wurde nicht gefahren).

---

## 3. Datengrundlage & Provenienz

Eine zentrale Disziplin des Projekts ist die strikte Trennung **echter** von **modellierter** Datenbasis.
Es wird nie über NaN gemittelt; Reihen mit identischem Werte-Hash werden vor jeder Zählung dedupliziert.

**Validierungskorpus (echt):** 51 eingelesene Einträge → nach Inhalts-Deduplikation **46 eindeutige
Reihen**, nach Korrelations-Deduplikation (Pearson r ≥ 0,98, transitive Cluster) **38 unabhängige
Netz-/Regionalcluster**. Aufteilung: 43 deutsche, 3 französische Reihen (Quelle:
`data_cache/real/corpus_index.json`, `network_independence.md`). Es handelt sich um nach §12/§23c bzw.
national veröffentlichte Lastgänge realer Netzbetreiber (u. a. Hilden, Herne, EVDB, Neuruppin,
Bitterfeld-Wolfen, Waren, TEN Thüringer Energienetze, neu.sw Neubrandenburg, Passau; Enedis FR, RTE FR).

**Bewusst gesondert behandelt:**
- **SLP-Summenlast** (1 Reihe): ein veröffentlichter Standardlastprofil-Summengang ist nahezu
  deterministisch; die niedrige MAPE dort (Hilden 1,8 %, Skill +34 %) ist „zu einfach" und wird NICHT als
  Realnachweis geführt. Das ehrliche Headline ist der Netzumsatz, nicht die SLP-Summe.
- **Signierte Differenzbilanz-Reihen** (7 Reihen): Mittelwert ≈ 0 → MAPE bedeutungslos; dort führt der
  Skill/MAE, nicht die Prozentzahl.
- **Modellierte „City"-Profile**: synthetische Profile (Cross-City r ≈ 0,89–0,98) dienen nur der
  Entwicklung, nicht als Beweis. Sie sind nicht Teil des Validierungskorpus.
- **Nationale Last (ENTSO-E)**: getrennt gehalten (`data_cache/intl/`) und nie als Verteilnetz gezählt.

**Preisdaten (echt):** reBAP 2024 (regelzonenübergreifend, netztransparenz.de, Viertelstunden) und
Day-ahead-Spot 2024 (`data_cache/real/`).

---

## 4. Prognosemethode

Das Modell ist bewusst einfach und robust statt komplex:

**Basis — Saisonal-Naiv:** die Last der Vorwoche zum gleichen Wochentag und zur gleichen Stunde. Eine
starke, transparente Baseline.

**Korrektur — ShrunkCorrector:** eine Ridge-Regression auf leakage-sichere Kalender- und Lag-Features,
mit **Shrinkage Richtung Baseline**. Pro Fit wird auf einem Out-of-sample-Tail ein Faktor s∈[0,1]
gewählt, der den MAE minimiert. Hilft die Korrektur nicht, geht s→0 und das Modell fällt auf die Baseline
zurück. Das ist die zentrale Vertrauenseigenschaft: **NetzPilot wird nie deutlich schlechter als die
triviale Baseline** — entscheidend für ein Werkzeug, das im Betrieb unbeaufsichtigt läuft.

**Unsicherheit — Conformalized Quantile Regression (CQR):** liefert P10/P50/P90 mit verteilungsfreier
marginaler Coverage-Garantie. (Quelle: `MODEL_CARD.md`.)

**Feiertagsbewusste Basis (2026-06):** war der Vorwochen-Referenztag ein Feiertag, ist die Saisonal-Naiv-
Basis systematisch verzerrt (~17–20 % zu niedrig auf Aggregat-Last). Statt eines gefitteten Flags (das auf
Industrie-Reihen katastrophal überfittete und verworfen wurde) weicht NetzPilot **deterministisch** auf die
nächste Nicht-Feiertags-Referenz gleichen Wochentags aus (d−14, d−21, …). Parameterfrei, leakage-sicher,
bit-identisch an allen anderen Tagen. Gemessen: +2,7…3,2 % MAE auf Aggregat-Reihen, ≈neutral auf Industrie.
(`scripts/verify_holiday_base.py`, Demo `holiday_base_demo.md`.)

**Online-Residuen-Feedback (2026-06):** die Modell-Residuen sind über alle echten Reihen signifikant
lag-1-autokorreliert (+0,18…+0,44) — persistente Regime-Fehler, die die Features nicht voll abbilden. Ein
Anteil ρ des zuletzt **beobachteten** Residuums wird auf die nächste Prognose addiert (Level-Shift auf
P10/P50/P90 gemeinsam); ρ wird online auf einem nachlaufenden Fenster getunt und zur 0 geschrumpft —
adaptiv: starke Autokorrelation → Gewinn, schwache → No-Op. Gemessen: 5/5 echte Reihen besser,
mean +1,6 % MAE, Pinball nie schlechter. (`scripts/verify_residual_feedback.py`, Demo
`residual_feedback_demo.md`.)

**Coverage-Kalibrierung, online-rollend und asymmetrisch (2026-06):** die rohen P10/P90 sind je Reihe
unterschiedlich und zweiseitig fehlkalibriert; zusätzlich sind die Fehler rechtsschief (Lastspitzen drücken
über P90). NetzPilot skaliert daher die untere und obere Bandhälfte **getrennt**, mit je Tag aus dem
nachlaufenden 28-Tage-Fenster getunten, geschrumpften Faktoren (leakage-sicher: nie der Zieltag). Gemessen
@84 Testtagen: mittlerer Coverage-Fehler ~halbiert (4,73→2,66 Punkte), beide Tails ≈10 % (Soll), Pinball in
keinem Fall schlechter. Naive und kalibrierte Coverage werden immer nebeneinander berichtet.
(`scripts/verify_rolling_calibration.py`, `verify_asymmetric_calibration.py`.)

**Kumulativ:** die P50-Mechanismen komponieren additiv (4-Arm-Messung, gleiche Eval-Tage): Aggregat-Reihen
**+3,1…3,2 % MAE gesamt** (Hilden +3,17 %, Herne +3,12 %), Industrie ≈neutral — keine negative Interaktion.
(`STAND_PROGNOSEKERN_2026-06-03.md`.)

---

## 4a. Temporale Reconciliation statt Spannungsebenen-Zwang

MinT-Reconciliation wird in NetzPilot nur auf Hierarchien angewandt, deren Summenidentitaet an den
Rohdaten gilt. Die naheliegende Idee, veroeffentlichte Netzlastreihen verschiedener Spannungsebenen als
`Parent = Sum(Children)` zu reconciliieren, wurde an echten Daten getestet und verworfen: diese Reihen
sind eine vertikale Kaskade mit Entnahme je Ebene, keine additive Hierarchie.

| gepruefte Identitaet (TEN 2025, 35.040 Viertelstunden) | Ergebnis | mittlere rel. Abweichung |
|---|---|---:|
| HS = HSU + MS | nein | 38,8 % |
| HS = HSU + MSU + NS | nein | 70,6 % |
| MS = MSU + NS | nein | 108,3 % |
| MSU = NS | Quasi-Duplikat, kein Mehrwert | 2,2 % |

Konsequenz: Eine cross-sektionale Spannungsebenen-Reconciliation wuerde reale Daten auf eine falsche
physikalische Constraint biegen. NetzPilot nutzt MinT stattdessen auf der temporal exakten Achse einer
einzigen 15-min-Reihe: Tagesenergie = Summe der 24 Stunden = Summe der 96 Viertelstunden. Auf Hilden
Netzumsatz 2025 (`data_cache/benchmark/reconcile_temporal_demo.md`) sinkt der maximale
Koharenzfehler im 14-Tage-Holdout von 44,369193 MWh auf 0,000000 MWh; MAE wird je Ebene ehrlich
ausgewiesen.

---

## 5. Evaluationsprotokoll

Vier Regeln, die jeden Validierungslauf binden (Quelle: `MODEL_CARD.md`, `benchmark_suite.py`):

1. **Leakage-sicher:** rolling-origin (expanding window) — das Modell sieht beim Vorhersagen nie den
   Zieltag. Sämtliche berichteten Zahlen sind Out-of-sample.
2. **Baselines verpflichtend:** Persistenz UND Saisonal-Naiv in jedem Lauf. Skill wird immer relativ zu
   beiden ausgewiesen.
3. **Signifikanz:** paired Block-Bootstrap mit Block = ganzer Tag (respektiert die Intraday-Korrelation),
   95 %-Konfidenzintervall des Skill + Wahrscheinlichkeit „Modell besser".
4. **NaN/Inf-sicher:** nicht-finite Werte werden vor jeder Mittelung verworfen.

5. **Ausreichende Test-Power (Audit 2026-06-03):** das frühere Default-Fenster von 28 Testtagen war
   statistisch unterpowert — 11 echte Reihen galten fälschlich als „nicht signifikant" (Punktschätzung
   stabil, CI nur zu breit). Bei **84 Testtagen (12 Wochen)** wurden im finalen Voll-Board (2026-06-04)
   **44 von 46 Reihen signifikant**; zwei Herne-NS-Quasi-Duplikate (+3,1 %, P(besser)=92 %) bleiben
   ehrlich n.s. Headline-Fenster ist seither 84 Tage.

Reproduktion: `python scripts/benchmark_suite.py` → `benchmark_table.md` + `benchmark_results.json`
(Seed 20260601, **84 Testtage** rollierend, 10 000 Bootstrap-Resamples). Voll-Stack-Variante mit
Residuen-Feedback + Online-Kalibrierung: `--residual-feedback --calibrate` → separat benannte Dateien
(`…_rf_cal.json/.md`); Fortschritts-Dateien sind je Flag-Kombination getrennt, ein Resume kann
Mechanismen nie mischen.

---

## 6. Validierungsergebnisse (Prognose)

Auf dem statistisch sauberen **84-Tage-Fenster** (siehe §5, Regel 5) schlägt NetzPilot Saisonal-Naiv
in **44 von 46 Reihen signifikant** (mittlerer Skill **+23,7 %**, Median +21,4 %; Ausnahmen: zwei
Herne-NS-Quasi-Duplikate, +3,1 % bei P(besser)=92 % — ehrliche Grenzfälle). Voll-Board final gerechnet
am 2026-06-04 (venv, Seed 20260601, 10 000 Bootstrap-Resamples). Die verteidigbaren Echtdaten-Headlines
(echter Netzbezug, kein SLP-Summengang; 84 Testtage):

| Reihe | Mittl. Last | MAPE | Skill vs S-Naiv (CI95) | Cov P10–P90 (Board) |
|---|---:|---:|---|---:|
| Hilden — Netzumsatz 2025 | 29,2 MW | 3,5 % | **+18,9 % [+10,5, +26,7] *** | 84,3 % |
| Herne — Bezug 110/10 kV 2024 | 49,3 MW | 3,9 % | +27,5 % [+17,6, +36,7] * | 81,8 % |
| EVDB — Lastgang NS 2024 | 6,0 MW | 4,8 % | +37,9 % [+30,5, +44,5] * | 85,5 % |

Lesart: „Skill" = Fehlerreduktion gegenüber der Baseline. `*` = signifikant (CI95-Untergrenze > 0).
Die frühere Aussage „Hilden n.s." war ein **Power-Artefakt des 28-Tage-Fensters** — bei 84 Tagen ist der
Vorsprung klar signifikant. Das Board enthält die **feiertagsbewusste Basis** (§4) — sichtbar z. B. an
Hilden +16,4 → +18,9 % gegenüber dem Vor-T48-Stand — aber bewusst **ohne** Online-Residuen-Feedback und
ohne Online-Kalibrierung: deren Zusatzgewinne (+1,6 % bzw. Coverage-Fehler halbiert, §4) sind separat
leakage-sicher belegt und kommen additiv obendrauf. Die Board-Coverage ist daher die **unkalibrierte**
Band-Coverage (Korpus-Mittel 78,9 % bei Soll 80 %).

**Französische Aggregate (Enedis, RTE):** hohe Skills gegen Saisonal-Naiv (+45,2 % / +51,3 % / +37,0 %),
aber gegen **Persistenz** bleiben zwei der drei Reihen nicht signifikant (−5,5 % / +0,4 %; nur eco2mix
+11,5 % sig) und die Band-Coverage ist schwach (57–62 %). Grund: die Regional-Aggregate sind so glatt,
dass Persistenz schwer zu schlagen ist. Das wird offen benannt — es ist kein Verteilnetz-Nachweis,
sondern zeigt die Grenze der Methode bei sehr glatten Reihen. (Quelle: `benchmark_table.md`.)

**Wetter:** Eine leakage-sichere Wetter-Integration brachte **keinen** signifikanten Genauigkeitsgewinn.
Es gibt kein „Wetter-Wunder"; diese Sackgasse ist dokumentiert statt verschwiegen.

---

## 7. §14a-Koordination

Bei einem prognostizierten Netzengpass dimmt die Branchenpraxis alle steuerbaren Einrichtungen pauschal
auf die garantierte Mindestleistung (4,2 kW). NetzPilot rechnet stattdessen die **minimale, faire**
Abregelung:

**Wasserfüll-Optimierung** (`control/optimize.py`): verteilt die nötige Leistungsreduktion so, dass die
Netzgrenze exakt gehalten wird, jede Einrichtung ihre §14a-Mindestleistung behält und die Gesamt-
abregelung **beweisbar minimal** ist (analytisch exakt per Wasserfüllung, kein Solver; verifiziert in
`verify_optimize_heterogen.py`). Wie viel das gegenüber pauschalem Dimmen konkret spart, hängt vom
Engpass ab — die end-to-end gemessene Größenordnung liefert der rollierende Re-Dispatch (nächster Absatz).

**Rollierender Re-Dispatch** (`control/redispatch.py`): statt eines starren Vortagsplans wird der Eingriff
stündlich mit der jeweils aktuellen Prognose nachgeführt — kein unnötiges Dimmen in Stunden, die sich
kurzfristig als unkritisch erweisen. Auf 3 echten Reihen im Mittel **74,3 % weniger Abregelenergie** als
pauschale Dauerdimmung, Netzgrenze und 4,2-kW-Floor in 3/3 Fällen gehalten. **Ehrliche Einschränkung:**
die Engpassschwelle ist hier synthetisch (knapp unter die Tagesspitze gesetzt; echte Schwelle = reale
Netzkapazität im Pilot), und die Prognosebasis ist als statische Day-ahead-P50 gekennzeichnet — der
größere Intraday-Vorteil ist damit noch nicht bewiesen. (Quelle: `redispatch_demo.md`.)

**Heterogene Flotte** (`optimize_setpoints_heterogen`): generalisiert die Optimierung auf einen Mix aus
Wärmepumpe/Wallbox/Speicher mit je eigener Mindestleistung und Gewichtung — minimal, fair und §14a-sicher,
ohne das verifizierte homogene Verhalten zu ändern.

**Audit-Trail:** `service/audit_ledger.py` dokumentiert Redispatch-, Dispatch- und Tarif-Eingriffe als
append-only JSONL-Hash-Kette. Jeder Eintrag enthaelt Grund, Dauer, magnitude_kw, betroffene Limits und
Regelversion; `verify_chain` erkennt nachtraegliche Aenderungen mit Bruchindex. Der druckbare Nachweis
enthaelt den Ketten-Kopf-Hash und optional eine HMAC-SHA256-Signatur. Ehrliche Grenze: Das ist ein
**manipulationssicherer Audit-Trail (Hash-Kette)**, nicht BNetzA-zertifiziert und keine Behauptung
juristischer Rechtssicherheit. Die Diskriminierungsfreiheit entsteht aus der Wasserfuellung; das Ledger
dokumentiert sie nachpruefbar.

NetzPilot berührt dabei nie selbst das Smart-Meter-Gateway: es erzeugt White-Label-Fahrpläne für den aEMT.

---

## 8. Ökonomie: Bilanzkreis-/Ausgleichsenergie

Die naheliegende €-Erzählung wäre „spart X €/Jahr". Eine ehrliche Analyse zeigt: **so einfach ist es
nicht.** Bessere Prognosen senken nur dann Ausgleichsenergiekosten, wenn die vermiedenen Abweichungen mit
teuren Preisstunden zusammenfallen — eine Korrelation, die pro Netz und Preisjahr verschieden ist.

NetzPilot rechnet die **echte Viertelstunden-Abrechnung**: pro Periode der signierte Prognosefehler mal
dem reBAP-minus-Spot-Aufschlag (`eval/bilanzkreis.py`), zerlegt in einen Bias- und einen
Korrelationsterm. Auf langen 2024-Fenstern (≈300 gemessene Tage, leakage-sicher rolling-origin):

| Reihe | realisiert €/Jahr | lineare Näherung €/Jahr | MC-Band P5/P50/P95 | P(spart) |
|---|---:|---:|---|---:|
| Herne 110/10 kV | 1 566 | 29 784 | [−136 724, 1 635, 133 913] | 50,8 % |
| EVDB NS | 43 732 | 16 787 | [15 462, 42 471, 77 632] | 99,9 % |

Bei EVDB ist der Nutzen robust positiv; bei Herne praktisch ein Münzwurf mit großer Varianz. 30-Tage-
Blöcke schwanken bei Herne zwischen −607 000 und +719 000 €/Jahr — d. h. eine kurze Messung kann jede
Zahl liefern. Das **Monte-Carlo-Band** (Tages-Block-Bootstrap, `eval/mc_savings.py`) macht diese
Unsicherheit explizit. (Quelle: `bilanzkreis_demo.md`.)

**Produktkonsequenz:** Der €-Nutzen ist keine feste Versprechung, sondern wird **pro Stadtwerk auf dessen
echten Daten** gerechnet und mit Unsicherheitsband ausgewiesen. Genau diese ehrliche, mandantenspezifische
Rechnung ist der Mehrwert — nicht eine plakative Sparzahl. Weitere Vorbehalte: kein Intraday-Handel
modelliert (die Zahl ist eine obere Schranke der Exposure), und das reBAP-Vorzeichen ist nicht steuerbar
(der Erwartungswert ist bei unverzerrtem, unkorreliertem Fehler ~0 → Nutzen = Downside-Schutz).

---

## 9. Entscheidungswert der Prognose-Unsicherheit (Newsvendor-Prinzip)

Der eigentliche Hebel von NetzPilot ist nicht die Prognose*genauigkeit* an sich, sondern die *Nutzung
der Unsicherheitsverteilung für eine günstigere Entscheidung*. Am klarsten zeigt sich das an der
Bilanzkreis-Nominierung: sind Unter- und Überspeisung **asymmetrisch** bepreist (c_short ≠ c_long), ist
die kostenoptimale Day-ahead-Nominierung nicht der Median (P50), sondern das **τ-Quantil** der
Prognoseverteilung mit τ = c_short / (c_short + c_long). Das ist die Minimalstelle des Pinball-Loss
(Newsvendor-Lehrsatz). Ein rein punktprognose-getriebener Dispatch nominiert P50 und lässt bei
Kostenasymmetrie systematisch Geld liegen.

Belegt auf echter SMARD-Netzlast, leakage-sicher (die Residuen-Quantile jedes Testtags nutzen nur
frühere Tage — dieselbe rolling-origin-Regel wie der Benchmark). Realisierte Kostenreduktion der
τ-Quantil- gegenüber der P50-Nominierung über mehrere Asymmetrie-Verhältnisse:

| c_short / c_long | 1,0 | 1,5 | 2,0 | 3,0 | 5,0 |
|---|---:|---:|---:|---:|---:|
| τ | 0,50 | 0,60 | 0,67 | 0,75 | 0,83 |
| Einsparung vs. P50 | 0,0 % | 2,86 % | 8,14 % | 19,82 % | 37,88 % |

Bei symmetrischer Bepreisung (Verhältnis 1) ist die Einsparung exakt 0 % (Sanity-Check, per Konstruktion
identisch zu P50); mit wachsender Asymmetrie steigt sie monoton. Ein wichtiger, kontraintuitiver Befund:
die prozentuale Einsparung ist **invariant gegen die Fehlergröße** (doppelte Streuung → identische %) und
wird allein von der **Form der Fehlerverteilung** (Schiefe/Tails) **× der Kostenasymmetrie** bestimmt —
„ein verrauschteres Portfolio spart mehr" ist also falsch.

Das ist der theoriegestützte, robuste Teil der €-Geschichte — anders als der korrelationsabhängige,
reihenspezifische Effekt aus §8. **Ehrliche Grenzen:** die absoluten Eurobeträge des Experiments beziehen
sich auf das bundesweite Bilanzvolumen und sind rein illustrativ — **keine NetzPilot-Umsatzzahl**;
übertragbar ist allein die *prozentuale* Aussage. Eine Monetarisierung setzt voraus, dass (a) die
Prognose-Quantile kalibriert sind (bei uns gut auf deutschen DSO-Reihen, schwach auf sehr glatten
Aggregaten) und (b) die reale reBAP-Asymmetrie sowie die Fehler-Form des konkreten Kunden aus einem
Piloten vorliegen. (Quelle: `dispatch_experiment/`, in der Sandbox reproduziert.)

---

## 10. Betrieb: Drift-Erkennung

Damit eine im Backtest kalibrierte Prognose im Dauerbetrieb nicht unbemerkt degradiert (neue Anlagen,
Verhaltens- oder Sensoränderungen), vergleicht NetzPilot die jüngsten Live-Fehler gegen die
Referenzverteilung (`eval/drift.py`): PSI (Industriestandard), KS (verteilungsfrei), Bias- und
MAE-Verhältnis sowie die Coverage der P10/P90-Intervalle. Bei Überschreiten dokumentierter Schwellen wird
„beobachten" bzw. „Drift — neu kalibrieren" gemeldet. Bewusst wird nur **gewarnt, nicht automatisch
neu trainiert** — Drift ist ein Hinweis, kein Beweis; die Entscheidung bleibt beim Betreiber.

---

## 11. Limitationen & Ehrlichkeits-Register

In einem Dokument konsolidiert, weil Glaubwürdigkeit vom offenen Umgang mit Grenzen abhängt:

1. **Kein Live-Pilot.** Alle Lastgänge sind öffentliche DSO-Reihen als Proxy für die Bilanzkreis-Entnahme.
   Die kundeneigene RLM-Gesamtlast kann abweichen. Das ist die zentrale offene Validierung.
2. **Kein Vergleich gegen kommerzielle Profitools.** Keine Behauptung „genauer als KISTERS/Seven2one".
3. **€-Zahlen aus einem Preisjahr (2024)** und mit milder Annualisierung (scale ≈ 1,2). Ein anderes
   Preisregime (z. B. die Volatilität 2022) ist darin nicht enthalten.
4. **§14a-Engpassschwellen sind in den Demos synthetisch.** Die echte Schwelle = reale Netzkapazität im
   Pilot.
5. **Sehr glatte Aggregate** (FR-Regionen) schlägt die Methode nicht gegen Persistenz; die CQR-Coverage
   ist dort schwach.
6. **Wetter** brachte leakage-sicher keinen signifikanten Gewinn.
7. **SLP-Summenlast und signierte Differenzbilanzen** werden nicht als MAPE-Headlines geführt.
8. **Newsvendor-/τ-Quantil-Einsparung (§9)** ist prozentual und theoriegestützt belegt, aber die
   absoluten Eurobeträge sind illustrativ (bundesweites Volumen, kein Umsatz); die Monetarisierung
   hängt an kalibrierten Quantilen plus realer reBAP-Asymmetrie und Fehler-Form aus einem Piloten.
9. **Gefittete Feiertags-Zusatz-Flags (Brückentag, „Vorwoche-war-Feiertag") wurden geprüft und
   verworfen:** wirkungslos bzw. auf Industrie-Last katastrophal überfittet (1–4 Events/Jahr sind nicht
   robust fitbar). Verschifft wurde nur die deterministische, parameterfreie Basis-Reparatur (§4).
10. **lag-7-Residuen-Feedback** (wöchentliche Mean-Reversion, real und signifikant) wurde gemessen, aber
    als marginal/fragil **zurückgestellt** (~+0,7 % Mittel, eine Reihe leicht negativ) — Revisit mit
    Mehrjahresdaten. Feiertags-/Persistenz-Mechanismen helfen Industrie-Reihen generell nicht (≈neutral).

---

## 12. Was ein Pilot beweisen würde

Ein einziges Stadtwerk mit (a) echter RLM-Gesamtlast des Bilanzkreises, (b) echter reBAP-/
Bilanzkreis-Abrechnung und (c) realer Netzkapazität als §14a-Schwelle würde drei Dinge schließen, die
heute offen sind: die Übertragbarkeit der Prognosegüte von öffentlichen DSO-Reihen auf die kundeneigene
Last, den realisierten €-Nutzen aus echter Ausgleichsenergie statt aus einem Proxy, und den
Intraday-Vorteil des rollierenden §14a-Re-Dispatch mit echten, stündlich aktualisierten Prognosebahnen.
Bis dahin gilt die hier dokumentierte, bewusst konservative Lesart.

**Live-Track-Record (seit 2026-06):** Die Infrastruktur für diesen Beweis läuft bereits. Jede ausgegebene
Prognose wird ZUM AUSGABEZEITPUNKT in eine manipulationssichere Hash-Kette geschrieben (gleiche Mechanik
wie der §14a-Audit-Trail) — damit ist **prüfbar, dass jede Prognose vor ihrem Zieltag existierte** (kein
Hindsight, kein Backfill). Eintreffende Ist-Werte werden automatisch abgerechnet (MAE/Bias/Coverage je Tag,
Aggregat, im Cockpit als „Live-Track-Record" mit Ketten-Status). Das ersetzt im Laufenden den Backtest
durch gelebte Prognosen gegen gelebte Realität — immer mit n Tagen + Pending ausgewiesen, ohne
Zeitraum-Cherry-Picking. Ehrliche Grenze: Hash-Kette belegt Unverändertheit seit Aufzeichnung, keine
juristische Zertifizierung. (`netzpilot/service/forecast_store.py`, `scripts/verify_forecast_store.py`.)

---

*Reproduzierbarkeit: Alle genannten Zahlen entstammen `scripts/benchmark_suite.py`,
`scripts/build_proof_pack.py`, `data_cache/benchmark/*` und den Verifikationsskripten `scripts/verify_*.py`.
Die Kern-Engines (Prognose-Korrektur, §14a-Optimierung, Bilanzkreis-Abrechnung, MC-Band, Drift) sind durch
eigenständige `verify_*`-Skripte abgesichert.*
