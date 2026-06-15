# NetzPilot — Ergebnisbericht (Stand 2026-05-30)

*Prognose- und §14a-Koordinationssoftware für kleine Stadtwerke. Dieser Bericht ist bewusst
ehrlich gehalten: jede Zahl ist mit ihrem Verifikationsstatus versehen, und die Grenzen sind
genauso prominent wie die Erfolge. Geschrieben für eine Zweitprüfung / einen kritischen Reviewer.*

## Verifikationskonvention
Jede Zahl ist eine von zwei Klassen:

- **[S] in der Sandbox von Claude selbst nachgerechnet** (numpy/pandas/stdlib, auf den echten Dateien) —
  voll reproduziert.
- **[H] Host-Lauf (Codex)**, von Claude aus den Ergebnis-/Cache-Dateien gelesen und auf Plausibilität,
  Leakage-Sicherheit und Provenienz geprüft — aber nicht selbst neu trainiert (Sandbox hat kein
  lightgbm/pvlib/Internet).

Nichts in diesem Bericht ist „geschätzt" oder geschönt. Wo ein Ergebnis schwach oder nur ein oberer
Rand ist, steht das so da.

---

## 1. Zusammenfassung (TL;DR)

NetzPilot besteht aus einer **leakage-sicheren Day-ahead-Prognose** (Last, Residuallast, Erzeugung) mit
**kalibrierten Unsicherheitsbändern** und einem **§14a-Steuerkreis-Prototyp**.

**Belastbar:**
- Nationale Last (SMARD, 2 Jahre): MAPE **3,36 %**, Skill **+55,8 %** vs. saisonal-naiv. **[H]**
- Erstes echtes deutsches Stadtwerk-Profil (gemessen): Hilden **Netzumsatz** 2025, MAPE **4,36 %**, Skill
  **+16,5 %** vs. saisonal-naiv. (Die SLP-Summenlast mit 1,76 %/+34 % ist ein synthetisches Profil und
  zählt NICHT als Prognosegüte-Beleg.) **[H]**
- 50 Stadt-Profile: MAPE ~**6,7 %**, Skill **+3,9 %** vs. saisonal-naiv (positiv in **50/50**). **[S]**
- Unsicherheitsbänder (CQR) sind **stadtweit kalibriert**: 80-/90-%-Coverage im Mittel 80,1 / 90,2 %,
  jede der 50 Städte innerhalb ±5 pp. **[S]**
- §14a-Steuerkreis end-to-end demonstriert (illegale Fahrpläne werden abgelehnt). **[S]**

**Ehrlich begrenzt:**
- Die 50 Stadt-CSVs sind **modellierte/synthetische Profile, keine gemessenen Stadtwerke-Daten**
  (drei unabhängige Belege, s. §3).
- Der Vorsprung gegen saisonal-naiv ist auf Stadt-/Kleinlast **dünn (~+4 %)** — die ehrliche Realität
  kleiner Lasten.
- Der früher berichtete **„+17,9 % durch Wetter" (T10) war Leakage** (bewiesen, s. §6). Leakage-sicher
  ist der Wetter-Lift **≈ 0** und nur auf wärmepumpen-/temperaturgekoppelter Last leicht positiv.
- Noch keine kundeneigenen Pilotdaten, keine realen reBAP-Kosten, kein echtes SMGW.

**Kernaussage:** Der belastbare Wert ist die **Prognose-Wedge + kalibrierte Bänder + §14a-Koordination**,
nicht ein wetterbasierter Genauigkeits-Moat. Der nächste echte Schritt sind **kundeneigene Pilotdaten mit
realer reBAP-/Bilanzkreis-Abrechnung**.

---

## 2. Was NetzPilot ist

Eine schlanke Prognose-/Berechnungs-Engine: NetzPilot rechnet Day-ahead-Prognosen + Fahrpläne und gibt
diese per REST/JSON an einen **zertifizierten White-Label-aEMT** (z. B. GWAdriga/GreenPocket/PPC) weiter,
der die Steuerung über CLS/SMGW/HEMS ausführt. NetzPilot selbst fasst **kein** SMGW an — das ist
regulatorisch und sicherheitstechnisch die richtige Trennung.

Strategie: **„Land mit der Prognose-Wedge, expandiere in §14a-/DER-Koordination."**

---

## 3. Datenbasis & Provenienz (ehrlich)

| Datensatz | Art | Verwendung | Provenienz |
|---|---|---|---|
| SMARD Netzlast + PV/Wind, 2 J | **echt** (ÜNB) | nationale Last/Residuallast/Erzeugung | öffentlich, real |
| Stadtwerke Hilden 2025 | **echt** (öffentliches Stadtwerk) | erstes echtes deutsches Lastgang-Ergebnis (SLP/Netzumsatz/DBA) | öffentlich, §12/§23c-Veröffentlichungspflicht |
| EAM Netz 2024 | **echt** (öffentlich) | Differenzbilanzierung BG1/BG2/BG3 als Robustheitscheck | öffentlich, §12 Abs. 3 StromNZV |
| `training_cities/` (50 CSV) | **modelliert/synthetisch** | Stadt-Skalierung der Pipeline | s. u. — KEINE Messdaten |
| OPSD/CoSSMic Konstanz | **echt** (11 Haushalte) | Kleinlast-Proxy (T9/T10) | real, aber 2013–2017 |
| HEAPO (Kanton Zürich) | **echt** (1.408 WP-Haushalte) | leakage-sicherer Wettertest (T12) | real, 2018–2024 |

**Warum die 50 Stadt-CSVs modelliert sind — drei unabhängige Belege [S]:**
1. **Identische Mittelwerte:** 12 von 50 Städten teilen einen Ø-Lastwert exakt auf 0,1 MW — fünf Städte
   liegen alle bei **genau 69,6 MW** (Leverkusen, Ludwigshafen, Oldenburg, Osnabrück, Solingen). Reale
   Messlast kann nicht so zusammenfallen.
2. **Cross-City-Korrelation** der normierten Tagesform r ≈ 0,89–0,98 (Münster–Berlin 0,98).
3. **Dateinamen** mehrerer Geschwisterdateien tragen „_mock"; zusätzlich zeigt T11 (§6), dass echtes
   lokales Wetter diese Last nicht vorhersagt → **keine echte Wetter-Kopplung** = Template-Charakter.

→ Sehr nützlich zum Entwickeln/Kalibrieren der Pipeline, **aber kein Beleg für Kunde/Gutachter**.

---

## 4. Prognose-Ergebnisse

### 4.1 Nationale Last & Residuallast (SMARD, 2-Jahres-Backtest) [H]
| Ziel | MAE | MAPE | Skill vs. saisonal-naiv |
|---|---|---|---|
| Last | 1758,8 MW | 3,36 % | **+55,8 %** |
| Residuallast (direkt) | 2707,1 MW | — | +52,1 % |
| Residuallast (physik. PV/Wind-Pfad, pvlib/windpowerlib + Bias) | 2796,4 MW | — | +50,5 % |

Reproduktion v1 (12-Wochen, ridge-korrigiert) **[S]**: MAE 1411,4 MW (saisonal-naiv 1472,4; Persistenz
3816,2), Coverage 81,5 %.

### 4.2 50 Stadt-Profile (rolling-origin, 14 Testtage je Stadt) [S]
| Metrik | Median | Range | positiv |
|---|---|---|---|
| MAPE (Last) | 6,67 % | 6,15–7,73 % | — |
| Skill vs. saisonal-naiv (Last) | **+3,9 %** | +1,2…+8,3 % | **50/50** |
| Skill vs. saisonal-naiv (Residuallast) | **+3,9 %** | +1,1…+16,3 % | **50/50** |
| Skill vs. Persistenz | +54,8 % | +49,8…+58,9 % | 50/50 |

MAPE liegt erwartungsgemäß zwischen national (3,4 %) und einem groben Kleinlast-Proxy (~15 %). Der
Vorsprung gegen saisonal-naiv ist **dünn, aber konsistent** (jede der 50 Städte schlägt beide Baselines).

### 4.3 Statistische Signifikanz (paired Block-Bootstrap, Block = Tag) [S]
- **National v1** (12 Wochen): vs. saisonal-naiv +4,1 % **nicht signifikant** (CI95 [−13,1; +17,5]);
  vs. Persistenz +63,0 % signifikant. Erst T3 (2 J + Wetter, +55,8 %) ist statistisch belastbar.
- **Pro Stadt** (n = 14 Testtage, 10/50 gerechnet): vs. Persistenz **10/10 signifikant**; vs. saisonal-naiv
  **4/10** — genau die Städte mit Skill ≥ 5 %. Auf 14 Tagen ist ein Einzel-Vorsprung von ~+2–4 % nicht von
  0 unterscheidbar (CI-Halbweite ~±5 pp).
- **Aggregat:** 50/50 Städte positiv → Vorzeichentest p ≈ 10⁻¹⁵ (Caveat: Städte korreliert, effektives
  N < 50). **Die belastbare Aussage ist das Aggregat, nicht die Einzelstadt.**

### 4.4 Kalibrierung (CQR) [S national: H]
- National **[H]**: Last 81,2 / 87,6 %, Residuallast 82,6 / 90,9 % (Soll 80/90 %).
- 50 Städte **[S]**: 80-%-Coverage Median 79,8 % (Mittel 80,1 %), 90-%-Coverage Median 90,2 %;
  **jede der 50 Städte innerhalb ±5 pp** des Sollwerts. Die Bänder sind stadtweit belastbar, nicht nur
  national. (MAPIE/EnbPI wurde getestet, blieb unterkalibriert → belastbar ist rolling CQR.)

### 4.5 Erstes echtes Stadtwerk (Hilden, öffentlich) [H]
T14 hat sechs öffentlich herunterladbare Volljahres-Lastgänge echter deutscher Netzbetreiber in
`data_cache/real/` gezogen und mit `scripts/pilot_in_a_box.py` leakage-sicher ausgewertet (364/365 Tage
Historie, 28 rollierende Testtage, saisonal-naiv und Persistenz als Baselines).

| Datensatz | Art | Ø Last | MAE | MAPE | Skill vs. S-Naiv | Skill vs. Persistenz | CQR 80/90 |
|---|---|---:|---:|---:|---:|---:|---:|
| **Hilden Netzumsatz 2025** | **gemessener Netzdurchsatz (echte Last)** | 29,2 MW | 1,3 MW | **4,36 %** | **+16,5 %** | +40,2 % | 74,4 / 84,5 % |
| Hilden DBA 2025 | signierte Differenzbilanz | 0,31 MW | 0,5 MW | 190 %* | +24,5 % | +9,8 % | 72,8 / 83,8 % |
| EAM BG1 2024 | signierte Differenzbilanz | -2,9 MW | 16,1 MW | 226 %* | +19,9 % | +17,5 % | 75,0 / 85,9 % |
| EAM BG2 2024 | signierte Differenzbilanz | 0,22 MW | 0,4 MW | 168 %* | +19,8 % | +20,7 % | 76,2 / 87,6 % |
| EAM BG3 2024 | signierte Differenzbilanz | 1,47 MW | 1,7 MW | 205 %* | +3,2 % | +20,0 % | 73,8 / 84,2 % |
| Hilden SLP-Summenlast 2025 † | **synthetisches SLP-Profil** | 12,0 MW | 0,2 MW | 1,76 % | +34,0 % | +48,5 % | 77,4 / 84,2 % |

`*` MAPE ist bei signierten Differenzbilanz-/DBA-Reihen mit Nulldurchgängen wenig aussagekräftig; dort sind
MAE, MASE, Skill gegen Baselines und Coverage die relevante Lesart.
`†` **SLP-Summenlast ist KEINE gemessene Nachfrage**, sondern die Summe der BDEW-Standardlastprofile
(Profil × Jahresverbrauch × Dynamisierung) — quasi-deterministisch. Die 1,76 % MAPE liegt sogar *unter*
der nationalen gemessenen Last (3,36 %) — das verräterische Zeichen für ein „zu leichtes" synthetisches
Profil. Nur Funktionsnachweis des Loaders, **kein** Beleg echter Prognosegüte.

**Headline (ehrlich):** Auf dem **gemessenen** Stadtwerke-Hilden-**Netzumsatz** 2025 erreicht NetzPilot
**MAPE 4,36 %** und schlägt saisonal-naiv um **+16,5 %** (Persistenz +40,2 %), mit kalibrierten Bändern
(Coverage ~74/85 %). Das ist der erste belastbare Nachweis auf **echter, gemessener** deutscher Stadtwerk-
Last — plausibel knapp über der glatten nationalen Last (3,36 %), wie für ein kleineres Netz erwartbar.
Die **Differenzbilanz (DBA)** ist die bilanzkreis-/reBAP-relevante Größe (signiert → Skill/MAE statt MAPE;
Hilden DBA +24,5 % vs. saisonal-naiv). Caveat: Netzumsatz/DBA sind veröffentlichte Durchsatz-/Restprofile
(§12/§23c), nicht die volle kundeneigene RLM-Netzlast und nicht mit realen Bilanzkreis-Kosten verknüpft;
die Euro-Schätzung bleibt ein transparenter Proxy bis zur realen reBAP-Abrechnung.

---

## 5. §14a-Steuerkreis (Prototyp) [S]
End-to-end in der Sandbox demonstriert: NetzPilot → Mock-aEMT (stdlib `http.server`) → simuliertes HEMS →
Wallbox. Im Engpassfenster wird die Wallbox 11 → **4,2 kW** gedrosselt; **illegale Fahrpläne (< 4,2 kW
§14a-Mindestleistung) werden vom aEMT mit HTTP 422 abgelehnt**. Das ist ein greifbarer Moat-Baustein —
ohne echtes SMGW/Zertifizierung, klar als Simulation gekennzeichnet.

---

## 6. Der Wetter-Lift — die ehrliche Geschichte (T10 → T11 → T12)

Dieser Abschnitt ist das wichtigste Ehrlichkeits-Kapitel des Projekts.

1. **T10 (alt): „+17,9 % durch lokales Wetter"** auf dem Konstanz-Kleinlastproxy — sah nach dem großen
   Hebel aus. **[H]**
2. **Leakage-Beweis [S/H]:** Der T10-Lauf nutzte **2017er** Konstanz-Wetter über den „Historical-Forecast"-
   Endpoint — aber dessen Archiv beginnt erst ~**Juli 2022**. Der Recheck zeigt: das 2017-Wetter ist
   **bit-identisch** zum Open-Meteo-Archiv (ERA5): `mean_abs_diff = 0,0`, `max_abs_diff = 0,0`, n = 168 je
   Variable. → Es war **Ist-/Reanalyse-Wetter (perfect foresight) = Leakage**. **Der +17,9 % ist ein
   optimistischer oberer Rand, kein leakage-sicherer Forecast-Beleg.**
3. **T11 (50 synth. Städte) [H, von Claude verifiziert]:** echtes lokales Forecast-Wetter hebt den Skill
   **nicht** (Median −0,5 pp; positiv 11/50; signifikant 1/50). Kein Pipeline-Fehler — sondern Beleg, dass
   die synthetischen Profile keine echte Wetter-Kopplung enthalten (drittes Provenienz-Signal).
4. **T12 (HEAPO, echte CH-Wärmepumpen, leakage-sicher ≥ 2022-07) [H, von Claude verifiziert]:** Median-Lift
   **−0,15 pp** über 26 Aggregationsrecords; signifikant besser nur 2/26. **Aber:** die Wärmepumpen-
   **Einzellast** trendet leicht **positiv** (+0,3…+1,6 pp, plausibel: Heizstrom ∝ Temperatur). Offene
   Verfeinerung (T13): per-MeteoSwiss-Station-Wetter statt einem Zürich-Punkt; zweiter echter Datensatz.

5. **T13 (HEAPO per-MeteoSwiss-Station-Wetter, leakage-sicher) [H, verifiziert]:** 5/8 Stationen exakt zu
   MeteoSwiss gematcht (bit-identische Temperatur); Wärmepumpen-Lift steigt auf **+0,6 pp** (Median), bleibt
   aber **nicht signifikant** (p=0,105; 5/23). Total-Last ~0. Per-Station schlägt den Zürich-Proxy um +0,2 pp.
   (Offenbach-Zweitdatensatz: data.zip 103 GB, Download blockiert.)

**Schluss:** Leakage-sicher ist der Wetter-Lift **klein (≈ 0–1 pp)** und allenfalls auf temperatur-
gekoppelter Last relevant — **kein Produkt-Moat**. Der Wert von NetzPilot liegt in der Prognose-Wedge,
der Kalibrierung und der §14a-Koordination.

---

## 7. Ehrliche Grenzen
- **Stadt-Daten modelliert**, nicht gemessen (§3) → Stadt-Ergebnisse validieren die Pipeline, nicht reale
  Performance.
- **Noch keine kundeneigenen Pilotdaten** (15-min-Lastgang, Kundensegmente) und **keine realen reBAP-/Bilanzkreis-
  Kosten** → Hilden/EAM belegen öffentliche echte Lastgänge, aber noch keine belastbare kundenspezifische
  €-Nutzenrechnung. Euro-Werte bleiben Proxy.
- **Wetter-Lift** leakage-sicher klein (§6); echter Beleg braucht reale, temperaturgekoppelte 2022+-Daten.
- **Physikalischer PV/Wind-Pfad** ist MVP (kein echtes Anlagenregister, nur Wetterproxy + Bias-Korrektur).
- **Kein echtes SMGW**; produktiv nur über zertifizierten White-Label-aEMT.
- **Open-Meteo-Lizenz** (kommerziell) vor Produktangebot klären.

---

## 8. Methodik (leakage-sicher by design)
- **Rolling-origin-Backtest**, Modell sieht nie Zukunft; saisonal-naiv (`load(d−7)`) als Korrekturziel.
- **Baselines Pflicht:** saisonal-naiv **und** Persistenz, immer mitberichtet.
- **DST-robust:** Lokaltag-Aufbereitung (`to_daily_local`) statt UTC-Reshape (sonst Stundenversatz über die
  Sommerzeit).
- **Unsicherheit:** Conformalized Quantile Regression (CQR), Coverage gemessen, nicht angenommen.
- **Signifikanz:** paired Block-Bootstrap, Block = ganzer Tag.
- **Wetter = Forecast:** Open-Meteo **Historical-Forecast** (Archiv ab ~2022-07); ältere Perioden sind
  reanalysebasiert und werden klar als „perfect-foresight upper bound" gelabelt, nie vermischt.

---

## 9. Reproduktion
`scripts/run_backtest.py` (v1) · `run_t8_cqr.py` (CQR national) · `eval_v1_significance.py`,
`eval_t9_significance.py` (Signifikanz) · `run_t10_small_utility.py` (T10) · `run_t11_city_weather.py`
(T11) · `run_t12_weather_lift.py` (T12 + T10-Leakage-Recheck) · `eval_cities_all_load.py`,
`eval_cities_all_residual.py`, `eval_cities_all_cqr.py`, `eval_cities_significance.py` (50-Städte) ·
`pilot_in_a_box.py` (T14 echte öffentliche Lastgänge) · `demo_control_loop.py` (§14a). Detailbefunde: `Notizen/_cities_finding.md`,
`Notizen/_t10_weather_leakage_check.md`, `Notizen/_t12_verification.md`.

---

## 10. Nächste Schritte
1. **Kundeneigene Pilotdaten** priorisieren (15-min-Last, Erzeugung, Kundensegmente, reale reBAP-Kosten) —
   Hilden zeigt: auf echten öffentlichen Stadtwerk-Daten funktioniert die Wedge; der eigentliche Engpass ist
   jetzt die kundenspezifische Nutzenrechnung.
2. **Outreach mit Hilden-Netzumsatz-Headline** starten: 4,36 % MAPE und +16,5 % vs. saisonal-naiv auf
   echtem gemessenem Netzdurchsatz, plus ehrliches Caveat (Durchsatz-/Restprofil statt kompletter RLM-Netzlast).
3. **Erzeugungsmodell** schärfen (regionale Kapazitäten, mehrere Wetterpunkte).
4. **§14a** produktiv über zertifizierten White-Label-aEMT anbinden.

---

## 11. Quellen / Datensätze
SMARD (ÜNB-Netzlast, PV/Wind) · Open-Meteo Historical-Forecast (Archiv ab ~2022-07) ·
OPSD Household Data / CoSSMic Konstanz (`data.open-power-system-data.org/household_data/`) ·
HEAPO (Zenodo 10.5281/zenodo.15056919, arXiv 2503.16993) · §14a EnWG / BSI TR-03109-5.
Stadtwerke Hilden Veröffentlichungspflichten Strom (SLP-Summenlast, Netzumsatz, DBA 2025) ·
EAM Netz Veröffentlichungspflichten Strom / §12 Abs. 3 StromNZV (Differenzbilanzierung BG1-BG3 2024).
