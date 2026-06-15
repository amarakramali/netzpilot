# NetzPilot — Entwicklungs-Historie & Detailergebnisse

Leakage-sichere Last-, Erzeugungs- und Residuallastprognose fuer kleine deutsche Stadtwerke.

> Dies ist die ausfuehrliche Entwicklungs-Dokumentation (Aufgaben T2–T44, alle Messwerte,
> Artefakt-Pfade). Fuer den kompakten Ueberblick siehe **[README.md](README.md)**.

> **Privates, proprietäres Projekt.** Siehe [LICENSE](LICENSE) — alle Rechte vorbehalten,
> keine Nutzung ohne ausdrückliche Genehmigung.

## Aktueller Funktionsstand (Juni 2026)

Kurzüberblick über die Kernfähigkeiten; die detaillierte Entwicklungs-Historie steht weiter unten (T2–T44) und in `Notizen/`.

- **Day-ahead-Prognose** mit kalibrierten P10/P50/P90-Bändern (rolling-origin, leakage-sicher; Pflicht-Baseline Saisonal-Naiv). Auf 46 echten DSO-Reihen in 44 signifikant besser als die Vorwochen-Regel.
- **Mehrtages-Horizont** D+1…D+3 (ein Fit, rekursiv) und **Intraday-Update** (Resttag-Korrektur aus den heutigen Ist-Werten).
- **§14a-EnWG-Koordination:** faire, minimale Abregelung (Water-Filling), rollierender Re-Dispatch, zeitvariables Netzentgelt (Modul 3), VPP-Pool.
- **§14a-Compliance:** Monats-Meldebogen (VNBdigital-Pflichtfelder) + bedarfsnormalisierter Diskriminierungsfreiheits-Nachweis, aus einem hash-verketteten Eingriffs-Ledger.
- **Bilanzkreis-Ökonomie:** realisierte reBAP-/Spot-Abrechnung mit Unsicherheitsband statt Plakatzahl.
- **Daten-Eingang:** robuster CSV-/Excel-Loader **und MSCONS-Leser** (EDIFACT-Lastgang aus der Marktkommunikation, read-only).
- **Blind-Challenge:** Board-identischer Sofort-Backtest auf fremden Daten (~15 s), persistiert nichts.
- **Bedienung:** Single-File-Cockpit (`/cockpit`), druckbarer Ergebnisbericht mit Live-Track-Record, täglicher SMARD-Live-Lauf.

Konsolidierter Selbsttest: `python scripts/run_all_checks.py` (alle Unit-/Leakage-/Integrationstests + UI-Harness).

## Starten (ein Doppelklick) — echte Software mit Oberflaeche
- **Windows:** Doppelklick auf **`Start_NetzPilot.bat`**
- **macOS/Linux:** `./start_netzpilot.sh`

Der Starter richtet beim ersten Mal automatisch alles ein (virtuelle Umgebung + Pakete), startet das
**echte Python-Backend** (`netzpilot/service`) und oeffnet den Browser auf <http://127.0.0.1:8000/>.
Dort: CSV-Lastgang hochladen -> Day-ahead-Prognose mit kalibrierten Baendern + §14a-Fahrplan, gerechnet
von der echten Engine (keine Browser-Nachbildung). Voraussetzung: Python 3.10+ installiert.

Die Dateien `NetzPilot_Tool.html` / `NetzPilot_Cockpit.html` sind nur Offline-Schaufenster ohne Backend.

## Schnellstart (manuell / Entwicklung)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

.\.venv\Scripts\python.exe scripts\run_backtest.py --data "prognose_engine_v1\data\wk*.json"
.\.venv\Scripts\python.exe scripts\build_data_cache.py --start 2022-01-01 --end 2024-01-01 --cache-dir data_cache
.\.venv\Scripts\python.exe scripts\run_t3_lightgbm.py
.\.venv\Scripts\python.exe scripts\run_t8_cqr.py
.\.venv\Scripts\python.exe scripts\run_t10_small_utility.py
.\.venv\Scripts\python.exe scripts\run_t13_weather_lift.py
.\.venv\Scripts\python.exe scripts\pilot_in_a_box.py --csv data_cache\real\SLP-Summenlast-Lastgang-2025.csv --ts-col Text --load-col Reihe1 --unit kW --region NW --name "Stadtwerke Hilden SLP-Summenlast 2025" --out data_cache\pilot\hilden_slp_2025 --keep-days 365 --n-test 28
.\.venv\Scripts\python.exe scripts\build_report.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Wichtigste Ergebnisse
- v1 reproduziert: MAE 1411.4 MW, MAPE 2.56 %, Coverage 81.5 % auf dem 12-Wochen-Sample.
- T3 Last, 2-Jahres-Cache: MAE 1758.8 MW, MAPE 3.36 %, Skill +55.8 % vs. saisonal-naiv.
- T4 Residuallast direkt: MAE 2707.1 MW, Skill +52.1 % vs. saisonal-naiv.
- T4 physikalisch: pvlib/windpowerlib-Erzeugungsprognose + Bias-Korrektur, residual MAE 2796.4 MW, Skill +50.5 % vs. saisonal-naiv.
- T8 CQR: Coverage im Zielbereich, Last 81.2 % / 87.6 %, Residuallast 82.6 % / 90.9 % fuer 80 % / 90 % Intervalle.
- T9 Klein-Stadtwerk-Proxy: OPSD/CoSSMic-Konstanz, MAPE ca. 15 %, Skill nur +4-6 % vs. saisonal-naiv; deutlich haerter als nationale Last.
- T9 Signifikanz: +5.8 % vs. saisonal-naiv ist auf 28 Test-Tagen nicht signifikant; vs. Persistenz signifikant.
- T10 Klein-Stadtwerk verbessert: lokale Wetterfeatures + Small-Load-Lags, MAE 4.3 MW, MAPE 12.50 %, Skill +17.9 % vs. saisonal-naiv. Nach T12-Recheck ist diese Zahl ein perfect-foresight Upper Bound, nicht leakage-sicher.
- T11 Stadt-Wetter-Pipeline: 50 synthetische Stadtprofile mit lokalen Open-Meteo Historical-Forecast-Features. Median-Lift vs. cached no-weather -0.50 pp; nur 1/50 Staedte signifikant besser. Coverage-Median 80.55 % / 90.20 %. Ergebnis validiert die Pipeline, nicht reale Performance.
- T12 Wetter-Lift real/leakage-sicher: HEAPO ab 2022-07, 26 Aggregationsrecords. Median-Lift durch lokale Historical-Forecast-Wetterfeatures -0.15 pp; Heatpump-Subset +0.30 pp, Total-Subset -0.50 pp. T10/Konstanz-2017 ist bitgleich mit Archive-Wetter.
- T13 HEAPO per-Station: 52 leakage-sichere Records mit exakt zugeordneten MeteoSwiss-Stationskoordinaten fuer 91.1 % der HEAPO-Haushalte. Heatpump-Median-Lift +0.60 pp, aber nicht signifikant (p=0.105); Total 0.00 pp. Wetter ist kein Produkt-Moat, sondern hoechstens kleiner Bonus fuer temperaturgekoppelte Last.
- T14 erstes echtes deutsches Stadtwerk: Stadtwerke Hilden SLP-Summenlast 2025, MAPE 1.76 %, Skill +34.0 % vs. saisonal-naiv; Netzumsatz 2025 MAPE 4.36 %, Skill +16.5 %. EAM/DBA-Differenzlasten sind signiert, dort MAPE wenig aussagekraeftig, Skill bleibt positiv.
- T15 Outreach-Hooks: fuenf weitere oeffentliche DSO-Lastgaenge gerechnet. Beste neue Hooks: EVDB NS 2024 mit MAPE 4.2 %, Skill +38.7 % vs. saisonal-naiv; Stadtwerke Herne Bezug vorgelagerte Ebene 2024 mit MAPE 4.0 %, Skill +26.7 %. Versandfertige Bulletsets: `outreach/*_bullets.md`.
- T16 zweiter 2022+-Wettercheck: Offenbach/Dryad bleibt per Download 401/403 blockiert; Fallback auf reale T15-DSO-Netzlasten. Lokales Historical-Forecast-Wetter liefert auf glatter Netzlast keinen robusten Lift: Median -2.4 pp, 0/3 signifikant.
- T17 Mehrspalten-Guard: `pilot_in_a_box.py` bricht bei mehreren plausiblen Lastspalten ohne `--load-col` ab und bietet `--list-columns`. Alle Outreach-Zahlen wurden mit explizit gepinnter Spalte/Ebene neu gerechnet; Herne 110/10kV reproduziert +26.7 %, 10kV waere nur +11.5 %.
- T18 zweite Outreach-Welle: Stadtwerke Neuruppin, NG Bitterfeld-Wolfen und Stadtwerke Waren aus offiziellen Quellen geladen, mit expliziter Spalte/Ebene gerechnet und per JS-Kern quergecheckt. Neue versandfertige Hooks: Bitterfeld-Wolfen NS/MS-NS, Waren Bezug vNB/NA MS, Neuruppin MS.
- T19 echte reBAP-Preise: Netztransparenz-Export 2024 normalisiert (`data_cache/real/rebap_2024.csv`, 35136 Viertelstunden). `pilot_in_a_box.py --rebap-csv` und der Service geben jetzt Median + P25-P75-EUR-Band aus; UI zeigt eine reBAP-Einsparungskachel. Median |reBAP| 2024: 99.35 EUR/MWh.
- T20 reBAP-Spot-Korrektur: Spot-DA DE-LU 2024 via Energy-Charts/SMARD geladen (`data_cache/real/spot_da_2024.csv`) und QH-aligned an reBAP gejoint. Headline im Dienst/UI ist jetzt die erwartete Einsparung ueber den signierten Mittel-Aufschlag `mean(reBAP-Spot)` (2024: 7.26 EUR/MWh); `|reBAP-Spot|` ist Risiko-/Stressband (Median 58.08 EUR/MWh), `|reBAP|` nur Upper Bound (Median 99.35 EUR/MWh).
- T28-T29 Daten-Moat: Korpus auf 46 valide öffentliche DSO-/Regionallast-Reihen gehoben (`data_cache/real/corpus_index.json`): 43 DE-Reihen, 2 FR-Enedis-DSO-Aggregate und 1 FR-éCO2mix-TSO-Regionallast. Korrelations-Dedup clustert zu 38 unabhängigen Netz-/Regionalclustern (`data_cache/benchmark/network_independence.md`). Pool-Prior: 38 DSO-Reihen (`n_houses=38`; éCO2mix ausgeschlossen). Erweiterter Benchmark: 46/46 ok, 35 signifikant besser als Saisonal-Naiv; Enedis FR +47.0 % und +51.9 % vs. S-Naiv.
- T30 internationale Methoden-Demo: separate ENTSO-E-Power-Stats-Nationallast für DE/NL/AT/CH/FR unter `data_cache/intl/`, strikt nicht im DSO-Korpus. Ergebnis: Saisonal-Naiv in 5/5 Ländern signifikant geschlagen; nationale Aggregatlast, kein Verteilnetz-Beweis.
- T31/T32 Paragraph-14a-Upgrade: Dienst liefert optional `out["redispatch"]` mit
  `forecast_basis="day_ahead_p50_static"`; Demo auf 3 echten DSO-Reihen spart im Mittel 74.3 %
  Abregelenergie gegenueber pauschaler Dauerdimmung bei gehaltener Netzgrenze. Heterogene steuVE-
  Flotten koennen je Anlage `floor_kw` und `weight` nutzen; der homogene Altpfad bleibt unveraendert.
- T33-T35 Bilanzkreis-Settlement: realisierte reBAP+Spot-Abrechnung aus rolling-origin Forecasts statt
  nur linearer Naeherung. T35 ersetzt die 14-Tage-Hochrechnung durch lange 2024-Fenster: Herne
  303 Tage, scale 1.205, realisiert 1.566 EUR/Jahr vs. linear 29.784 EUR/Jahr
  (`prob_positive=50.8%`); EVDB NS 305 Tage, scale 1.197, realisiert 43.732 EUR/Jahr
  vs. linear 16.787 EUR/Jahr (`prob_positive=99.9%`).
- T36 Drift-Monitoring: `run_forecast(..., drift_monitoring=True)` liefert additiv `out["drift"]`
  mit PSI/KS/Bias/MAE-Drift, CQR-Coverage und `needs_recalibration`; Referenz-/Recent-Residuen werden
  versioniert persistiert. Demo: Herne real stabil, synthetischer Bias/Scale-Fall drift.
- T37 zeitvariables Netzentgelt: `run_forecast(..., grid_fee_eur_per_kwh=[...],
  tariff_energy_kwh=..., tariff_p_max_kw=...)` liefert additiv `out["tariff_schedule"]`; Redispatch-Caps
  sind harte Netzgrenze. Demo: 40-kWh/11-kW-Abendlast spart im illustrativen Profil 5.06 EUR, mit Cap
  noch 0.92 EUR.
- T38 Quantil-Dispatch: `run_forecast(..., dispatch_plan_enabled=True, ...)` liefert additiv
  `out["dispatch_plan"]`: Netzgrenze halten, steuVE-Budget guenstig platzieren, Bilanzkreis per
  Newsvendor-Quantil nominieren. Demo Herne: grid_safe, Coverage 82.1 %, illustrative
  Newsvendor-Ersparnis 537.50 EUR.
- T39 Input-Validierung + Asset-Overload: `run_forecast(..., validate_input=True)` liefert
  `out["input_validation"]` mit quality_score/Issues und nutzt ersetzbare cleaned-Werte vor der Engine.
  Optional `asset_rating_kw` liefert `out["overload"]` und `out["hosting_capacity"]` fuer ein einzelnes
  Netzasset. Demo Herne: angenommene Grenze 60.70 MW, 7 Risikostunden, erwartete Ueberlast 2649.1 kWh,
  Hosting-Capacity 0.0 kW; Validate-Demo erkennt injizierte Defekte und meldet Frozen-Phasen nur.
- T40 CVaR-Dispatch + IEC-Thermik: `dispatch_risk_beta>0` ersetzt optional die Newsvendor-Nominierung
  durch eine CVaR-risk-averse Nominierung und weist Expected/CVaR-Kosten nebeneinander aus. Optional
  `thermal_rating_kw` liefert IEC-60076-7-artige Hotspot- und Lebensdauer-Risiken fuer ein Einzelasset.
  Demo Neuruppin: angenommene 9.265-kW-Trafo-Bemessung, 3 thermische Risikostunden, max.
  Hotspot-Ueberschreitungs-Wkt 7.1 %, erwarteter Lebensdauerverbrauch 78.689 h. EVT bleibt optional
  research-grade und wurde nicht ueberstuerzt eingebaut.
- T41 EEBUS-LPC + Mehr-/Mindermengen: `submit_to_aemt=True, aemt_adapter="eebus_lpc"` liefert
  `out["fahrplan_lpc"]` mit EEBUS-LPC-Payload (Transport/SMGW extern). Optional
  `mmm_price_eur_mwh` liefert `out["mmm"]` aus dem rolling-origin Forecast-vs.-Ist-Vergleich.
  Demo Hilden: LPC `status=MAPPED`, 1 Limit, 4200 W Verbrauchsgrenze/Failsafe; MMM reduziert das
  absolute Volumen gegen Saisonal-Naiv um 222.0388 MWh (bei Demo-Annahme 60 EUR/MWh).
- T42 VPP-Pool + operatives Cockpit: `pool_assets` + `pool_shared_cap_kw` liefert
  `out["pool_dispatch"]` fuer faire Periodenkappung mehrerer steuVE unter einer gemeinsamen Grenze.
  `rating_kw` fuehrt im Cockpit eine einzige Netzgrenze fuer Redispatch/Dispatch und Asset-Risiko.
  Demo Hilden: 3 Pool-Assets, Engpassstunde 24 kW, Limits 8/8/8 kW, grid_safe=true. `/cockpit`
  zeigt echte JSON-Felder fuer Forecast, Drift, Paragraph-14a, Netzrisiko, EUR/MMM und Pool.
- T43 Temporale MinT-Reconciliation: `reconcile_temporal=True` liefert additiv
  `out["reconcile_temporal"]` fuer koharente P50-Forecasts ueber Tag, 24 Stunden und 96 Viertelstunden.
  Demo Hilden Netzumsatz: Koharenzfehler max 44.369193 MWh vor Reconcile, 0.000000 danach; MAE je
  Ebene ehrlich gemessen. Spannungsebenen-Reconciliation wurde an echten Daten verworfen, weil die
  veroeffentlichten Ebenenreihen keine additive Hierarchie bilden.
- T44 Paragraph-14a-Audit-Ledger: `audit_ledger_path=...` schreibt vorhandene Redispatch-/Dispatch-/
  Tarif-Eingriffe additiv in eine append-only Hash-Kette und liefert `out["audit"]` mit head hash,
  HMAC-Signatur und Chain-Status. Demo: 2 Redispatch-Eintraege, chain_ok=true. Beschriftung bewusst:
  manipulationssicherer Audit-Trail (Hash-Kette), nicht BNetzA-zertifiziert/rechtssicher.
- T7 Paragraph-14a-Demo: Mock-aEMT lehnt <4.2 kW ab und drosselt ein simuliertes HEMS.

## Zentrale Artefakte
- Gesamtbericht: `data_cache/report/netzpilot_report.html`
- T2 Daten/Provenienz: `data_cache/t2_2022-01-01_2024-01-01/`
- T3 Last: `data_cache/t3_lightgbm/`
- T4 Residuallast: `data_cache/t4_residual/`
- T4 physikalische Erzeugung: `data_cache/t4_physical_generation/`
- T8 CQR: `data_cache/t8_cqr/`
- T9 Klein-Stadtwerk: `data_cache/t9_small_utility/`, `data_cache/t9_small_utility_weather/`
- T9 Signifikanz: `data_cache/t9_significance/`
- T10 Klein-Stadtwerk verbessert: `data_cache/t10_small_utility/`
- T10 Signifikanz: `data_cache/t10_significance/`
- T11 Stadt-Wetter: `data_cache/cities_weather_eval.jsonl`, `data_cache/cities_weather_summary.json`, `data_cache/city_coords.json`, `data_cache/city_weather/`
- T12 Wetter-Lift: `data_cache/t12_weather_lift/heapo_eval.jsonl`, `data_cache/t12_weather_lift/t12_summary.json`, `data_cache/t12_weather_lift/t10_weather_reverify.json`
- T13 Per-Station-Wetter: `data_cache/t13_weather_lift/heapo_per_station_eval.jsonl`, `data_cache/t13_weather_lift/t13_summary.json`, `data_cache/t13_weather_lift/station_mapping.json`, `data_cache/t13_weather_lift/dryad_download_attempt.json`
- T14 echte Lastgaenge: `data_cache/real/*.csv`, `data_cache/pilot/hilden_slp_2025/`, `data_cache/pilot/hilden_netzumsatz_2025/`, `data_cache/pilot/hilden_dba_2025/`, `data_cache/pilot/eam_bg*_2024/`
- T15 neue Hook-Lastgaenge: `data_cache/real/SOURCES.md`, `data_cache/pilot/evdb_*_2024/`, `data_cache/pilot/herne_*_2024/`, `outreach/hook_overview.md`, `outreach/*_bullets.md`
- T16 Wetter-Lift zweiter Datensatz: `scripts/run_t16_weather_lift.py`, `data_cache/t16_weather_lift/dso_weather_eval.jsonl`, `data_cache/t16_weather_lift/t16_summary.json`, `data_cache/t16_weather_lift/dryad_download_attempt.json`
- T17 explizite Spalten/Ebenen: `scripts/pilot_in_a_box.py --list-columns`, `data_cache/pilot/t17_*`, `Notizen/_t15_t16_verification.md`, `outreach/hook_overview.md`
- T18 zweite Outreach-Welle: `data_cache/pilot/t18_*`, `data_cache/t18_js_crosscheck/`, `scripts/run_t18_js_crosscheck.js`, `outreach/hook_overview.md`, `outreach/*_bullets.md`
- T19 echte reBAP-Economics: `scripts/download_rebap_2024.py`, `netzpilot/data/rebap.py`, `data_cache/real/rebap_2024.csv`, `data_cache/real/rebap_2024_source.json`
- T20 reBAP-Spot-Economics: `scripts/fetch_spot_da_2024.py`, `scripts/verify_rebap_economics.py`, `netzpilot/data/spot_da.py`, `data_cache/real/spot_da_2024.csv`, `data_cache/real/spot_da_2024_source.json`
- T28/T29 echter DSO-/Regionalkorpus: `scripts/build_corpus_index.py`, `scripts/fetch_fr_public_loads.py`, `data_cache/real/corpus_index.json`, `data_cache/pool/pool_prior.json`, `data_cache/benchmark/network_independence.md`, `data_cache/benchmark/benchmark_table.md`, `data_cache/benchmark/benchmark_results.json`, `data_cache/benchmark/MODEL_CARD.md`
- T30 internationale Nationallast-Demo: `scripts/fetch_entsoe_power_stats.py`, `scripts/eval_intl.py`, `data_cache/intl/SOURCES.md`, `data_cache/intl/intl_benchmark.md`, `data_cache/intl/intl_benchmark.json`
- T31/T32 Paragraph-14a-Redispatch: `netzpilot/control/redispatch.py`, `tests/test_redispatch.py`,
  `tests/test_optimize_heterogen.py`, `scripts/build_redispatch_demo.py`,
  `data_cache/benchmark/redispatch_demo.md`
- T33-T35 Bilanzkreis-Settlement: `netzpilot/eval/bilanzkreis_realized.py`,
  `tests/test_bilanzkreis.py`, `tests/test_mc_savings.py`, `scripts/build_bilanzkreis_demo.py`,
  `data_cache/benchmark/bilanzkreis_demo.md`
- T36 Drift-Monitoring: `netzpilot/service/drift_monitor.py`, `tests/test_drift.py`,
  `scripts/build_drift_demo.py`, `data_cache/benchmark/drift_demo.md`
- T37 Modul-3-Tarif: `netzpilot/service/tariff_schedule.py`, `tests/test_tariff.py`,
  `scripts/build_tariff_demo.py`, `data_cache/benchmark/tariff_demo.md`
- T38 Quantil-Dispatch: `netzpilot/service/dispatch_plan.py`, `tests/test_dispatch.py`,
  `scripts/build_dispatch_demo.py`, `data_cache/benchmark/dispatch_demo.md`
- T39 Input-Validierung + Asset-Overload: `netzpilot/service/input_validation.py`,
  `tests/test_validate.py`, `tests/test_overload.py`, `scripts/build_validate_demo.py`,
  `scripts/build_overload_demo.py`, `data_cache/benchmark/validate_demo.md`,
  `data_cache/benchmark/overload_demo.md`
- T40 CVaR-Dispatch + IEC-Thermik: `netzpilot/control/risk.py`,
  `netzpilot/grid/thermal.py`, `tests/test_cvar.py`, `tests/test_thermal.py`,
  `scripts/verify_cvar.py`, `scripts/verify_thermal.py`, `scripts/build_thermal_demo.py`,
  `data_cache/benchmark/thermal_demo.md`
- T41 EEBUS-LPC + Mehr-/Mindermengen: `netzpilot/control/eebus_lpc.py`,
  `netzpilot/eval/mehrmindermengen.py`, `tests/test_eebus_lpc.py`,
  `tests/test_mehrmindermengen.py`, `scripts/verify_eebus_lpc.py`,
  `scripts/verify_mehrmindermengen.py`, `scripts/build_lpc_demo.py`,
  `scripts/build_mmm_demo.py`, `data_cache/benchmark/lpc_demo.md`,
  `data_cache/benchmark/mmm_demo.md`
- T42 VPP-Pool + operatives Cockpit: `netzpilot/control/vpp_pool.py`,
  `netzpilot/service/cockpit.html`, `tests/test_vpp.py`, `scripts/verify_vpp.py`,
  `scripts/build_vpp_demo.py`, `scripts/build_cockpit_demo.py`,
  `data_cache/benchmark/vpp_demo.md`, `data_cache/benchmark/cockpit_demo.json`
- T43 Temporale MinT-Reconciliation: `netzpilot/service/reconcile_temporal.py`,
  `tests/test_reconcile.py`, `scripts/verify_reconcile.py`,
  `scripts/build_reconcile_temporal_demo.py`,
  `data_cache/benchmark/reconcile_temporal_demo.md`
- T44 Paragraph-14a-Audit-Ledger: `netzpilot/service/audit_ledger.py`,
  `tests/test_audit_ledger.py`, `scripts/build_audit_demo.py`,
  `data_cache/benchmark/audit_ledger_demo.jsonl`,
  `data_cache/benchmark/audit_ledger_demo.html`

## Grundregeln
Kein Leakage: Wetter im Backtest ist Open-Meteo Historical Forecast, nie Reanalyse/Ist-Wetter.
Baselines bleiben immer dabei. Rolling-Origin statt k-fold. DST und Feiertage werden beruecksichtigt.
Schwache Ergebnisse werden dokumentiert statt schoengerechnet.

## Struktur
```text
netzpilot/
  data/       SMARD, Open-Meteo, Residual-/Generation-/Small-Utility-Daten
  features/   leakage-sichere Features
  models/     Baselines, Ridge, LightGBM-Quantile
  eval/       Metriken, Rolling-Origin, CQR, Economics
  control/    Mock-aEMT und HEMS-Simulation
  report/     HTML-/Markdown-Bericht
scripts/      reproduzierbare Laeufe
tests/        Unit-/Leakage-/Integritaetstests
```

## Reproduzierbare Hauptlaeufe
```powershell
.\.venv\Scripts\python.exe scripts\run_t4_physical_generation.py
.\.venv\Scripts\python.exe scripts\run_t6_mapie.py --target load
.\.venv\Scripts\python.exe scripts\run_t6_mapie.py --target residual
.\.venv\Scripts\python.exe scripts\run_t9_small_utility.py --with-weather --out data_cache\t9_small_utility_weather
.\.venv\Scripts\python.exe scripts\eval_t9_significance.py
.\.venv\Scripts\python.exe scripts\run_t10_small_utility.py
.\.venv\Scripts\python.exe scripts\run_t11_city_weather.py
.\.venv\Scripts\python.exe scripts\run_t12_weather_lift.py
.\.venv\Scripts\python.exe scripts\run_t13_weather_lift.py
.\.venv\Scripts\python.exe scripts\pilot_in_a_box.py --csv data_cache\real\SLP-Summenlast-Lastgang-2025.csv --ts-col Text --load-col Reihe1 --unit kW --region NW --name "Stadtwerke Hilden SLP-Summenlast 2025" --out data_cache\pilot\hilden_slp_2025 --keep-days 365 --n-test 28
.\.venv\Scripts\python.exe scripts\build_redispatch_demo.py
.\.venv\Scripts\python.exe scripts\build_bilanzkreis_demo.py
.\.venv\Scripts\python.exe scripts\verify_drift.py
.\.venv\Scripts\python.exe scripts\build_drift_demo.py
.\.venv\Scripts\python.exe scripts\verify_tariff.py
.\.venv\Scripts\python.exe scripts\build_tariff_demo.py
.\.venv\Scripts\python.exe scripts\verify_dispatch.py
.\.venv\Scripts\python.exe scripts\build_dispatch_demo.py
.\.venv\Scripts\python.exe scripts\verify_validate.py
.\.venv\Scripts\python.exe scripts\verify_overload.py
.\.venv\Scripts\python.exe scripts\build_validate_demo.py
.\.venv\Scripts\python.exe scripts\build_overload_demo.py
.\.venv\Scripts\python.exe scripts\verify_cvar.py
.\.venv\Scripts\python.exe scripts\verify_thermal.py
.\.venv\Scripts\python.exe scripts\build_thermal_demo.py
.\.venv\Scripts\python.exe scripts\verify_eebus_lpc.py
.\.venv\Scripts\python.exe scripts\verify_mehrmindermengen.py
.\.venv\Scripts\python.exe scripts\build_lpc_demo.py
.\.venv\Scripts\python.exe scripts\build_mmm_demo.py
.\.venv\Scripts\python.exe scripts\verify_vpp.py
.\.venv\Scripts\python.exe scripts\build_vpp_demo.py
.\.venv\Scripts\python.exe scripts\build_cockpit_demo.py
.\.venv\Scripts\python.exe scripts\demo_control_loop.py
```
