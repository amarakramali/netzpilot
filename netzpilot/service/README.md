# NetzPilot — Dienst (Service-Schicht)

Macht aus der validierten Batch-Engine (`netzpilot/`) einen **betreibbaren Day-ahead-Dienst**:
Lastgang rein → kalibrierte Prognose (P10/P50/P90) + optional Residuallast + §14a-Fahrplan raus,
persistiert, per REST abrufbar. Die Rechen-Logik ist die **echte Engine** (`forecast_next_day`,
ShrunkCorrector, CQR) — keine Nachbildung.

## Komponenten
- `runner.py` — orchestriert verifizierten Loader → `to_daily_local` → `forecast_next_day` → §14a-`make_fahrplan`.
  Optionale Erzeugungs-CSV → **Residuallast = Last − Erzeugung** mit derselben Engine; §14a läuft dann auf der Residuallast.
  Optional `rolling_redispatch=True` liefert additiv `redispatch` auf Day-ahead-P50-Basis; `steuve_devices`
  erlaubt heterogene Floors/Gewichte je steuerbarer Verbrauchseinrichtung.
  Default `validate_input=True` plausibilisiert den stündlichen Eingangslastgang vor der Engine und
  nutzt ersetzbare `cleaned`-Werte, ohne die Original-CSV zu verändern.
  Optional `asset_rating_kw` liefert additiv `overload` und `hosting_capacity` für ein einzelnes
  Netzasset; wenn `congestion_threshold_mw` gesetzt ist, muss sie dieselbe Grenze abbilden.
  Optional `realized_economics=True` liefert `economics_realized` aus der rolling-origin
  Bilanzkreis-Abrechnung gegen Saisonal-Naiv (reBAP+Spot, Stundenregel, mit Bootstrap-Band).
- `app.py` — FastAPI-REST (`/forecast`, `/forecast/{utility}/latest`, `/forecast/{utility}/{date}`,
  `/history/{utility}`, `/report/{utility}/latest`, `/report/{utility}/{date}`, `/utilities`, `/health`).
- `report.py` — druckoptimierter Ein-Seiten-HTML-Bericht (Browser: Drucken → PDF; dependency-frei).
- `store.py` — dateibasierte Persistenz je Mandant (kein DB-Server; offline beim Stadtwerk lauffähig).
- `../../scripts/run_daily_forecast.py` — täglicher Scheduler-Einstieg (cron / Windows-Aufgabenplanung),
  Einzel-Mandant via Flags oder mehrere via `--config config/utilities.json` (Vorlage: `config/utilities.example.json`).

## Drift-Monitoring
Optional `drift_monitoring=True` liefert additiv `out["drift"]` mit PSI/KS/Bias/MAE-Signalen,
P10/P90-Coverage und `needs_recalibration`. Referenz- und Recent-Residuen werden versioniert durch
`drift_monitor.py` persistiert; der Dienst warnt nur und startet kein Auto-Retrain.

## Input-Validierung + Asset-Overload
Default `validate_input=True` liefert `out["input_validation"]` mit `quality_score`, Issue-Zählung
und Ersatzwert-Samples. Lücken, Ausreißer, negative und außerhalb der Plausibilitätsgrenze liegende
Werte werden ersetzt, wenn alle Ersatzwerte verfügbar sind; eingefrorene Phasen werden nur gemeldet.

Optional `asset_rating_kw` liefert `out["overload"]` und `out["hosting_capacity"]` aus der
rolling-origin-Prognoseverteilung. Das ist eine probabilistische Einzelasset-Ampel, kein
Netzlastfluss. `asset_rating_kw` und `congestion_threshold_mw * 1000` müssen zusammenpassen, wenn
beide gesetzt sind.

## Modul-3-Tarif
Optional `grid_fee_eur_per_kwh`, `tariff_energy_kwh` und `tariff_p_max_kw` liefern additiv
`out["tariff_schedule"]`. Wenn `rolling_redispatch=True` aktiv ist, nutzt der Tarifplan dessen
`cap_kw`-Grenzen; Netzsicherheit dominiert die Netzentgelt-Optimierung.

## Quantil-Dispatch
Optional `dispatch_plan_enabled=True` mit `dispatch_steuve_energy_kwh` und `dispatch_steuve_p_max_kw`
liefert additiv `out["dispatch_plan"]`. Der Plan nutzt trailing rolling-origin Residuen fuer die
Newsvendor-Nominierung und prueft seine Caps gegen Redispatch, wenn Redispatch aktiv ist.

Optional `dispatch_risk_beta>0` aktiviert eine CVaR-risk-averse Nominierung. Default `0.0` bleibt der
Newsvendor-Pfad. Im Ergebnis steht `out["dispatch_plan"]["risk_averse"]` mit Expected-Kosten, CVaR,
Objective und Delta gegen die Newsvendor-Nominierung.

## IEC-Thermik
Optional `thermal_rating_kw` liefert additiv `out["thermal"]` aus einem IEC-60076-7-artigen
Hotspot-/Alterungsmodell ueber die trailing rolling-origin Prognoseverteilung. `thermal_ambient_c`
kann ein Skalar oder 24h-Profil sein; sonst nutzt der Dienst `weather_csv` mit `temperature_2m` oder
faellt ehrlich auf 20 C Default-Annahme zurueck. Das ist Einzelasset-Thermik mit Standardparametern,
kein Netzlastfluss und kein Ersatz fuer echte Trafo-Daten.

Optional `rating_kw` ist die bevorzugte eine Rating-Wahrheit fuer Cockpit-/Betriebslaufe. Der Wert
setzt konsistent `congestion_threshold_mw`, `asset_rating_kw` und, falls angefordert,
`thermal_rating_kw`. Divergierende Grenzwerte werden abgelehnt.

## EEBUS-LPC-Uebergabe
Optional `submit_to_aemt=True` uebergibt den Fahrplan an den Adapter. Default `aemt_adapter="mock"`
bleibt der bisherige Mock-aEMT. Mit `aemt_adapter="eebus_lpc"` nutzt der Dienst den verifizierten
`EebusLpcAdapter` und liefert additiv `out["fahrplan_lpc"]`. Das ist die LPC-Datenabbildung;
EEBUS-Transport, SHIP/SPINE, SMGW und Anlagensteuerung bleiben extern.

## Mehr-/Mindermengen
Optional `mmm_price_eur_mwh` liefert additiv `out["mmm"]` aus einem trailing rolling-origin
Forecast-vs.-Ist-Vergleich. Ausgewiesen werden Mehr-/Mindermenge, Netto, absolutes MMM-Volumen und
Volumenreduktion gegen Saisonal-Naiv. Der MMM-Preis ist regulierte Eingabe/Config, kein erfundener
Default.

## VPP-/Pool-Dispatch
Optional `pool_assets` plus `pool_shared_cap_kw` liefert additiv `out["pool_dispatch"]` aus
`control.vpp_pool.pool_dispatch`. Die Assets muessen je Anlage `demand_kw` ueber den Horizont liefern;
die gemeinsame Cap-Reihe ist die harte Pool-Grenze. Das ist faire Periodenkappung und Aggregation,
keine zeituebergreifende Speicher-/SOC-Optimierung.

## Operatives Cockpit
`/cockpit` liefert eine Single-Page-Betriebsansicht aus echten Service-JSONs. Sie kann gespeicherte
JSONs laden oder einen Live-Lauf starten und fuehrt dabei `rating_kw` als einzige Netzgrenze.

## Start (Host, im `.venv`)
```bash
pip install -r requirements.txt
uvicorn netzpilot.service.app:app --host 0.0.0.0 --port 8000
# Doku/Swagger: http://localhost:8000/docs
```

## Beispiel-Aufrufe
```bash
# Prognose per Datei-Upload
curl -F "file=@data_cache/real/<lastgang>.csv" \
     -F "utility=Beispiel-Stadtwerk" -F "unit=kW" -F "ts_col=Text" -F "load_col=Reihe1" \
     -F "rebap_csv=data_cache/real/rebap_2024.csv" \
     -F "congestion_threshold_mw=45" -F "steuve_malo=DE0001234567890" \
     http://localhost:8000/forecast

# Prognose mit rollierendem Redispatch-Feld und steuVE-Bedarf
curl -F "csv_path=data_cache/real/<lastgang>.csv" \
     -F "utility=Beispiel-Stadtwerk" -F "unit=kW" -F "ts_col=Text" -F "load_col=Reihe1" \
     -F "congestion_threshold_mw=33" -F "steuve_malo=DE0001234567890" \
     -F "steuve_demands_kw=[1000,800,600]" -F "rolling_redispatch=true" \
     http://localhost:8000/forecast

# Prognose mit realisierter Bilanzkreis-Abrechnung + MC-Band
curl -F "csv_path=data_cache/real/<lastgang>.csv" \
     -F "utility=Beispiel 110/10kV" -F "unit=kW" -F "ts_col=Datum+von" -F "load_col=Load_1" \
     -F "rebap_csv=data_cache/real/rebap_2024.csv" \
     -F "spot_csv=data_cache/real/spot_da_2024.csv" -F "realized_economics=true" \
     http://localhost:8000/forecast

# Prognose mit Input-Gate + probabilistischer Asset-Ueberlast
curl -F "csv_path=data_cache/real/<lastgang>.csv" \
     -F "utility=Beispiel 110/10kV" -F "unit=kW" -F "ts_col=Datum+von" -F "load_col=Load_1" \
     -F "congestion_threshold_mw=60.7" -F "asset_rating_kw=60700" \
     -F "validate_input=true" http://localhost:8000/forecast

# Prognose mit CVaR-Dispatch und IEC-Thermik
curl -F "csv_path=data_cache/real/<lastgang>.csv" \
     -F "utility=ThermalDemo" -F "unit=kW" -F "ts_col=Datum+von" -F "load_col=Wert.11" \
     -F "dispatch_plan_enabled=true" -F "dispatch_steuve_energy_kwh=20" \
     -F "dispatch_steuve_p_max_kw=1000" -F "dispatch_risk_beta=0.6" \
     -F "thermal_rating_kw=9265" -F "thermal_hotspot_limit_c=120" \
     http://localhost:8000/forecast

# Prognose mit EEBUS-LPC-Mapping und Mehr-/Mindermengen
curl -F "csv_path=data_cache/real/<lastgang>.csv" \
     -F "utility=Beispiel W5W6" -F "unit=kW" -F "ts_col=Text" -F "load_col=Reihe1" \
     -F "congestion_threshold_mw=33" -F "steuve_malo=DE0001234567890" \
     -F "submit_to_aemt=true" -F "aemt_adapter=eebus_lpc" \
     -F "mmm_price_eur_mwh=60" http://localhost:8000/forecast

# Prognose mit einer Rating-Wahrheit und VPP-/Pool-Dispatch
curl -F "csv_path=data_cache/real/<lastgang>.csv" \
     -F "utility=Beispiel Pool" -F "unit=kW" -F "ts_col=Text" -F "load_col=Reihe1" \
     -F "rating_kw=33000" \
     -F 'pool_assets=[{"id":"A","demand_kw":[10,10],"floor_kw":4.2},{"id":"B","demand_kw":[10,10],"floor_kw":4.2}]' \
     -F "pool_shared_cap_kw=40,16" http://localhost:8000/forecast

# jüngste gespeicherte Prognose
curl http://localhost:8000/forecast/Beispiel-Stadtwerk/latest

# Prognose MIT Residuallast (Erzeugungs-CSV per Pfad)
curl -F "csv_path=data_cache/real/<lastgang>.csv" -F "utility=SW Beispiel" -F "unit=MW" \
     -F "generation_csv=data_cache/real/<erzeugung>.csv" -F "generation_unit=MW" \
     -F "congestion_threshold_mw=12" http://localhost:8000/forecast

# Verlauf + druckbarer Bericht (Browser öffnen, dann Drucken → Als PDF speichern)
curl http://localhost:8000/history/Beispiel-Stadtwerk
#   http://localhost:8000/report/Beispiel-Stadtwerk/latest

# Optional: temporale MinT-Reconciliation fuer 15-min-Lastgaenge
curl -X POST -F "csv_path=data_cache/real/<lastgang>.csv" \
     -F "unit=kW" -F "ts_col=Text" -F "load_col=Reihe1" \
     -F "reconcile_temporal=true" -F "reconcile_temporal_method=wls_struct" \
     http://localhost:8000/forecast

# Optional: Paragraph-14a-Audit-Ledger fuer Eingriffe
curl -X POST -F "csv_path=data_cache/real/<lastgang>.csv" \
     -F "unit=kW" -F "ts_col=Text" -F "load_col=Reihe1" \
     -F "congestion_threshold_mw=33" -F "rolling_redispatch=true" \
     -F "steuve_demands_kw=1000,800,600" \
     -F "audit_ledger_path=data_cache/audit/beispiel.jsonl" \
     -F "audit_signing_key=<secret-from-config>" \
     http://localhost:8000/forecast
```

## Täglicher Lauf (Scheduler)
```bash
python scripts/run_daily_forecast.py --csv <lastgang.csv> --utility "<Name>" \
    --unit kW --ts-col Text --load-col Reihe1 --congestion-threshold-mw 45 \
    --asset-rating-kw 45000 --thermal-rating-kw 45000 --dispatch-risk-beta 0.6 \
    --submit-to-aemt --aemt-adapter eebus_lpc --mmm-price-eur-mwh 60 \
    --steuve-malo DE000...
# oder mehrere Mandanten via --config config/utilities.json
```

## Verifiziert (2026-05-31)
- Runner + Scheduler + Persistenz auf einem echten Netzumsatz-Lastgang (364 Tage) end-to-end gelaufen:
  Prognose für Folgetag, P10≤P50≤P90, Engpasslogik + §14a-Fahrplan (4,2-kW-Mindestleistung) korrekt.
- API-Pfade in `tests/test_service.py` (TestClient); laufen im Host-`.venv` (fastapi/httpx dort installiert).
- Optionaler reBAP-Pfad: `rebap_csv=data_cache/real/rebap_2024.csv` erzeugt im Ergebnisfeld `economics`
  Median + P25-P75-EUR/Jahr auf Basis echter reBAP-2024-Viertelstundenwerte. Nutzen = Downside-Schutz,
  kein garantierter linearer Ertrag.
- T39: `verify_overload.py`, `verify_validate.py`, `tests/test_overload.py` und `tests/test_validate.py`
  grün; Demos liegen unter `data_cache/benchmark/overload_demo.md` und `validate_demo.md`.
- T40: `verify_cvar.py`, `verify_thermal.py`, `tests/test_cvar.py` und `tests/test_thermal.py`
  gruen; Demo liegt unter `data_cache/benchmark/thermal_demo.md`.
- T41: `verify_eebus_lpc.py`, `verify_mehrmindermengen.py`, `tests/test_eebus_lpc.py` und
  `tests/test_mehrmindermengen.py` gruen; Demos liegen unter `data_cache/benchmark/lpc_demo.md`
  und `mmm_demo.md`.
- T42: `verify_vpp.py`, `tests/test_vpp.py` gruen; Demos liegen unter
  `data_cache/benchmark/vpp_demo.md` und `cockpit_demo.json`; Cockpit unter `/cockpit`.
- T43: `verify_reconcile.py`, `tests/test_reconcile.py` gruen; temporale Demo liegt unter
  `data_cache/benchmark/reconcile_temporal_demo.md`; Servicefeld `reconcile_temporal` ist optional.
- T44: `tests/test_audit_ledger.py` gruen; Demo-Artefakte liegen unter
  `data_cache/benchmark/audit_ledger_demo.jsonl` und `audit_ledger_demo.html`. Servicefeld
  `audit_ledger_path` aktiviert das Ledger; `audit_signing_key` wird nur aus Config/Form gelesen.

## Ehrliche Grenzen
- Kein Auth/Mandanten-Trennung auf Transportebene (für lokalen/VPN-Betrieb gedacht; vor Internet-Exposition
  Auth + TLS ergänzen).
- §14a-Fahrplan ist ein **Entwurf**; produktiv geht er an einen zertifizierten White-Label-aEMT, nicht direkt ans SMGW.
- Engpassschwelle ist parametrisch (MW); echte Schwelle = Netzkapazität des Stadtwerks (im Pilot zu setzen).
- `overload`/`hosting_capacity` ist Einzelasset-Risiko gegen eine Bemessungsgrenze, kein Netzlastfluss.
- `thermal` nutzt Standardparameter und optional Wetterproxy; echte Trafo-Parameter, Vorbelastung und
  Umgebungssensorik muessen im Pilot gesetzt werden.
- CVaR-/EVT-Aussagen sind Quantilmodell-Annahmen. EVT ist nicht produktiv verdrahtet, sondern bleibt
  research-grade fuer spaetere Tailkalibrierung.
- EEBUS-LPC ist eine Datenabbildung. NetzPilot sendet nicht selbst per EEBUS und beruehrt kein SMGW.
- `mmm` nutzt den uebergebenen regulierten MMM-Preis; Demo-Preisannahmen sind keine
  Nutzenversprechen und MMM ist nicht reBAP-Ausgleichsenergie.
- `pool_dispatch` ist Periodenkappung/Aggregation, kein VPP-MILP und kein Speicherfahrplan.
- `reconcile_temporal` erzwingt nur die exakt gueltige Zeit-Hierarchie einer einzelnen 15-min-Reihe.
  Spannungsebenen werden nicht reconciled, weil echte Veroeffentlichungsreihen dort keine additive
  Parent/Child-Hierarchie bilden.
- `audit` ist ein manipulationssicherer Audit-Trail (Hash-Kette), nicht BNetzA-zertifiziert und keine
  juristische Rechtssicherheitsbehauptung. HMAC-Signaturen brauchen einen extern konfigurierten
  Schluessel; nie hartkodieren.
- Das Cockpit zeigt echte JSON-Felder; fehlende Felder bleiben sichtbar als nicht vorhanden.
