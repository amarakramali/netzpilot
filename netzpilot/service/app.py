# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""NetzPilot REST-Dienst (FastAPI). rev: +report/+history routes

Macht aus der Batch-Engine einen betreibbaren Service:
  POST /forecast              — Lastgang-CSV (Upload oder Pfad) -> Day-ahead-Prognose + §14a-Fahrplan, persistiert
  GET  /forecast/{utility}/latest   — jüngste gespeicherte Prognose
  GET  /forecast/{utility}/{date}   — Prognose eines bestimmten Laufs
  GET  /utilities             — bekannte Mandanten
  GET  /health                — Health-Check

Start:  uvicorn netzpilot.service.app:app --reload
Die eigentliche Rechen-Logik liegt in runner.run_forecast (echte Engine). Diese Datei ist nur Transport.
"""
from __future__ import annotations
import json
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

from netzpilot.intraday import intraday_update
from netzpilot.service.runner import run_forecast
from netzpilot.service.input_validation import validate_hourly_series
from netzpilot.service.store import ForecastStore
from netzpilot.service.report import render_report_html
from netzpilot.service.compliance_14a import render_meldebogen_html, render_fairness_html
from netzpilot.service.security import api_key_and_logging_middleware

app = FastAPI(title="NetzPilot", version="1.0",
              description="Leakage-sichere Day-ahead-Last-/Residuallastprognose + §14a-Fahrplan für Stadtwerke.")
# Optionaler API-Key-Schutz (NETZPILOT_API_KEY) + Request-Logging. Ohne Key gesetzt: offen (lokal).
app.middleware("http")(api_key_and_logging_middleware)
store = ForecastStore(os.environ.get("NETZPILOT_STORE", "data_cache/service_store"))

_UI_PATH = os.path.join(os.path.dirname(__file__), "ui.html")
_COCKPIT_PATH = os.path.join(os.path.dirname(__file__), "cockpit.html")


def _parse_float_list(raw: Optional[str]):
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = [x.strip() for x in raw.split(",") if x.strip()]
    if not isinstance(value, list):
        raise ValueError("steuve_demands_kw muss Liste oder Komma-Liste sein")
    return [float(x) for x in value]


def _parse_bool_list(raw: Optional[str]):
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = [x.strip() for x in raw.split(",") if x.strip()]
    if not isinstance(value, list):
        raise ValueError("tariff_available muss Liste oder Komma-Liste sein")
    return [str(x).strip().lower() in {"1", "true", "yes", "ja"} for x in value]


def _parse_device_list(raw: Optional[str]):
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError("steuve_devices muss ein JSON-Array sein")
    return value


def _parse_json_list(raw: Optional[str], name: str):
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{name} muss ein JSON-Array sein")
    return value


def _parse_intraday_actuals(raw: str) -> list[float]:
    """Kommaliste heutiger Ist-Stunden: leere/nichtnumerische Tokens werden zu NaN."""
    try:
        value = json.loads(raw)
        if not isinstance(value, list):
            value = [raw]
    except json.JSONDecodeError:
        value = raw.split(",")
    if not 1 <= len(value) <= 23:
        raise ValueError("actuals muss 1..23 Stunden enthalten.")
    out = []
    for x in value:
        s = "" if x is None else str(x).strip()
        if s == "":
            out.append(float("nan"))
            continue
        try:
            out.append(float(s))
        except ValueError:
            out.append(float("nan"))
    return out


@app.get("/")
def root():
    """EIN Einstieg: die Wurzel führt direkt ins operative Cockpit (verhindert Zwei-UI-Verwirrung).

    Die geführte Einsteiger-Seite (Dropzone, Schritt-für-Schritt) bleibt unter /einfach erreichbar.
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/cockpit", status_code=307)


@app.get("/einfach", response_class=HTMLResponse)
def ui():
    """Geführte Einsteiger-Oberfläche (ruft die ECHTE Engine über /forecast auf — keine Nachbildung)."""
    with open(_UI_PATH, encoding="utf-8") as f:
        html = f.read()
    # Deutlicher Querverweis aufs operative Cockpit (einmalig injiziert, Seite selbst bleibt unberührt).
    hint = ('<div style="margin:10px 0;padding:8px 12px;border:1px solid #2c4a6e;border-radius:8px;'
            'font-size:13px">Vollansicht mit Verlauf, Track-Record &amp; Blind-Challenge: '
            '<a href="/cockpit" style="color:#4cc2ff;font-weight:700">→ Operatives Cockpit</a></div>')
    return html.replace("<body>", "<body>" + hint, 1)


@app.get("/cockpit", response_class=HTMLResponse)
def cockpit():
    with open(_COCKPIT_PATH, encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "service": "netzpilot", "version": app.version}


@app.get("/sample.csv")
def sample_csv():
    """Liefert einen echten Beispiel-Lastgang (Hilden Netzumsatz) für den 'Beispiel laden'-Knopf.

    Damit jemand ohne eigenen Lastgang den vollen Ablauf sofort ausprobieren kann (Einheit kW,
    Zeitspalte 'Text', Lastspalte 'Reihe1' — wird von /inspect automatisch erkannt).
    """
    from fastapi.responses import FileResponse
    for p in ("data_cache/real/Netzumsatz-Lastgang-2025.csv",
              os.path.join(os.path.dirname(__file__), "..", "..",
                           "data_cache", "real", "Netzumsatz-Lastgang-2025.csv")):
        if os.path.exists(p):
            return FileResponse(p, media_type="text/csv", filename="beispiel_lastgang.csv")
    raise HTTPException(404, "Beispiel-CSV nicht gefunden (data_cache/real fehlt).")


@app.get("/utilities")
def utilities():
    return {"utilities": store.list_utilities()}


@app.post("/intraday")
async def intraday(
    utility: str = Form(...),
    actuals: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """Resttag-Update als Ansicht auf Store-Latest. Schreibt nie in den Store."""
    latest = store.latest(utility)
    if latest is None:
        raise HTTPException(404, f"Keine Prognose fuer '{utility}' gespeichert.")
    if actuals is None:
        if file is not None:
            raise HTTPException(422, "Intraday-Dateiupload ist noch nicht verdrahtet; bitte actuals-Kommaliste senden.")
        raise HTTPException(422, "actuals-Kommaliste angeben.")
    try:
        values = _parse_intraday_actuals(actuals)
        upd = intraday_update(latest.get("forecast") or [], values)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return JSONResponse({**upd, "forecast_date": latest.get("forecast_date"), "utility": utility})


@app.post("/challenge")
async def challenge(
    unit: str = Form("MW"),
    region: str = Form("NW"),
    ts_col: Optional[str] = Form(None),
    load_col: Optional[str] = Form(None),
    n_test: int = Form(84),
    n_boot: int = Form(4000),
    csv_path: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """Blind-Challenge: leakage-sicherer Backtest + Signifikanz auf einer fremden Datei.

    Persistiert NICHTS (kein Store-Eintrag) — reiner Sofort-Beweis. Datei-Upload ODER csv_path.
    """
    from netzpilot.service.challenge import run_challenge
    tmp = None
    try:
        if file is not None:
            suffix = os.path.splitext(file.filename or "load.csv")[1] or ".csv"
            fd, tmp = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                f.write(await file.read())
            path = tmp
            src_name = file.filename or "upload.csv"
        elif csv_path:
            if not os.path.exists(csv_path):
                raise HTTPException(404, f"csv_path nicht gefunden: {csv_path}")
            path = csv_path
            src_name = None
        else:
            raise HTTPException(400, "Entweder 'file' (Upload) oder 'csv_path' angeben.")
        try:
            out = run_challenge(path, ts_col=ts_col, load_col=load_col, unit=unit,
                                region=region, n_test=n_test, n_boot=n_boot)
        except SystemExit as e:      # robust_load_csv signalisiert Spaltenfehler so
            raise HTTPException(422, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))
        if src_name:
            out["source_file"] = src_name
        return out
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


@app.get("/files")
def list_load_files():
    """Lastgang-Dateien aus data_cache/real für die Cockpit-Auswahl.

    Bewusst eng: NUR dieser feste Ordner, NUR Dateinamen (kein Pfad-Parameter, kein Traversal),
    nur Lastgang-Endungen. Der Browser darf den Startordner des nativen Datei-Dialogs nicht
    setzen — diese Server-Auswahl ist der saubere Ersatz dafür.
    """
    d = "data_cache/real"
    exts = (".csv", ".xlsx", ".xls", ".xlsm")
    if not os.path.isdir(d):
        return {"dir": d, "files": []}
    return {"dir": d, "files": sorted(f for f in os.listdir(d)
                                      if f.lower().endswith(exts)
                                      and os.path.isfile(os.path.join(d, f)))}


@app.post("/inspect")
async def inspect(unit: str = Form("kW"), file: UploadFile = File(...)):
    """Onboarding-Hilfe: erkennt Zeit-/Lastspalten einer hochgeladenen CSV und schlägt sie vor.

    Damit ein Stadtwerk-Mitarbeiter NICHT raten muss, welche Spalte die Last ist. Die UI ruft das
    direkt nach dem Drag&Drop und füllt Zeit-/Lastspalte automatisch. Bei mehreren Lastspalten
    (Netzebenen) zeigt sie eine Auswahl statt blind zu raten.
    """
    from scripts.pilot_in_a_box import inspect_load_columns, robust_load_csv
    fd, tmp = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())
        try:
            info = inspect_load_columns(tmp, unit=unit)
        except SystemExit as e:
            raise HTTPException(422, str(e))
        except Exception as e:
            raise HTTPException(422, f"CSV nicht lesbar: {e}")
        cands = info.get("load_candidates", [])
        # Vorschlag: bevorzugt benannte Lastspalten (LOAD_HINTS greift schon im Loader), sonst die erste.
        suggested = cands[0]["name"] if cands else None
        input_validation = None
        if suggested:
            try:
                hourly, _ts, _lc, _meta = robust_load_csv(
                    tmp,
                    ts_col=info.get("ts_col"),
                    load_col=suggested,
                    unit=unit,
                    return_meta=True,
                )
                _cleaned, input_validation = validate_hourly_series(hourly, apply_cleaned=False)
            except Exception as e:
                input_validation = {
                    "enabled": True,
                    "status": "not_available",
                    "reason": str(e),
                    "original_preserved": True,
                }
        return JSONResponse({
            "ts_col": info.get("ts_col"),
            "timestamp_parse_rate": info.get("timestamp_parse_rate"),
            "load_candidates": cands,
            "suggested_load_col": suggested,
            "n_candidates": len(cands),
            "input_validation": input_validation,
        })
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.post("/forecast")
async def forecast(
    utility: str = Form("default"),
    region: str = Form("NW"),
    unit: str = Form("MW"),
    ts_col: Optional[str] = Form(None),
    load_col: Optional[str] = Form(None),
    congestion_threshold_mw: Optional[float] = Form(None),
    steuve_malo: Optional[str] = Form(None),
    steuve_demands_kw: Optional[str] = Form(None),
    steuve_devices: Optional[str] = Form(None),
    rolling_redispatch: bool = Form(False),
    realized_economics: bool = Form(False),
    drift_monitoring: bool = Form(False),
    validate_input: bool = Form(True),
    validate_allow_negative: bool = Form(False),
    validate_max_plausible: Optional[float] = Form(None),
    asset_rating_kw: Optional[float] = Form(None),
    overload_risk_alpha: float = Form(0.05),
    thermal_rating_kw: Optional[float] = Form(None),
    thermal_ambient_c: Optional[float] = Form(None),
    thermal_hotspot_limit_c: float = Form(120.0),
    thermal_risk_alpha: float = Form(0.05),
    grid_fee_eur_per_kwh: Optional[str] = Form(None),
    tariff_energy_kwh: Optional[float] = Form(None),
    tariff_p_max_kw: Optional[float] = Form(None),
    tariff_available: Optional[str] = Form(None),
    tariff_available_start_hour: Optional[int] = Form(None),
    tariff_available_end_hour: Optional[int] = Form(None),
    dispatch_plan_enabled: bool = Form(False),
    dispatch_steuve_energy_kwh: Optional[float] = Form(None),
    dispatch_steuve_p_max_kw: Optional[float] = Form(None),
    dispatch_c_short: float = Form(0.20),
    dispatch_c_long: float = Form(0.10),
    dispatch_risk_beta: float = Form(0.0),
    dispatch_risk_alpha: float = Form(0.95),
    rebap_csv: Optional[str] = Form(None),
    spot_csv: Optional[str] = Form(None),
    generation_csv: Optional[str] = Form(None),
    generation_unit: str = Form("MW"),
    generation_ts_col: Optional[str] = Form(None),
    generation_load_col: Optional[str] = Form(None),
    submit_to_aemt: bool = Form(False),
    aemt_adapter: str = Form("mock"),
    mmm_price_eur_mwh: Optional[float] = Form(None),
    rating_kw: Optional[float] = Form(None),
    pool_assets: Optional[str] = Form(None),
    pool_shared_cap_kw: Optional[str] = Form(None),
    reconcile_temporal: bool = Form(False),
    reconcile_temporal_method: str = Form("wls_struct"),
    reconcile_temporal_n_test: int = Form(7),
    audit_ledger_path: Optional[str] = Form(None),
    audit_signing_key: Optional[str] = Form(None),
    audit_rule_version: str = Form("netzpilot-paragraph14a-v1"),
    forecast_store_path: Optional[str] = Form(None),
    residual_feedback: Optional[bool] = Form(None),
    horizon_days: int = Form(1),
    horizon_bands: str = Form("k1"),
    holiday_add: Optional[str] = Form(None),
    holiday_remove: Optional[str] = Form(None),
    csv_path: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """Erzeugt + speichert die Day-ahead-Prognose. Lastgang per Datei-Upload ODER --csv_path.

    Optionale Erzeugungs-CSV (generation_csv, per Pfad) aktiviert die Residuallast-Prognose.
    """
    tmp = None
    try:
        if horizon_days < 1 or horizon_days > 7:
            raise HTTPException(422, "horizon_days muss in 1..7 liegen.")
        if horizon_bands not in ("k1", "per_horizon"):
            raise HTTPException(422, "horizon_bands muss 'k1' oder 'per_horizon' sein.")
        if horizon_days < 2 and horizon_bands != "k1":
            raise HTTPException(422, "horizon_bands='per_horizon' ist nur mit horizon_days>=2 gueltig.")
        if file is not None:
            suffix = os.path.splitext(file.filename or "load.csv")[1] or ".csv"
            fd, tmp = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                f.write(await file.read())
            path = tmp
        elif csv_path:
            if not os.path.exists(csv_path):
                raise HTTPException(404, f"csv_path nicht gefunden: {csv_path}")
            path = csv_path
        else:
            raise HTTPException(400, "Entweder 'file' (Upload) oder 'csv_path' angeben.")
        if rebap_csv and not os.path.exists(rebap_csv):
            raise HTTPException(404, f"rebap_csv nicht gefunden: {rebap_csv}")
        if spot_csv and not os.path.exists(spot_csv):
            raise HTTPException(404, f"spot_csv nicht gefunden: {spot_csv}")
        if generation_csv and not os.path.exists(generation_csv):
            raise HTTPException(404, f"generation_csv nicht gefunden: {generation_csv}")

        try:
            demands = _parse_float_list(steuve_demands_kw)
            devices = _parse_device_list(steuve_devices)
            pool_asset_list = _parse_json_list(pool_assets, "pool_assets")
            pool_cap = _parse_float_list(pool_shared_cap_kw)
            grid_fee = _parse_float_list(grid_fee_eur_per_kwh)
            tariff_avail = _parse_bool_list(tariff_available)
            result = run_forecast(
                path, utility=utility, region=region, unit=unit,
                ts_col=ts_col, load_col=load_col,
                congestion_threshold_mw=congestion_threshold_mw, steuve_malo=steuve_malo,
                steuve_demands_kw=demands, steuve_devices=devices,
                rolling_redispatch=rolling_redispatch,
                rebap_csv=rebap_csv, spot_csv=spot_csv,
                generation_csv=generation_csv, generation_unit=generation_unit,
                generation_ts_col=generation_ts_col, generation_load_col=generation_load_col,
                realized_economics=realized_economics, submit_to_aemt=submit_to_aemt,
                aemt_adapter=aemt_adapter,
                mmm_price_eur_mwh=mmm_price_eur_mwh,
                rating_kw=rating_kw,
                drift_monitoring=drift_monitoring,
                validate_input=validate_input,
                validate_allow_negative=validate_allow_negative,
                validate_max_plausible=validate_max_plausible,
                asset_rating_kw=asset_rating_kw,
                overload_risk_alpha=overload_risk_alpha,
                thermal_rating_kw=thermal_rating_kw,
                thermal_ambient_c=thermal_ambient_c,
                thermal_hotspot_limit_c=thermal_hotspot_limit_c,
                thermal_risk_alpha=thermal_risk_alpha,
                grid_fee_eur_per_kwh=grid_fee,
                tariff_energy_kwh=tariff_energy_kwh,
                tariff_p_max_kw=tariff_p_max_kw,
                tariff_available=tariff_avail,
                tariff_available_start_hour=tariff_available_start_hour,
                tariff_available_end_hour=tariff_available_end_hour,
                dispatch_plan_enabled=dispatch_plan_enabled,
                dispatch_steuve_energy_kwh=dispatch_steuve_energy_kwh,
                dispatch_steuve_p_max_kw=dispatch_steuve_p_max_kw,
                dispatch_c_short=dispatch_c_short,
                dispatch_c_long=dispatch_c_long,
                dispatch_risk_beta=dispatch_risk_beta,
                dispatch_risk_alpha=dispatch_risk_alpha,
                pool_assets=pool_asset_list,
                pool_shared_cap_kw=pool_cap,
                reconcile_temporal=reconcile_temporal,
                reconcile_temporal_method=reconcile_temporal_method,
                reconcile_temporal_n_test=reconcile_temporal_n_test,
                audit_ledger_path=audit_ledger_path,
                audit_signing_key=audit_signing_key,
                audit_rule_version=audit_rule_version,
                forecast_store_path=forecast_store_path,
                residual_feedback=residual_feedback,
                horizon_days=horizon_days,
                horizon_bands=horizon_bands,
                holiday_add=[s.strip() for s in holiday_add.split(",") if s.strip()] if holiday_add else None,
                holiday_remove=[s.strip() for s in holiday_remove.split(",") if s.strip()] if holiday_remove else None,
            )
        except SystemExit as e:   # robust_load_csv signalisiert Mehrspalten/Spaltenfehler so
            raise HTTPException(422, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))

        saved = store.save(utility, result)
        result["_stored_at"] = saved
        return JSONResponse(result)
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


@app.get("/forecast/{utility}/latest")
def latest(utility: str):
    r = store.latest(utility)
    if r is None:
        raise HTTPException(404, f"Keine Prognose für '{utility}' gespeichert.")
    return r


@app.get("/history/{utility}")
def history(utility: str):
    """Liste aller gespeicherten Prognosetage eines Mandanten (für den Verlauf-Tab)."""
    return {"utility": utility, "dates": store.history(utility)}


@app.get("/report/{utility}/latest", response_class=HTMLResponse)
def report_latest(utility: str):
    """Druckoptimierter Ein-Seiten-Bericht der jüngsten Prognose (Browser: Drucken → PDF)."""
    r = store.latest(utility)
    if r is None:
        raise HTTPException(404, f"Keine Prognose für '{utility}' gespeichert.")
    return render_report_html(r)


@app.get("/report/{utility}/{date}", response_class=HTMLResponse)
def report_by_date(utility: str, date: str):
    """Druckoptimierter Bericht eines bestimmten Prognosetags."""
    r = store.get(utility, date)
    if r is None:
        raise HTTPException(404, f"Keine Prognose für '{utility}' am {date}.")
    return render_report_html(r)


@app.get("/forecast/{utility}/{date}")
def by_date(utility: str, date: str):
    r = store.get(utility, date)
    if r is None:
        raise HTTPException(404, f"Keine Prognose für '{utility}' am {date}.")
    return r


def _ledger_path(utility: str) -> str:
    """§14a-Audit-Ledger je Mandant (neben dessen Store-Ordner). Wird vom Lauf mit
    audit_ledger_path=<dieser Pfad> befüllt, sobald §14a-Eingriffe protokolliert werden."""
    return os.path.join(store._utility_dir(utility), "ledger.jsonl")


@app.get("/compliance/meldebogen/{utility}/{year}/{month}", response_class=HTMLResponse)
def compliance_meldebogen(utility: str, year: int, month: int):
    """W12: §14a-Monats-Meldebogen (VNBdigital-Pflichtfelder) als druckbarer Bericht."""
    path = _ledger_path(utility)
    if not os.path.exists(path):
        raise HTTPException(404, f"Kein §14a-Audit-Ledger für '{utility}' — noch keine protokollierten Eingriffe.")
    return render_meldebogen_html(path, year, month, utility=utility)


@app.get("/compliance/fairness/{utility}", response_class=HTMLResponse)
def compliance_fairness(utility: str):
    """W13: §14a-Diskriminierungsfreiheits-Auditbericht als druckbarer Bericht."""
    path = _ledger_path(utility)
    if not os.path.exists(path):
        raise HTTPException(404, f"Kein §14a-Audit-Ledger für '{utility}' — noch keine protokollierten Eingriffe.")
    return render_fairness_html(path, utility=utility)
