# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Runner: orchestriert die ECHTE netzpilot-Engine zu einem Service-Aufruf.

Lastgang (CSV-Pfad oder load2d) -> leakage-sichere Day-ahead-Prognose (forecast_next_day,
CQR-kalibriert) -> optional §14a-Fahrplan bei prognostiziertem Engpass. Keine Re-Implementierung:
nutzt robust_load_csv (verifizierter Loader), forecast_next_day und control.schema.make_fahrplan.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from netzpilot.features.build import apply_holiday_overrides, get_holidays, to_daily_local
from netzpilot.eval.backtest import rolling_origin
from netzpilot.eval.economics import saving_from_real_rebap, saving_from_rebap_spot, expected_saving_eur
from netzpilot.eval.mehrmindermengen import compare_forecasts_mmm
from netzpilot.forecast import forecast_next_day
from netzpilot.horizon import forecast_days
from netzpilot.models.robust_corrector import ShrunkCorrector
from netzpilot.control.schema import (
    make_fahrplan,
    MIN_GUARANTEED_KW,
    devices_are_heterogeneous,
    normalize_steuve_devices,
    steuve_demands_from_devices,
)
from netzpilot.control.optimize import optimize_setpoints, optimize_setpoints_heterogen, naive_shed_kw
from netzpilot.control.redispatch import rolling_redispatch as compute_rolling_redispatch, from_single_path
from netzpilot.control.vpp_pool import pool_dispatch
from netzpilot.grid.overload import hosting_capacity_kw, overload_forecast
from netzpilot.grid.thermal import probabilistic_thermal_risk
from netzpilot.models.pooled_corrector import PooledCorrector
from netzpilot.models.pool_prior import load_prior
from netzpilot.data.rebap import load_rebap
from netzpilot.data.spot_da import load_rebap_spot_pairs
from netzpilot.service.audit_ledger import append_forecast_audit
from netzpilot.service.forecast_store import record_forecast, realized_track_record
from netzpilot.service.dispatch_plan import build_dispatch_plan
from netzpilot.service.drift_monitor import build_drift_payload
from netzpilot.service.input_validation import validate_hourly_series
from netzpilot.service.reconcile_temporal import build_temporal_reconciliation_payload
from netzpilot.service.tariff_schedule import build_tariff_schedule
from scripts.pilot_in_a_box import robust_load_csv


def _load2d_from_csv(csv_path: str, ts_col=None, load_col=None, unit="MW", *,
                     validate_input: bool = False, validate_allow_negative: bool = False,
                     validate_max_plausible=None, return_validation: bool = False):
    hourly, _ts, _lc, meta = robust_load_csv(csv_path, ts_col=ts_col, load_col=load_col,
                                             unit=unit, return_meta=True)
    input_validation = {"enabled": False, "original_preserved": True}
    if validate_input:
        hourly, input_validation = validate_hourly_series(
            hourly,
            allow_negative=validate_allow_negative,
            max_plausible=validate_max_plausible,
            apply_cleaned=True,
        )
    load2d, days, _ = to_daily_local(hourly)
    if return_validation:
        return load2d, days, meta, hourly, input_validation
    return load2d, days, meta, hourly


def _corrector_factory(prior, load2d, used_pool):
    """Wählt den Korrektor: gepoolt (Pool-Prior, bei wenig Historie) oder Standard (ShrunkCorrector).

    Der Pool-Prior lebt im LAST-NORMALISIERTEN Raum. Die Wochen-Residuen r=load-load_vorwoche skalieren
    linear mit der Last → der Prior-Koeffizientenvektor wird mit der mittleren Hauslast hochskaliert,
    damit er in denselben (Roh-MW-)Raum passt wie der eigene Fit.
    """
    import numpy as _np
    if not used_pool or not prior:
        return lambda: ShrunkCorrector(10.0)
    scale = float(_np.mean(load2d[8:])) if len(load2d) > 8 else float(_np.mean(load2d))
    w_pool_scaled = _np.asarray(prior["w_pool"], dtype=float) * scale
    return lambda: PooledCorrector(w_pool_scaled, lam=10.0, tau_days=30.0)


def _residual_from_csvs(load_hourly, gen_csv, *, gen_ts_col=None, gen_load_col=None,
                        gen_unit="MW"):
    """Residuallast = Last - Erzeugung (PV+Wind), stuendlich auf gemeinsamer UTC-Achse.

    Laedt die Erzeugungsreihe mit demselben robusten Loader, schneidet sie auf die
    gemeinsamen Stunden der Lastreihe (inner join) und bildet die Differenz. KEINE neue
    Prognose-Mathematik: die abgeleitete Residuallast-Reihe wird spaeter mit derselben
    leakage-sicheren Engine (forecast_next_day) prognostiziert.

    Vorzeichen: Erzeugung positiv -> Residuallast = Last - Erzeugung (hoch = Netzbezug,
    negativ = Rueckspeisung). Gibt (residual2d, resid_days, gen_meta, n_common_hours).
    """
    gen_hourly, _ts, _lc, gen_meta = robust_load_csv(
        gen_csv, ts_col=gen_ts_col, load_col=gen_load_col, unit=gen_unit, return_meta=True)
    return _residual_from_series(load_hourly, gen_hourly, gen_meta)


def _residual_from_series(load_hourly, gen_hourly, gen_meta):
    """Residuallast aus bereits geladener Last- + Erzeugungs-Stundenreihe (gemeinsame UTC-Achse)."""
    import pandas as _pd
    if gen_hourly.index.tz is None:
        gen_hourly = gen_hourly.copy()
        gen_hourly.index = _pd.to_datetime(gen_hourly.index, utc=True)
    common = load_hourly.index.intersection(gen_hourly.index)
    if len(common) < 24 * 60:
        raise ValueError(
            f"Zu wenig gemeinsame Stunden Last & Erzeugung ({len(common)}); "
            "mind. ~60 Tage Ueberlappung noetig."
        )
    residual_hourly = load_hourly.loc[common] - gen_hourly.loc[common]
    residual2d, resid_days, _ = to_daily_local(residual_hourly)
    return residual2d, resid_days, gen_meta, int(len(common))


def _naive_shed_devices_kw(devices: list[dict]) -> float:
    if devices_are_heterogeneous(devices):
        return round(sum(
            max(0.0, float(d["demand_kw"]) - float(d.get("floor_kw", MIN_GUARANTEED_KW)))
            for d in devices
        ), 3)
    return naive_shed_kw(steuve_demands_from_devices(devices))


def _optimize_devices(devices: list[dict], cap_kw: float) -> tuple[dict, str]:
    if devices_are_heterogeneous(devices):
        return optimize_setpoints_heterogen(devices, cap_kw), "heterogeneous weighted shedding"
    return optimize_setpoints(steuve_demands_from_devices(devices), cap_kw), "fair-min water-filling"


def _setpoints_from_limits(devices: list[dict], limits_kw: list[float], start: str, end: str) -> list[dict]:
    hetero = devices_are_heterogeneous(devices)
    setpoints = []
    for i, (dev, lim) in enumerate(zip(devices, limits_kw)):
        floor_kw = float(dev.get("floor_kw", MIN_GUARANTEED_KW))
        sp = {
            "start_utc": start,
            "end_utc": end,
            "p_limit_kw": round(max(floor_kw, float(lim)), 3),
        }
        if hetero:
            sp["floor_kw"] = round(floor_kw, 3)
            sp["weight"] = round(float(dev.get("weight", 1.0)), 6)
            sp["demand_kw"] = round(float(dev["demand_kw"]), 3)
            sp["device_index"] = i
            if dev.get("device_id"):
                sp["device_id"] = dev["device_id"]
        setpoints.append(sp)
    return setpoints


def _congestion_and_fahrplan(hours, fc_date, threshold_mw, steuve_malo,
                             steuve_demands_kw=None, steuve_devices=None):
    """Erkennt Engpassfenster (P90 > Schwelle) und baut optional einen §14a-Fahrplan-Entwurf.

    Wird auf die §14a-relevante Groesse angewandt: Residuallast wenn vorhanden, sonst Last.

    steuve_demands_kw (optional): angeforderte Leistungen je steuerbarer Verbrauchseinrichtung [kW].
    Sind sie bekannt, rechnet NetzPilot den OPTIMALEN (minimal+fair abgeregelten) Fahrplan statt
    pauschal alle auf 4,2 kW zu dimmen (control.optimize). Sonst Fallback auf die Pauschal-Dimmung.
    """
    if threshold_mw is None:
        return None, None
    peak_hours = [h for h in hours if h["p90"] > threshold_mw]
    if not peak_hours:
        return None, None
    first, last = peak_hours[0]["hour"], peak_hours[-1]["hour"]
    d = pd.Timestamp(fc_date)
    start = d.replace(hour=first).isoformat()
    end = (d.replace(hour=last) + pd.Timedelta(hours=1)).isoformat()
    cong = {
        "threshold_mw": threshold_mw,
        "window_hours": [h["hour"] for h in peak_hours],
        "max_p90_mw": max(h["p90"] for h in peak_hours),
    }
    fahrplan = None
    if steuve_malo:
        devices = normalize_steuve_devices(steuve_demands_kw, steuve_devices)
        if devices:
            # OPTIMIERT: minimale, faire Abregelung. Netzentlastungsziel = wie viel die steuVE-Summe
            # runter muss, damit P90 die Schwelle haelt (Ueberlast in kW auf die steuVE umgelegt).
            overload_mw = max(h["p90"] for h in peak_hours) - threshold_mw
            total_demand_kw = float(sum(steuve_demands_from_devices(devices)))
            cap_kw = max(0.0, total_demand_kw - overload_mw * 1000.0)
            opt, method = _optimize_devices(devices, cap_kw)
            sps = _setpoints_from_limits(devices, opt["limits_kw"], start, end)
            fahrplan = make_fahrplan(malo=steuve_malo, setpoints=sps, reason="forecast_congestion")
            fahrplan["optimization"] = {
                "method": method,
                "heterogeneous": devices_are_heterogeneous(devices),
                "n_steuve": len(devices),
                "cap_kw": round(cap_kw, 1),
                "total_shed_kw": opt["total_shed_kw"],
                "naive_shed_kw": _naive_shed_devices_kw(devices),
                "feasible_under_14a": opt["feasible"],
            }
        else:
            # Fallback: pauschale Dimmung auf garantierte Mindestleistung (ein Setpoint).
            fahrplan = make_fahrplan(
                malo=steuve_malo,
                setpoints=[{"start_utc": start, "end_utc": end, "p_limit_kw": MIN_GUARANTEED_KW}],
                reason="forecast_congestion",
            )
    return cong, fahrplan


def _rolling_residuals_by_hour(load2d, days, holiday_set, *, max_history_days: int = 120) -> tuple[list[list[float]], int, int]:
    keep = min(max_history_days, len(load2d))
    first = 8
    max_test = keep - first - 1
    if max_test < 1:
        raise ValueError("Zu wenig Historie fuer rolling-origin Residuen.")
    n_test = min(42, max(7, keep - 60), max_test)
    R, _summary = rolling_origin(
        load2d[-keep:],
        days[-keep:],
        lambda: ShrunkCorrector(10.0),
        n_test=n_test,
        holiday_set=holiday_set,
    )
    residuals_kw = (
        np.asarray(R["actual"], float) - np.asarray(R["model"], float)
    ) * 1000.0
    return [[float(x) for x in residuals_kw[h::24]] for h in range(24)], n_test, keep


def _thermal_ambient24(thermal_ambient_c, *, weather_csv: str | None, forecast_date) -> tuple[list[float], str]:
    if thermal_ambient_c is not None:
        if isinstance(thermal_ambient_c, (int, float)):
            return [float(thermal_ambient_c)] * 24, "provided_scalar"
        vals = [float(x) for x in thermal_ambient_c]
        if len(vals) != 24:
            raise ValueError("thermal_ambient_c must be a scalar or 24 values.")
        return vals, "provided_24h"
    if weather_csv:
        df = pd.read_csv(weather_csv)
        ts_col = None
        for c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
            if parsed.notna().mean() > 0.7:
                ts_col = c
                break
        if ts_col and "temperature_2m" in df.columns:
            idx = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dt.tz_convert("Europe/Berlin")
            temp = pd.Series(pd.to_numeric(df["temperature_2m"], errors="coerce").to_numpy(float), index=idx)
            target = pd.Timestamp(forecast_date).date()
            by_day = temp[temp.index.date == target].dropna()
            if len(by_day) >= 24:
                return [float(x) for x in by_day.iloc[:24].to_list()], "weather_csv_temperature_2m"
            recent = temp.dropna()
            if len(recent) >= 24:
                return [float(x) for x in recent.iloc[-24:].to_list()], "weather_csv_temperature_2m_last24_fallback"
    return [20.0] * 24, "default_20c_assumption"


def _normalize_cap_series(raw, *, name: str, horizon: int = 24) -> list[float]:
    if raw is None:
        raise ValueError(f"{name} fehlt.")
    if isinstance(raw, (int, float)):
        return [float(raw)] * horizon
    values = [float(x) for x in raw]
    if not values:
        raise ValueError(f"{name} darf nicht leer sein.")
    return values


def _apply_rating_truth(*, rating_kw, congestion_threshold_mw, asset_rating_kw, thermal_rating_kw) -> tuple:
    """Reconcile the single asset-limit truth across grid-facing service paths."""
    sources = []
    if rating_kw is not None:
        sources.append(("rating_kw", float(rating_kw)))
    if asset_rating_kw is not None:
        sources.append(("asset_rating_kw", float(asset_rating_kw)))
    if congestion_threshold_mw is not None:
        sources.append(("congestion_threshold_mw", float(congestion_threshold_mw) * 1000.0))
    if thermal_rating_kw is not None:
        sources.append(("thermal_rating_kw", float(thermal_rating_kw)))
    for name, value in sources:
        if value <= 0:
            raise ValueError(f"{name} muss > 0 sein.")
    unified = float(rating_kw) if rating_kw is not None else None
    if unified is not None:
        for name, value in sources:
            if abs(value - unified) > 1e-6:
                raise ValueError(
                    "rating_kw muss mit allen Netzgrenzen uebereinstimmen "
                    f"({name}={value:.3f} kW != rating_kw={unified:.3f} kW)."
                )
        congestion_threshold_mw = unified / 1000.0
        asset_rating_kw = unified
        if thermal_rating_kw is not None:
            thermal_rating_kw = unified
    elif sources:
        unified = sources[0][1]
        for name, value in sources[1:]:
            if abs(value - unified) > 1e-6:
                raise ValueError(
                    "Netzgrenzen muessen eine Rating-Wahrheit teilen "
                    f"({name}={value:.3f} kW != {sources[0][0]}={unified:.3f} kW)."
                )
    return congestion_threshold_mw, asset_rating_kw, thermal_rating_kw, unified


def run_forecast(csv_path: str, *, utility: str = "default", region: str = "NW",
                 unit: str = "MW", ts_col=None, load_col=None,
                 congestion_threshold_mw: float | None = None,
                 steuve_malo: str | None = None, steuve_demands_kw=None,
                 steuve_devices=None, rolling_redispatch: bool = False,
                 rebap_prices=None, rebap_csv: str | None = None,
                 spot_prices=None, spot_csv: str | None = None,
                 generation_csv: str | None = None, generation_unit: str = "MW",
                 generation_ts_col=None, generation_load_col=None,
                 weather_csv: str | None = None, pv_capacity_mw: float = 0.0,
                 wind_capacity_mw: float = 0.0, latitude: float = 51.16, longitude: float = 10.45,
                 realized_economics: bool = False, submit_to_aemt: bool = False,
                 aemt_adapter: str = "mock",
                 mmm_price_eur_mwh: float | None = None,
                 rating_kw: float | None = None,
                 drift_monitoring: bool = False,
                 drift_store_dir: str = "data_cache/drift",
                 validate_input: bool = True,
                 validate_allow_negative: bool = False,
                 validate_max_plausible: float | None = None,
                 asset_rating_kw: float | None = None,
                 overload_risk_alpha: float = 0.05,
                 thermal_rating_kw: float | None = None,
                 thermal_ambient_c=None,
                 thermal_hotspot_limit_c: float = 120.0,
                 thermal_risk_alpha: float = 0.05,
                 grid_fee_eur_per_kwh=None,
                 tariff_energy_kwh: float | None = None,
                 tariff_p_max_kw: float | None = None,
                 tariff_available=None,
                 tariff_available_start_hour: int | None = None,
                 tariff_available_end_hour: int | None = None,
                 dispatch_plan_enabled: bool = False,
                 dispatch_steuve_energy_kwh: float | None = None,
                 dispatch_steuve_p_max_kw: float | None = None,
                 dispatch_c_short: float = 0.20,
                 dispatch_c_long: float = 0.10,
                 dispatch_risk_beta: float = 0.0,
                 dispatch_risk_alpha: float = 0.95,
                 pool_assets=None,
                 pool_shared_cap_kw=None,
                 reconcile_temporal: bool = False,
                 reconcile_temporal_method: str = "wls_struct",
                 reconcile_temporal_n_test: int = 7,
                 audit_ledger_path: str | None = None,
                 audit_signing_key: str | None = None,
                 audit_rule_version: str = "netzpilot-paragraph14a-v1",
                 forecast_store_path: str | None = None,
                 residual_feedback: bool | None = None,
                 horizon_days: int = 1,
                 horizon_bands: str = "k1",
                 holiday_add=None,
                 holiday_remove=None) -> dict:
    """Erzeugt die Day-ahead-Prognose (P10/P50/P90) für morgen und – falls ein Engpass
    prognostiziert wird – einen §14a-konformen Fahrplan-Entwurf.

    congestion_threshold_mw: P90 > Schwelle in einer Stunde => Engpassfenster.

    Residuallast (optional): wird eine Erzeugungs-CSV (PV+Wind) uebergeben, prognostiziert der
    Dienst zusaetzlich die RESIDUALLAST = Last - Erzeugung mit DERSELBEN Engine (keine neue
    Prognose-Mathematik). Die §14a-Engpass-/Fahrplanlogik laeuft dann auf der Residuallast —
    der physikalisch korrekten Netzgroesse. Ohne Erzeugung unveraendert auf der Last.

    economics-Logik (drei ehrliche Stufen, wenn reBAP UND Spot-DA-Preise vorliegen):
      economics_expected    = ERWARTUNGSWERT ueber signierten Mittel-Aufschlag mean(reBAP-Spot)
                              — die ehrliche Headline-€-Zahl (2024 ~7 EUR/MWh).
      economics             = Risiko-/Stressband ueber |reBAP - Spot| (Median + P25-P75);
                              misst Volatilitaet, ueberschaetzt den ERWARTETEN Nutzen.
      economics_upper_bound = |reBAP| absolut — ueberschaetzter oberster Rand.
    Ohne Spot fallback auf |reBAP|, klar als Upper Bound gelabelt.
    """
    horizon_days = int(horizon_days)
    if not 1 <= horizon_days <= 7:
        raise ValueError("horizon_days muss in 1..7 liegen.")
    if horizon_bands not in ("k1", "per_horizon"):
        raise ValueError("horizon_bands muss 'k1' oder 'per_horizon' sein.")
    if horizon_days < 2 and horizon_bands != "k1":
        raise ValueError("horizon_bands='per_horizon' ist nur mit horizon_days>=2 gueltig.")

    congestion_threshold_mw, asset_rating_kw, thermal_rating_kw, unified_rating_kw = _apply_rating_truth(
        rating_kw=rating_kw,
        congestion_threshold_mw=congestion_threshold_mw,
        asset_rating_kw=asset_rating_kw,
        thermal_rating_kw=thermal_rating_kw,
    )
    load2d, days, meta, load_hourly, input_validation = _load2d_from_csv(
        csv_path,
        ts_col=ts_col,
        load_col=load_col,
        unit=unit,
        validate_input=validate_input,
        validate_allow_negative=validate_allow_negative,
        validate_max_plausible=validate_max_plausible,
        return_validation=True,
    )
    # Pool-Prior (falls vorhanden) erlaubt verlässliche Prognosen schon ab WENIG Historie —
    # genau dort wirkt der Multi-Mandanten-Effekt. Ohne Prior bleibt die 60-Tage-Mindestgrenze.
    prior = load_prior()
    min_days = 21 if prior else 60
    if len(load2d) < min_days:
        raise ValueError(f"Zu wenig vollständige Tage ({len(load2d)}); mind. ~{min_days} nötig.")
    # Kalenderjahre = Datenjahre PLUS Zieljahre (D+1..D+horizon): endet die Historie am 31.12.,
    # liegt der Prognosetag im Folgejahr — ohne diese Erweiterung hätte z. B. Neujahr KEIN
    # Feiertags-Flag (Jahresgrenzen-Bug, gefunden 2026-06-05 beim Override-Audit).
    _years = {pd.Timestamp(d).year for d in days}
    _years |= {(pd.Timestamp(days[-1]) + pd.Timedelta(days=k)).year for k in range(1, horizon_days + 1)}
    hol = get_holidays(sorted(_years), region)
    if holiday_add or holiday_remove:
        # Nutzer-Override: explizit markierte Tage (z. B. Brückentag, lokaler Feiertag) fließen in
        # DENSELBEN holiday_set wie der Kalender — Feature, Anker und Training bleiben konsistent.
        hol = apply_holiday_overrides(hol, holiday_add, holiday_remove)
    used_pool = bool(prior and len(load2d) < 60)

    # T51: opt-in Forecast-Store. Vor der neuen Prognose: realisierten Track-Record gegen vorhandene
    # Actuals auswerten und ggf. exaktes Vortagsresiduum (last_residual) fuer T50 nutzen.
    track_record_payload = None
    last_residual_from_store = None
    if forecast_store_path:
        try:
            actuals_by_date = {str(pd.Timestamp(days[i]).date()): [float(x) for x in load2d[i]]
                               for i in range(len(load2d))}
            tr = realized_track_record(forecast_store_path, actuals_by_date)
            last_residual_from_store = tr.get("last_residual")
            n_realized = (tr["aggregate"] or {}).get("n_days_realized", 0) if tr.get("aggregate") else 0
            track_record_payload = {
                "chain_ok": bool(tr["chain_ok"]),
                "n_forecasts_stored": int(tr["n_forecasts_stored"]),
                "n_realized": int(n_realized),
                "n_pending": max(0, int(tr["n_forecasts_stored"]) - int(n_realized)),
                "n_duplicates_superseded": int(tr.get("n_duplicates_superseded", 0)),
                "n_skipped_period_mismatch": int(tr.get("n_skipped_period_mismatch", 0)),
                "aggregate": tr.get("aggregate"),
                "last_residual_date": tr.get("last_residual_date"),
                "last_30_days": (tr.get("days") or [])[-30:],
                "note": tr.get("note"),
            }
        except Exception as e:
            track_record_payload = {
                "chain_ok": False, "n_forecasts_stored": 0, "n_realized": 0, "n_pending": 0,
                "error": str(e),
                "note": "Track-Record konnte nicht ausgewertet werden — Store ggf. neu.",
            }

    # T51-Nachtrag: residual_feedback explizit steuerbar (None = auto: an wenn Store-Pfad gesetzt).
    # Damit ist reines Logging ohne P50-Aenderung moeglich (Store + residual_feedback=False).
    _rf_active = bool(forecast_store_path) if residual_feedback is None else bool(residual_feedback)
    corrector_factory = _corrector_factory(prior, load2d, used_pool)
    fc = forecast_next_day(load2d, days, corrector_factory, holiday_set=hol,
                           calibrate=True,
                           residual_feedback=_rf_active,
                           last_residual=last_residual_from_store if _rf_active else None)
    horizon_payload = None
    if horizon_days >= 2:
        fd = forecast_days(load2d, days, corrector_factory, horizon=horizon_days, holiday_set=hol,
                           bands=horizon_bands)
        horizon_payload = {
            "days": [d for d in fd["days"] if int(d.get("horizon", 0)) >= 2],
            "bands_mode": fd["bands_mode"],
            "bands": fd["bands"],
            "bands_note": fd["bands"],
            "issued_after": fd["issued_after"],
            "note": "D+1 produktiv bleibt out['forecast'] (kalibriert + RF); horizon.days beginnt bei D+2.",
        }
    out_pool_info = {"used_pool_prior": used_pool,
                     "pool_n_houses": (prior or {}).get("n_houses") if used_pool else None}
    if rebap_prices is None and rebap_csv:
        rebap_prices = load_rebap(rebap_csv)
    if spot_prices is None and spot_csv and rebap_csv:
        # Beide CSVs auf gemeinsamer QH-Achse paaren (NaN-Paare gedroppt)
        rebap_prices, spot_prices = load_rebap_spot_pairs(rebap_csv, spot_csv)

    hours = fc["hours"]
    mean_load = float(np.mean(load2d[-7:]))  # Referenz: Mittel der letzten Woche
    out = {
        "utility": utility,
        "forecast_date": fc["date"],
        "unit": "MW",
        "load_column": meta.get("load_col"),
        "load_level": meta.get("load_level"),
        "n_days_history": int(len(load2d)),
        "recent_mean_load_mw": round(mean_load, 3),
        "pool_prior": out_pool_info,
        "input_validation": input_validation,
        "asset_limit": (
            {
                "rating_kw": round(float(unified_rating_kw), 3),
                "rating_mw": round(float(unified_rating_kw) / 1000.0, 6),
                "source": "rating_kw" if rating_kw is not None else "reconciled_existing_limits",
                "feeds": {
                    "congestion_threshold_mw": (
                        float(congestion_threshold_mw) if congestion_threshold_mw is not None else None
                    ),
                    "asset_rating_kw": (
                        round(float(asset_rating_kw), 3) if asset_rating_kw is not None else None
                    ),
                    "thermal_rating_kw": (
                        round(float(thermal_rating_kw), 3) if thermal_rating_kw is not None else None
                    ),
                },
                "note": "Eine Rating-Wahrheit fuer Redispatch/Dispatch-Cap und Asset-Risiko.",
            }
            if unified_rating_kw is not None else None
        ),
        "forecast": hours,
        "coverage_scale_used": fc.get("coverage_scale_used"),
        "coverage_calibrated": bool(fc.get("coverage_calibrated", False)),
        "residual_feedback": fc.get("residual_feedback"),    # T50/T51
        "track_record": track_record_payload,                # T51
        "horizon": horizon_payload,                           # T52: D+2..D+H, D+1 bleibt forecast
        "holiday_overrides": ({
            "added": sorted(str(pd.Timestamp(x).date()) for x in (holiday_add or [])),
            "removed": sorted(str(pd.Timestamp(x).date()) for x in (holiday_remove or [])),
            "target_is_holiday": pd.Timestamp(fc["date"]).date() in hol,
            "caveat": ("Override = Nutzer-Annahme (Tag wie Feiertag behandeln bzw. nicht); "
                       "gemessen belegt ist das Verhalten echter Feiertage."),
        } if (holiday_add or holiday_remove) else None),
        "residual_forecast": None,
        "economics_expected": None,
        "economics": None,
        "economics_upper_bound": None,
        "economics_realized": None,
        "drift": None,
        "overload": None,
        "hosting_capacity": None,
        "thermal": None,
        "tariff_schedule": None,
        "dispatch_plan": None,
        "mmm": None,
        "pool_dispatch": None,
        "reconcile_temporal": None,
        "audit": None,
        "congestion": None,
        "fahrplan": None,
        "fahrplan_lpc": None,
        "redispatch": None,
    }

    if reconcile_temporal:
        try:
            out["reconcile_temporal"] = build_temporal_reconciliation_payload(
                csv_path,
                ts_col=ts_col or "Text",
                load_col=load_col or meta.get("load_col") or "Reihe1",
                unit=unit,
                region=region,
                method=reconcile_temporal_method,
                n_test=int(reconcile_temporal_n_test),
            )
        except ValueError as e:
            out["reconcile_temporal"] = {
                "status": "not_available",
                "reason": str(e),
                "method": reconcile_temporal_method,
                "source": "requires a complete 15-minute loadgang with 96 slots per normal day",
            }

    # --- Residuallast (optional): Last - Erzeugung, gleiche Engine ---
    # Erzeugung aus fertiger CSV ODER physikalisch aus Wetter-CSV + installierten Kapazitäten.
    residual_hours = None
    gen_source = None
    if generation_csv:
        residual2d, resid_days, gen_meta, n_common = _residual_from_csvs(
            load_hourly, generation_csv, gen_ts_col=generation_ts_col,
            gen_load_col=generation_load_col, gen_unit=generation_unit)
        gen_source = "csv"
    elif weather_csv and (pv_capacity_mw > 0 or wind_capacity_mw > 0):
        from netzpilot.data.generation_forecast import generation_from_weather_csv
        gen_hourly = generation_from_weather_csv(
            weather_csv, pv_capacity_mw=pv_capacity_mw, wind_capacity_mw=wind_capacity_mw,
            latitude=latitude, longitude=longitude)
        residual2d, resid_days, gen_meta, n_common = _residual_from_series(
            load_hourly, gen_hourly,
            {"load_col": f"PV {pv_capacity_mw} MW + Wind {wind_capacity_mw} MW (aus Wetter)"})
        gen_source = "weather"
    if residual_hours is None and gen_source is not None:
        _years_r = {pd.Timestamp(d).year for d in resid_days}
        _years_r.add((pd.Timestamp(resid_days[-1]) + pd.Timedelta(days=1)).year)   # Zieljahr (s. o.)
        hol_r = get_holidays(sorted(_years_r), region)
        if holiday_add or holiday_remove:
            hol_r = apply_holiday_overrides(hol_r, holiday_add, holiday_remove)
        fc_r = forecast_next_day(residual2d, resid_days, lambda: ShrunkCorrector(10.0),
                                 holiday_set=hol_r, calibrate=True)
        residual_hours = fc_r["hours"]
        out["residual_forecast"] = {
            "forecast_date": fc_r["date"],
            "n_days_history": int(len(residual2d)),
            "n_common_hours": n_common,
            "generation_column": gen_meta.get("load_col"),
            "generation_source": gen_source,
            "recent_mean_residual_mw": round(float(np.mean(residual2d[-7:])), 3),
            "forecast": residual_hours,
            "definition": "Residuallast = Last - Erzeugung (PV+Wind); §14a-relevante Netzgroesse.",
            "note_economics": "Die €-Oekonomie unten basiert weiterhin auf der LAST (nicht der "
                              "Residuallast). Belastbar waere die Bilanzkreis-Abweichung auf dem "
                              "Netzbezug — das braucht reale Bilanzkreisdaten des Stadtwerks.",
        }
    if rebap_prices is not None and len(load2d) >= 45:
        keep = min(120, len(load2d))
        bt_load2d, bt_days = load2d[-keep:], days[-keep:]
        n_test = min(14, max(7, keep - 30))
        R, _sm = rolling_origin(bt_load2d, bt_days, lambda: ShrunkCorrector(10.0),
                                n_test=n_test, holiday_set=hol)
        if realized_economics:
            if rebap_csv and spot_csv:
                from netzpilot.eval.bilanzkreis_realized import realized_settlement_from_backtest
                try:
                    out["economics_realized"] = realized_settlement_from_backtest(
                        R, bt_days, n_test, rebap_csv, spot_csv)
                except ValueError as e:
                    out["economics_realized"] = {"status": "not_available", "reason": str(e)}
            else:
                out["economics_realized"] = {
                    "status": "not_available",
                    "reason": "realized_economics braucht rebap_csv und spot_csv fuer Zeit-Alignment.",
                }
        actual = np.asarray(R["actual"], float)
        mae_model = float(np.mean(np.abs(np.asarray(R["model"], float) - actual)))
        mae_snaive = float(np.mean(np.abs(np.asarray(R["snaive"], float) - actual)))
        delta_mae = max(0.0, mae_snaive - mae_model)

        upper = saving_from_real_rebap(delta_mae, rebap_prices)
        upper.update({
            "method": "trailing rolling-origin backtest vs. seasonal-naive",
            "n_test_days": n_test,
            "rebap_source": rebap_csv or "provided prices",
            "caveat": "|reBAP|-Annahme ueberschaetzt den realen Hebel (gespart wird nur der Aufschlag).",
        })
        if spot_prices is not None:
            meta_econ = {
                "method": "trailing rolling-origin backtest vs. seasonal-naive",
                "n_test_days": n_test,
                "rebap_source": rebap_csv or "provided prices",
                "spot_source": spot_csv or "provided prices",
            }
            expected = expected_saving_eur(delta_mae, rebap_prices, spot_prices)
            expected.update(meta_econ)
            expected["caveat"] = (
                "ERWARTUNGSWERT (signierter Mittel-Aufschlag mean(reBAP-Spot)). Downside-Schutz, "
                "kein garantierter linearer Ertrag. Belastbarste Zahl bleibt die reale "
                "Bilanzkreis-Abrechnung des Stadtwerks."
            )
            band = saving_from_rebap_spot(delta_mae, rebap_prices, spot_prices)
            band.update(meta_econ)
            band["caveat"] = (
                "Risiko-/Stressband |reBAP - Spot| (Volatilitaet, beide Richtungen) — "
                "ueberschaetzt den ERWARTETEN Nutzen; ehrliche Headline = economics_expected."
            )
            out["economics_expected"] = expected
            out["economics"] = band
            out["economics_upper_bound"] = upper
        else:
            upper["caveat"] = (
                "Ohne Spot-DA-Reihe nur |reBAP|-Annahme verfuegbar — UEBERSCHAETZT den Nutzen "
                "(Spot-Kosten haetten ohnehin gezahlt werden muessen). Spot-CSV liefern fuer "
                "belastbare Aufschlag-Headline."
            )
            out["economics"] = upper

    if mmm_price_eur_mwh is not None:
        price = float(mmm_price_eur_mwh)
        if len(load2d) < 45:
            out["mmm"] = {
                "status": "not_available",
                "reason": f"Zu wenig Historie fuer MMM-Backtest ({len(load2d)} Tage; mind. ~45 noetig).",
                "mmm_price_eur_mwh": round(price, 4),
            }
        else:
            keep = min(120, len(load2d))
            bt_load2d, bt_days = load2d[-keep:], days[-keep:]
            n_test = min(42, max(14, keep - 60))
            R_mmm, _sm_mmm = rolling_origin(
                bt_load2d,
                bt_days,
                lambda: ShrunkCorrector(10.0),
                n_test=n_test,
                holiday_set=hol,
            )
            actual = np.asarray(R_mmm["actual"], float)
            snaive = np.asarray(R_mmm["snaive"], float)
            model = np.asarray(R_mmm["model"], float)
            cmp = compare_forecasts_mmm(actual, snaive, model, price, dt_h=1.0)
            out["mmm"] = {
                "status": "available",
                "mmm_price_eur_mwh": round(price, 4),
                "dt_h": 1.0,
                "n_test_days": n_test,
                "history_days_used": keep,
                "forecast_a": "seasonal_naive",
                "forecast_b": "NetzPilot",
                "abs_volumen_snaive_mwh": cmp["abs_volumen_a_mwh"],
                "abs_volumen_netzpilot_mwh": cmp["abs_volumen_b_mwh"],
                "abs_volumen_reduktion_mwh": cmp["abs_volumen_reduktion_mwh"],
                "abs_volumen_reduktion_at_price_eur": round(
                    cmp["abs_volumen_reduktion_mwh"] * price, 2
                ),
                "netto_snaive_mwh": cmp["netto_a_mwh"],
                "netto_netzpilot_mwh": cmp["netto_b_mwh"],
                "report_snaive": cmp["report_a"],
                "report_netzpilot": cmp["report_b"],
                "basis": (
                    "Trailing rolling-origin Backtest: Forecast/Seasonal-Naive vs. Ist, "
                    "MW je Stunde als MWh mit dt_h=1.0."
                ),
                "caveat": (
                    "Mehr-/Mindermengen sind EDM-Reconciliation am regulierten MMM-Preis; "
                    "MMM ist nicht QH-Ausgleichsenergie/reBAP. Preis ist Eingabe/Config."
                ),
            }

    # Drift monitoring (optional, additive): realized rolling-origin forecast errors against a
    # persisted reference distribution. This only warns; it never starts automatic retraining.
    if drift_monitoring:
        try:
            keep = min(120, len(load2d))
            bt_load2d, bt_days = load2d[-keep:], days[-keep:]
            n_test = min(42, max(14, keep - 30))
            R_drift, _sm_drift = rolling_origin(
                bt_load2d,
                bt_days,
                lambda: ShrunkCorrector(10.0),
                n_test=n_test,
                holiday_set=hol,
            )
            out["drift"] = build_drift_payload(
                R_drift,
                utility=utility,
                base_dir=drift_store_dir,
                reference_days=28,
                recent_days=14,
                min_recent_days=7,
                metadata={
                    "forecast_date": fc["date"],
                    "utility": utility,
                    "csv_path": csv_path,
                    "unit": "MW",
                    "n_test_days": n_test,
                    "history_days_used": keep,
                    "model": "ShrunkCorrector(10.0)",
                },
            )
        except ValueError as e:
            out["drift"] = {
                "status": "not_available",
                "needs_recalibration": False,
                "reasons": [str(e)],
                "coverage": None,
                "action": "warn_only_no_auto_retraining",
            }

    # §14a-Engpass auf der physikalisch korrekten Groesse: Residuallast wenn vorhanden, sonst Last.
    cong_hours = residual_hours if residual_hours is not None else hours
    cong_date = out["residual_forecast"]["forecast_date"] if residual_hours is not None else fc["date"]
    out["congestion"], out["fahrplan"] = _congestion_and_fahrplan(
        cong_hours, cong_date, congestion_threshold_mw, steuve_malo,
        steuve_demands_kw, steuve_devices)
    if out["congestion"] is not None:
        out["congestion"]["basis"] = "residual" if residual_hours is not None else "load"

    if asset_rating_kw is not None:
        rating_kw = float(asset_rating_kw)
        if congestion_threshold_mw is not None:
            threshold_kw = float(congestion_threshold_mw) * 1000.0
            if abs(rating_kw - threshold_kw) > 1e-6:
                raise ValueError(
                    "asset_rating_kw muss zur congestion_threshold_mw passen "
                    f"({rating_kw:.3f} kW != {threshold_kw:.3f} kW)."
                )
        basis = "residual" if residual_hours is not None else "load"
        overload_load2d = residual2d if residual_hours is not None else load2d
        overload_days = resid_days if residual_hours is not None else days
        overload_holidays = hol_r if residual_hours is not None else hol
        residuals_by_hour, n_test_overload, history_overload = _rolling_residuals_by_hour(
            overload_load2d,
            overload_days,
            overload_holidays,
        )
        point_kw = [float(h["p50"]) * 1000.0 for h in cong_hours]
        overload_payload = overload_forecast(
            point_kw,
            residuals_by_hour,
            rating_kw,
            dt_h=1.0,
            risk_alpha=overload_risk_alpha,
        )
        overload_payload.update({
            "basis": basis,
            "forecast_basis": "day_ahead_p50_static",
            "residual_source": "trailing rolling-origin residuals",
            "n_test_days": n_test_overload,
            "history_days_used": history_overload,
            "limit_consistency": {
                "asset_rating_kw": round(rating_kw, 3),
                "congestion_threshold_mw": (
                    float(congestion_threshold_mw) if congestion_threshold_mw is not None else None
                ),
                "consistent": True if congestion_threshold_mw is not None else None,
            },
            "caveat": "Einzelasset-Risiko aus Prognoseverteilung; kein Netzlastfluss/GIS-Modell.",
        })
        hosting_payload = hosting_capacity_kw(
            point_kw,
            residuals_by_hour,
            rating_kw,
            risk_alpha=overload_risk_alpha,
        )
        hosting_payload.update({
            "basis": basis,
            "forecast_basis": "day_ahead_p50_static",
            "residual_source": "trailing rolling-origin residuals",
            "n_test_days": n_test_overload,
            "history_days_used": history_overload,
            "asset_rating_kw": round(rating_kw, 3),
            "caveat": "Konservative koinzidente Zusatzlast; kein Netzlastfluss.",
        })
        out["overload"] = overload_payload
        out["hosting_capacity"] = hosting_payload

    if thermal_rating_kw is not None:
        rating_kw = float(thermal_rating_kw)
        if asset_rating_kw is not None and abs(rating_kw - float(asset_rating_kw)) > 1e-6:
            raise ValueError(
                "thermal_rating_kw muss zu asset_rating_kw passen "
                f"({rating_kw:.3f} kW != {float(asset_rating_kw):.3f} kW)."
            )
        if congestion_threshold_mw is not None:
            threshold_kw = float(congestion_threshold_mw) * 1000.0
            if abs(rating_kw - threshold_kw) > 1e-6:
                raise ValueError(
                    "thermal_rating_kw muss zur congestion_threshold_mw passen "
                    f"({rating_kw:.3f} kW != {threshold_kw:.3f} kW)."
                )
        basis = "residual" if residual_hours is not None else "load"
        thermal_load2d = residual2d if residual_hours is not None else load2d
        thermal_days = resid_days if residual_hours is not None else days
        thermal_holidays = hol_r if residual_hours is not None else hol
        residuals_by_hour, n_test_thermal, history_thermal = _rolling_residuals_by_hour(
            thermal_load2d,
            thermal_days,
            thermal_holidays,
        )
        point_kw = [float(h["p50"]) * 1000.0 for h in cong_hours]
        ambient24, ambient_source = _thermal_ambient24(
            thermal_ambient_c,
            weather_csv=weather_csv,
            forecast_date=cong_date,
        )
        thermal_payload = probabilistic_thermal_risk(
            point_kw,
            residuals_by_hour,
            rating_kw,
            ambient24,
            hotspot_limit_c=thermal_hotspot_limit_c,
            risk_alpha=thermal_risk_alpha,
        )
        thermal_payload.update({
            "basis": basis,
            "forecast_basis": "day_ahead_p50_static",
            "residual_source": "trailing rolling-origin residuals",
            "n_test_days": n_test_thermal,
            "history_days_used": history_thermal,
            "ambient_temperature_c": [round(float(x), 3) for x in ambient24],
            "ambient_source": ambient_source,
            "limit_consistency": {
                "thermal_rating_kw": round(rating_kw, 3),
                "asset_rating_kw": round(float(asset_rating_kw), 3) if asset_rating_kw is not None else None,
                "congestion_threshold_mw": (
                    float(congestion_threshold_mw) if congestion_threshold_mw is not None else None
                ),
                "consistent": True if (asset_rating_kw is not None or congestion_threshold_mw is not None) else None,
            },
            "caveat": (
                "Einzelasset-Thermik mit Standardparametern; echte Trafo-Parameter und "
                "Umgebungstemperaturen im Pilot setzen. Kein Netzlastfluss."
            ),
        })
        out["thermal"] = thermal_payload

    devices = normalize_steuve_devices(steuve_demands_kw, steuve_devices)
    if rolling_redispatch and congestion_threshold_mw is not None and devices:
        load24_kw = [float(h["p50"]) * 1000.0 for h in cong_hours]
        rp = compute_rolling_redispatch(
            from_single_path(load24_kw),
            float(congestion_threshold_mw) * 1000.0,
            steuve_demands_kw,
            steuve_devices=devices,
        )
        rp.update({
            "forecast_basis": "day_ahead_p50_static",
            "basis": "residual" if residual_hours is not None else "load",
            "threshold_mw": float(congestion_threshold_mw),
            "threshold_kw": round(float(congestion_threshold_mw) * 1000.0, 3),
            "note": (
                "Static day-ahead P50 approximation via from_single_path; not a true intraday "
                "updated forecast path."
            ),
        })
        out["redispatch"] = rp

    if dispatch_plan_enabled:
        if congestion_threshold_mw is None:
            raise ValueError("dispatch_plan braucht congestion_threshold_mw als Netzgrenze.")
        if dispatch_steuve_energy_kwh is None or dispatch_steuve_p_max_kw is None:
            raise ValueError("dispatch_plan braucht dispatch_steuve_energy_kwh und dispatch_steuve_p_max_kw.")
        keep = min(120, len(load2d))
        bt_load2d, bt_days = load2d[-keep:], days[-keep:]
        n_test = min(42, max(14, keep - 60))
        R_dispatch, _sm_dispatch = rolling_origin(
            bt_load2d,
            bt_days,
            lambda: ShrunkCorrector(10.0),
            n_test=n_test,
            holiday_set=hol,
        )
        residuals_kw = (np.asarray(R_dispatch["actual"], float) - np.asarray(R_dispatch["model"], float)) * 1000.0
        residuals_by_hour = [[float(x) for x in residuals_kw[h::24]] for h in range(24)]
        load24_kw = [float(h["p50"]) * 1000.0 for h in cong_hours]
        pmax_kw = float(dispatch_steuve_p_max_kw)
        base_point_kw = [max(0.0, x - pmax_kw) for x in load24_kw]
        dispatch_redispatch = out["redispatch"]
        if dispatch_redispatch is None:
            dispatch_redispatch = compute_rolling_redispatch(
                from_single_path(load24_kw),
                float(congestion_threshold_mw) * 1000.0,
                [pmax_kw],
            )
        out["dispatch_plan"] = build_dispatch_plan(
            base_point_kw,
            residuals_by_hour,
            float(congestion_threshold_mw) * 1000.0,
            steuve_energy_kwh=dispatch_steuve_energy_kwh,
            steuve_p_max_kw=pmax_kw,
            grid_fee_eur_per_kwh=grid_fee_eur_per_kwh,
            c_short=dispatch_c_short,
            c_long=dispatch_c_long,
            risk_beta=dispatch_risk_beta,
            risk_alpha=dispatch_risk_alpha,
            redispatch=dispatch_redispatch,
            metadata={
                "forecast_basis": "day_ahead_p50_static",
                "residual_source": "trailing rolling-origin residuals",
                "n_test_days": n_test,
                "history_days_used": keep,
                "threshold_mw": float(congestion_threshold_mw),
            },
        )

    if grid_fee_eur_per_kwh is not None and tariff_energy_kwh is not None and tariff_p_max_kw is not None:
        out["tariff_schedule"] = build_tariff_schedule(
            grid_fee_eur_per_kwh,
            tariff_energy_kwh,
            tariff_p_max_kw,
            redispatch=out["redispatch"],
            available=tariff_available,
            available_start_hour=tariff_available_start_hour,
            available_end_hour=tariff_available_end_hour,
        )

    # §14a-Regelkreis end-to-end: Fahrplan an (Mock-)aEMT übergeben und Quittung mitliefern.
    # Rollentrennung: NetzPilot übergibt nur, der aEMT steuert. Default aus -> kein Verhalten geändert.
    if pool_assets is not None or pool_shared_cap_kw is not None:
        if pool_assets is None or pool_shared_cap_kw is None:
            raise ValueError("pool_dispatch braucht pool_assets und pool_shared_cap_kw.")
        if len(pool_assets) < 2:
            raise ValueError("pool_dispatch braucht mindestens zwei Pool-Assets.")
        shared_cap = _normalize_cap_series(pool_shared_cap_kw, name="pool_shared_cap_kw")
        pool_payload = pool_dispatch(pool_assets, shared_cap, dt_h=1.0)
        pool_payload.update({
            "forecast_basis": "provided_pool_asset_demands",
            "cap_source": "pool_shared_cap_kw",
            "rating_truth_kw": round(float(unified_rating_kw), 3) if unified_rating_kw is not None else None,
            "caveat": (
                "Faire Periodenkappung und Pool-Aggregation. Keine zeituebergreifende Optimierung, "
                "kein Speicher-/SOC-Modell."
            ),
        })
        out["pool_dispatch"] = pool_payload

    if audit_ledger_path:
        out["audit"] = append_forecast_audit(
            audit_ledger_path,
            out,
            signing_key=audit_signing_key,
            rule_version=audit_rule_version,
        )

    out["aemt_ack"] = None
    if submit_to_aemt and out["fahrplan"] is not None:
        adapter_key = str(aemt_adapter or "mock").strip().lower()
        if adapter_key not in {"mock", "eebus_lpc"}:
            raise ValueError("aemt_adapter muss 'mock' oder 'eebus_lpc' sein.")
        from netzpilot.control.aemt_adapter import AemtError, MockAemt
        try:
            if adapter_key == "eebus_lpc":
                from netzpilot.control.eebus_lpc import EebusLpcAdapter
                out["aemt_ack"] = EebusLpcAdapter().submit(out["fahrplan"])
                out["fahrplan_lpc"] = out["aemt_ack"].get("lpc")
            else:
                out["aemt_ack"] = MockAemt().submit(out["fahrplan"])
        except AemtError as e:
            out["aemt_ack"] = {"status": "REJECTED", "reason": str(e)}

    # T51: ausgegebene Prognose tamper-evident im Forecast-Store ablegen (opt-in). Nach Erfolg der
    # produktiven Wertschoepfung — damit fehlerhafte Laeufe nicht den Store kontaminieren.
    if forecast_store_path:
        try:
            extras = {
                "utility": utility,
                "region": region,
                "coverage_scale_used": fc.get("coverage_scale_used"),
                "residual_feedback": fc.get("residual_feedback"),
            }
            entry = record_forecast(forecast_store_path, fc, series_id=utility, extras=extras)
            out["forecast_store"] = {
                "path": forecast_store_path,
                "target_date": fc["date"],
                "entry_hash": entry.get("entry_hash"),
                "note": ("Hash-verkettet, vorab ausgegeben. Manipulationssicher = Hash-Kette, "
                         "keine juristische Zertifizierung."),
            }
        except ValueError as e:
            out["forecast_store"] = {"path": forecast_store_path, "error": str(e)}
    return out
