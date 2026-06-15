# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Täglicher Day-ahead-Lauf (Scheduler-Einstieg) — ein Mandant oder mehrere via Config.

Beispiel (cron/Task Scheduler, z.B. 06:00):
  python scripts/run_daily_forecast.py --csv data_cache/real/Netzumsatz-Lastgang-2025.csv \
      --utility "Stadtwerke Hilden" --unit kW --ts-col Text --load-col Reihe1 \
      --congestion-threshold-mw 45 --steuve-malo DE0001234567890

Oder mit JSON-Config (mehrere Mandanten):
  python scripts/run_daily_forecast.py --config config/utilities.json
Config-Format: [{"utility","csv","unit","region","ts_col","load_col","congestion_threshold_mw","steuve_malo"}]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.service.runner import run_forecast
from netzpilot.service.store import ForecastStore


def _parse_float_list(raw):
    if raw is None or isinstance(raw, list):
        return raw
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_bool(raw) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "ja"}


def _parse_bool_list(raw):
    if raw is None or isinstance(raw, list):
        return raw
    return [str(x).strip().lower() in {"1", "true", "yes", "ja"}
            for x in str(raw).split(",") if str(x).strip()]


def _parse_json_list(raw):
    if raw is None or isinstance(raw, list):
        return raw
    return json.loads(raw)


def _run_one(cfg: dict, store: ForecastStore) -> dict:
    res = run_forecast(
        cfg["csv"], utility=cfg.get("utility", "default"), region=cfg.get("region", "NW"),
        unit=cfg.get("unit", "MW"), ts_col=cfg.get("ts_col"), load_col=cfg.get("load_col"),
        congestion_threshold_mw=cfg.get("congestion_threshold_mw"), steuve_malo=cfg.get("steuve_malo"),
        steuve_demands_kw=_parse_float_list(cfg.get("steuve_demands_kw")),
        steuve_devices=cfg.get("steuve_devices"),
        rolling_redispatch=_parse_bool(cfg.get("rolling_redispatch")),
        rebap_csv=cfg.get("rebap_csv"), spot_csv=cfg.get("spot_csv"),
        generation_csv=cfg.get("generation_csv"), generation_unit=cfg.get("generation_unit", "MW"),
        generation_ts_col=cfg.get("generation_ts_col"), generation_load_col=cfg.get("generation_load_col"),
        realized_economics=_parse_bool(cfg.get("realized_economics")),
        submit_to_aemt=_parse_bool(cfg.get("submit_to_aemt")),
        aemt_adapter=cfg.get("aemt_adapter", "mock"),
        mmm_price_eur_mwh=cfg.get("mmm_price_eur_mwh"),
        rating_kw=cfg.get("rating_kw"),
        drift_monitoring=_parse_bool(cfg.get("drift_monitoring")),
        drift_store_dir=cfg.get("drift_store_dir", "data_cache/drift"),
        validate_input=_parse_bool(cfg.get("validate_input", True)),
        validate_allow_negative=_parse_bool(cfg.get("validate_allow_negative")),
        validate_max_plausible=cfg.get("validate_max_plausible"),
        asset_rating_kw=cfg.get("asset_rating_kw"),
        overload_risk_alpha=cfg.get("overload_risk_alpha", 0.05),
        thermal_rating_kw=cfg.get("thermal_rating_kw"),
        thermal_ambient_c=cfg.get("thermal_ambient_c"),
        thermal_hotspot_limit_c=cfg.get("thermal_hotspot_limit_c", 120.0),
        thermal_risk_alpha=cfg.get("thermal_risk_alpha", 0.05),
        grid_fee_eur_per_kwh=_parse_float_list(cfg.get("grid_fee_eur_per_kwh")),
        tariff_energy_kwh=cfg.get("tariff_energy_kwh"),
        tariff_p_max_kw=cfg.get("tariff_p_max_kw"),
        tariff_available=_parse_bool_list(cfg.get("tariff_available")),
        tariff_available_start_hour=cfg.get("tariff_available_start_hour"),
        tariff_available_end_hour=cfg.get("tariff_available_end_hour"),
        dispatch_plan_enabled=_parse_bool(cfg.get("dispatch_plan_enabled")),
        dispatch_steuve_energy_kwh=cfg.get("dispatch_steuve_energy_kwh"),
        dispatch_steuve_p_max_kw=cfg.get("dispatch_steuve_p_max_kw"),
        dispatch_c_short=cfg.get("dispatch_c_short", 0.20),
        dispatch_c_long=cfg.get("dispatch_c_long", 0.10),
        dispatch_risk_beta=cfg.get("dispatch_risk_beta", 0.0),
        dispatch_risk_alpha=cfg.get("dispatch_risk_alpha", 0.95),
        pool_assets=_parse_json_list(cfg.get("pool_assets")),
        pool_shared_cap_kw=_parse_float_list(cfg.get("pool_shared_cap_kw")),
    )
    path = store.save(res["utility"], res)
    cong = res["congestion"]
    basis = f" [{cong['basis']}]" if cong and cong.get("basis") else ""
    resid = "" if res.get("residual_forecast") is None else " +Residuallast"
    drift = res.get("drift")
    drift_msg = ""
    if drift:
        alarm = " REKALIBRIERUNG PRUEFEN" if drift.get("needs_recalibration") else ""
        drift_msg = f"; Drift={drift.get('status')}{alarm}"
    tariff = res.get("tariff_schedule")
    tariff_msg = ""
    if tariff:
        tariff_msg = f"; Tarifersparnis={tariff.get('saving_eur', 0):.2f} EUR"
    dispatch = res.get("dispatch_plan")
    dispatch_msg = ""
    if dispatch:
        dispatch_msg = f"; Dispatch-Newsvendor={dispatch.get('newsvendor_saving_eur', 0):.2f} EUR"
        if dispatch.get("risk_averse"):
            risk = dispatch["risk_averse"]
            dispatch_msg += (
                f"; CVaR-beta={risk.get('beta', 0):.2f}"
                f" dCVaR={risk.get('risk_cvar_delta_vs_newsvendor_eur', 0):.2f} EUR"
            )
    validation = res.get("input_validation") or {}
    validation_msg = ""
    if validation.get("enabled"):
        validation_msg = (
            f"; Datenqualitaet={validation.get('quality_score', 0) * 100:.1f}%"
            f"/{validation.get('n_issues_total', 0)} Issues"
        )
    overload = res.get("overload")
    overload_msg = ""
    if overload:
        overload_msg = (
            f"; Overload-Risiko={overload.get('max_exceedance_prob', 0) * 100:.1f}%"
            f"/{overload.get('hours_at_risk', 0)}h"
        )
    hosting = res.get("hosting_capacity")
    hosting_msg = ""
    if hosting:
        hosting_msg = f"; Hosting={hosting.get('hosting_capacity_kw', 0):.1f} kW"
    thermal = res.get("thermal")
    thermal_msg = ""
    if thermal:
        thermal_msg = (
            f"; Hotspot-Risiko={thermal.get('max_exceedance_prob', 0) * 100:.1f}%"
            f"/Aging={thermal.get('expected_loss_of_life_h_total', 0):.2f}h"
        )
    lpc = res.get("fahrplan_lpc")
    lpc_msg = ""
    if lpc:
        lpc_msg = f"; LPC={lpc.get('n_limits', 0)} Limits"
    mmm = res.get("mmm")
    mmm_msg = ""
    if mmm and mmm.get("status") == "available":
        mmm_msg = f"; MMM-Volumenreduktion={mmm.get('abs_volumen_reduktion_mwh', 0):.2f} MWh"
    pool = res.get("pool_dispatch")
    pool_msg = ""
    if pool:
        pool_msg = (
            f"; Pool-Shed={pool.get('pool_shed_kwh', 0):.2f} kWh"
            f"/safe={str(pool.get('grid_safe')).lower()}"
        )
    print(f"[{res['utility']}] {res['forecast_date']}: P50 max "
          f"{max(h['p50'] for h in res['forecast']):.1f} MW{resid}; "
          f"{'ENGPASS '+str(cong['window_hours'])+basis if cong else 'kein Engpass'}"
          f"{validation_msg}{drift_msg}{overload_msg}{hosting_msg}{thermal_msg}{lpc_msg}"
          f"{mmm_msg}{pool_msg}{tariff_msg}{dispatch_msg}; -> {path}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--csv"); ap.add_argument("--utility", default="default")
    ap.add_argument("--region", default="NW"); ap.add_argument("--unit", default="MW")
    ap.add_argument("--ts-col"); ap.add_argument("--load-col")
    ap.add_argument("--congestion-threshold-mw", type=float)
    ap.add_argument("--steuve-malo")
    ap.add_argument("--steuve-demands-kw")
    ap.add_argument("--rolling-redispatch", action="store_true")
    ap.add_argument("--realized-economics", action="store_true")
    ap.add_argument("--drift-monitoring", action="store_true")
    ap.add_argument("--drift-store-dir", default="data_cache/drift")
    ap.add_argument("--no-validate-input", dest="validate_input", action="store_false", default=True)
    ap.add_argument("--validate-allow-negative", action="store_true")
    ap.add_argument("--validate-max-plausible", type=float)
    ap.add_argument("--asset-rating-kw", type=float)
    ap.add_argument("--overload-risk-alpha", type=float, default=0.05)
    ap.add_argument("--thermal-rating-kw", type=float)
    ap.add_argument("--thermal-ambient-c", type=float)
    ap.add_argument("--thermal-hotspot-limit-c", type=float, default=120.0)
    ap.add_argument("--thermal-risk-alpha", type=float, default=0.05)
    ap.add_argument("--grid-fee-eur-per-kwh")
    ap.add_argument("--tariff-energy-kwh", type=float)
    ap.add_argument("--tariff-p-max-kw", type=float)
    ap.add_argument("--tariff-available")
    ap.add_argument("--tariff-available-start-hour", type=int)
    ap.add_argument("--tariff-available-end-hour", type=int)
    ap.add_argument("--dispatch-plan", action="store_true")
    ap.add_argument("--dispatch-steuve-energy-kwh", type=float)
    ap.add_argument("--dispatch-steuve-p-max-kw", type=float)
    ap.add_argument("--dispatch-c-short", type=float, default=0.20)
    ap.add_argument("--dispatch-c-long", type=float, default=0.10)
    ap.add_argument("--dispatch-risk-beta", type=float, default=0.0)
    ap.add_argument("--dispatch-risk-alpha", type=float, default=0.95)
    ap.add_argument("--submit-to-aemt", action="store_true")
    ap.add_argument("--aemt-adapter", choices=["mock", "eebus_lpc"], default="mock")
    ap.add_argument("--mmm-price-eur-mwh", type=float)
    ap.add_argument("--rating-kw", type=float)
    ap.add_argument("--pool-assets")
    ap.add_argument("--pool-shared-cap-kw")
    ap.add_argument("--store", default=os.environ.get("NETZPILOT_STORE", "data_cache/service_store"))
    args = ap.parse_args()

    store = ForecastStore(args.store)
    if args.config:
        cfgs = json.load(open(args.config, encoding="utf-8"))
    elif args.csv:
        cfgs = [{"csv": args.csv, "utility": args.utility, "region": args.region, "unit": args.unit,
                 "ts_col": args.ts_col, "load_col": args.load_col,
                 "congestion_threshold_mw": args.congestion_threshold_mw,
                 "steuve_malo": args.steuve_malo,
                 "steuve_demands_kw": _parse_float_list(args.steuve_demands_kw),
                 "rolling_redispatch": args.rolling_redispatch,
                 "realized_economics": args.realized_economics,
                 "submit_to_aemt": args.submit_to_aemt,
                 "aemt_adapter": args.aemt_adapter,
                 "mmm_price_eur_mwh": args.mmm_price_eur_mwh,
                 "rating_kw": args.rating_kw,
                 "drift_monitoring": args.drift_monitoring,
                 "drift_store_dir": args.drift_store_dir,
                 "validate_input": args.validate_input,
                 "validate_allow_negative": args.validate_allow_negative,
                 "validate_max_plausible": args.validate_max_plausible,
                 "asset_rating_kw": args.asset_rating_kw,
                 "overload_risk_alpha": args.overload_risk_alpha,
                 "thermal_rating_kw": args.thermal_rating_kw,
                 "thermal_ambient_c": args.thermal_ambient_c,
                 "thermal_hotspot_limit_c": args.thermal_hotspot_limit_c,
                 "thermal_risk_alpha": args.thermal_risk_alpha,
                 "grid_fee_eur_per_kwh": _parse_float_list(args.grid_fee_eur_per_kwh),
                 "tariff_energy_kwh": args.tariff_energy_kwh,
                 "tariff_p_max_kw": args.tariff_p_max_kw,
                 "tariff_available": _parse_bool_list(args.tariff_available),
                 "tariff_available_start_hour": args.tariff_available_start_hour,
                 "tariff_available_end_hour": args.tariff_available_end_hour,
                 "dispatch_plan_enabled": args.dispatch_plan,
                 "dispatch_steuve_energy_kwh": args.dispatch_steuve_energy_kwh,
                 "dispatch_steuve_p_max_kw": args.dispatch_steuve_p_max_kw,
                 "dispatch_c_short": args.dispatch_c_short,
                 "dispatch_c_long": args.dispatch_c_long,
                 "dispatch_risk_beta": args.dispatch_risk_beta,
                 "dispatch_risk_alpha": args.dispatch_risk_alpha,
                 "pool_assets": _parse_json_list(args.pool_assets),
                 "pool_shared_cap_kw": _parse_float_list(args.pool_shared_cap_kw)}]
    else:
        ap.error("Entweder --config oder --csv angeben.")
    for cfg in cfgs:
        try:
            _run_one(cfg, store)
        except Exception as e:
            print(f"[FEHLER] {cfg.get('utility','?')}: {e}")


if __name__ == "__main__":
    main()
