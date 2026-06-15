"""Forecast-Verlauf-Store — beweisbar VORAB ausgegebene Prognosen + Live-Track-Record.

Zweck (Ehrlichkeits-Infrastruktur): jede ausgegebene Day-ahead-Prognose wird ZUM AUSGABEZEITPUNKT in eine
manipulationssichere Hash-Kette geschrieben (Wiederverwendung von service/audit_ledger, T44). Damit ist
beweisbar, dass die Prognose VOR dem Zieltag existierte — kein Hindsight, kein Backfill. Sobald die
Ist-Werte vorliegen, wird der realisierte Track-Record berechnet (MAE/Coverage/Bias je Tag) — der ehrlichste
Leistungsnachweis, den es gibt: nicht Backtest, sondern gelebte Prognosen gegen gelebte Realität.

Zusatznutzen: das EXAKTE Residuum der zuletzt ausgegebenen (und realisierten) Prognose steht für das
Online-Residuen-Feedback (T50) bereit — statt der Rekonstruktion aus dem aktuellen Fit.

Reine stdlib/numpy; Kette/Format kommen aus audit_ledger (canonical_json, append mit Ketten-Check).
Ehrliche Grenze: „manipulationssicher" = Hash-Kette (Unverändertheit seit Aufzeichnung), keine
juristische Zertifizierung.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .audit_ledger import append_entry, load_entries, verify_chain

FORECAST_KIND = "issued_forecast"


def record_forecast(store_path: str, fp: dict, series_id: str, extras: dict | None = None) -> dict:
    """Ausgegebene Prognose (forecast_next_day-Output) tamper-evident speichern.

    fp: dict mit "date", "periods_per_day", "hours" = [{hour, p10, p50, p90}, …].
    Rückgabe: der gespeicherte Ketten-Eintrag (inkl. entry_hash).
    """
    hours = fp.get("hours")
    if not hours:
        raise ValueError("fp ohne 'hours' — kein forecast_next_day-Output?")
    payload = {
        "kind": FORECAST_KIND,
        "issued_at_utc": datetime.now(timezone.utc).isoformat(),
        "series_id": str(series_id),
        "target_date": str(fp["date"]),
        "periods_per_day": int(fp.get("periods_per_day", len(hours))),
        "p10": [float(h["p10"]) for h in hours],
        "p50": [float(h["p50"]) for h in hours],
        "p90": [float(h["p90"]) for h in hours],
        "extras": extras or {},
    }
    return append_entry(store_path, payload)


def realized_track_record(store_path: str, actuals_by_date: dict) -> dict:
    """Track-Record: gespeicherte Prognosen gegen eingetroffene Ist-Werte auswerten.

    actuals_by_date: {target_date(str): Sequenz der Ist-Werte (Länge = periods_per_day)}.
    Je Zieltag zählt der ZULETZT ausgegebene Eintrag (operativ maßgeblich); Tage ohne Actuals werden
    übersprungen (pending). Rückgabe: days (je Tag MAE/Coverage/Bias), aggregate, chain_ok,
    last_residual (EXAKTES Residuum des jüngsten realisierten Zieltags, fürs Residuen-Feedback).
    """
    chain = verify_chain(store_path)
    entries = load_entries(store_path) if chain["n"] else []
    latest = {}
    n_dupes = 0
    for e in entries:
        p = e.get("payload", {})
        if p.get("kind") != FORECAST_KIND:
            continue
        if p["target_date"] in latest:
            n_dupes += 1
        latest[p["target_date"]] = p           # letzter Eintrag je Zieltag gewinnt
    days = []
    skipped_mismatch = 0
    for td in sorted(latest):
        p = latest[td]
        if td not in actuals_by_date:
            continue                            # noch keine Ist-Werte -> pending
        a = np.asarray(actuals_by_date[td], float)
        p50 = np.asarray(p["p50"], float)
        if a.shape != p50.shape:
            skipped_mismatch += 1
            continue
        p10 = np.asarray(p["p10"], float); p90 = np.asarray(p["p90"], float)
        resid = a - p50
        days.append({
            "target_date": td,
            "issued_at_utc": p["issued_at_utc"],
            "n_periods": int(a.size),
            "mae": float(np.mean(np.abs(resid))),
            "bias": float(np.mean(resid)),
            "coverage_p10_p90_pct": float(np.mean((a >= p10) & (a <= p90)) * 100.0),
            "residual": [float(x) for x in resid],
        })
    agg = None
    if days:
        agg = {
            "n_days_realized": len(days),
            "mae_mean": float(np.mean([d["mae"] for d in days])),
            "bias_mean": float(np.mean([d["bias"] for d in days])),
            "coverage_mean_pct": float(np.mean([d["coverage_p10_p90_pct"] for d in days])),
        }
    return {
        "chain_ok": bool(chain["ok"]),
        "n_forecasts_stored": len(latest),
        "n_duplicates_superseded": n_dupes,
        "n_skipped_period_mismatch": skipped_mismatch,
        "days": days,
        "aggregate": agg,
        "last_residual": days[-1]["residual"] if days else None,
        "last_residual_date": days[-1]["target_date"] if days else None,
        "note": "Hash-verkettete, VORAB ausgegebene Prognosen vs. Ist — Live-Track-Record, kein Backtest. "
                "Manipulationssicher = Hash-Kette, keine juristische Zertifizierung.",
    }
