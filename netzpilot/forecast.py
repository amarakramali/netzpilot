"""Operative Day-ahead-Prognose: erzeugt fuer den NAECHSTEN Tag P10/P50/P90 (kalibriert via CQR).

Macht aus der Backtest-Engine ein Werkzeug: gegeben die Lasthistorie -> Fahrplan fuer morgen.
Leakage-sicher: nutzt nur Vergangenheit + Kalender (+ optional Wetter-FORECAST des Zieltags).

T45: Optional zusaetzlich Coverage-Kalibrierung (geschrumpfte Band-Skalierung). Auf dem
Vergangenheitsfenster fit_end..ND werden p10/p50/p90 + Actuals ohnehin berechnet (CQR-scores);
daraus wird via coverage_scale ein Skalenfaktor s tuned (mit Shrinkage gegen Ueberschiessen)
und mit apply_scale auf das naechste-Tag-Band angewendet. p50 bleibt unangetastet. Mit
`calibrate=False` (default) verhaelt sich die Funktion bit-genau wie vor T45.

T48: Saisonal-Naiv-Anker feiertagsbewusst — `holiday_aware_base` waehlt bei d-7-Feiertag die
naechste NICHT-Feiertags-Vorwochenreferenz (d-14, d-21, …). Ohne `holiday_set` bit-identisch
zum alten `base(load2d, t) = load2d[t-7]`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .features.build import build_features
from .features.holiday_base import holiday_aware_base, holiday_aware_resid_target
from .eval.conformal import _conf_quantile
from .eval.coverage_calibration import coverage_scale, apply_scale
from .models.residual_feedback import online_residual_feedback


def forecast_next_day(load2d, days, corrector_factory, first=8, alpha=0.2, cal_days=28,
                      weather2d=None, holiday_set=None, next_weather=None,
                      round_digits=1, calibrate=False, cal_shrink=0.5,
                      residual_feedback=False, residual_feedback_window=28,
                      residual_feedback_shrink=0.5, last_residual=None) -> dict:
    ND = len(load2d)
    H = int(load2d.shape[1])
    next_date = pd.Timestamp(days[-1]) + pd.Timedelta(days=1)
    days_ext = pd.DatetimeIndex(list(pd.DatetimeIndex(days)) + [next_date])
    fit_end = ND - cal_days if ND - cal_days > first + 2 else ND

    Xtr = np.vstack([build_features(load2d, days, t, weather2d, holiday_set) for t in range(first, fit_end)])
    ytr = np.concatenate([holiday_aware_resid_target(load2d, t, days, holiday_set)
                          for t in range(first, fit_end)])
    model = corrector_factory().fit(Xtr, ytr)

    fitted = np.concatenate([holiday_aware_base(load2d, t, days, holiday_set)
                             for t in range(first, fit_end)]) + model.predict(Xtr)
    res_tr = np.concatenate([load2d[t] for t in range(first, fit_end)]) - fitted
    rq_lo = {h: float(np.quantile(res_tr[h::H], alpha / 2)) for h in range(H)}
    rq_hi = {h: float(np.quantile(res_tr[h::H], 1 - alpha / 2)) for h in range(H)}

    scores = []
    cal_actual, cal_p10_rq, cal_p50, cal_p90_rq = [], [], [], []  # T45: fuer coverage_scale
    _need_past = calibrate or residual_feedback     # T50: cal_p50/cal_actual auch fuer RF noetig
    for t in range(fit_end, ND):
        yh = holiday_aware_base(load2d, t, days, holiday_set) + \
             model.predict(build_features(load2d, days, t, weather2d, holiday_set))
        lo = np.array([yh[h] + rq_lo[h] for h in range(H)]); hi = np.array([yh[h] + rq_hi[h] for h in range(H)])
        scores += list(np.maximum(lo - load2d[t], load2d[t] - hi))
        if _need_past:
            cal_actual.extend(load2d[t].tolist())
            cal_p10_rq.extend(lo.tolist())
            cal_p50.extend(yh.tolist())
            cal_p90_rq.extend(hi.tolist())
    c = _conf_quantile(scores, alpha) if scores else 0.0

    w = weather2d
    if weather2d is not None and next_weather is not None:
        w = np.concatenate([weather2d, next_weather[None, :, :]], axis=0)
    Xd = build_features(load2d, days_ext, ND, w, holiday_set)
    # T48: Endprognose-Anker ebenfalls feiertagsbewusst. days_ext erweitert den Kalender um den
    # Zieltag; holiday_aware_base nutzt days[ND-7], days[ND-14], … — alles strikt < ND (leakage-frei).
    yhat = holiday_aware_base(load2d, ND, days_ext, holiday_set) + model.predict(Xd)

    # T50: optionales Residuen-Feedback — Level-Shift δ = ρ·(actual_gestern - forecast_gestern) auf yhat.
    # ρ aus dem nachlaufenden cal_p50/cal_actual-Fenster (online, gleiche Logik wie residual_feedback.py).
    # T51: Wenn `last_residual` (aus dem Forecast-Store) uebergeben wird, EXAKTES Vortagsresiduum nutzen
    # statt der Rekonstruktion aus dem aktuellen Fit; ρ_next wird trotzdem auf dem Tuning-Fenster bestimmt.
    delta_next = np.zeros(H)
    rho_next = 0.0
    residual_source = "reconstructed"
    if residual_feedback and len(cal_actual) >= H:
        f_past_2d = np.asarray(cal_p50, float).reshape(-1, H)
        a_past_2d = np.asarray(cal_actual, float).reshape(-1, H)
        n_past = f_past_2d.shape[0]
        rf_min_window = min(14, max(7, residual_feedback_window // 2))
        if n_past >= rf_min_window + 1:
            # Dummy-Tag anhaengen, sodass rho_arr[-1] aus past-Fenster [n_past-window, n_past) gefittet wird.
            f_ext = np.vstack([f_past_2d, np.zeros((1, H), dtype=float)])
            a_ext = np.vstack([a_past_2d, np.zeros((1, H), dtype=float)])
            rho_arr, _, _ = online_residual_feedback(
                f_ext, a_ext,
                window=residual_feedback_window,
                shrink=residual_feedback_shrink,
                min_window=rf_min_window)
            rho_next = float(rho_arr[-1])
            # T51: exaktes Vortagsresiduum aus dem Store (falls vorhanden) — sonst Rekonstruktion
            if last_residual is not None:
                lr = np.asarray(last_residual, float).reshape(-1)
                if lr.shape[0] == H:
                    delta_next = rho_next * lr
                    residual_source = "store"
                else:
                    delta_next = rho_next * (a_past_2d[-1] - f_past_2d[-1])
            else:
                delta_next = rho_next * (a_past_2d[-1] - f_past_2d[-1])
        yhat = yhat + delta_next

    p10 = np.array([yhat[h] + rq_lo[h] - c for h in range(H)])
    p90 = np.array([yhat[h] + rq_hi[h] + c for h in range(H)])

    s_used = 1.0
    coverage_calibrated = False
    if calibrate and len(cal_actual) > 0:
        # Tune s auf produktionsaehnlichem Vergangenheitsband (rq +/- c) — gleiches Format wie das
        # naechste-Tag-Band. Anwenden auf das naechste-Tag-Band. p50 (yhat) wird NICHT skaliert.
        cal_p10 = np.asarray(cal_p10_rq, float) - c
        cal_p90 = np.asarray(cal_p90_rq, float) + c
        s_used = coverage_scale(np.asarray(cal_actual, float), cal_p10,
                                np.asarray(cal_p50, float), cal_p90,
                                target=1.0 - alpha, shrink=cal_shrink)
        p10, p90 = apply_scale(p10, yhat, p90, s_used)
        coverage_calibrated = True

    p10 = np.minimum(p10, yhat); p90 = np.maximum(p90, yhat)   # Monotonie sichern

    def _fmt(x):
        x = float(x)
        return x if round_digits is None else round(x, round_digits)

    out = {"date": str(next_date.date()),
           "periods_per_day": H,
           "hours": [{"hour": h, "p10": _fmt(p10[h]), "p50": _fmt(yhat[h]),
                      "p90": _fmt(p90[h])} for h in range(H)]}
    if calibrate:
        out["coverage_scale_used"] = float(s_used)
        out["coverage_calibrated"] = bool(coverage_calibrated)
    if residual_feedback:
        out["residual_feedback"] = {
            "method": "online-rolling residual feedback (lag-1)",
            "rho": round(float(rho_next), 4),
            "delta_mean_MW": round(float(np.mean(delta_next)), 3),
            "delta_abs_mean_MW": round(float(np.mean(np.abs(delta_next))), 3),
            "applied": bool(rho_next != 0.0),
            "window": int(residual_feedback_window),
            "shrink": float(residual_feedback_shrink),
            "residual_source": residual_source,    # T51: "store" vs. "reconstructed"
        }
    return out
