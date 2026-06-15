"""Leakage-sicheres Rolling-Origin-Backtest (expanding window).

Vergleicht: Persistenz, Saisonal-Naiv und ein Korrekturmodell (Ridge v1 oder LGBM, T3).
P10/P90 ueber stundenbedingte Residuenquantile aus dem Trainingsfenster.

T45: Optional zusaetzlich KALIBRIERTE Coverage neben der naiven. s wird auf einem
Validierungs-Split STRIKT VOR dem Testfenster getunt (load2d[:ND-n_test]) und auf
das Testband angewendet (geschrumpfte Band-Skalierung, leakage-sicher).
"""
from __future__ import annotations
import numpy as np
from ..features.build import build_features
from ..features.holiday_base import holiday_aware_base, holiday_aware_resid_target
from ..models.baselines import persistence, seasonal_naive
from . import metrics as M
from .coverage_calibration import (
    coverage_scale, apply_scale, rolling_coverage_scale, rolling_asymmetric_scale,
)
from ..models.residual_feedback import online_residual_feedback

def rolling_origin(load2d, days, corrector_factory, first=8, n_test=28,
                   weather2d=None, holiday_set=None, feature_fn=build_features,
                   calibrate=False, cal_shrink=0.5, cal_val_days=None,
                   residual_feedback=False, residual_feedback_window=28,
                   residual_feedback_shrink=0.5):
    ND = len(load2d)
    test_days = list(range(ND - n_test, ND))
    rows = {k: [] for k in ["persist", "snaive", "model", "actual", "hour", "p10", "p90"]}
    for d in test_days:
        Xtr = np.vstack([feature_fn(load2d, days, t, weather2d, holiday_set) for t in range(first, d)])
        ytr = np.concatenate([holiday_aware_resid_target(load2d, t, days, holiday_set)
                              for t in range(first, d)])
        model = corrector_factory().fit(Xtr, ytr)
        fitted = np.concatenate([holiday_aware_base(load2d, t, days, holiday_set)
                                 for t in range(first, d)]) + model.predict(Xtr)
        actual_tr = np.concatenate([load2d[t] for t in range(first, d)])
        res = actual_tr - fitted
        q10 = {h: np.quantile(res[h::24], 0.10) for h in range(24)}
        q90 = {h: np.quantile(res[h::24], 0.90) for h in range(24)}
        yhat = holiday_aware_base(load2d, d, days, holiday_set) + \
               model.predict(feature_fn(load2d, days, d, weather2d, holiday_set))
        yd = load2d[d]
        for h in range(24):
            # T48: Baselines persistence/seasonal_naive bleiben UNVERAENDERT (Vergleichsmassstab).
            rows["persist"].append(persistence(load2d, d)[h])
            rows["snaive"].append(seasonal_naive(load2d, d)[h])
            rows["model"].append(yhat[h]); rows["actual"].append(yd[h]); rows["hour"].append(h)
            rows["p10"].append(yhat[h] + q10[h]); rows["p90"].append(yhat[h] + q90[h])
    R = {k: np.array(v) for k, v in rows.items()}
    a = R["actual"]

    # T50: optional Online-Residuen-Feedback (Level-Shift) — leakage-sicher, BEFORE Kalibrierung.
    # delta_d = rho_d * (actual[d-1] - forecast[d-1]) wird auf P10/P50/P90 von Tag d gemeinsam
    # addiert (Bandmitte korrigiert, Bandbreite unveraendert). False = bit-identisch zum alten Pfad.
    rf_info = None
    naive_metrics = None
    if residual_feedback:
        H = int(load2d.shape[1])
        f2 = R["model"].reshape(-1, H)
        a2 = R["actual"].reshape(-1, H)
        rho, delta, corrected = online_residual_feedback(
            f2, a2, window=residual_feedback_window, shrink=residual_feedback_shrink)
        # NAIVe Metriken VOR der Korrektur — fuer ehrlichen Nebeneinander-Vergleich
        naive_metrics = {
            "model_MAE_MW_naiv": round(M.mae(R["model"], a), 1),
            "Pinball_avg_naiv": round(float(np.mean([
                M.pinball(a, R["p10"], .1), M.pinball(a, R["model"], .5), M.pinball(a, R["p90"], .9)])), 1),
            "Coverage_P10_P90_naiv_%": round(M.coverage(a, R["p10"], R["p90"]), 1),
        }
        # Level-Shift auf alle drei Quantile gemeinsam
        R["model"] = corrected.ravel()
        R["p10"] = (R["p10"].reshape(-1, H) + delta).ravel()
        R["p90"] = (R["p90"].reshape(-1, H) + delta).ravel()
        # ρ aus dem Warmup ist 0; aktive ρ ab Tag residual_feedback_window // 2 (min_window default 14)
        active = max(14, residual_feedback_window // 2)
        rf_info = {
            "method": "online-rolling residual feedback (lag-1)",
            "window": int(residual_feedback_window),
            "shrink": float(residual_feedback_shrink),
            "rho_mean": round(float(np.mean(rho[active:])) if len(rho) > active else 0.0, 4),
            "rho_median": round(float(np.median(rho[active:])) if len(rho) > active else 0.0, 4),
            "delta_abs_mean_MW": round(float(np.mean(np.abs(delta))), 3),
            "n_days_with_rho_gt_0": int(np.sum(rho > 0.0)),
        }

    scale = float(np.mean(np.abs(load2d[first:ND - n_test] - load2d[first - 1:ND - n_test - 1])))
    mp, ms = M.mae(R["persist"], a), M.mae(R["snaive"], a)
    tab = {}
    for name in ["persist", "snaive", "model"]:
        p = R[name]
        tab[name] = {"MAE_MW": round(M.mae(p, a), 1), "RMSE_MW": round(M.rmse(p, a), 1),
                     "MAPE_%": round(M.mape(p, a), 2), "MASE": round(M.mase(p, a, scale), 3),
                     "Skill_vs_Persistenz_%": round(M.skill(M.mae(p, a), mp), 1),
                     "Skill_vs_SaisonalNaiv_%": round(M.skill(M.mae(p, a), ms), 1)}
    summary = {"test_tage": n_test, "test_vorhersagen": len(a), "horizont": "Day-ahead 24h",
               "metriken": tab,
               "probabilistisch": {
                   "Pinball_avg": round(np.mean([M.pinball(a, R["p10"], .1), M.pinball(a, R["model"], .5),
                                                 M.pinball(a, R["p90"], .9)]), 1),
                   "Coverage_P10_P90_%": round(M.coverage(a, R["p10"], R["p90"]), 1), "Soll_%": 80},
               "MASE_Skala_MW": round(scale, 1)}
    if rf_info is not None:
        summary["residual_feedback"] = rf_info
        summary["metriken_naiv"] = naive_metrics
    if calibrate:
        summary["probabilistisch"].update(_calibration_summary(
            load2d, days, corrector_factory, R, first, n_test,
            weather2d, holiday_set, feature_fn, cal_shrink, cal_val_days))
    return R, summary


DEFAULT_CAL_VAL_RECENT = 28  # T46: s wird auf den letzten ~28 Tagen UNMITTELBAR vor dem Testfenster
                              # getunt — unabhaengig von n_test. Das spiegelt forecast_next_day, wo
                              # ebenfalls auf einem recent Kalibrierfenster (~cal_days) getunt wird.
                              # T45 hatte den Default auf n_test (= 84 im Headline-Scoreboard), was
                              # den Validierungs-Split ~168 Tage zurueckreichen liess → Saison-
                              # Mismatch, s≈1, Kalibrierung wirkungslos. Recent 28 ist robust.


def _calibration_summary(load2d, days, corrector_factory, R_test, first, n_test,
                         weather2d, holiday_set, feature_fn, cal_shrink, cal_val_days):
    """T47: ONLINE-rollende Coverage-Kalibrierung — s je Testtag aus dem nachlaufenden Fenster.

    Adaptiert an Drift; schlaegt das T46-Einzelfenster (5 echte Reihen, n_test=84:
    naiv 6.42 → T46 5.86 → online 3.56). Verschlechtert keine Reihe.

    Effizient: berechnet zusaetzlich nur `window` Warmup-Tage VOR dem Testfenster und nutzt
    fuer die Testtage die bereits vorliegenden Baender (R_test). Leakage-sicher: jeder Tag
    nutzt nur strikt davor liegende Tage zur s-Bestimmung.

    cal_val_days: Fenstergroesse (Default DEFAULT_CAL_VAL_RECENT=28). cal_shrink: Shrinkage
    der s_opt-Schaetzung zur 1 (Default 0.5, gemessenes Optimum)."""
    H = int(load2d.shape[1])
    ND = len(load2d)
    window = int(cal_val_days) if cal_val_days is not None else DEFAULT_CAL_VAL_RECENT
    warmup_end = ND - n_test
    if window < 7 or warmup_end - window <= first:
        return {"Coverage_P10_P90_kalibriert_%": None, "Pinball_avg_kalibriert": None,
                "coverage_scale_used": None,
                "kalibrier_hinweis": (
                    f"Zu wenig Warmup-Vorlauf (window={window}, verfuegbar={warmup_end - first}); "
                    "keine Kalibrierung.")}
    # Warmup-Baender berechnen (die `window` Tage UNMITTELBAR vor dem Testfenster); R_test fuer
    # die Test-Tage wiederverwenden. Beide Mengen sind exakt dieselben Baender, die ein einzelner
    # rolling_origin(n_test=window+n_test)-Aufruf liefern wuerde — Training endet je Tag d strikt vor d.
    R_warm, _ = rolling_origin(load2d[:warmup_end], days[:warmup_end], corrector_factory,
                               first=first, n_test=window, weather2d=weather2d,
                               holiday_set=holiday_set, feature_fn=feature_fn)

    def _stack(r):
        return (np.asarray(r["actual"], float).reshape(-1, H),
                np.asarray(r["p10"], float).reshape(-1, H),
                np.asarray(r["model"], float).reshape(-1, H),
                np.asarray(r["p90"], float).reshape(-1, H))

    a_w, lo_w, p50_w, hi_w = _stack(R_warm)
    a_t, lo_t, p50_t, hi_t = _stack(R_test)
    actual_2d = np.concatenate([a_w, a_t], axis=0)
    p10_2d = np.concatenate([lo_w, lo_t], axis=0)
    p50_2d = np.concatenate([p50_w, p50_t], axis=0)
    p90_2d = np.concatenate([hi_w, hi_t], axis=0)

    min_win = min(14, max(7, window // 2))
    # T49: asymmetrische Kalibrierung (lo/hi getrennt) als Default — faengt rechtsschiefe Lastfehler
    # ohne den symmetrischen Pfad zu verschlechtern (gemessen: Pinball asym <= sym in jedem Fall).
    s_lo_arr, s_hi_arr, lo_2d, hi_2d = rolling_asymmetric_scale(
        actual_2d, p10_2d, p50_2d, p90_2d,
        window=window, target_tail=0.1, shrink=cal_shrink, min_window=min_win)

    # Test-Portion: letzte n_test Tage
    a_test = actual_2d[-n_test:].ravel()
    p10_naive = p10_2d[-n_test:].ravel()
    p90_naive = p90_2d[-n_test:].ravel()
    lo_cal = lo_2d[-n_test:].ravel()
    hi_cal = hi_2d[-n_test:].ravel()
    p50_test = p50_2d[-n_test:].ravel()
    s_lo_test = s_lo_arr[-n_test:]
    s_hi_test = s_hi_arr[-n_test:]

    cov_cal = M.coverage(a_test, lo_cal, hi_cal)
    pin_cal = float(np.mean([M.pinball(a_test, lo_cal, 0.1),
                             M.pinball(a_test, p50_test, 0.5),
                             M.pinball(a_test, hi_cal, 0.9)]))
    # Beide Tail-Anteile (Soll je 10 %) — macht die Schiefe sichtbar
    frac_below_naiv = float(np.mean(a_test < p10_naive) * 100.0)
    frac_above_naiv = float(np.mean(a_test > p90_naive) * 100.0)
    frac_below_cal = float(np.mean(a_test < lo_cal) * 100.0)
    frac_above_cal = float(np.mean(a_test > hi_cal) * 100.0)
    s_mean = float(np.mean(0.5 * (s_lo_test + s_hi_test)))   # Backward-Compat-Mittel
    return {"Coverage_P10_P90_kalibriert_%": round(cov_cal, 1),
            "Pinball_avg_kalibriert": round(pin_cal, 1),
            "frac_below_P10_%": round(frac_below_naiv, 1),
            "frac_above_P90_%": round(frac_above_naiv, 1),
            "frac_below_P10_kalibriert_%": round(frac_below_cal, 1),
            "frac_above_P90_kalibriert_%": round(frac_above_cal, 1),
            "coverage_scale_used": round(s_mean, 3),     # Backward-Compat (T47-Tests)
            "coverage_scale_lo_used": round(float(np.mean(s_lo_test)), 3),
            "coverage_scale_hi_used": round(float(np.mean(s_hi_test)), 3),
            "coverage_scale_lo_median": round(float(np.median(s_lo_test)), 3),
            "coverage_scale_hi_median": round(float(np.median(s_hi_test)), 3),
            "coverage_scale_median": round(float(np.median(0.5 * (s_lo_test + s_hi_test))), 3),
            "coverage_scale_method": "online-rolling-asymmetric",
            "kalibrier_window_tage": int(window),
            "kalibrier_min_window": int(min_win),
            "kalibrier_val_tage": int(window),     # Backward-Compat (T46-Tests)
            "kalibrier_shrink": float(cal_shrink)}

def _summary(R, load2d, first, n_test, quantile_levels=(0.1, 0.5, 0.9)):
    a = R["actual"]
    scale = float(np.mean(np.abs(load2d[first:len(load2d) - n_test] - load2d[first - 1:len(load2d) - n_test - 1])))
    mp, ms = M.mae(R["persist"], a), M.mae(R["snaive"], a)
    tab = {}
    for name in ["persist", "snaive", "model"]:
        p = R[name]
        tab[name] = {"MAE_MW": round(M.mae(p, a), 1), "RMSE_MW": round(M.rmse(p, a), 1),
                     "MAPE_%": round(M.mape(p, a), 2), "MASE": round(M.mase(p, a, scale), 3),
                     "Skill_vs_Persistenz_%": round(M.skill(M.mae(p, a), mp), 1),
                     "Skill_vs_SaisonalNaiv_%": round(M.skill(M.mae(p, a), ms), 1)}
    quantiles = np.vstack([R["p10"], R["model"], R["p90"]])
    pinball_avg = float(np.mean([M.pinball(a, quantiles[i], t) for i, t in enumerate(quantile_levels)]))
    summary = {"test_tage": n_test, "test_vorhersagen": len(a), "horizont": "Day-ahead 24h",
               "metriken": tab,
               "probabilistisch": {
                   "Pinball_avg": round(pinball_avg, 1),
                   "CRPS_proxy": round(2.0 * pinball_avg, 1),
                   "Coverage_P10_P90_%": round(M.coverage(a, R["p10"], R["p90"]), 1), "Soll_%": 80},
               "MASE_Skala_MW": round(scale, 1)}
    return summary

def conformal_adjustment(scores, target_coverage):
    """Finite-sample CQR correction quantile for nonconformity scores."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return 0.0
    ordered = np.sort(scores)
    k = int(np.ceil((scores.size + 1) * target_coverage))
    k = min(max(k, 1), scores.size)
    return float(max(0.0, ordered[k - 1]))

def _pred(preds, alpha):
    key = min(preds.keys(), key=lambda a: abs(a - alpha))
    return preds[key]

def rolling_origin_quantile(load2d, days, corrector_factory, first=8, n_test=28,
                            weather2d=None, holiday_set=None, retrain_every=1,
                            calibration_tail_days=0, target_coverage=0.8):
    """Rolling-origin backtest with native P10/P50/P90 residual-quantile models."""
    ND = len(load2d)
    test_days = list(range(ND - n_test, ND))
    rows = {k: [] for k in ["persist", "snaive", "model", "actual", "hour", "p10", "p90"]}
    model = None
    last_fit_day = None
    qhat = 0.0
    for d in test_days:
        if model is None or (d - last_fit_day) >= retrain_every:
            fit_end = d
            if calibration_tail_days:
                fit_end = max(first + 1, d - calibration_tail_days)
            Xtr = np.vstack([build_features(load2d, days, t, weather2d, holiday_set) for t in range(first, fit_end)])
            ytr = np.concatenate([holiday_aware_resid_target(load2d, t, days, holiday_set)
                                  for t in range(first, fit_end)])
            model = corrector_factory().fit(Xtr, ytr)
            last_fit_day = d
            qhat = 0.0
            if calibration_tail_days and fit_end < d:
                scores = []
                for c in range(fit_end, d):
                    Xc = build_features(load2d, days, c, weather2d, holiday_set)
                    cpreds = model.predict(Xc)
                    cb = holiday_aware_base(load2d, c, days, holiday_set)
                    cq10 = cb + cpreds[0.1]
                    cq50 = cb + cpreds[0.5]
                    cq90 = cb + cpreds[0.9]
                    cstacked = np.sort(np.vstack([cq10, cq50, cq90]), axis=0)
                    yc = load2d[c]
                    scores.extend(np.maximum.reduce([cstacked[0] - yc, yc - cstacked[2], np.zeros(24)]))
                qhat = float(np.quantile(scores, target_coverage, method="higher"))
        Xd = build_features(load2d, days, d, weather2d, holiday_set)
        preds = model.predict(Xd)
        db = holiday_aware_base(load2d, d, days, holiday_set)
        q10 = db + preds[0.1]
        q50 = db + preds[0.5]
        q90 = db + preds[0.9]
        stacked = np.sort(np.vstack([q10, q50, q90]), axis=0)
        stacked[0] -= qhat
        stacked[2] += qhat
        yd = load2d[d]
        for h in range(24):
            rows["persist"].append(persistence(load2d, d)[h])
            rows["snaive"].append(seasonal_naive(load2d, d)[h])
            rows["p10"].append(stacked[0, h])
            rows["model"].append(stacked[1, h])
            rows["p90"].append(stacked[2, h])
            rows["actual"].append(yd[h])
            rows["hour"].append(h)
    R = {k: np.array(v) for k, v in rows.items()}
    summary = _summary(R, load2d, first, n_test)
    summary["quantile_calibration"] = {
        "method": "training-tail conformal widening" if calibration_tail_days else "none",
        "calibration_tail_days": calibration_tail_days,
        "target_coverage": target_coverage,
    }
    return R, summary

def rolling_origin_cqr(load2d, days, corrector_factory, first=8, n_test=28,
                       weather2d=None, holiday_set=None, retrain_every=7,
                       calibration_window_days=56, interval_coverage=0.8,
                       online_eta=0.0, feature_fn=build_features, per_hour_cal=False):
    """Rolling-origin CQR with a strictly past calibration window for each target day.

    per_hour_cal: wenn True, wird das Konformitaetsquantil PRO STUNDE gebildet (24 separate qhat)
    statt ueber alle Stunden gepoolt. Das behebt die typische Unterdeckung in den volatilen
    Spitzenstunden (nachts ist die Last ruhig -> ein gemeinsames Band ist dort zu breit und
    tagsueber zu eng). Methodisch sauberere Kalibrierung; Default False (Abwaertskompatibilitaet)."""
    ND = len(load2d)
    test_days = list(range(ND - n_test, ND))
    qlo_level = round((1.0 - interval_coverage) / 2.0, 6)
    qhi_level = round(1.0 - qlo_level, 6)
    rows = {k: [] for k in ["persist", "snaive", "model", "actual", "hour", "p10", "p90"]}
    model = None
    last_fit_day = None
    effective_coverage = interval_coverage
    effective_history = []
    for d in test_days:
        if model is None or (d - last_fit_day) >= retrain_every:
            train_end = max(first + 1, d - calibration_window_days)
            Xtr = np.vstack([feature_fn(load2d, days, t, weather2d, holiday_set) for t in range(first, train_end)])
            ytr = np.concatenate([holiday_aware_resid_target(load2d, t, days, holiday_set)
                                  for t in range(first, train_end)])
            model = corrector_factory((qlo_level, 0.5, qhi_level)).fit(Xtr, ytr)
            last_fit_day = d

        scores = []                       # gepoolt (Default)
        scores_h = {h: [] for h in range(24)}   # pro Stunde (per_hour_cal)
        cal_start = max(first, d - calibration_window_days)
        for c in range(cal_start, d):
            Xc = feature_fn(load2d, days, c, weather2d, holiday_set)
            cpreds = model.predict(Xc)
            cb = holiday_aware_base(load2d, c, days, holiday_set)
            cqlo = cb + _pred(cpreds, qlo_level)
            cq50 = cb + _pred(cpreds, 0.5)
            cqhi = cb + _pred(cpreds, qhi_level)
            cstacked = np.sort(np.vstack([cqlo, cq50, cqhi]), axis=0)
            yc = load2d[c]
            s_c = np.maximum(cstacked[0] - yc, yc - cstacked[2])
            scores.extend(s_c)
            for h in range(24):
                scores_h[h].append(s_c[h])
        if per_hour_cal:
            qhat = np.array([conformal_adjustment(scores_h[h], effective_coverage) for h in range(24)])
        else:
            qhat = conformal_adjustment(scores, effective_coverage)

        Xd = feature_fn(load2d, days, d, weather2d, holiday_set)
        preds = model.predict(Xd)
        db = holiday_aware_base(load2d, d, days, holiday_set)
        qlo = db + _pred(preds, qlo_level)
        q50 = db + _pred(preds, 0.5)
        qhi = db + _pred(preds, qhi_level)
        stacked = np.sort(np.vstack([qlo, q50, qhi]), axis=0)
        stacked[0] -= qhat                # qhat ist Skalar (gepoolt) ODER [24]-Array (per-hour) -> broadcastet
        stacked[2] += qhat
        yd = load2d[d]
        day_coverage = float(np.mean((yd >= stacked[0]) & (yd <= stacked[2])))
        for h in range(24):
            rows["persist"].append(persistence(load2d, d)[h])
            rows["snaive"].append(seasonal_naive(load2d, d)[h])
            rows["p10"].append(stacked[0, h])
            rows["model"].append(stacked[1, h])
            rows["p90"].append(stacked[2, h])
            rows["actual"].append(yd[h])
            rows["hour"].append(h)
        if online_eta:
            target_miss_rate = 1.0 - interval_coverage
            observed_miss_rate = 1.0 - day_coverage
            effective_coverage += online_eta * (observed_miss_rate - target_miss_rate)
            effective_coverage = float(np.clip(effective_coverage, interval_coverage, 0.995))
        effective_history.append(effective_coverage)
    R = {k: np.array(v) for k, v in rows.items()}
    summary = _summary(R, load2d, first, n_test, quantile_levels=(qlo_level, 0.5, qhi_level))
    label = f"P{int(round(qlo_level * 100))}-P{int(round(qhi_level * 100))}"
    summary["probabilistisch"]["Interval_Label"] = label
    summary["probabilistisch"]["Soll_%"] = round(interval_coverage * 100, 1)
    summary["probabilistisch"]["Coverage_Interval_%"] = summary["probabilistisch"]["Coverage_P10_P90_%"]
    summary["quantile_calibration"] = {
        "method": "rolling CQR" if not online_eta else "rolling CQR with online score-quantile update",
        "calibration_window_days": calibration_window_days,
        "target_coverage": interval_coverage,
        "quantile_levels": [qlo_level, 0.5, qhi_level],
        "online_eta": online_eta,
        "effective_coverage_mean": round(float(np.mean(effective_history)), 4) if effective_history else interval_coverage,
        "effective_coverage_last": round(float(effective_history[-1]), 4) if effective_history else interval_coverage,
    }
    return R, summary
