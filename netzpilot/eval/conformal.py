"""Konforme Vorhersageintervalle fuer den Day-ahead-Korrektor (reine numpy-Implementierung).

Enthaelt:
- rolling_origin_conformal: split-/EnbPI-Stil (Residuenquantile, optional online) — UNTERDECKT oft.
- rolling_origin_cqr:       Conformalized Quantile Regression (CQR) — kalibriert die Baender auf
                            ~Soll-Coverage (verteilungsfrei, marginale Garantie). DE-RISKT T8.

Leakage-sicher: Fit- und Kalibrierfenster liegen strikt vor dem Zieltag.
"""
from __future__ import annotations
import numpy as np
from ..features.build import build_features
from ..features.holiday_base import holiday_aware_base, holiday_aware_resid_target
from ..models.baselines import persistence, seasonal_naive
from . import metrics as M


def _conf_quantile(scores, alpha):
    """CQR-Konformitaetsquantil auf Level (1-alpha)*(1+1/n) (endliche-Stichproben-Korrektur)."""
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, (1 - alpha) * (1 + 1.0 / n))
    return float(np.quantile(np.asarray(scores, float), level))


def rolling_origin_conformal(load2d, days, corrector_factory, first=8, n_test=28,
                             alpha=0.2, cal_days=28, weather2d=None, holiday_set=None,
                             online=True, per_hour=False):
    ND = len(load2d)
    test_days = list(range(ND - n_test, ND))
    rows = {k: [] for k in ["persist", "snaive", "model", "actual", "hour", "lo", "hi"]}
    pool = {h: [] for h in range(24)}
    for d in test_days:
        fit_end = d - cal_days
        if fit_end <= first + 1:
            fit_end, cal_range = d, range(first, d)
        else:
            cal_range = range(fit_end, d)
        Xtr = np.vstack([build_features(load2d, days, t, weather2d, holiday_set) for t in range(first, fit_end)])
        ytr = np.concatenate([holiday_aware_resid_target(load2d, t, days, holiday_set)
                              for t in range(first, fit_end)])
        model = corrector_factory().fit(Xtr, ytr)
        cal = {h: [] for h in range(24)}
        for t in cal_range:
            r = load2d[t] - (holiday_aware_base(load2d, t, days, holiday_set)
                             + model.predict(build_features(load2d, days, t, weather2d, holiday_set)))
            for h in range(24):
                cal[h].append(r[h])
        if per_hour:
            ql = {h: float(np.quantile(cal[h] + pool[h], alpha / 2)) for h in range(24)}
            qu = {h: float(np.quantile(cal[h] + pool[h], 1 - alpha / 2)) for h in range(24)}
        else:
            allr = [v for h in range(24) for v in cal[h]] + [v for h in range(24) for v in pool[h]]
            lo_q, hi_q = float(np.quantile(allr, alpha / 2)), float(np.quantile(allr, 1 - alpha / 2))
            ql = {h: lo_q for h in range(24)}; qu = {h: hi_q for h in range(24)}
        yhat = holiday_aware_base(load2d, d, days, holiday_set) + \
               model.predict(build_features(load2d, days, d, weather2d, holiday_set))
        yd = load2d[d]
        for h in range(24):
            rows["persist"].append(persistence(load2d, d)[h]); rows["snaive"].append(seasonal_naive(load2d, d)[h])
            rows["model"].append(yhat[h]); rows["actual"].append(yd[h]); rows["hour"].append(h)
            rows["lo"].append(yhat[h] + ql[h]); rows["hi"].append(yhat[h] + qu[h])
        if online:
            for h in range(24):
                pool[h].append(yd[h] - yhat[h])
    R = {k: np.array(v) for k, v in rows.items()}
    a = R["actual"]
    return R, {"alpha": alpha, "nominal_%": round((1 - alpha) * 100, 1),
               "coverage_%": round(M.coverage(a, R["lo"], R["hi"]), 1),
               "mean_width_MW": round(float(np.mean(R["hi"] - R["lo"])), 1),
               "method": "conformal", "per_hour": per_hour, "online": online}


def rolling_origin_cqr(load2d, days, corrector_factory, first=8, n_test=28, alpha=0.2,
                       cal_days=28, weather2d=None, holiday_set=None, per_hour=False, online=True):
    """CQR: Basisintervall aus Fit-Residuenquantilen, dann auf vorgelagertem Kalibrierset
    per Konformitaetsscore E=max(lo-y, y-hi) auf ~Soll-Coverage justiert."""
    ND = len(load2d)
    test_days = list(range(ND - n_test, ND))
    rows = {k: [] for k in ["persist", "snaive", "model", "actual", "hour", "lo", "hi"]}
    pool = {h: [] for h in range(24)}

    for d in test_days:
        fit_end = d - cal_days
        if fit_end <= first + 2:
            fit_end = d - max(7, cal_days // 2)
        cal_range = range(fit_end, d)
        Xtr = np.vstack([build_features(load2d, days, t, weather2d, holiday_set) for t in range(first, fit_end)])
        ytr = np.concatenate([holiday_aware_resid_target(load2d, t, days, holiday_set)
                              for t in range(first, fit_end)])
        model = corrector_factory().fit(Xtr, ytr)
        fitted = np.concatenate([holiday_aware_base(load2d, t, days, holiday_set)
                                 for t in range(first, fit_end)]) + model.predict(Xtr)
        res_tr = np.concatenate([load2d[t] for t in range(first, fit_end)]) - fitted
        rq_lo = {h: float(np.quantile(res_tr[h::24], alpha / 2)) for h in range(24)}
        rq_hi = {h: float(np.quantile(res_tr[h::24], 1 - alpha / 2)) for h in range(24)}

        def base_interval(t):
            yhat = holiday_aware_base(load2d, t, days, holiday_set) + \
                   model.predict(build_features(load2d, days, t, weather2d, holiday_set))
            lo = np.array([yhat[h] + rq_lo[h] for h in range(24)])
            hi = np.array([yhat[h] + rq_hi[h] for h in range(24)])
            return yhat, lo, hi

        sc = {h: [] for h in range(24)}
        for t in cal_range:
            _, lo, hi = base_interval(t); y = load2d[t]
            E = np.maximum(lo - y, y - hi)
            for h in range(24):
                sc[h].append(E[h])
        if per_hour:
            c = {h: _conf_quantile(sc[h] + pool[h], alpha) for h in range(24)}
        else:
            allsc = [v for h in range(24) for v in sc[h]] + [v for h in range(24) for v in pool[h]]
            cv = _conf_quantile(allsc, alpha); c = {h: cv for h in range(24)}

        yhat, lo, hi = base_interval(d); yd = load2d[d]
        for h in range(24):
            rows["persist"].append(persistence(load2d, d)[h]); rows["snaive"].append(seasonal_naive(load2d, d)[h])
            rows["model"].append(yhat[h]); rows["actual"].append(yd[h]); rows["hour"].append(h)
            rows["lo"].append(lo[h] - c[h]); rows["hi"].append(hi[h] + c[h])
        if online:
            E = np.maximum(lo - yd, yd - hi)
            for h in range(24):
                pool[h].append(E[h])

    R = {k: np.array(v) for k, v in rows.items()}
    a = R["actual"]
    return R, {"alpha": alpha, "nominal_%": round((1 - alpha) * 100, 1),
               "coverage_%": round(M.coverage(a, R["lo"], R["hi"]), 1),
               "mean_width_MW": round(float(np.mean(R["hi"] - R["lo"])), 1),
               "method": "CQR", "per_hour": per_hour, "online": online}


def rolling_origin_cqr_targeted(load2d, days, corrector_factory, target_coverage=0.9,
                                first=8, n_test=28, cal_days=28, weather2d=None,
                                holiday_set=None, per_hour=False, online=True,
                                split_frac=0.5):
    """CQR mit COVERAGE-TARGETING: justiert alpha datengetrieben, damit die GEMESSENE Coverage das
    Ziel trifft — nicht nur die nominale.

    Hintergrund (ehrlich): reine CQR unterdeckt bei kurzer Historie systematisch (Basis-Residuenquantile
    unterschaetzen die Streuung; die verteilungsfreie Garantie ist nur asymptotisch scharf). Hier wird
    auf einem VERGANGENEN Validierungs-Split gemessen, wie gross die Luecke ist, und das effektive alpha
    entsprechend gesenkt (breitere Baender). So stehen 90% drauf UND 90% sind drin — der ehrliche Preis
    ist etwas groessere Bandbreite. Leakage-sicher: das Targeting nutzt nur Tage VOR dem Testfenster.
    """
    ND = len(load2d)
    n_val = max(7, int(n_test * split_frac))
    val_end = ND - n_test
    val_start = val_end - n_val
    alpha_nominal = 1.0 - target_coverage
    chosen_alpha = alpha_nominal
    if val_start > first + cal_days + 2:
        # Auf dem Validierungs-Split das alpha suchen, dessen gemessene Coverage >= Ziel ist.
        val_load = load2d[:val_end]
        val_days = days[:val_end]
        for alpha in (alpha_nominal, alpha_nominal * 0.7, alpha_nominal * 0.5,
                      alpha_nominal * 0.35, alpha_nominal * 0.25):
            _, s = rolling_origin_cqr(val_load, val_days, corrector_factory, first=first,
                                      n_test=n_val, alpha=alpha, cal_days=cal_days,
                                      weather2d=weather2d, holiday_set=holiday_set,
                                      per_hour=per_hour, online=online)
            if s["coverage_%"] >= target_coverage * 100:
                chosen_alpha = alpha
                break
            chosen_alpha = alpha
    R, summary = rolling_origin_cqr(load2d, days, corrector_factory, first=first, n_test=n_test,
                                    alpha=chosen_alpha, cal_days=cal_days, weather2d=weather2d,
                                    holiday_set=holiday_set, per_hour=per_hour, online=online)
    summary["target_coverage_%"] = round(target_coverage * 100, 1)
    summary["alpha_targeted"] = round(chosen_alpha, 4)
    summary["method"] = "CQR + Coverage-Targeting"
    return R, summary
