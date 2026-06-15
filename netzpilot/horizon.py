# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Mehrtages-Horizont: D+1 … D+H aus EINEM leakage-sicheren Fit (operative Betriebslücke).

Warum: Ein Stadtwerk nominiert freitags für Sa+So+Mo — Day-ahead allein reicht im Betrieb nicht.

Methode (bewusst minimal, keine neue Mathematik):
- EIN Korrektor-Fit auf der ECHTEN Historie — exakt wie `forecast_next_day` (gleiche Features,
  gleiche feiertagsbewusste Basis, gleiches Fit-Fenster). Pseudo-Tage entstehen erst DANACH.
- Rekursive Mehrschritt-Prognose: für k=2..H wird die eigene P50-Prognose als Pseudo-Tag an die
  Lastmatrix angehängt — sie speist NUR die Features (Lag-1 u. ä.), NIE das Fit. Der
  Saisonal-Naiv-Anker (Lag-7, feiertagsbewusst) nutzt für k<=7 ausschließlich ECHTE Tage.
- k=1 ist BIT-IDENTISCH zu `forecast_next_day(..., calibrate=False, residual_feedback=False)`
  (gleicher Fit, gleiche Features) — das ist der zentrale Korrektheitsbeweis in verify_horizon.

Ehrliche v1-Grenze (gemessen statt versprochen):
- Volle, kalibrierte P10/P90-Bänder gibt es in v1 nur für k=1 (der produktive Pfad).
  Für k>=2 liefert v1 die P50-Punktprognose; die Band-Kalibrierung je Horizont folgt erst,
  wenn die Horizont-Residuen gemessen sind (`rolling_horizon_backtest` liefert genau das).
  Felder sind entsprechend benannt/gelabelt — kein stilles „Band gilt schon".

Reine numpy/pandas auf den bestehenden Bausteinen. Additiv: forecast.py bleibt unberührt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .eval.conformal import _conf_quantile
from .features.build import build_features
from .features.holiday_base import holiday_aware_base, holiday_aware_resid_target
from .models.baselines import seasonal_naive


def _fit_like_next_day(load2d, days, corrector_factory, first, cal_days,
                       weather2d, holiday_set):
    """Exakt der Fit aus forecast_next_day (gleiches Fenster, gleiche Targets)."""
    ND = len(load2d)
    fit_end = ND - cal_days if ND - cal_days > first + 2 else ND
    Xtr = np.vstack([build_features(load2d, days, t, weather2d, holiday_set)
                     for t in range(first, fit_end)])
    ytr = np.concatenate([holiday_aware_resid_target(load2d, t, days, holiday_set)
                          for t in range(first, fit_end)])
    model = corrector_factory().fit(Xtr, ytr)
    fitted = np.concatenate([holiday_aware_base(load2d, t, days, holiday_set)
                             for t in range(first, fit_end)]) + model.predict(Xtr)
    res_tr = np.concatenate([load2d[t] for t in range(first, fit_end)]) - fitted
    return model, fit_end, res_tr


def _horizon_band_params(load2d, days, model, fit_end, horizon, rq_lo, rq_hi, alpha,
                         weather2d, holiday_set):
    """Bandparameter je Horizont k>=2 aus REKURSIVEN k-Schritt-Residuen des Kalibrierfensters.

    Je Ausgabetag j im Kalibrierfenster wird mit dem FESTEN Modell rekursiv j+1..j+H prognostiziert
    (Pseudo-Tage nur Features) und das Residuum gegen das echte Ist je k gesammelt. Daraus je k:
    - scale s_k >= 1: gepoolte Quantilbreite relativ zu k=1 — die STUNDENFORM bleibt das verifizierte
      1-Schritt-Profil (rq_lo/rq_hi), nur die Breite wächst gemessen mit dem Horizont;
    - c_k: CQR-Konformitätsquantil auf dem mit s_k skalierten Basisband (finite-sample-Korrektur).
    k=1 bleibt absichtlich der EXAKTE Produktionspfad (rq ± c) — Bit-Parität zu forecast_next_day.
    Leakage-sicher: ausschließlich Tage < ND; jedes Residuum nutzt nur Daten <= Ausgabetag j.
    """
    ND, H = load2d.shape
    days_idx = pd.DatetimeIndex(days)
    resid = {k: [] for k in range(1, horizon + 1)}
    for j in range(fit_end, ND - 1):
        ext = load2d[:j + 1]
        dext = days_idx[:j + 1]
        kmax = min(horizon, ND - 1 - j)
        for k in range(1, kmax + 1):
            nd = pd.Timestamp(dext[-1]) + pd.Timedelta(days=1)
            dext = pd.DatetimeIndex(list(dext) + [nd])
            t = len(ext)
            yh = holiday_aware_base(ext, t, dext, holiday_set) + \
                model.predict(build_features(ext, dext, t, weather2d, holiday_set))
            resid[k].append(load2d[j + k] - yh)
            ext = np.vstack([ext, yh[None, :]])
    pool1 = np.concatenate(resid[1]) if resid[1] else np.array([0.0])
    w1 = float(np.quantile(pool1, 1 - alpha / 2) - np.quantile(pool1, alpha / 2))
    w1 = w1 if w1 > 1e-12 else 1.0
    params = {}
    for k in range(2, horizon + 1):
        if not resid[k]:
            params[k] = {"scale": 1.0, "c": 0.0, "n_cal_days": 0}
            continue
        pool = np.concatenate(resid[k])
        wk = float(np.quantile(pool, 1 - alpha / 2) - np.quantile(pool, alpha / 2))
        s = max(1.0, wk / w1)               # nie schmaler als der 1-Schritt (ehrliche Untergrenze)
        scores = []
        for r in resid[k]:
            lo = np.array([s * rq_lo[h] for h in range(H)])
            hi = np.array([s * rq_hi[h] for h in range(H)])
            scores += list(np.maximum(lo - r, r - hi))
        c_k = _conf_quantile(scores, alpha) if scores else 0.0
        params[k] = {"scale": round(float(s), 4), "c": float(c_k), "n_cal_days": len(resid[k])}
    return params


def forecast_days(load2d, days, corrector_factory, horizon: int = 3, *,
                  first: int = 8, alpha: float = 0.2, cal_days: int = 28,
                  weather2d=None, holiday_set=None, round_digits: int | None = 1,
                  bands: str = "k1") -> dict:
    """P50 für D+1..D+horizon (rekursiv). Bänder:
    - bands="k1" (Default, T52-Kontrakt): volles P10/P90 nur für k=1, k>=2 nur P50.
    - bands="per_horizon": zusätzlich kalibrierte Bänder für k>=2 (gemessen breiter via s_k + c_k
      aus rekursiven Kalibrierfenster-Residuen); k=1 bleibt bit-identisch zum Produktionsband.
    """
    if horizon < 1:
        raise ValueError("horizon muss >= 1 sein.")
    if horizon > 7:
        raise ValueError("horizon > 7 nicht unterstützt: ab k=8 wäre der Lag-7-Anker selbst Pseudo.")
    if bands not in ("k1", "per_horizon"):
        raise ValueError("bands muss 'k1' oder 'per_horizon' sein.")
    load2d = np.asarray(load2d, float)
    ND, H = load2d.shape

    model, fit_end, res_tr = _fit_like_next_day(
        load2d, days, corrector_factory, first, cal_days, weather2d, holiday_set)
    rq_lo = {h: float(np.quantile(res_tr[h::H], alpha / 2)) for h in range(H)}
    rq_hi = {h: float(np.quantile(res_tr[h::H], 1 - alpha / 2)) for h in range(H)}
    # CQR-Konformitaets-Aufschlag c — exakt wie forecast_next_day (Scores auf dem Kalibrierfenster):
    scores = []
    for t in range(fit_end, ND):
        yh = holiday_aware_base(load2d, t, pd.DatetimeIndex(days), holiday_set) + \
            model.predict(build_features(load2d, days, t, weather2d, holiday_set))
        lo = np.array([yh[h] + rq_lo[h] for h in range(H)])
        hi = np.array([yh[h] + rq_hi[h] for h in range(H)])
        scores += list(np.maximum(lo - load2d[t], load2d[t] - hi))
    c = _conf_quantile(scores, alpha) if scores else 0.0

    band_params = (_horizon_band_params(load2d, days, model, fit_end, horizon,
                                        rq_lo, rq_hi, alpha, weather2d, holiday_set)
                   if (bands == "per_horizon" and horizon >= 2) else {})

    def _fmt(x):
        x = float(x)
        return x if round_digits is None else round(x, round_digits)

    ext = load2d
    days_ext = pd.DatetimeIndex(days)
    out_days = []
    for k in range(1, horizon + 1):
        next_date = pd.Timestamp(days_ext[-1]) + pd.Timedelta(days=1)
        days_ext = pd.DatetimeIndex(list(days_ext) + [next_date])
        t = len(ext)                       # Zieltag-Index im erweiterten Array
        yhat = holiday_aware_base(ext, t, days_ext, holiday_set) + \
            model.predict(build_features(ext, days_ext, t, weather2d, holiday_set))
        day = {"date": str(next_date.date()), "horizon": int(k),
               "hours": [{"hour": h, "p50": _fmt(yhat[h])} for h in range(H)]}
        if k == 1:                          # produktiver 1-Schritt: exakt forecast_next_day-Band
            for h in range(H):              # inkl. Monotonie-Klemme
                day["hours"][h]["p10"] = _fmt(min(yhat[h] + rq_lo[h] - c, yhat[h]))
                day["hours"][h]["p90"] = _fmt(max(yhat[h] + rq_hi[h] + c, yhat[h]))
        elif bands == "per_horizon":        # k>=2: gemessen breiteres, konform justiertes Band
            bp = band_params.get(k, {"scale": 1.0, "c": 0.0, "n_cal_days": 0})
            s, ck = float(bp["scale"]), float(bp["c"])
            for h in range(H):
                day["hours"][h]["p10"] = _fmt(min(yhat[h] + s * rq_lo[h] - ck, yhat[h]))
                day["hours"][h]["p90"] = _fmt(max(yhat[h] + s * rq_hi[h] + ck, yhat[h]))
            day["band"] = {"scale": bp["scale"], "conf_c": round(float(bp["c"]), 4),
                           "n_cal_days": int(bp.get("n_cal_days", 0)),
                           "basis": "rekursive k-Schritt-Residuen des Kalibrierfensters"}
        out_days.append(day)
        ext = np.vstack([ext, yhat[None, :]])   # Pseudo-Tag: NUR Features, nie Fit

    bands_note = ("k=1 voll (Trainings-Residuenquantile + CQR-c); k>=2 nur P50 — Band-Kalibrierung je "
                  "Horizont folgt aus gemessenen Horizont-Residuen (rolling_horizon_backtest)."
                  if bands == "k1" else
                  "alle Horizonte mit kalibriertem Band: k=1 = Produktionsband (bit-identisch); "
                  "k>=2 = 1-Schritt-Stundenform × gemessenem Breitenfaktor s_k (>=1) + CQR-c_k aus "
                  "rekursiven Kalibrierfenster-Residuen.")
    return {"horizon": int(horizon),
            "issued_after": str(pd.Timestamp(days[-1]).date()),
            "days": out_days,
            "bands_mode": bands,
            "bands": bands_note,
            "fit": "ein Fit auf echter Historie; Pseudo-Tage speisen nur Features (leakage-frei)"}


def rolling_horizon_backtest(load2d, days, corrector_factory, horizon: int = 3, *,
                             n_test: int = 42, first: int = 8, cal_days: int = 28,
                             weather2d=None, holiday_set=None) -> dict:
    """Leakage-sichere Horizont-Messung: je Ausgabetag d werden d+1..d+H rekursiv prognostiziert
    (nur Daten <= d), gegen Ist und Saisonal-Naiv-k verglichen. Liefert je k: MAE/MAPE/Skill.

    Saisonal-Naiv für Zieltag d+k = load[d+k-7] — am Ausgabetag bekannt (k<=7), identische
    Vergleichsbasis wie im Board, nur horizontgerecht verschoben.
    """
    if not 1 <= horizon <= 7:
        raise ValueError("horizon in 1..7")
    load2d = np.asarray(load2d, float)
    ND, H = load2d.shape
    issue_days = list(range(ND - n_test - horizon, ND - horizon))
    if issue_days[0] <= first + cal_days:
        raise ValueError(f"Zu wenig Historie für n_test={n_test}, horizon={horizon}.")

    abs_err = {k: [] for k in range(1, horizon + 1)}      # je k: Liste von Tages-MAE
    abs_err_snv = {k: [] for k in range(1, horizon + 1)}
    ape = {k: [] for k in range(1, horizon + 1)}
    for d in issue_days:                                   # Ausgabetag: Daten bis EINSCHL. d-? -> bis d
        hist = load2d[:d + 1]
        hist_days = pd.DatetimeIndex(days[:d + 1])
        model, _fe, _res = _fit_like_next_day(hist, hist_days, corrector_factory,
                                              first, cal_days, weather2d, holiday_set)
        ext, days_ext = hist, hist_days
        for k in range(1, horizon + 1):
            nd = pd.Timestamp(days_ext[-1]) + pd.Timedelta(days=1)
            days_ext = pd.DatetimeIndex(list(days_ext) + [nd])
            t = len(ext)
            yhat = holiday_aware_base(ext, t, days_ext, holiday_set) + \
                model.predict(build_features(ext, days_ext, t, weather2d, holiday_set))
            actual = load2d[d + k]
            snv = seasonal_naive(load2d, d + k)            # Lag-7: am Ausgabetag bekannt (k<=7)
            abs_err[k].append(float(np.mean(np.abs(yhat - actual))))
            abs_err_snv[k].append(float(np.mean(np.abs(snv - actual))))
            denom = np.where(np.abs(actual) < 1e-9, np.nan, np.abs(actual))
            ape[k].append(float(np.nanmean(np.abs(yhat - actual) / denom) * 100.0))
            ext = np.vstack([ext, yhat[None, :]])
    per_k = {}
    for k in range(1, horizon + 1):
        mae = float(np.mean(abs_err[k])); mae_s = float(np.mean(abs_err_snv[k]))
        per_k[k] = {"mae_mw": round(mae, 4), "mae_snaive_mw": round(mae_s, 4),
                    "mape_pct": round(float(np.nanmean(ape[k])), 2),
                    "skill_vs_snaive_pct": round((1.0 - mae / mae_s) * 100.0, 1),
                    "n_days": len(abs_err[k])}
    return {"n_test": n_test, "horizon": horizon, "per_horizon": per_k,
            "method": "rekursiv, ein Fit je Ausgabetag (expanding), Pseudo-Tage nur Features; "
                      "Baseline = Saisonal-Naiv je Zielhorizont (Lag-7, am Ausgabetag bekannt)"}


def horizon_band_backtest(load2d, days, corrector_factory, horizon: int = 3, *,
                          n_test: int = 42, first: int = 8, cal_days: int = 28,
                          alpha: float = 0.2, weather2d=None, holiday_set=None) -> dict:
    """Coverage-Messung der per-Horizont-Bänder: je Ausgabetag d wird die KOMPLETTE
    forecast_days(bands="per_horizon")-Mechanik nur aus Daten <= d gebaut (Fit, rq, c, s_k, c_k)
    und gegen die echten Tage d+1..d+H abgerechnet. Liefert je k: Coverage %, mittlere Bandbreite,
    mittleren Breitenfaktor. Soll-Coverage = (1-alpha)*100.
    """
    if not 1 <= horizon <= 7:
        raise ValueError("horizon in 1..7")
    load2d = np.asarray(load2d, float)
    ND, H = load2d.shape
    issue_days = list(range(ND - n_test - horizon, ND - horizon))
    if issue_days[0] <= first + cal_days + 2:
        raise ValueError(f"Zu wenig Historie für n_test={n_test}, horizon={horizon}.")

    cov = {k: [] for k in range(1, horizon + 1)}
    width = {k: [] for k in range(1, horizon + 1)}
    scales = {k: [] for k in range(2, horizon + 1)}
    for d in issue_days:
        hist = load2d[:d + 1]
        hist_days = pd.DatetimeIndex(days[:d + 1])
        model, fit_end, res_tr = _fit_like_next_day(hist, hist_days, corrector_factory,
                                                    first, cal_days, weather2d, holiday_set)
        rq_lo = {h: float(np.quantile(res_tr[h::H], alpha / 2)) for h in range(H)}
        rq_hi = {h: float(np.quantile(res_tr[h::H], 1 - alpha / 2)) for h in range(H)}
        scores = []
        for t in range(fit_end, len(hist)):
            yh = holiday_aware_base(hist, t, hist_days, holiday_set) + \
                model.predict(build_features(hist, hist_days, t, weather2d, holiday_set))
            lo = np.array([yh[h] + rq_lo[h] for h in range(H)])
            hi = np.array([yh[h] + rq_hi[h] for h in range(H)])
            scores += list(np.maximum(lo - hist[t], hist[t] - hi))
        c = _conf_quantile(scores, alpha) if scores else 0.0
        bp = _horizon_band_params(hist, hist_days, model, fit_end, horizon,
                                  rq_lo, rq_hi, alpha, weather2d, holiday_set)
        ext, days_ext = hist, hist_days
        for k in range(1, horizon + 1):
            nd = pd.Timestamp(days_ext[-1]) + pd.Timedelta(days=1)
            days_ext = pd.DatetimeIndex(list(days_ext) + [nd])
            t = len(ext)
            yhat = holiday_aware_base(ext, t, days_ext, holiday_set) + \
                model.predict(build_features(ext, days_ext, t, weather2d, holiday_set))
            if k == 1:
                lo = np.minimum(yhat + np.array([rq_lo[h] for h in range(H)]) - c, yhat)
                hi = np.maximum(yhat + np.array([rq_hi[h] for h in range(H)]) + c, yhat)
            else:
                p = bp.get(k, {"scale": 1.0, "c": 0.0})
                s, ck = float(p["scale"]), float(p["c"])
                lo = np.minimum(yhat + s * np.array([rq_lo[h] for h in range(H)]) - ck, yhat)
                hi = np.maximum(yhat + s * np.array([rq_hi[h] for h in range(H)]) + ck, yhat)
                scales[k].append(s)
            actual = load2d[d + k]
            cov[k].append(float(np.mean((actual >= lo) & (actual <= hi)) * 100.0))
            width[k].append(float(np.mean(hi - lo)))
            ext = np.vstack([ext, yhat[None, :]])
    per_k = {}
    for k in range(1, horizon + 1):
        per_k[k] = {"coverage_pct": round(float(np.mean(cov[k])), 1),
                    "mean_width_mw": round(float(np.mean(width[k])), 3),
                    "mean_scale": round(float(np.mean(scales[k])), 3) if k >= 2 else 1.0,
                    "n_days": len(cov[k])}
    return {"n_test": n_test, "horizon": horizon, "soll_coverage_pct": round((1 - alpha) * 100, 1),
            "per_horizon": per_k}
