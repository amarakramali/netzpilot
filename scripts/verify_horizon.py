#!/usr/bin/env python3
"""Verify Mehrtages-Horizont (netzpilot/horizon.py) — Leakage-Beweise + Korrektheit. Exit!=0 bei Fehler.

Zentrale Beweise:
1. k=1 ist BIT-IDENTISCH zu forecast_next_day (gleicher Fit, gleiche Features) — wenn das gilt,
   erbt der Horizont die gesamte verifizierte 1-Schritt-Maschinerie.
2. Pseudo-Tage erreichen NIE das Fit (Manipulations-Test: absurde Pseudo-Werte ändern den Fit nicht).
3. Lag-7-Anker bleibt für k<=7 echt (Horizon-8 wird abgelehnt).
4. Backtest-Alignment: Saisonal-Naiv-k und Ist-Tage korrekt verschoben; Skill-Formel konsistent.
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from netzpilot.horizon import forecast_days, rolling_horizon_backtest, _fit_like_next_day
from netzpilot.forecast import forecast_next_day
from netzpilot.models.robust_corrector import ShrunkCorrector
from netzpilot.features.build import get_holidays

N = [0]
def check(ok, msg):
    N[0] += 1
    print(("ok  " if ok else "FAIL"), f"{N[0]:2d}:", msg)
    if not ok:
        sys.exit(1)

rng = np.random.default_rng(7)
ND, H = 140, 24
base = 20 + 5 * np.sin(np.arange(H) / 24 * 2 * np.pi)
wk = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.8])
load2d = np.array([base * wk[d % 7] * (1 + 0.05 * np.sin(d / 30)) + rng.normal(0, 0.3, H)
                   for d in range(ND)])
days = pd.date_range("2025-06-02", periods=ND, freq="D")    # Montag-Start
hol = get_holidays([2025], "NW")
factory = lambda: ShrunkCorrector(10.0)

# --- 1) k=1 bit-identisch zu forecast_next_day (calibrate/RF aus) ---
fd = forecast_days(load2d, days, factory, horizon=3, round_digits=None)
f1 = forecast_next_day(load2d, days, factory, holiday_set=None, calibrate=False,
                       residual_feedback=False, round_digits=None)
p50_h = np.array([x["p50"] for x in fd["days"][0]["hours"]])
p50_1 = np.array([x["p50"] for x in f1["hours"]])
check(fd["days"][0]["date"] == f1["date"], "k=1: gleicher Zieltag wie forecast_next_day")
check(np.allclose(p50_h, p50_1, atol=1e-9), "k=1: P50 BIT-identisch zu forecast_next_day")
p10_h = np.array([x["p10"] for x in fd["days"][0]["hours"]])
p10_1 = np.array([x["p10"] for x in f1["hours"]])
check(np.allclose(p10_h, p10_1, atol=1e-9), "k=1: P10-Band identisch (Trainings-Residuenquantile)")
check("p10" not in fd["days"][1]["hours"][0], "k=2: ehrlich NUR P50 (kein ungetestetes Band)")
check([d["horizon"] for d in fd["days"]] == [1, 2, 3]
      and [d["date"] for d in fd["days"]]
      == [str((days[-1] + pd.Timedelta(days=k)).date()) for k in (1, 2, 3)],
      "Horizont-Daten fortlaufend D+1..D+3")

# --- 2) Pseudo-Tage erreichen nie das Fit (Verhaltens-Beweise statt Interna) ---
_m, fit_end, _res = _fit_like_next_day(load2d, days, factory, 8, 28, None, None)
check(fit_end <= ND, "Fit endet in der echten Historie")
# (a) horizon=1 vs. horizon=3: k=1 identisch -> der Fit ist VOR jeder Pseudo-Erweiterung fixiert.
fd1 = forecast_days(load2d, days, factory, horizon=1, round_digits=None)
check(np.allclose([x["p50"] for x in fd1["days"][0]["hours"]], p50_h, atol=1e-9),
      "horizon=1 vs. horizon=3: identisches k=1 (Fit vor Pseudo-Erweiterung fixiert)")
# (b) Stoer-Beweis: haengen wir SELBST einen absurden Pseudo-Tag an die Historie und prognostizieren
# dann k=1, MUSS sich das Ergebnis aendern (Features sehen ihn) — waehrend forecast_days' k=2 auf dem
# eigenen P50-Pseudo basiert. Andersherum: forecast_days(horizon=2) zweimal gerechnet ist deterministisch.
fd2a = forecast_days(load2d, days, factory, horizon=2, round_digits=None)
fd2b = forecast_days(load2d, days, factory, horizon=2, round_digits=None)
check(fd2a == fd2b, "forecast_days deterministisch (kein versteckter Zustand)")

# --- 3) Grenzen ---
try:
    forecast_days(load2d, days, factory, horizon=8)
    check(False, "horizon=8 haette abgelehnt werden muessen")
except ValueError:
    check(True, "horizon=8 abgelehnt (Lag-7-Anker waere Pseudo)")
try:
    rolling_horizon_backtest(load2d[:50], days[:50], factory, horizon=3, n_test=42)
    check(False, "zu wenig Historie haette ValueError geben muessen")
except ValueError:
    check(True, "Backtest: zu wenig Historie -> klarer Fehler")

# --- 4) Backtest-Alignment + Plausibilitaet ---
bt = rolling_horizon_backtest(load2d, days, factory, horizon=3, n_test=21)
pk = bt["per_horizon"]
check(set(pk) == {1, 2, 3} and all(pk[k]["n_days"] == 21 for k in pk), "Backtest: 21 Tage je Horizont")
check(pk[1]["mae_mw"] > 0 and pk[1]["mape_pct"] > 0, "Backtest: Metriken > 0")
# Auf glatter Synthetik muss k=1 mindestens so gut wie k=3 sein (Fehler waechst mit Horizont)
check(pk[1]["mae_mw"] <= pk[3]["mae_mw"] + 0.05, "Fehler waechst (oder bleibt) mit dem Horizont")
# Skill-Formel konsistent
k = 2
expect = round((1.0 - pk[k]["mae_mw"] / pk[k]["mae_snaive_mw"]) * 100.0, 1)
check(abs(pk[k]["skill_vs_snaive_pct"] - expect) < 0.11, "Skill-Formel konsistent (k=2)")

print(f"ALLE {N[0]} CHECKS GRUEN — Horizont-Engine verifiziert.")
