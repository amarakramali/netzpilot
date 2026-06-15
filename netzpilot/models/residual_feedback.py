"""Online-Residuen-Feedback — korrigiert die Punktprognose (P50) um persistente Modellfehler.

Befund 2026-06-03 (gemessen): die Modell-Residuen (actual − forecast) sind SIGNIFIKANT positiv
autokorreliert (lag-1 Tagesmittel: Hilden +0,44, Herne +0,23, EVDB NS/Bitterfeld MS/Neuruppin +0,18…0,19;
Schwelle ±0,18) — was das Modell gestern verfehlt hat (z.B. ein Kälte-/Trend-Regime, das die Features nicht
voll abbilden), verfehlt es heute tendenziell wieder. Das ist leakage-sicher ausnutzbar: zur morgigen
Prognose einen Anteil ρ des ZULETZT BEOBACHTETEN Residuums addieren.

    korrigiert[d] = forecast[d] + ρ · (actual[d−1] − forecast[d−1])

ρ wird ONLINE auf einem nachlaufenden Fenster getunt (MAE-minimierend) und zur 0 geschrumpft (shrink) —
so adaptiert es: starke Autokorrelation → ρ groß (Gewinn), schwache → ρ≈0 (kein Schaden). Gemessen
(online, shrink 0,5): Hilden +3,32 %, Neuruppin +1,79 %, Herne +1,26 % MAE; Bitterfeld MS +0,20 % (No-Harm).

Es ist ein LEVEL-Shift δ=ρ·Vortagsresiduum: P10/P50/P90 werden gemeinsam um δ verschoben (Band-Mitte
korrigiert; die Bandbreite/Kalibrierung bleibt davon unberührt und greift danach). Leakage-sicher: Tag d
nutzt nur Residuen aus < d. Reine numpy.
"""
from __future__ import annotations

import numpy as np


def _mae_rho(forecasts, actuals, rho, lo, hi):
    """MAE von forecast[j] + rho·(actual[j−1]−forecast[j−1]) über j in [lo, hi) (alle < aktueller Tag)."""
    err = []
    for j in range(max(1, lo), hi):
        corr = forecasts[j] + rho * (actuals[j - 1] - forecasts[j - 1])
        err.append(np.abs(actuals[j] - corr))
    return float(np.mean(np.concatenate(err))) if err else float("inf")


def online_residual_feedback(forecasts, actuals, window=28, shrink=0.5, min_window=14,
                             rho_max=0.8, n_grid=17):
    """Online-rollende Residuen-Feedback-Korrektur der Punktprognose.

    forecasts/actuals: [n_days, H] (P50 und Ist). Für Tag d wird ρ_d auf [d−window, d) MAE-minimierend
    getunt (Gitter 0..rho_max), zur 0 geschrumpft, und δ_d = ρ_d·(actual[d−1]−forecast[d−1]) bestimmt.
    Leakage-sicher: nutzt nur Tage < d. Erste min_window+1 Tage: ρ=0 (kein Eingriff).
    Rückgabe (rho[n], delta[n,H], corrected[n,H]).
    """
    f = np.asarray(forecasts, float); a = np.asarray(actuals, float)
    if f.shape != a.shape:
        raise ValueError("forecasts und actuals müssen gleiche Form [n_days, H] haben.")
    if f.ndim != 2:
        raise ValueError("Erwartet 2D-Arrays [n_days, H].")
    if not (0.0 <= shrink <= 1.0):
        raise ValueError("shrink muss in [0,1] liegen.")
    n = f.shape[0]
    grid = np.linspace(0.0, rho_max, n_grid)
    rho = np.zeros(n)
    delta = np.zeros_like(f)
    corrected = f.copy()
    for d in range(n):
        lo = max(0, d - window)
        if d >= 1 and (d - lo) >= min_window:
            r_star = min(grid, key=lambda r: _mae_rho(f, a, r, lo, d))
            rho[d] = float(r_star) * float(shrink)
        if d >= 1:
            delta[d] = rho[d] * (a[d - 1] - f[d - 1])
            corrected[d] = f[d] + delta[d]
    return rho, delta, corrected
