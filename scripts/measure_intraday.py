#!/usr/bin/env python3
"""Intraday-Hebel MESSEN, bevor irgendetwas verdrahtet wird (Juni-Disziplin: Mess-Schleife).

Frage: Um Stunde h des Prognosetags sind die Stunden 0..h-1 bereits Ist — wie viel besser wird der
RESTTAG (h..23), wenn die Day-ahead-P50 um einen Level-Shift aus den heutigen Residuen korrigiert wird?

    δ_h = shrink · w-Mittel( actual[0:h] − dayahead[0:h] ),   korrigiert[h:] = dayahead[h:] + δ_h

Leakage-frei per Konstruktion: δ_h nutzt ausschließlich Stunden < h DESSELBEN Tages; die
Day-ahead-Basis stammt aus dem leakage-sicheren rolling_origin-Backtest (nur Vergangenheit).

Varianten: Gewichtung mean (alle Stunden gleich) vs. ewm3 (Halbwertszeit 3 h, jüngste Stunden zählen
mehr) × shrink ∈ {0.5, 0.75, 1.0}. Bewertet je Update-Stunde h: MAE des Resttags vs. statisch,
plus No-Harm-Quote (Anteil Tage, die sich verschlechtern). KEINE Parameterwahl auf demselben
Datensatz verschweigen: alle Zellen werden ausgegeben; der Default wird als der über ALLE Reihen
robusteste gewählt und so dokumentiert.

Aufruf:  python scripts/measure_intraday.py   (Sandbox: via /tmp-Paket; venv: direkt)
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from netzpilot.eval.backtest import rolling_origin
from netzpilot.features.build import get_holidays, to_daily_local
from netzpilot.models.robust_corrector import ShrunkCorrector
from scripts.pilot_in_a_box import robust_load_csv

SERIES = [
    ("Hilden Netzumsatz", "data_cache/real/Netzumsatz-Lastgang-2025.csv", "Text", "Reihe1", "NW"),
    ("EVDB NS", "data_cache/real/evdb_lastgang_ns_2024.csv", None, "Wert", "NI"),
    ("Herne 110/10kV", "data_cache/real/herne_bezug_vorgelagerte_ebene_2024.csv", None, "Load_1", "NW"),
]
N_TEST = 84
UPDATE_HOURS = (3, 6, 9, 12, 15, 18, 21)
SHRINKS = (0.5, 0.75, 1.0)


def _weights(h: int, kind: str) -> np.ndarray:
    if kind == "mean":
        return np.full(h, 1.0 / h)
    half = 3.0                                    # ewm3: Halbwertszeit 3 Stunden
    w = 0.5 ** ((h - 1 - np.arange(h)) / half)
    return w / w.sum()


def evaluate(model2d: np.ndarray, actual2d: np.ndarray):
    """je (kind, shrink, h): (MAE_rest_statisch, MAE_rest_korrigiert, %Tage besser)."""
    resid = actual2d - model2d
    out = {}
    for kind in ("mean", "ewm3"):
        for s in SHRINKS:
            for h in UPDATE_HOURS:
                w = _weights(h, kind)
                delta = s * (resid[:, :h] @ w)                  # [n_days]
                rest_static = np.abs(resid[:, h:])
                rest_corr = np.abs(resid[:, h:] - delta[:, None])
                mae_s = float(rest_static.mean())
                mae_c = float(rest_corr.mean())
                per_day_better = float(np.mean(rest_corr.mean(1) < rest_static.mean(1)) * 100)
                out[(kind, s, h)] = (mae_s, mae_c, per_day_better)
    return out


def main() -> int:
    agg = {}
    for name, csv, ts, col, reg in SERIES:
        hourly, _, _, _ = robust_load_csv(csv, ts_col=ts, load_col=col, unit="kW", return_meta=True)
        load2d, days, _ = to_daily_local(hourly)
        hol = get_holidays(sorted({d.year for d in days}), reg)
        R, _ = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0),
                              n_test=N_TEST, holiday_set=hol)
        H = load2d.shape[1]
        model2d = np.asarray(R["model"], float).reshape(-1, H)
        actual2d = np.asarray(R["actual"], float).reshape(-1, H)
        res = evaluate(model2d, actual2d)
        agg[name] = res
        print(f"\n== {name} (n_test={N_TEST}) — Resttag-MAE-Reduktion in % (positiv = besser) ==")
        print("    h: " + "  ".join(f"{h:>5d}" for h in UPDATE_HOURS))
        for kind in ("mean", "ewm3"):
            for s in SHRINKS:
                cells = []
                for h in UPDATE_HOURS:
                    ms, mc, pb = res[(kind, s, h)]
                    cells.append(f"{(1-mc/ms)*100:+5.1f}")
                print(f"{kind:>5s} s={s:.2f}: " + "  ".join(cells))
        # No-Harm-Zeile fuer die mittlere Variante
        nh = "  ".join(f"{res[('ewm3',0.75,h)][2]:5.1f}" for h in UPDATE_HOURS)
        print(f"  %Tage besser (ewm3, s=0.75): {nh}")

    print("\n== ROBUSTHEITS-RANKING (Mittel der MAE-Reduktion über alle Reihen, je Variante) ==")
    ranking = []
    for kind in ("mean", "ewm3"):
        for s in SHRINKS:
            vals = []
            for name in agg:
                for h in UPDATE_HOURS:
                    ms, mc, _ = agg[name][(kind, s, h)]
                    vals.append((1 - mc / ms) * 100)
            worst = min(
                np.mean([(1 - agg[name][(kind, s, h)][1] / agg[name][(kind, s, h)][0]) * 100
                         for h in UPDATE_HOURS]) for name in agg)
            ranking.append((float(np.mean(vals)), float(worst), kind, s))
    for mean_gain, worst_series, kind, s in sorted(ranking, reverse=True):
        print(f"  {kind:>5s} s={s:.2f}: mittlere Reduktion {mean_gain:+.2f} % | schwächste Reihe {worst_series:+.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
