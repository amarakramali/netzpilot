# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Forecaster-Validierung auf staedtischen Lastgaengen (leichtgewichtig) + Provenienz-Check."""
import sys, os, itertools, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from netzpilot.features.build import get_holidays
from netzpilot.eval.backtest import rolling_origin
from netzpilot.eval.conformal import rolling_origin_cqr
from netzpilot.models.robust_corrector import ShrunkCorrector
DATA = "netzpilot/data/training_cities"

def load_city(path, keep_days=None):
    df = pd.read_csv(path, usecols=["timestamp", "load_mw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    h = df.set_index("timestamp")["load_mw"].astype(float).sort_index().resample("1h").mean()
    d = pd.DataFrame({"v": h.values}, index=h.index); d["date"] = d.index.normalize(); d["hour"] = d.index.hour
    g = {dt: x.sort_values("hour")["v"].to_numpy() for dt, x in d.groupby("date")
         if len(x) == 24 and sorted(x["hour"].tolist()) == list(range(24))}
    good = sorted(g); load2d = np.array([g[k] for k in good]); days = pd.to_datetime([k.date() for k in good])
    if keep_days: load2d, days = load2d[-keep_days:], days[-keep_days:]
    return load2d, days

prov = ["Muenster", "Bielefeld", "Berlin", "Kiel"]
shapes = {}
for c in prov:
    l2, _ = load_city(f"{DATA}/{c}_Netz_Lastgang_2024.csv", keep_days=120); fl = l2.flatten(); shapes[c] = fl / fl.mean()
print("=== Provenienz: Korrelation normierter Lastformen (120 Tage) ===", flush=True)
for a, b in itertools.combinations(shapes, 2):
    n = min(len(shapes[a]), len(shapes[b]))
    print(f"  {a:10s} vs {b:10s}: r={np.corrcoef(shapes[a][:n], shapes[b][:n])[0,1]:.3f}", flush=True)

print("\n=== Forecast je Stadt (ShrunkCorrector, 110 Tage, 14-Tage-Test) ===", flush=True)
print(f"{'Stadt':10s} {'mean':>7s} {'MAE':>6s} {'MAPE%':>6s} {'Skill_snv':>9s}", flush=True)
out = []
for c in ["Muenster", "Bielefeld", "Berlin", "Kiel"]:
    load2d, days = load_city(f"{DATA}/{c}_Netz_Lastgang_2024.csv", keep_days=110)
    hol = get_holidays(sorted({d.year for d in days}), "NW")
    _, sm = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=14, holiday_set=hol)
    m = sm["metriken"]["model"]
    out.append({"city": c, "mean_mw": round(float(load2d.mean()), 1), **m})
    print(f"{c:10s} {load2d.mean():7.1f} {m['MAE_MW']:6.2f} {m['MAPE_%']:6.2f} {m['Skill_vs_SaisonalNaiv_%']:+8.1f}%", flush=True)
# CQR-Coverage nur fuer eine Stadt
load2d, days = load_city(f"{DATA}/Muenster_Netz_Lastgang_2024.csv", keep_days=110)
hol = get_holidays(sorted({d.year for d in days}), "NW")
_, cq = rolling_origin_cqr(load2d, days, lambda: ShrunkCorrector(10.0), alpha=0.2, cal_days=21, n_test=14, holiday_set=hol)
print(f"\nMuenster CQR-Coverage (Soll 80%): {cq['coverage_%']}%", flush=True)
json.dump(out, open("data_cache/cities_eval.json", "w"), indent=2)
