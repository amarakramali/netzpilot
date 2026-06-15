"""Residuallast (Last - PV - Wind) je Stadt prognostizieren (Sandbox-rechenbar)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from netzpilot.features.build import get_holidays
from netzpilot.eval.backtest import rolling_origin
from netzpilot.eval.conformal import rolling_origin_cqr
from netzpilot.models.robust_corrector import ShrunkCorrector
DATA = "netzpilot/data/training_cities"

def load_residual(path, keep_days=110):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    res = df["load_mw"].astype(float) - df.get("pv_feedin_mw", 0.0).astype(float) - df.get("wind_feedin_mw", 0.0).astype(float)
    s = pd.Series(res.values, index=df["timestamp"]).sort_index()
    h = s.resample("1h").mean()
    d = pd.DataFrame({"v": h.values}, index=h.index); d["date"] = d.index.normalize(); d["hour"] = d.index.hour
    g = {dt: x.sort_values("hour")["v"].to_numpy() for dt, x in d.groupby("date")
         if len(x) == 24 and sorted(x["hour"].tolist()) == list(range(24))}
    good = sorted(g); load2d = np.array([g[k] for k in good]); days = pd.to_datetime([k.date() for k in good])
    return load2d[-keep_days:], days[-keep_days:]

print(f"{'Stadt':10s} {'Ø Resid':>8s} {'MAE':>6s} {'Skill_snv':>9s} {'Skill_pers':>10s}", flush=True)
out = []
for c in ["Muenster", "Bielefeld", "Berlin", "Kiel"]:
    load2d, days = load_residual(f"{DATA}/{c}_Netz_Lastgang_2024.csv")
    hol = get_holidays(sorted({d.year for d in days}), "NW")
    _, sm = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=14, holiday_set=hol)
    m = sm["metriken"]["model"]
    out.append({"city": c, "mean_residual_mw": round(float(load2d.mean()), 1), **m})
    print(f"{c:10s} {load2d.mean():8.1f} {m['MAE_MW']:6.2f} {m['Skill_vs_SaisonalNaiv_%']:+8.1f}% {m['Skill_vs_Persistenz_%']:+9.1f}%", flush=True)
load2d, days = load_residual(f"{DATA}/Muenster_Netz_Lastgang_2024.csv")
hol = get_holidays(sorted({d.year for d in days}), "NW")
_, cq = rolling_origin_cqr(load2d, days, lambda: ShrunkCorrector(10.0), alpha=0.2, cal_days=21, n_test=14, holiday_set=hol)
print(f"\nMuenster Residual CQR-Coverage (Soll 80%): {cq['coverage_%']}%", flush=True)
json.dump(out, open("data_cache/cities_residual_eval.json", "w"), indent=2)
