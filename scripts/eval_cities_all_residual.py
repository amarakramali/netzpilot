"""Resumable 50-Staedte Residuallast (Last-PV-Wind) Auswertung -> JSONL."""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from netzpilot.features.build import get_holidays
from netzpilot.eval.backtest import rolling_origin
from netzpilot.models.robust_corrector import ShrunkCorrector
DATA="netzpilot/data/training_cities"; OUT="data_cache/cities_all_residual.jsonl"; N_TEST=14; KEEP=110
def load_res(path):
    df=pd.read_csv(path); df["timestamp"]=pd.to_datetime(df["timestamp"])
    res=df["load_mw"].astype(float)-df["pv_feedin_mw"].astype(float)-df["wind_feedin_mw"].astype(float)
    h=pd.Series(res.values,index=df["timestamp"]).sort_index().resample("1h").mean()
    d=pd.DataFrame({"v":h.values},index=h.index); d["date"]=d.index.normalize(); d["hour"]=d.index.hour
    g={dt:x.sort_values("hour")["v"].to_numpy() for dt,x in d.groupby("date")
       if len(x)==24 and sorted(x["hour"].tolist())==list(range(24))}
    good=sorted(g); a=np.array([g[k] for k in good]); days=pd.to_datetime([k.date() for k in good])
    return a[-KEEP:], days[-KEEP:]
os.makedirs("data_cache",exist_ok=True); done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        if l.strip(): done.add(json.loads(l)["city"])
files=sorted(glob.glob(f"{DATA}/*_Netz_Lastgang_2024.csv"))
todo=[f for f in files if os.path.basename(f).split("_")[0] not in done]
print(f"done={len(done)} todo={len(todo)}",flush=True)
with open(OUT,"a") as fo:
    for f in todo:
        city=os.path.basename(f).split("_")[0]; a,days=load_res(f)
        hol=get_holidays(sorted({d.year for d in days}),"NW")
        _,sm=rolling_origin(a,days,lambda:ShrunkCorrector(10.0),n_test=N_TEST,holiday_set=hol)
        m=sm["metriken"]["model"]
        rec={"city":city,"mean_mw":round(float(a.mean()),1),"MAE_MW":m["MAE_MW"],
             "skill_snv_%":m["Skill_vs_SaisonalNaiv_%"],"skill_pers_%":m["Skill_vs_Persistenz_%"]}
        fo.write(json.dumps(rec)+"\n"); fo.flush()
        print(f"{city:14s} mean={rec['mean_mw']:7.1f} MAE={rec['MAE_MW']:6.2f} snv={rec['skill_snv_%']:+5.1f}% pers={rec['skill_pers_%']:+5.1f}%",flush=True)
print("ALL DONE" if not todo else "batch finished",flush=True)
