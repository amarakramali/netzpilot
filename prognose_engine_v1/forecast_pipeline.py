"""
NetzPilot Prognose-Engine - v1 Vertical Slice (echte SMARD-Daten, DE Netzlast, stuendlich)
Day-ahead 24h. Reines numpy/pandas. Leakage-sicheres Rolling-Origin-Backtest.
Baselines: Persistenz (t-24h), Saisonal-Naiv (t-168h).
Modell: Saisonal-Naiv + Ridge-Korrektur der Wochenabweichung (P50) + stundenbedingte Residuenquantile (P10/P90).
"""
import json, glob, numpy as np, pandas as pd

# 1) echte Daten laden
pairs=[]
for f in sorted(glob.glob("data/wk*.json")): pairs += json.load(open(f))
ts=np.array([p[0] for p in pairs],dtype=np.int64); load=np.array([p[1] for p in pairs],float)
assert (np.diff(ts)==3600000).all()
idx=pd.to_datetime(ts,unit="ms",utc=True).tz_convert("Europe/Berlin")
NH=len(load); ND=NH//24; load2d=load.reshape(ND,24)
days=pd.to_datetime([idx[d*24].date() for d in range(ND)])
HOLID={pd.Timestamp("2024-01-01").date()}
def hol(d): return 1.0 if days[d].date() in HOLID else 0.0

# 2) Features fuer Korrekturmodell  r(d,h)=load(d,h)-load(d-7,h)  (alles <= Tag d-1 23:00 bekannt)
def feats(d):
    dev_prev=load2d[d-1]-load2d[d-8]                 # gestern vs. Vorwoche-gestern (je Stunde)
    dev_mean=dev_prev.mean()                          # Niveauversatz dieser Woche
    trend=load2d[d-1].mean()-load2d[d-8].mean()
    dow=days[d].dayofweek; wknd=1.0 if dow>=5 else 0.0; h_=hol(d)
    X=[]
    for h in range(24):
        X.append([1.0,
            dev_prev[h], dev_mean, trend,
            load2d[d-1,h]-load2d[d-7,h],
            np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),np.sin(4*np.pi*h/24),np.cos(4*np.pi*h/24),
            np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7), wknd, h_])
    return np.array(X)
def base(d): return load2d[d-7].copy()        # Saisonal-Naiv als Basis
def resid_target(d): return load2d[d]-load2d[d-7]

FIRST=8; TEST=list(range(ND-28,ND)); LAM=10.0
def fit(X,y,lam):
    mu=X[:,1:].mean(0); sd=X[:,1:].std(0); sd[sd==0]=1
    Xs=np.hstack([np.ones((len(X),1)),(X[:,1:]-mu)/sd])
    A=Xs.T@Xs+lam*np.eye(Xs.shape[1]); A[0,0]-=lam
    return (np.linalg.solve(A,Xs.T@y),mu,sd)
def pred(m,X):
    w,mu,sd=m; return np.hstack([np.ones((len(X),1)),(X[:,1:]-mu)/sd])@w

rows={k:[] for k in ["persist","snaive","ridge","actual","hour","p10","p90"]}
for d in TEST:
    Xtr=np.vstack([feats(t) for t in range(FIRST,d)])
    ytr=np.concatenate([resid_target(t) for t in range(FIRST,d)])
    m=fit(Xtr,ytr,LAM)
    fitted=np.concatenate([base(t) for t in range(FIRST,d)])+pred(m,Xtr)
    actual_tr=np.concatenate([load2d[t] for t in range(FIRST,d)])
    res=actual_tr-fitted
    q10={h:np.quantile(res[h::24],0.10) for h in range(24)}
    q90={h:np.quantile(res[h::24],0.90) for h in range(24)}
    yhat=base(d)+pred(m,feats(d)); yd=load2d[d]
    for h in range(24):
        rows["persist"].append(load2d[d-1,h]); rows["snaive"].append(load2d[d-7,h])
        rows["ridge"].append(yhat[h]); rows["actual"].append(yd[h]); rows["hour"].append(h)
        rows["p10"].append(yhat[h]+q10[h]); rows["p90"].append(yhat[h]+q90[h])
R={k:np.array(v) for k,v in rows.items()}; a=R["actual"]
mae=lambda p:float(np.mean(np.abs(p-a))); rmse=lambda p:float(np.sqrt(np.mean((p-a)**2)))
mape=lambda p:float(np.mean(np.abs((p-a)/a))*100)
scale=np.mean(np.abs(load2d[FIRST:ND-28]-load2d[FIRST-1:ND-29]))
pin=lambda qp,t:float(np.mean(np.maximum(t*(a-qp),(t-1)*(a-qp))))
mp=mae(R["persist"]); ms=mae(R["snaive"])
tab={}
for n in ["persist","snaive","ridge"]:
    p=R[n]; tab[n]={"MAE_MW":round(mae(p),1),"RMSE_MW":round(rmse(p),1),"MAPE_%":round(mape(p),2),
        "MASE":round(mae(p)/scale,3),"Skill_vs_Persistenz_%":round((1-mae(p)/mp)*100,1),
        "Skill_vs_SaisonalNaiv_%":round((1-mae(p)/ms)*100,1)}
summary={"datensatz":"SMARD Netzlast DE, stuendlich (echt)","zeitraum":f"{days[0].date()}..{days[-1].date()}",
  "tage":ND,"test_tage":len(TEST),"test_vorhersagen":len(a),"horizont":"Day-ahead 24h",
  "mittlere_last_MW":round(float(load.mean())),"metriken":tab,
  "probabilistisch":{"Pinball_avg":round(float(np.mean([pin(R['p10'],.1),pin(R['ridge'],.5),pin(R['p90'],.9)])),1),
    "Coverage_P10_P90_%":round(float(np.mean((a>=R['p10'])&(a<=R['p90']))*100),1),"Soll_%":80}}
json.dump(summary,open("forecast_results.json","w"),indent=2,ensure_ascii=False)
np.savez("forecast_arrays.npz",**R,test_day_starts=np.array([days[d].value for d in TEST]))
print(json.dumps(summary,indent=2,ensure_ascii=False))
