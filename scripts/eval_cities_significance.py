"""Resumable Per-Stadt-Signifikanz (paired Block-Bootstrap, Block = ganzer Tag).

Fuer jede der 50 training_cities: rolling-origin Punktprognose (ShrunkCorrector),
dann paired block-bootstrap der taeglichen MAE gegen Saisonal-Naiv und Persistenz.
Schreibt inkrementell nach data_cache/cities_significance.jsonl (resumable).

Frage: Bei wie vielen Staedten ist der Vorsprung EINZELN signifikant (n=14 Testtage)?
Wiederverwendet die bestehende Bootstrap-Maschinerie aus eval_v1_significance.
"""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from netzpilot.features.build import get_holidays
from netzpilot.eval.backtest import rolling_origin
from netzpilot.models.robust_corrector import ShrunkCorrector
from scripts.eval_v1_significance import ci95, paired_block_bootstrap

DATA = "netzpilot/data/training_cities"
OUT = "data_cache/cities_significance_v2.jsonl"
KEEP, NT, NBOOT, SEED = 110, 14, 3000, 20260530


def load_city(path):
    df = pd.read_csv(path, usecols=["timestamp", "load_mw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    h = df.set_index("timestamp")["load_mw"].astype(float).sort_index().resample("1h").mean()
    d = pd.DataFrame({"v": h.values}, index=h.index)
    d["date"] = d.index.normalize(); d["hour"] = d.index.hour
    g = {dt: x.sort_values("hour")["v"].to_numpy() for dt, x in d.groupby("date")
         if len(x) == 24 and sorted(x["hour"].tolist()) == list(range(24))}
    good = sorted(g)
    a = np.array([g[k] for k in good]); days = pd.to_datetime([k.date() for k in good])
    return a[-KEEP:], days[-KEEP:]


def daily_mae(R, name):
    a = np.asarray(R["actual"], float)
    p = np.asarray(R[name], float)
    return np.abs(p - a).reshape(-1, 24).mean(axis=1)


def main():
    os.makedirs("data_cache", exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            if line.strip():
                done.add(json.loads(line)["city"])
    files = sorted(glob.glob(f"{DATA}/*_Netz_Lastgang_2024.csv"))
    todo = [f for f in files if os.path.basename(f).split("_")[0] not in done]
    print(f"done={len(done)} todo={len(todo)}", flush=True)
    with open(OUT, "a") as fo:
        for f in todo:
            city = os.path.basename(f).split("_")[0]
            a, days = load_city(f)
            hol = get_holidays(sorted({d.year for d in days}), "NW")
            R, _ = rolling_origin(a, days, lambda: ShrunkCorrector(10.0), n_test=NT, holiday_set=hol)
            dm_model = daily_mae(R, "model")
            rng = np.random.default_rng(SEED)
            rec = {"city": city, "n_test_days": int(len(dm_model))}
            for ref in ("snaive", "persist"):
                dm_ref = daily_mae(R, ref)
                skill, _ = paired_block_bootstrap(dm_model, dm_ref, rng, NBOOT)
                point = (1.0 - dm_model.sum() / dm_ref.sum()) * 100.0
                lo, hi = np.percentile(skill, [2.5, 97.5])
                rec[f"vs_{ref}"] = {
                    "skill_%": round(float(point), 2),
                    "ci95_%": [round(float(lo), 2), round(float(hi), 2)],
                    "sig_5pct": bool(lo > 0.0),
                    "P_besser_%": round(float(np.mean(skill > 0) * 100), 1),
                }
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            s = rec["vs_snaive"]
            print(f"{city:14s} snv {s['skill_%']:+5.1f}% CI[{s['ci95_%'][0]:+.1f},{s['ci95_%'][1]:+.1f}] "
                  f"sig={s['sig_5pct']}", flush=True)
    print("ALL DONE" if not todo else "batch finished", flush=True)


if __name__ == "__main__":
    main()
