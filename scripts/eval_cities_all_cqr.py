# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Resumable 50-Staedte CQR-Kalibrierung: Coverage bei nominal 80% und 90% je Stadt.

Schreibt inkrementell nach data_cache/cities_all_cqr.jsonl (eine Zeile je Stadt),
ueberspringt Erledigtes -> timeout-sicher, beliebig oft neu startbar.
"""
import sys, os, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from netzpilot.features.build import get_holidays
from netzpilot.eval.conformal import rolling_origin_cqr
from netzpilot.models.robust_corrector import ShrunkCorrector

DATA = "netzpilot/data/training_cities"
OUT = "data_cache/cities_all_cqr.jsonl"
KEEP, NT, CAL = 110, 14, 21


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


def width_of(cq):
    for k in ("avg_width_MW", "width_MW", "mean_width_MW", "avg_width", "width"):
        if k in cq:
            return cq[k]
    return None


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
            rec = {"city": city}
            for nom, alpha in [(80, 0.2), (90, 0.1)]:
                _, cq = rolling_origin_cqr(a, days, lambda: ShrunkCorrector(10.0),
                                           alpha=alpha, cal_days=CAL, n_test=NT, holiday_set=hol)
                rec[f"cov{nom}"] = cq.get("coverage_%")
                rec[f"w{nom}"] = width_of(cq)
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(f"{city:14s} cov80={rec['cov80']}%  cov90={rec['cov90']}%", flush=True)
    print("ALL DONE" if not todo else "batch finished", flush=True)


if __name__ == "__main__":
    main()
