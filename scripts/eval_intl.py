#!/usr/bin/env python3
"""Separate international benchmark for national aggregate TSO load (T30).

This is intentionally separate from the DSO corpus:
- reads only data_cache/intl/entsoe_*.csv
- writes only data_cache/intl/intl_benchmark.*
- never touches corpus_index.json or the pool prior
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from netzpilot.eval.backtest import rolling_origin
from netzpilot.features.build import to_daily_local
from netzpilot.models.robust_corrector import ShrunkCorrector

SEED = 20260601
COUNTRIES = [
    ("DE", "Germany"),
    ("NL", "Netherlands"),
    ("AT", "Austria"),
    ("CH", "Switzerland"),
    ("FR", "France"),
]


def daily_mae(result, name):
    ae = np.abs(result[name] - result["actual"]).reshape(-1, 24)
    return np.nanmean(ae, axis=1)


def paired_block_bootstrap(ae_model, ae_ref, rng, n_boot):
    mask = np.isfinite(ae_model) & np.isfinite(ae_ref)
    ae_model, ae_ref = ae_model[mask], ae_ref[mask]
    n = len(ae_model)
    if n < 3:
        return None
    idx = rng.integers(0, n, size=(n_boot, n))
    sm = ae_model[idx].sum(axis=1)
    sr = ae_ref[idx].sum(axis=1)
    sr = np.where(sr == 0, np.nan, sr)
    skill = (1.0 - sm / sr) * 100.0
    dmae = (ae_ref[idx] - ae_model[idx]).mean(axis=1)
    return skill, dmae, n


def pct(x, q):
    return float(np.nanpercentile(x, q))


def bootstrap_summary(ae_model, ae_ref, rng, n_boot):
    res = paired_block_bootstrap(ae_model, ae_ref, rng, n_boot)
    if res is None:
        return None
    skill, dmae, n = res
    skill_point = (1.0 - np.nansum(ae_model) / np.nansum(ae_ref)) * 100.0
    return {
        "skill_point_%": round(float(skill_point), 1),
        "skill_ci95_%": [round(pct(skill, 2.5), 1), round(pct(skill, 97.5), 1)],
        "P_model_besser_%": round(float(np.nanmean(skill > 0) * 100), 1),
        "dMAE_mean_MW": round(float(np.nanmean(dmae)), 3),
        "n_test_days": int(n),
        "signifikant_5pct": bool(pct(skill, 2.5) > 0),
    }


def load_country(path):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    load = pd.to_numeric(df["load_mw"], errors="coerce")
    s = pd.Series(load.to_numpy(dtype=float), index=ts).dropna().sort_index()
    return s[~s.index.duplicated(keep="first")]


def evaluate_country(code, name, path, n_test, n_boot):
    hourly = load_country(path)
    load2d, days, _ = to_daily_local(hourly)
    if len(load2d) < n_test + 30:
        return {"country_code": code, "country": name, "error": f"too few complete days ({len(load2d)})"}
    result, summary = rolling_origin(
        load2d, days, lambda: ShrunkCorrector(10.0), n_test=n_test, holiday_set=set())
    ae = {m: daily_mae(result, m) for m in ["model", "snaive", "persist"]}
    rng = np.random.default_rng(SEED)
    model_metrics = summary["metriken"]["model"]
    mape = model_metrics["MAPE_%"]
    return {
        "country_code": code,
        "country": name,
        "path": path,
        "source_label": "ENTSO-E Power Statistics monthly hourly load values 2024",
        "data_label": "national aggregate TSO load; not distribution-network load",
        "n_input_rows": int(len(hourly)),
        "n_complete_local_days": int(len(load2d)),
        "n_test": int(n_test),
        "mean_load_MW": round(float(np.nanmean(load2d)), 1),
        "MAE_MW": model_metrics["MAE_MW"],
        "MAPE_%": None if not math.isfinite(mape) else mape,
        "MASE": model_metrics["MASE"],
        "coverage_P10_P90_%": summary["probabilistisch"]["Coverage_P10_P90_%"],
        "bootstrap": {
            "snaive": bootstrap_summary(ae["model"], ae["snaive"], rng, n_boot),
            "persist": bootstrap_summary(ae["model"], ae["persist"], rng, n_boot),
        },
    }


def fmt_ci(b):
    if b is None:
        return "n/a"
    lo, hi = b["skill_ci95_%"]
    star = "*" if b["signifikant_5pct"] else "n.s."
    return f"{b['skill_point_%']:+.1f}% [{lo:+.1f},{hi:+.1f}] {star}"


def build_table(results):
    lines = [
        "| Country | Mean load | MAPE | MASE | Skill vs S-Naiv (CI95) | Skill vs Persistenz (CI95) | Cov 80% |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['country']} ({r['country_code']}) | - | - | - | ERROR: {r['error']} | - | - |")
            continue
        mape = "-" if r["MAPE_%"] is None else f"{r['MAPE_%']:.1f}%"
        lines.append(
            f"| {r['country']} ({r['country_code']}) | {r['mean_load_MW']:.1f} MW | {mape} | {r['MASE']} | "
            f"{fmt_ci(r['bootstrap']['snaive'])} | {fmt_ci(r['bootstrap']['persist'])} | "
            f"{r['coverage_P10_P90_%']:.0f}% |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default="data_cache/intl")
    ap.add_argument("--out", default="data_cache/intl")
    ap.add_argument("--n-test", type=int, default=28)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = []
    for code, name in COUNTRIES:
        path = os.path.join(args.source_dir, f"entsoe_{code.lower()}_2024.csv")
        if not os.path.exists(path):
            results.append({"country_code": code, "country": name, "error": f"missing {path}"})
            continue
        print(f"[run] {code} {name}")
        results.append(evaluate_country(code, name, path, args.n_test, args.n_boot))

    ok = [r for r in results if "error" not in r]
    n_sig = sum(1 for r in ok if (r["bootstrap"].get("snaive") or {}).get("signifikant_5pct"))
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_test": args.n_test,
        "n_boot": args.n_boot,
        "method": "rolling-origin ShrunkCorrector + paired block-bootstrap over whole days",
        "strict_scope": "national aggregate TSO load only; excluded from DSO corpus and pool prior",
        "n_countries": len(results),
        "n_ok": len(ok),
        "n_signifikant_vs_snaive_5pct": n_sig,
        "results": results,
    }
    with open(os.path.join(args.out, "intl_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    table = build_table(results)
    md = f"""# NetzPilot - International National-Load Benchmark

*Generated {out['generated_utc']} · Seed {SEED} · {args.n_test} rolling test days · {args.n_boot} bootstrap resamples.*

Scope: **national aggregate TSO load only** from ENTSO-E Power Statistics. This is a cross-country
method demo, not distribution-network evidence, and is intentionally separate from `data_cache/real/`.

{table}

**Summary:** Engine beats seasonal naive significantly in {n_sig}/{len(ok)} countries on national aggregate load.
National load is smoother than DSO load; do not count these rows as distribution networks or pool-prior evidence.
"""
    with open(os.path.join(args.out, "intl_benchmark.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(table)
    print(f"-> {args.out}/intl_benchmark.md + intl_benchmark.json")


if __name__ == "__main__":
    main()
