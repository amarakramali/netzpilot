"""T9 aggregation-level sensitivity: forecast skill vs. idiosyncratic load volatility.

Self-contained, dependency-light (numpy/pandas/stdlib only — runs in the sandbox).
Isolates ONE effect: how day-ahead forecast quality depends on portfolio aggregation
level for a small utility. The load is a SYNTHETIC small-utility proxy (clearly
labelled — NOT a real-data validation) built on the REAL national SMARD shape; the only
swept knob is the idiosyncratic multiplicative AR(1) volatility ``sigma``. Lower sigma =
more aggregation (a bigger, smoother portfolio); higher sigma = a small, volatile one.

It reuses the engine's leakage-safe rolling-origin protocol and keeps BOTH baselines
(persistence, seasonal-naive), so every number is comparable to the main backtest.

Honest reading of the output:
- MAPE rises monotonically as the portfolio shrinks -> small load is absolutely harder.
- Skill vs. persistence stays large at every level (persistence is a weak baseline).
- Skill vs. seasonal-naive is ~0 and within seed-noise at high aggregation; it grows at
  low aggregation ONLY because this generator's AR(1) idiosyncratic noise (rho=0.9) is
  partly predictable and the feature model exploits recent-lag autocorrelation that
  week-ago seasonal-naive ignores. With white-noise idiosyncrasy this trend would flatten.
  => Do NOT read this as "small utilities are easier to beat seasonal-naive on."

Output: data_cache/t9_aggregation_sweep/{rows.jsonl,results.json}. Resumable: completed
sigmas are skipped on re-run.

Run:
  python scripts/run_t9_aggregation_sweep.py
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.data.smard import load_local_json
from netzpilot.data.synthetic_smallutility import make_small_utility_load
from netzpilot.eval.backtest import rolling_origin
from netzpilot.features.build import get_holidays, to_daily
from netzpilot.models.ridge_correction import RidgeCorrector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGMAS = [0.02, 0.04, 0.06, 0.10, 0.15, 0.22, 0.32, 0.45]
SEEDS = list(range(6))
PEAK_MW, AR_RHO, N_TEST, REGION = 25.0, 0.9, 28, "NW"
# Optional "effective independent customers" reading: sigma_N ~= sigma_1 / sqrt(N).
# sigma_1 is the assumed single-consumer relative volatility; the N column is therefore
# an optimistic upper bound (real consumers are correlated via weather/calendar).
SIGMA1 = 0.45


def main() -> None:
    out_dir = os.path.join(ROOT, "data_cache", "t9_aggregation_sweep")
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, "rows.jsonl")

    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["sigma"])
                except Exception:  # noqa: BLE001
                    pass

    national = load_local_json(os.path.join(ROOT, "prognose_engine_v1", "data", "wk*.json"))
    _, ref_days = to_daily(make_small_utility_load(national, seed=0))
    holidays = get_holidays(sorted({d.year for d in ref_days}), REGION)
    print(
        f"base: real SMARD national, {len(national)} h ({len(national)//24} d); "
        f"sigmas={SIGMAS}; seeds={len(SEEDS)}; n_test={N_TEST}",
        flush=True,
    )

    for sigma in SIGMAS:
        if sigma in done:
            print(f"skip sigma={sigma} (done)", flush=True)
            continue
        sk_sn, sk_pe, mape, mae = [], [], [], []
        for seed in SEEDS:
            syn = make_small_utility_load(
                national, peak_mw=PEAK_MW, ar_rho=AR_RHO, noise_sigma=sigma, seed=seed
            )
            load2d, days = to_daily(syn)
            _, summary = rolling_origin(
                load2d, days, lambda: RidgeCorrector(lam=10.0),
                n_test=N_TEST, holiday_set=holidays,
            )
            m = summary["metriken"]["model"]
            sk_sn.append(m["Skill_vs_SaisonalNaiv_%"])
            sk_pe.append(m["Skill_vs_Persistenz_%"])
            mape.append(m["MAPE_%"])
            mae.append(m["MAE_MW"])
        win = sum(1 for x in sk_sn if x > 0)
        row = {
            "sigma": sigma,
            "approx_N_independent": round((SIGMA1 / sigma) ** 2, 1),
            "MAPE_%_mean": round(st.mean(mape), 2),
            "skill_vs_snaive_%_mean": round(st.mean(sk_sn), 2),
            "skill_vs_snaive_%_std": round(st.pstdev(sk_sn), 2),
            "skill_vs_snaive_%_min": round(min(sk_sn), 2),
            "skill_vs_snaive_%_max": round(max(sk_sn), 2),
            "skill_vs_snaive_winrate": f"{win}/{len(SEEDS)}",
            "skill_vs_persist_%_mean": round(st.mean(sk_pe), 2),
            "model_MAE_MW_mean": round(st.mean(mae), 3),
        }
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"sigma={sigma:>4} N~{row['approx_N_independent']:>6} MAPE {row['MAPE_%_mean']:>6}% "
            f"skill/S-Naiv {row['skill_vs_snaive_%_mean']:>6}% +-{row['skill_vs_snaive_%_std']:>4} "
            f"(win {row['skill_vs_snaive_winrate']}) skill/Persist {row['skill_vs_persist_%_mean']:>5}%",
            flush=True,
        )

    rows = [json.loads(line) for line in open(jsonl, encoding="utf-8") if line.strip()]
    rows = sorted({r["sigma"]: r for r in rows}.values(), key=lambda r: r["sigma"])
    if len(rows) == len(SIGMAS):
        pos = [r for r in rows if r["skill_vs_snaive_%_mean"] > 0]
        out = {
            "description": "SYNTHETIC small-utility aggregation sensitivity (proxy, NOT real-data validation)",
            "base_data": "prognose_engine_v1/data/wk*.json (real SMARD national load, 84 d)",
            "generator": "netzpilot.data.synthetic_smallutility.make_small_utility_load",
            "generator_params": {"peak_mw": PEAK_MW, "ar_rho": AR_RHO, "residential_share": 0.55, "n_large": 3},
            "protocol": "rolling_origin (leakage-safe, Ridge lam=10); baselines persist + snaive",
            "n_test": N_TEST,
            "region_holidays": REGION,
            "seeds": SEEDS,
            "sigmas": SIGMAS,
            "N_reading_assumption": f"sigma_N ~= sigma_1/sqrt(N), sigma_1={SIGMA1} (independent consumers; optimistic upper bound on N)",
            "caveat": "Skill-vs-snaive uptrend at high sigma is generator-dependent (AR(1) rho=0.9 noise is partly predictable). Not a claim that small utilities beat seasonal-naive more easily.",
            "rows": rows,
            "crossover_sigma_skill_vs_snaive_positive": (
                max(pos, key=lambda r: r["sigma"])["sigma"] if pos else None
            ),
        }
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("\nALL DONE -> results.json", flush=True)


if __name__ == "__main__":
    main()
