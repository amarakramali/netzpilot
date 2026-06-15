"""Split-conformal calibration for already generated quantile arrays."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.eval import metrics as M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="data_cache/t3_lightgbm_56d/arrays.npz")
    ap.add_argument("--calibration-days", type=int, default=28)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--out", default="data_cache/t6_split_conformal")
    ap.add_argument("--no-online-update", action="store_true")
    args = ap.parse_args()

    data = np.load(args.arrays)
    actual = data["actual"]
    model = data["model"]
    p10 = data["p10"]
    p90 = data["p90"]
    split = args.calibration_days * 24
    if split <= 0 or split >= len(actual):
        raise ValueError("Calibration split must leave a non-empty evaluation set")

    conformity = np.maximum(p10[:split] - actual[:split], actual[:split] - p90[:split])
    conformity = np.maximum(conformity, 0.0)
    eval_actual = actual[split:]
    eval_model = model[split:]
    eval_p10_raw = p10[split:]
    eval_p90_raw = p90[split:]
    eval_p10 = np.empty_like(eval_p10_raw)
    eval_p90 = np.empty_like(eval_p90_raw)
    qhats = []
    if args.no_online_update:
        qhat = float(np.quantile(conformity, 1 - args.alpha, method="higher"))
        eval_p10[:] = eval_p10_raw - qhat
        eval_p90[:] = eval_p90_raw + qhat
        qhats.append(qhat)
    else:
        scores = list(conformity)
        for start in range(0, len(eval_actual), 24):
            end = start + 24
            qhat = float(np.quantile(scores, 1 - args.alpha, method="higher"))
            qhats.append(qhat)
            eval_p10[start:end] = eval_p10_raw[start:end] - qhat
            eval_p90[start:end] = eval_p90_raw[start:end] + qhat
            new_scores = np.maximum(
                eval_p10_raw[start:end] - eval_actual[start:end],
                eval_actual[start:end] - eval_p90_raw[start:end],
            )
            scores.extend(np.maximum(new_scores, 0.0))
    summary = {
        "method": "split conformal widening of LightGBM P10/P90",
        "source_arrays": args.arrays,
        "calibration_days": args.calibration_days,
        "evaluation_days": int(len(eval_actual) / 24),
        "alpha": args.alpha,
        "qhat_MW_mean": round(float(np.mean(qhats)), 1),
        "qhat_MW_last": round(float(qhats[-1]), 1),
        "online_update": not args.no_online_update,
        "MAE_MW": round(M.mae(eval_model, eval_actual), 1),
        "RMSE_MW": round(M.rmse(eval_model, eval_actual), 1),
        "MAPE_%": round(M.mape(eval_model, eval_actual), 2),
        "Coverage_P10_P90_%": round(M.coverage(eval_actual, eval_p10, eval_p90), 1),
        "Soll_%": round((1 - args.alpha) * 100, 1),
        "Pinball_avg": round(np.mean([
            M.pinball(eval_actual, eval_p10, 0.1),
            M.pinball(eval_actual, eval_model, 0.5),
            M.pinball(eval_actual, eval_p90, 0.9),
        ]), 1),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "arrays.npz", actual=eval_actual, model=eval_model, p10=eval_p10, p90=eval_p90)
    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
