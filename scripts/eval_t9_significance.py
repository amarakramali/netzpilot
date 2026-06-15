# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Bootstrap significance for T9 small-utility cached backtests.

The script uses only cached rolling-origin arrays from T9. It does not retrain or
change the evaluation protocol: each bootstrap block is one complete 24-hour test
day, paired across model and baseline errors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_v1_significance import ci95, paired_block_bootstrap

SEED = 20260530
N_BOOT = 10000
DEFAULT_INPUTS = [
    ("small_utility_no_weather", Path("data_cache/t9_small_utility/small_utility_80_arrays.npz")),
    ("small_utility_weather", Path("data_cache/t9_small_utility_weather/small_utility_80_arrays.npz")),
]


def daily_mae(arrays, name: str) -> np.ndarray:
    """Daily MAE [n_days] from flat rolling-origin hourly arrays."""
    actual = np.asarray(arrays["actual"], dtype=float)
    pred = np.asarray(arrays[name], dtype=float)
    if actual.shape != pred.shape:
        raise ValueError(f"{name} shape {pred.shape} does not match actual {actual.shape}")
    if len(actual) % 24 != 0:
        raise ValueError(f"{name} length {len(actual)} is not a whole number of days")
    return np.abs(pred - actual).reshape(-1, 24).mean(axis=1)


def summarize_npz(label: str, path: Path, n_boot: int, seed: int) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    arrays = np.load(path)
    ae = {name: daily_mae(arrays, name) for name in ("model", "snaive", "persist")}
    rng = np.random.default_rng(seed)

    out = {
        "label": label,
        "arrays": str(path),
        "n_test_days": int(len(ae["model"])),
        "daily_mae_MW": {k: round(float(v.mean()), 3) for k, v in ae.items()},
        "bootstrap": {},
    }

    for ref in ("snaive", "persist"):
        skill, dmae = paired_block_bootstrap(ae["model"], ae[ref], rng, n_boot)
        skill_point = (1.0 - ae["model"].sum() / ae[ref].sum()) * 100.0
        lo, hi = np.percentile(skill, [2.5, 97.5])
        out["bootstrap"][f"model_vs_{ref}"] = {
            "skill_point_%": round(float(skill_point), 2),
            "skill_ci95_%": ci95(skill),
            "P(model_besser)_%": round(float(np.mean(skill > 0) * 100), 1),
            "dMAE_mean_MW": round(float(dmae.mean()), 3),
            "dMAE_ci95_MW": [
                round(float(np.percentile(dmae, 2.5)), 3),
                round(float(np.percentile(dmae, 97.5)), 3),
            ],
            "tage_modell_gewinnt_%": round(float(np.mean(ae["model"] < ae[ref]) * 100), 1),
            "signifikant_5pct": bool(lo > 0.0),
        }

    return out


def parse_inputs(values: list[str]) -> list[tuple[str, Path]]:
    if not values:
        return DEFAULT_INPUTS
    parsed = []
    for item in values:
        if "=" not in item:
            raise ValueError("--input must be LABEL=PATH")
        label, path = item.split("=", 1)
        parsed.append((label, Path(path)))
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", default=[], help="LABEL=PATH to arrays.npz")
    ap.add_argument("--out", default="data_cache/t9_significance")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    runs = [
        summarize_npz(label, path, args.n_boot, args.seed)
        for label, path in parse_inputs(args.input)
    ]
    out = {
        "frage": "Ist der T9-Klein-Last-Skill gegen die Baselines von 0 unterscheidbar?",
        "methode": "paired block-bootstrap, Block = ganzer Tag (24 h)",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "runs": runs,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
