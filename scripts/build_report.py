"""Build the T5 HTML report and plots from cached backtest outputs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.report.html_report import build_html_report


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _plot_metric_bars(summaries: list[tuple[str, dict]], out: Path) -> str:
    labels = [label for label, _ in summaries]
    maes = [summary["metriken"]["model"]["MAE_MW"] for _, summary in summaries]
    coverage = []
    targets = []
    for _, summary in summaries:
        cov = summary["probabilistisch"].get("Coverage_Interval_%", summary["probabilistisch"]["Coverage_P10_P90_%"])
        target = summary["probabilistisch"]["Soll_%"]
        coverage.append(float(cov) if isinstance(cov, (int, float)) else np.nan)
        targets.append(float(target) if isinstance(target, (int, float)) else np.nan)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].bar(x, maes, color="#2a6f97")
    axes[0].set_ylabel("MAE [MW]")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_title("Punktfehler")

    axes[1].bar(x, coverage, color="#4f8f55", label="Coverage")
    axes[1].scatter(x, targets, color="#a23b3b", label="Ziel", zorder=3)
    axes[1].set_ylabel("Coverage [%]")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].set_title("Intervallkalibrierung")

    path = out / "metrics_coverage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _plot_timeseries(arrays_path: Path, out: Path, name: str) -> str:
    arr = np.load(arrays_path)
    hours = np.arange(min(168, len(arr["actual"])))
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.plot(hours, arr["actual"][: len(hours)], label="Actual", color="#182026", linewidth=1.8)
    ax.plot(hours, arr["model"][: len(hours)], label="Model P50", color="#2a6f97", linewidth=1.5)
    ax.fill_between(hours, arr["p10"][: len(hours)], arr["p90"][: len(hours)], color="#9cc7df", alpha=0.35, label="Interval")
    ax.set_title(name)
    ax.set_xlabel("Test hour")
    ax.set_ylabel("MW")
    ax.legend(loc="best")
    path = out / f"{name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data_cache/t2_2022-01-01_2024-01-01")
    ap.add_argument("--out", default="data_cache/report")
    args = ap.parse_args()

    out = Path(args.out)
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    t3 = _load_json(Path("data_cache/t3_lightgbm/results.json"))
    t4 = _load_json(Path("data_cache/t4_residual/results.json"))
    t8 = _load_json(Path("data_cache/t8_cqr/results.json"))["runs"]
    summaries = [("T3 Load native P10-P90", t3), ("T4 Residual native P10-P90", t4)]
    t4_phys_path = Path("data_cache/t4_physical_generation/results.json")
    if t4_phys_path.exists():
        t4_phys = _load_json(t4_phys_path)
        summaries.append((
            "T4 Physical PV/Wind residual point",
            {
                "metriken": t4_phys["residual_metrics"],
                "probabilistisch": {
                    "Interval_Label": "point only",
                    "Coverage_Interval_%": "n/a",
                    "Coverage_P10_P90_%": "n/a",
                    "Soll_%": "n/a",
                },
            },
        ))
    for run in t8:
        label = f"T8 {run['target']} {run['probabilistisch']['Interval_Label']}"
        summaries.append((label, run))
    t9_path = Path("data_cache/t9_small_utility/results.json")
    t9_runs = []
    if t9_path.exists():
        t9_runs = _load_json(t9_path)["runs"]
        for run in t9_runs:
            label = f"T9 {run['target']} {run['probabilistisch']['Interval_Label']}"
            summaries.append((label, run))
    t9_weather_path = Path("data_cache/t9_small_utility_weather/results.json")
    t9_weather_runs = []
    if t9_weather_path.exists():
        t9_weather_runs = _load_json(t9_weather_path)["runs"]
        for run in t9_weather_runs:
            label = f"T9 weather {run['target']} {run['probabilistisch']['Interval_Label']}"
            summaries.append((label, run))
    t10_path = Path("data_cache/t10_small_utility/results.json")
    t10_runs = []
    if t10_path.exists():
        t10_runs = _load_json(t10_path)["runs"]
        for run in t10_runs:
            label = f"T10 improved {run['target']} {run['probabilistisch']['Interval_Label']}"
            summaries.append((label, run))

    provenance = _load_json(Path(args.cache_dir) / "provenance.json")
    integrity = _load_json(Path(args.cache_dir) / "integrity_report.json")

    plot_files = [_plot_metric_bars(summaries, plots_dir)]
    plot_files.append(_plot_timeseries(Path("data_cache/t8_cqr/load_80_arrays.npz"), plots_dir, "Load CQR P10-P90"))
    plot_files.append(_plot_timeseries(Path("data_cache/t8_cqr/residual_80_arrays.npz"), plots_dir, "Residual CQR P10-P90"))
    if t9_runs:
        plot_files.append(_plot_timeseries(Path("data_cache/t9_small_utility/small_utility_80_arrays.npz"), plots_dir, "Small Utility CQR P10-P90"))
    if t9_weather_runs:
        plot_files.append(_plot_timeseries(Path("data_cache/t9_small_utility_weather/small_utility_80_arrays.npz"), plots_dir, "Small Utility Weather CQR P10-P90"))
    if t10_runs:
        plot_files.append(_plot_timeseries(Path("data_cache/t10_small_utility/small_utility_t10_80_arrays.npz"), plots_dir, "Small Utility T10 CQR P10-P90"))
    plot_refs = [f"plots/{name}" for name in plot_files]

    html = build_html_report(summaries, provenance, integrity, plot_refs)
    with open(out / "netzpilot_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    with open(out / "report_inputs.json", "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "provenance": provenance, "integrity": integrity}, f, indent=2, ensure_ascii=False)
    print(json.dumps({"html": str(out / "netzpilot_report.html"), "plots": plot_refs}, indent=2))


if __name__ == "__main__":
    main()
