"""Dependency-free Markdown result report."""
from __future__ import annotations

import json


def build_markdown(summary: dict, title="NetzPilot - Prognose-Engine - Ergebnis") -> str:
    t = summary["metriken"]
    pr = summary["probabilistisch"]

    def row(n, lab):
        m = t[n]
        return (
            f"| {lab} | {m['MAE_MW']:.0f} | {m['RMSE_MW']:.0f} | {m['MAPE_%']:.2f} "
            f"| {m['MASE']:.3f} | {m['Skill_vs_Persistenz_%']:.0f}% "
            f"| {m['Skill_vs_SaisonalNaiv_%']:.1f}% |"
        )

    lines = [
        f"# {title}", "",
        f"Horizont: {summary['horizont']} - Test: {summary['test_tage']} Tage / {summary['test_vorhersagen']} Stunden",
        "", "| Verfahren | MAE [MW] | RMSE [MW] | MAPE [%] | MASE | Skill vs Pers. | Skill vs S-Naiv |",
        "|---|---|---|---|---|---|---|",
        row("persist", "Persistenz (t-24h)"), row("snaive", "Saisonal-Naiv (t-168h)"),
        row("model", "NetzPilot (Korrektur)"), "",
        (
            f"Probabilistisch: Pinball_avg {pr['Pinball_avg']} - "
            f"CRPS_proxy {pr.get('CRPS_proxy', 'n/a')} - "
            f"{pr.get('Interval_Label', 'P10-P90')}-Coverage {pr['Coverage_P10_P90_%']}% "
            f"(Ziel {pr['Soll_%']}%)"
        ),
    ]
    if "v1_reference" in summary:
        v1 = summary["v1_reference"]
        lines.extend([
            "",
            "V1-Referenz: "
            f"MAE {v1['MAE_MW']} MW, MAPE {v1['MAPE_%']}%, "
            f"P10-P90-Coverage {v1['Coverage_P10_P90_%']}%, "
            f"Skill vs S-Naiv {v1['Skill_vs_SaisonalNaiv_%']}% "
            f"({v1['window']}; {v1['note']})",
        ])
    meta = []
    if "target" in summary:
        meta.append(f"Zielgroesse: {summary['target']}")
    if "cache_dir" in summary:
        meta.append(f"Cache: {summary['cache_dir']}")
    if "feature_set" in summary:
        meta.append(f"Features: {summary['feature_set']}")
    if "weather_source" in summary:
        meta.append(f"Wetter: {summary['weather_source']}")
    if meta:
        lines.extend(["", "Provenienz: " + " - ".join(meta)])
    if "method_limitations" in summary:
        lines.extend(["", "Methodische Grenze: " + summary["method_limitations"]])
    lines.extend(["", "_Baselines bleiben Pflicht; Wetter im Backtest als Forecast, nicht Istwert._"])
    return "\n".join(lines)


def write_report(summary: dict, md_path: str, json_path: str | None = None):
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown(summary))
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
