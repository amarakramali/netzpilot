"""Small HTML report builder for cached NetzPilot backtest results."""
from __future__ import annotations

from html import escape


def _metric_row(label: str, summary: dict) -> str:
    model = summary["metriken"]["model"]
    prob = summary["probabilistisch"]
    return (
        "<tr>"
        f"<td>{escape(label)}</td>"
        f"<td>{model['MAE_MW']}</td>"
        f"<td>{model['RMSE_MW']}</td>"
        f"<td>{model['MAPE_%']}</td>"
        f"<td>{model['Skill_vs_Persistenz_%']}%</td>"
        f"<td>{model['Skill_vs_SaisonalNaiv_%']}%</td>"
        f"<td>{escape(prob.get('Interval_Label', 'P10-P90'))}</td>"
        f"<td>{prob.get('Coverage_Interval_%', prob['Coverage_P10_P90_%'])}% / {prob['Soll_%']}%</td>"
        "</tr>"
    )


def build_html_report(summaries: list[tuple[str, dict]], provenance: dict, integrity: dict, plots: list[str]) -> str:
    rows = "\n".join(_metric_row(label, summary) for label, summary in summaries)
    plot_imgs = "\n".join(f'<figure><img src="{escape(path)}" alt="{escape(path)}"></figure>' for path in plots)
    weather_rule = provenance.get("weather_training_rule", "Historical Forecast API only.")
    generated = provenance.get("generated_at_utc", "unknown")
    start = provenance.get("start_utc_inclusive", "unknown")
    end = provenance.get("end_utc_exclusive", "unknown")
    load_report = integrity.get("smard_load_hour", {})
    gap_count = load_report.get("gap_count", "unknown")
    non_finite = load_report.get("non_finite_count", "unknown")
    return f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>NetzPilot Prognose-Engine Bericht</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #182026; }}
    h1, h2 {{ color: #0b3d57; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef4f7; }}
    .note {{ background: #f7f9fb; border-left: 4px solid #2a6f97; padding: 12px 14px; }}
    figure {{ margin: 18px 0 28px; }}
    img {{ max-width: 100%; border: 1px solid #d8dee4; }}
  </style>
</head>
<body>
  <h1>NetzPilot Prognose-Engine</h1>
  <p class="note">Baselines bleiben Pflicht. Wetter im Backtest ist Open-Meteo Historical Forecast, nicht Reanalyse/Ist-Wetter.</p>

  <h2>Metriken</h2>
  <table>
    <thead>
      <tr><th>Lauf</th><th>MAE MW</th><th>RMSE MW</th><th>MAPE %</th><th>Skill vs Persistenz</th><th>Skill vs S-Naiv</th><th>Intervall</th><th>Coverage / Ziel</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>

  <h2>Plots</h2>
  {plot_imgs}

  <h2>Provenienz</h2>
  <ul>
    <li>Datenstand UTC: {escape(str(start))} bis {escape(str(end))}, erzeugt {escape(str(generated))}</li>
    <li>Wetterregel: {escape(str(weather_rule))}</li>
    <li>SMARD Last hourly: gaps={escape(str(gap_count))}, non_finite={escape(str(non_finite))}</li>
    <li>Modell: LightGBM quantile residual correction + rolling CQR where noted; fixed random_state=42.</li>
  </ul>

  <h2>Einordnung</h2>
  <p>T4 ist ein Residuallast-MVP mit direkter SMARD-Erzeugungsreihe. Physikalische PV-/Windprognosen mit pvlib/windpowerlib bleiben ein Ausbaupfad.</p>
  <p>Der neue T4-Physical-Vergleich nutzt pvlib/windpowerlib-Proxies plus Bias-Korrektur. Er ist methodisch sauberer fuer Erzeugung, aber als Point-Forecast etwas schwaecher als die direkte Residuallast-Modellierung.</p>
  <p>T8 kalibriert die zuvor zu engen Quantilbaender leakage-sicher ueber ein vorgelagertes CQR-Kalibrierfenster. MAPIE/EnbPI wurde gepinnt und getestet, blieb aber unterkalibriert und ist aktuell nicht der empfohlene Pfad.</p>
  <p>T9 ist ein oeffentlicher Zielmarkt-Proxy aus OPSD/CoSSMic-Konstanz, kein echter Stadtwerke-Pilot. Der deutlich hoehere Prozentfehler ist deshalb realistisch und wichtig fuer die Produkteinordnung. Die erste Wettervariante half nur leicht; T10 kombiniert lokale Forecast-Wetterfeatures mit einem kleinlastspezifischen Lag-Feature-Set und erreicht auf dem Proxy einen signifikant positiven zweistelligen Skill gegen Saisonal-Naiv.</p>
</body>
</html>
"""
