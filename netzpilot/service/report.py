"""Druckoptimierter Ein-Seiten-Bericht (HTML) aus einem gespeicherten Prognose-Lauf.

Bewusst dependency-frei: rendert reines, in sich geschlossenes HTML mit A4-Druck-CSS.
Der Anwender erzeugt das PDF ueber den Browser (Drucken -> Als PDF speichern). Das passt zur
Offline-Produktphilosophie (kein reportlab/weasyprint/Systemlibs, laeuft beim Stadtwerk lokal).

Eine echte Server-seitige PDF-Erzeugung (One-Click-Download) waere als Erweiterung moeglich,
ist hier aber bewusst nicht eingebaut, um die Abhaengigkeiten schlank zu halten.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone


def _eur(v) -> str:
    if v is None:
        return "–"
    return f"{int(round(v)):,}".replace(",", ".") + " €"


def _fmt(v, nd=1) -> str:
    if v is None:
        return "–"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _sparkline_svg(hours, key="p50", w=520, h=90, stroke="#1f9c84") -> str:
    """Kleine Inline-SVG-Linie der P50-Reihe — druckbar, ohne JS/CDN."""
    vals = [float(x[key]) for x in hours]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = len(vals)
    pad = 6
    def X(i):
        return pad + (w - 2 * pad) * i / (n - 1)
    def Y(v):
        return pad + (h - 2 * pad) * (1 - (v - lo) / rng)
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
    zero = ""
    if lo < 0 < hi:
        zy = Y(0)
        zero = f'<line x1="{pad}" y1="{zy:.1f}" x2="{w-pad}" y2="{zy:.1f}" stroke="#bbb" stroke-dasharray="3 3"/>'
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none">'
            f'{zero}<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="2"/></svg>')


def _forecast_rows(hours) -> str:
    out = []
    for x in hours:
        out.append(
            f"<tr><td>{int(x['hour']):02d}:00</td>"
            f"<td class='num'>{_fmt(x['p10'])}</td>"
            f"<td class='num'><b>{_fmt(x['p50'])}</b></td>"
            f"<td class='num'>{_fmt(x['p90'])}</td></tr>"
        )
    return "".join(out)


def _forecast_table(hours) -> str:
    """Eine vollständige Stundentabelle (eigener Kopf) für einen Stundenbereich."""
    return ("<table><thead><tr><th>Std</th><th class='num'>P10</th>"
            "<th class='num'>P50</th><th class='num'>P90</th></tr></thead>"
            f"<tbody>{_forecast_rows(hours)}</tbody></table>")


def _economics_block(rec) -> str:
    exp = rec.get("economics_expected")
    band = rec.get("economics")
    ub = rec.get("economics_upper_bound")
    if not exp and not band:
        return "<p class='muted'>Keine reBAP-/Spot-Reihe übergeben — keine €-Schätzung in diesem Lauf.</p>"
    rows = []
    if exp:
        rows.append(
            f"<tr><td><b>Erwartete Einsparung/Jahr</b><div class='sub'>signierter Mittel-Aufschlag "
            f"{_fmt(exp.get('signed_mean_spread_eur_mwh'), 2)} €/MWh — ehrlichste Erwartung</div></td>"
            f"<td class='num big'>{_eur(exp.get('expected_eur_per_year'))}</td></tr>"
        )
    if band:
        rows.append(
            f"<tr><td>Risiko-/Stressband (|reBAP − Spot|)<div class='sub'>Volatilität, beide Richtungen — "
            f"nicht der Erwartungswert</div></td>"
            f"<td class='num'>{_eur(band.get('eur_per_year_point_median'))} "
            f"<span class='sub'>({_eur(band.get('eur_per_year_p25'))} – {_eur(band.get('eur_per_year_p75'))})</span></td></tr>"
        )
    if ub:
        rows.append(
            f"<tr><td>Oberer Rand (|reBAP| absolut)<div class='sub'>überschätzt — gespart wird nur der Aufschlag</div></td>"
            f"<td class='num'>{_eur(ub.get('eur_per_year_point_median'))}</td></tr>"
        )
    return (
        "<table class='econ'><tbody>" + "".join(rows) + "</tbody></table>"
        "<p class='caveat'>reBAP-Nutzen = Downside-Schutz, kein garantierter linearer Ertrag. "
        "Die belastbarste Zahl bleibt die reale Bilanzkreis-Abrechnung des Stadtwerks.</p>"
    )


def _congestion_block(rec) -> str:
    c = rec.get("congestion")
    fp = rec.get("fahrplan")
    if not c:
        return "<p class='good'>Kein §14a-Engpass prognostiziert — kein netzorientierter Eingriff nötig.</p>"
    basis = c.get("basis", "load")
    basis_txt = "Residuallast" if basis == "residual" else "Last"
    wh = c.get("window_hours", [])
    win = f"{wh[0]:02d}:00–{wh[-1]+1:02d}:00" if wh else "–"
    out = [
        f"<p><b>Engpass prognostiziert</b> ({basis_txt}-Basis): Fenster <b>{win}</b>, "
        f"max P90 {_fmt(c.get('max_p90_mw'))} MW über Schwelle {_fmt(c.get('threshold_mw'))} MW.</p>"
    ]
    if fp:
        sps = fp.get("setpoints", [])
        rows = "".join(
            f"<tr><td>{html.escape(str(s.get('start_utc','')).replace('T',' '))} → "
            f"{html.escape(str(s.get('end_utc','')).replace('T',' '))}</td>"
            f"<td class='num'>{_fmt(s.get('p_limit_kw'))} kW</td></tr>"
            for s in sps
        )
        out.append(
            "<table class='fp'><thead><tr><th>Fenster (UTC)</th><th>P-Limit</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            "<p class='caveat'>§14a-Fahrplan-<b>Entwurf</b>: produktiv geht er an einen zertifizierten "
            "White-Label-aEMT, nicht direkt ans SMGW. 4,2-kW-Mindestleistung erzwungen.</p>"
        )
    return "".join(out)


def _track_record_block(rec) -> str:
    """Live-Track-Record-Sektion (nur wenn ein Forecast-Store aktiv war) — der Beweis, der mit
    jedem Betriebstag wächst: vorab ausgegebene, hash-verkettete Prognosen gegen das spätere Ist."""
    tr = rec.get("track_record")
    if not tr:
        return ""
    chain_ok = bool(tr.get("chain_ok"))
    chain = "Hash-Kette intakt" if chain_ok else "HASH-KETTE GEBROCHEN"
    chain_cls = "good" if chain_ok else "bad"
    agg = tr.get("aggregate") or {}
    days = (tr.get("last_30_days") or [])[-10:]
    rows = "".join(
        f"<tr><td>{html.escape(str(d.get('target_date', '')))}</td>"
        f"<td>{html.escape(str(d.get('issued_at_utc', '')).replace('T', ' ')[:16])}</td>"
        f"<td class='num'>{_fmt(d.get('mae'), 3)}</td>"
        f"<td class='num'>{_fmt(d.get('bias'), 3)}</td>"
        f"<td class='num'>{_fmt(d.get('coverage_p10_p90_pct'))}</td></tr>"
        for d in days
    )
    table = (
        "<table><thead><tr><th>Zieltag</th><th>ausgegeben (UTC)</th><th class='num'>MAE</th>"
        "<th class='num'>Bias</th><th class='num'>Coverage %</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    ) if days else "<p class='muted'>Noch keine realisierten Tage — Prognosen sind pending.</p>"
    return f"""
  <h2>Live-Track-Record (vorab ausgegebene Prognosen)</h2>
  <p class="{chain_cls}"><b>{chain}</b> · {int(tr.get('n_realized', 0))} realisierte Tage ·
    {int(tr.get('n_pending', 0))} pending · {int(tr.get('n_forecasts_stored', 0))} im Store</p>
  <div class="kpis">
    <div class="kpi"><div class="v">{_fmt(agg.get('mae_mean'), 3)}</div><div class="l">MAE realisierte Tage (MW)</div></div>
    <div class="kpi"><div class="v">{_fmt(agg.get('coverage_mean_pct'))} %</div><div class="l">Coverage P10–P90 (Soll 80)</div></div>
  </div>
  {table}
  <p class="caveat">Hash-verkettete, <b>vorab ausgegebene</b> Prognosen gegen das spätere Ist — Live-Nachweis,
    kein Backtest. Manipulationssicher = Hash-Kette, <b>keine juristische Zertifizierung</b>. Gezeigt werden
    immer alle realisierten Tage — kein Zeitraum-Cherry-Picking.</p>"""


def render_report_html(rec: dict) -> str:
    """Baue den vollständigen, in sich geschlossenen Druckbericht (HTML-String)."""
    utility = html.escape(str(rec.get("utility", "Stadtwerk")))
    date = html.escape(str(rec.get("forecast_date", "")))
    hours = rec.get("forecast", []) or []
    rf = rec.get("residual_forecast")
    peak = max((h["p50"] for h in hours), default=None)
    # Stundentabelle in zwei vollständige Tabellen (0–11 / 12–23), jede mit eigenem Kopf —
    # statt einer per CSS gebrochenen Tabelle (dort fehlte der Kopf über der rechten Hälfte).
    mid = (len(hours) + 1) // 2
    fc_tables = _forecast_table(hours[:mid]) + _forecast_table(hours[mid:]) if hours else ""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    resid_block = ""
    if rf:
        rhours = rf.get("forecast", []) or []
        rpeak = max((h["p50"] for h in rhours), default=None)
        resid_block = f"""
      <h2>Residuallast (Last − Erzeugung)</h2>
      <div class="kpis">
        <div class="kpi"><div class="v">{_fmt(rpeak)} MW</div><div class="l">Spitzen-P50 Residuallast</div></div>
        <div class="kpi"><div class="v">{_fmt(rf.get('recent_mean_residual_mw'))} MW</div><div class="l">⌀ Residuallast (Woche)</div></div>
        <div class="kpi"><div class="v">{int(rf.get('n_days_history', 0))}</div><div class="l">Tage Historie</div></div>
      </div>
      <div class="spark">{_sparkline_svg(rhours, stroke="#d08a1f")}</div>
      <p class="sub">{html.escape(str(rf.get('definition','')))}</p>"""

    return f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<title>NetzPilot-Bericht — {utility} — {date}</title>
<style>
  @page {{ size: A4; margin: 16mm; }}
  body {{ font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif; color:#1a2230; margin:0; line-height:1.45; }}
  .page {{ max-width: 800px; margin:0 auto; padding:18px; }}
  .top {{ display:flex; align-items:baseline; gap:12px; border-bottom:2px solid #1f9c84; padding-bottom:10px; }}
  .logo {{ font-size:22px; font-weight:800; }} .logo span {{ color:#1f9c84; }}
  .tag {{ color:#5a6b7d; font-size:13px; margin-left:auto; }}
  h1 {{ font-size:18px; margin:16px 0 2px; }} h2 {{ font-size:14px; margin:20px 0 8px; color:#33485e;
        text-transform:uppercase; letter-spacing:.05em; }}
  .meta {{ color:#5a6b7d; font-size:13px; margin-bottom:6px; }}
  .kpis {{ display:flex; gap:14px; flex-wrap:wrap; margin:6px 0 4px; }}
  .kpi {{ border:1px solid #dce3ec; border-radius:10px; padding:10px 14px; min-width:130px; }}
  .kpi .v {{ font-size:20px; font-weight:700; }} .kpi .l {{ color:#5a6b7d; font-size:11px; }}
  .spark {{ border:1px solid #eef2f6; border-radius:8px; padding:6px; margin:8px 0; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  td,th {{ text-align:left; padding:4px 8px; border-bottom:1px solid #eef2f6; }}
  th {{ color:#5a6b7d; font-size:11px; text-transform:uppercase; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.big {{ font-size:18px; font-weight:800; color:#1f9c84; }}
  table.econ td {{ padding:8px; }} .sub {{ color:#7a899a; font-size:11px; }}
  .caveat {{ color:#7a899a; font-size:11px; border-left:3px solid #e2e8f0; padding-left:10px; margin-top:8px; }}
  .good {{ color:#1f9c84; font-weight:600; }} .muted {{ color:#7a899a; }}
  .bad {{ color:#c0392b; font-weight:700; }}
  .fc-table {{ display:flex; gap:24px; align-items:flex-start; }}
  .fc-table > table {{ flex:1; table-layout:fixed; width:50%; }}
  .fc-table th:first-child, .fc-table td:first-child {{ width:34%; }}
  @media print {{ .fc-table {{ gap:18px; }} }}
  .noprint {{ margin:14px 0; }} button {{ background:#1f9c84; color:#fff; border:0; border-radius:8px;
        padding:9px 16px; font-weight:700; cursor:pointer; font-size:13px; }}
  @media print {{ .noprint {{ display:none; }} .page {{ max-width:none; }} }}
  footer {{ margin-top:22px; color:#9aa7b5; font-size:10.5px; border-top:1px solid #eef2f6; padding-top:8px; }}
</style></head>
<body><div class="page">
  <div class="top"><div class="logo">Netz<span>Pilot</span></div>
    <div class="tag">Day-ahead-Prognose &amp; §14a — automatischer Bericht</div></div>

  <h1>{utility}</h1>
  <div class="meta">Prognosetag: <b>{date}</b> · {int(rec.get('n_days_history',0))} Tage Historie (leakage-sicher)
    · Lastspalte: {html.escape(str(rec.get('load_column') or '—'))}
    {('· Ebene '+html.escape(str(rec.get('load_level')))) if rec.get('load_level') else ''}</div>

  <div class="noprint"><button onclick="window.print()">Als PDF speichern / drucken</button></div>

  <h2>Last-Prognose</h2>
  <div class="kpis">
    <div class="kpi"><div class="v">{_fmt(peak)} MW</div><div class="l">Spitzen-P50 (Last)</div></div>
    <div class="kpi"><div class="v">{_fmt(rec.get('recent_mean_load_mw'))} MW</div><div class="l">⌀ Last (Woche)</div></div>
  </div>
  <div class="spark">{_sparkline_svg(hours)}</div>
{resid_block}

  <h2>§14a-Netzsituation</h2>
  {_congestion_block(rec)}

  <h2>Wirtschaftlicher Hebel (Ausgleichsenergie)</h2>
  {_economics_block(rec)}
{_track_record_block(rec)}
  <h2>Stündliche Prognose (P10 / P50 / P90, MW)</h2>
  <div class="fc-table">{fc_tables}</div>

  <footer>Erzeugt {generated} · NetzPilot · Prognose = echte Engine (ShrunkCorrector + CQR), leakage-sicher.
    P10/P50/P90 sind kalibrierte Quantile. §14a-Fahrplan ist ein Entwurf für einen zertifizierten aEMT.</footer>
</div></body></html>"""


def render_audit_report_html(ledger_path: str, *, signing_key: str | None = None,
                             title: str = "Paragraph-14a Audit-Nachweis") -> str:
    """Render the audit evidence through the same browser-print-to-PDF report path."""
    from netzpilot.service.audit_ledger import render_audit_report_html as _render

    return _render(ledger_path, signing_key=signing_key, title=title)


def write_audit_report_html(ledger_path: str, out_path: str, *, signing_key: str | None = None,
                            title: str = "Paragraph-14a Audit-Nachweis") -> str:
    """Write the print-ready audit evidence HTML for PDF export via browser print."""
    from netzpilot.service.audit_ledger import write_audit_report_html as _write

    return _write(ledger_path, out_path, signing_key=signing_key, title=title)
