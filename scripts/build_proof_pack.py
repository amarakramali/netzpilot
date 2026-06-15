#!/usr/bin/env python3
"""Proof-Pack-Generator — EIN sauberes Beweis-Dokument für Stadtwerke.

Gießt zusammen, was sonst über mehrere Dateien verstreut ist:
  - die Benchmark-Tabelle über alle echten DSO-Lastgänge (Signifikanz, CI95),
  - die Modellkarte (Methodik + ehrliche Grenzen),
  - optional ein haus-spezifisches Pilot-Ergebnis (pilot_metrics.json),
in EIN druckoptimiertes A4-HTML. Der Browser macht daraus per "Drucken → Als PDF speichern" ein
echtes PDF — bewusst dependency-frei (kein reportlab/weasyprint), passend zur Offline-Philosophie.

Aufruf:
  python scripts/build_proof_pack.py                              # nur Benchmark + Modellkarte
  python scripts/build_proof_pack.py --pilot data_cache/pilot/t17_hilden_netzumsatz/pilot_metrics.json
  python scripts/build_proof_pack.py --out data_cache/benchmark/NetzPilot_Beweis.html
"""
from __future__ import annotations
import argparse
import html
import json
import os
from datetime import datetime, timezone


def _load(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _corpus_lookup(corpus_path="data_cache/real/corpus_index.json"):
    """key -> {country, network_kind} aus dem Korpus-Register (für Länder-/Ebenen-Label im Report)."""
    c = _load(corpus_path)
    out = {}
    if c:
        for e in c.get("entries", []):
            out[e.get("key")] = {"country": e.get("country"), "network_kind": e.get("network_kind")}
    return out


def _country_of(row, lookup):
    """Land robust bestimmen: erst Register, sonst aus key-Präfix (enedis_/eco2mix_ = FR)."""
    info = lookup.get(row.get("key"), {})
    if info.get("country"):
        return info["country"]
    key = (row.get("key") or "").lower()
    if key.startswith(("enedis_", "eco2mix_")):
        return "FR"
    return "DE"


def _ci(b):
    if not b:
        return "n/a"
    lo, hi = b["skill_ci95_%"]
    star = "✓" if b["signifikant_5pct"] else "○"
    return f"{b['skill_point_%']:+.1f}% [{lo:+.1f}, {hi:+.1f}] {star}"


def _bench_rows(results, lookup):
    out = []
    for r in results:
        if "error" in r:
            continue
        mape = "signiert" if (r.get("signed") or r.get("mape_meaningless")) else f"{r['MAPE_%']:.1f}%"
        snv = _ci((r.get("bootstrap") or {}).get("snaive"))
        per = _ci((r.get("bootstrap") or {}).get("persist"))
        country = _country_of(r, lookup)
        out.append(
            f"<tr><td>{html.escape(r['name'])}</td><td class='num'>{country}</td>"
            f"<td class='num'>{r['mean_load_MW']:.1f}</td>"
            f"<td class='num'>{mape}</td><td class='num'>{r['MASE']}</td>"
            f"<td>{snv}</td><td>{per}</td><td class='num'>{r['coverage_P10_P90_%']:.0f}%</td></tr>")
    return "\n".join(out)


def _pilot_block(pilot):
    if not pilot:
        return ""
    name = html.escape(str(pilot.get("dataset", "Pilot")))
    rows = [
        ("Ø Last", f"{pilot.get('mean_load_MW','–')} MW"),
        ("MAE", f"{pilot.get('MAE_MW','–')} MW"),
        ("MAPE", f"{pilot.get('MAPE_%','–')} %"),
        ("Skill vs. Saisonal-Naiv", f"{pilot.get('skill_vs_snaive_%','–')} %"),
        ("Skill vs. Persistenz", f"{pilot.get('skill_vs_persistenz_%','–')} %"),
        ("Coverage 80 / 90 %", f"{pilot.get('coverage80_%','–')} / {pilot.get('coverage90_%','–')} %"),
        ("Testtage / Historie", f"{pilot.get('n_test_days','–')} / {pilot.get('n_days_used','–')} Tage"),
    ]
    body = "".join(f"<tr><td>{k}</td><td class='num'>{v}</td></tr>" for k, v in rows)
    return f"""
  <h2>Haus-spezifisches Pilot-Ergebnis: {name}</h2>
  <p class="sub">Leakage-sicherer Backtest auf dem gelieferten/öffentlichen Lastgang dieses Hauses.</p>
  <table class="kv"><tbody>{body}</tbody></table>"""


def build_html(bench, card_md, pilot):
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lookup = _corpus_lookup()
    n_ok = bench.get("n_ok", 0) if bench else 0
    n_series = bench.get("n_unique_series", n_ok) if bench else 0
    n_indep = bench.get("n_independent_networks", n_ok) if bench else 0
    n_sig = bench.get("n_signifikant_vs_snaive_5pct", 0) if bench else 0
    # Echte Betreiber/Länder aus dem Register zählen (ehrliche Headline-Zahl).
    rows_ok = [r for r in bench["results"] if "error" not in r] if bench else []
    countries = sorted({_country_of(r, lookup) for r in rows_ok})
    countries_str = ", ".join(countries)
    de_ops = {(r.get("name") or "").split(" - ")[0] for r in rows_ok if _country_of(r, lookup) == "DE"}
    n_de_ops = len(de_ops)
    n_fr_regions = sum(1 for r in rows_ok if _country_of(r, lookup) == "FR"
                       and (r.get("key") or "").startswith("enedis_"))
    bench_rows = _bench_rows(bench["results"], lookup) if bench else ""
    card_html = ""
    if card_md:
        # Minimaler MD->HTML: ## -> h3, **..** -> b, - -> li. Bewusst simpel (kein MD-Paket).
        lines = []
        in_ul = False
        for ln in card_md.splitlines():
            s = ln.rstrip()
            if s.startswith("# "):
                continue
            if s.startswith("## "):
                if in_ul:
                    lines.append("</ul>"); in_ul = False
                lines.append(f"<h3>{html.escape(s[3:])}</h3>")
            elif s.startswith("- "):
                if not in_ul:
                    lines.append("<ul>"); in_ul = True
                item = html.escape(s[2:]).replace("**", "")
                lines.append(f"<li>{item}</li>")
            elif s.strip() == "":
                if in_ul:
                    lines.append("</ul>"); in_ul = False
            else:
                if in_ul:
                    lines.append("</ul>"); in_ul = False
                lines.append(f"<p class='sub'>{html.escape(s).replace('**','')}</p>")
        if in_ul:
            lines.append("</ul>")
        card_html = "\n".join(lines)

    return f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"><title>NetzPilot — Beweis-Pack</title>
<style>
  @page {{ size: A4; margin: 16mm; }}
  body {{ font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif; color:#1a2230; margin:0; line-height:1.5; }}
  .page {{ max-width:820px; margin:0 auto; padding:20px; }}
  .top {{ display:flex; align-items:baseline; gap:12px; border-bottom:2px solid #1f9c84; padding-bottom:10px; }}
  .logo {{ font-size:22px; font-weight:800; }} .logo span {{ color:#1f9c84; }}
  .tag {{ color:#5a6b7d; font-size:13px; margin-left:auto; }}
  h1 {{ font-size:19px; margin:16px 0 2px; }}
  h2 {{ font-size:14px; margin:22px 0 8px; color:#33485e; text-transform:uppercase; letter-spacing:.05em; }}
  h3 {{ font-size:13.5px; margin:14px 0 4px; color:#33485e; }}
  .sub {{ color:#5a6b7d; font-size:12.5px; margin:4px 0; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:6px; }}
  td,th {{ text-align:left; padding:5px 8px; border-bottom:1px solid #eef2f6; }}
  th {{ color:#5a6b7d; font-size:10.5px; text-transform:uppercase; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  table.kv td:first-child {{ color:#5a6b7d; }}
  .headline {{ background:#eaf7f3; border:1px solid #bfe6dc; border-radius:10px; padding:14px 16px; margin:14px 0; }}
  .headline b {{ color:#1f9c84; font-size:16px; }}
  ul {{ margin:4px 0 4px 0; padding-left:18px; }} li {{ font-size:12px; color:#41526a; margin:2px 0; }}
  .noprint {{ margin:14px 0; }} button {{ background:#1f9c84; color:#fff; border:0; border-radius:8px; padding:9px 16px; font-weight:700; cursor:pointer; }}
  @media print {{ .noprint {{ display:none; }} .page {{ max-width:none; }} }}
  footer {{ margin-top:22px; color:#9aa7b5; font-size:10px; border-top:1px solid #eef2f6; padding-top:8px; }}
</style></head>
<body><div class="page">
  <div class="top"><div class="logo">Netz<span>Pilot</span></div>
    <div class="tag">Beweis-Pack — Methodik &amp; Ergebnisse</div></div>
  <h1>Day-ahead-Lastprognose für Stadtwerke — was belegt ist</h1>
  <div class="noprint"><button onclick="window.print()">Als PDF speichern / drucken</button></div>

  <div class="headline">Getestet auf echten Verteilnetzdaten von <b>{n_de_ops} deutschen
  Netzbetreibern + {n_fr_regions} französischen Regionen</b> (Enedis). Leakage-sicher schlägt NetzPilot
  die Saisonal-Naiv-Baseline (branchenübliche Faustregel) in <b>{n_sig} von {n_series}</b> Lastreihen
  <b>statistisch signifikant</b> (95 %-Konfidenz).</div>

  <p class="sub"><b>Drei ehrliche Zahlen, sauber getrennt:</b> {n_series} eindeutige Lastreihen ·
  {n_indep} statistisch unabhängige Netz-/Regionalcluster (Korrelations-Dedup r≥0,98) ·
  {n_de_ops} deutsche Verteilnetzbetreiber + Frankreich (Enedis). Länder: {countries_str}.</p>

  <h2>Benchmark über echte Verteilnetz-Lastgänge (DE + FR)</h2>
  <table>
    <thead><tr><th>Datensatz</th><th class="num">Land</th><th class="num">Ø Last</th><th class="num">MAPE</th><th class="num">MASE</th>
      <th>Skill vs S-Naiv (CI95)</th><th>Skill vs Persistenz (CI95)</th><th class="num">Cov 80%</th></tr></thead>
    <tbody>{bench_rows}</tbody>
  </table>
  <p class="sub">Skill = Fehlerreduktion vs. Baseline. [CI95] = 95 %-Konfidenzintervall (paired Block-Bootstrap,
  Block = Tag). ✓ = unteres Ende &gt; 0 (signifikant); ○ = nicht signifikant. „signiert" = Differenzbilanz/
  Null-nahe Reihe (Mittel ≈ 0), dort führt Skill/MAE statt MAPE.</p>
  <p class="sub"><b>Ehrliche Einordnung:</b> Gegen die <i>Saisonal-Naiv-Baseline</i> (was ein Stadtwerk
  heute nutzt) gewinnt NetzPilot durchgängig. Bei der sehr glatten halbstündlichen FR-Regionalsumme ist
  die <i>Persistenz</i> (letzter Wert) eine harte Messlatte und wird dort nicht geschlagen — der Mehrwert
  liegt in der mehrtägigen Prognose + kalibrierten Bändern, nicht im 30-min-Nachlauf.</p>
{_pilot_block(pilot)}

  <h2>Methodik &amp; ehrliche Grenzen (Modellkarte)</h2>
  {card_html}

  <footer>Erzeugt {generated} · Reproduzierbar via <code>scripts/benchmark_suite.py</code> +
    <code>scripts/build_proof_pack.py</code>. Alle Zahlen aus leakage-sicheren Rolling-origin-Backtests
    mit verpflichtenden Baselines (28 Testtage, 10.000 Bootstrap-Resamples). Nationale Last (DE/NL/AT/CH/FR)
    wird separat in <code>data_cache/intl/</code> geführt und NICHT als Verteilnetz mitgezählt.
    NetzPilot — schlanke, ehrliche Prognose für kleine Stadtwerke.</footer>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data_cache/benchmark/benchmark_results.json")
    ap.add_argument("--card", default="data_cache/benchmark/MODEL_CARD.md")
    ap.add_argument("--pilot", default=None, help="optional: pilot_metrics.json eines Hauses")
    ap.add_argument("--out", default="data_cache/benchmark/NetzPilot_Beweis.html")
    a = ap.parse_args()

    bench = _load(a.bench)
    card_md = None
    if a.card and os.path.exists(a.card):
        with open(a.card, encoding="utf-8") as f:
            card_md = f.read()
    pilot = _load(a.pilot)
    if not bench:
        raise SystemExit(f"Benchmark-JSON fehlt ({a.bench}). Erst `python scripts/benchmark_suite.py` laufen lassen.")

    out_html = build_html(bench, card_md, pilot)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(out_html)
    print(f"-> {a.out}  ({len(out_html)} Zeichen)")
    print(f"   {bench.get('n_signifikant_vs_snaive_5pct',0)}/{bench.get('n_ok',0)} signifikant vs. Saisonal-Naiv.")
    print("   Im Browser öffnen -> Drucken -> Als PDF speichern.")


if __name__ == "__main__":
    main()
