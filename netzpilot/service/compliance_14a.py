# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""§14a-Compliance-Auswertungen aus dem Audit-Ledger (W12 + W13).

Zwei Berichte, beide ausschließlich aus dem hash-verketteten Eingriffs-Ledger
(`audit_ledger.py`) abgeleitet — keine neue Datenquelle, keine neue Prognose-Mathematik:

  W12 monthly_meldebogen  — Monats-Roll-up der §14a-Steuerungseingriffe in die Felder, die ein
                            Verteilnetzbetreiber monatlich veröffentlichen muss (Steuerungsart,
                            Anzahl Maßnahmen, Anzahl betroffener steuVE, Ø reduzierte Leistung,
                            Gesamtdauer). Erzeugt den Melde-INHALT; der Upload zu VNBdigital bleibt
                            extern und manuell.

  W13 fairness_report     — Diskriminierungsfreiheits-Prüfung: misst, ob die Drosselungen über die
                            steuerbaren Verbrauchseinrichtungen gleichmäßig verteilt waren.
                            EHRLICH bedarfsnormalisiert (Kappung relativ zum Bedarf), wenn der
                            Bedarf je Gerät im Ledger steht (redispatch.device_demands_kw → W13-Pfad);
                            sonst nur Gleichheit der gewährten Grenzen, klar als eingeschränkt gelabelt.

Ehrliche Grenzen (durchgängig):
  - Nicht BNetzA-zertifiziert; die Hash-Kette beweist nur Unveraendertheit seit Aufzeichnung.
  - Erfasst NUR Eingriffe, die durch NetzPilot liefen. Fremde Steuer-Quellsysteme muss das
    Stadtwerk selbst ergänzen.
  - Die finale Zuordnung zu den amtlichen Meldebogen-Kategorien bleibt Verantwortung des
    Netzbetreibers; dieses Modul liefert den belegten Inhalt, nicht die Rechtsverbindlichkeit.

Reine stdlib (wie audit_ledger.py) — lauffähig im dep-freien Test-Shim.
"""
from __future__ import annotations

import html
from typing import Iterable

from netzpilot.service.audit_ledger import (
    HONEST_LABEL,
    load_entries,
    verify_chain,
    _utc_now,
)

# Lesbarer Klartext je Eingriffstyp. KEINE Behauptung der exakten amtlichen Paragraphen-Zuordnung —
# die trifft der Netzbetreiber. Nur eine verständliche Beschriftung der internen decision_type-Werte.
STEUERUNGSART_LABEL = {
    "redispatch": "Netzorientierte Drosselung steuerbarer Verbrauchseinrichtungen",
    "dispatch": "Netzdienliche Einsatzplanung steuerbarer Verbrauchseinrichtungen",
    "tariff": "Netzentgelt-Zeitfenster (Modul 3)",
}

NOT_CERTIFIED = ("Nicht BNetzA-zertifiziert; Hash-Kette beweist nur Unveraendertheit seit "
                 "Aufzeichnung. Erfasst nur Eingriffe, die durch NetzPilot liefen.")


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _gini(values: Iterable[float]) -> float:
    """Gini-Koeffizient einer nicht-negativen Verteilung (0 = perfekt gleich ... 1 = maximal ungleich).

    Bei leerer Reihe oder Gesamtsumme 0 (niemand gedrosselt) → 0.0: triviale Gleichbehandlung.
    Negative Werte werden auf 0 geklemmt (Kappungen sind per Definition >= 0).
    """
    xs = sorted(max(0.0, _as_float(v)) for v in values)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s <= 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    return round((2.0 * cum) / (n * s) - (n + 1.0) / n, 4)


def _verified_payloads(ledger_path: str, *, decision_type: str | None = None,
                       utility: str | None = None) -> tuple[list[dict], dict]:
    """Liest die Ledger-Eintraege und gibt (payloads, chain_status). Bei gebrochener Kette werden
    die Payloads trotzdem zurueckgegeben, aber chain_status['ok'] ist False → der Aufrufer weist es
    ROT aus, statt vertrauenswuerdige Zahlen vorzutaeuschen."""
    status = verify_chain(ledger_path)
    payloads = []
    for entry in load_entries(ledger_path):
        p = entry.get("payload")
        if not isinstance(p, dict):
            continue
        if decision_type is not None and p.get("decision_type") != decision_type:
            continue
        if utility is not None and str(p.get("asset_id")) != str(utility):
            continue
        payloads.append(p)
    return payloads, status


def monthly_meldebogen(ledger_path: str, year: int, month: int, *,
                       utility: str | None = None) -> dict:
    """W12: Monats-Roll-up je Steuerungsart in die VNBdigital-Pflichtfelder.

    Filtert die Eingriffe nach Kalendermonat (ts_utc-Praefix YYYY-MM) und aggregiert je
    decision_type: Anzahl Maßnahmen, Anzahl betroffener steuVE (distinct Geraet mit echter
    Drosselung shed_kw>0; ohne Bedarf konservativ alle gelisteten), Ø reduzierte Leistung
    (mean magnitude_kw), Gesamtdauer (sum duration_min).
    """
    month_prefix = f"{int(year):04d}-{int(month):02d}"
    payloads, status = _verified_payloads(ledger_path, utility=utility)
    in_month = [p for p in payloads if str(p.get("ts_utc", ""))[:7] == month_prefix]

    by_art: dict[str, dict] = {}
    for p in in_month:
        art = str(p.get("decision_type") or "unbekannt")
        b = by_art.setdefault(art, {
            "decision_type": art,
            "steuerungsart": STEUERUNGSART_LABEL.get(art, art),
            "n_massnahmen": 0,
            "_magnitudes": [],
            "gesamtdauer_min": 0,
            "_devices_betroffen": set(),
            "_devices_gelistet": set(),
            "_bedarf_bekannt": True,
        })
        b["n_massnahmen"] += 1
        b["_magnitudes"].append(_as_float(p.get("magnitude_kw")))
        b["gesamtdauer_min"] += int(_as_float(p.get("duration_min")))
        for a in p.get("affected") or []:
            key = (str(p.get("asset_id")), a.get("device_index", a.get("asset")))
            b["_devices_gelistet"].add(key)
            if "shed_kw" in a:
                if _as_float(a.get("shed_kw")) > 1e-9:
                    b["_devices_betroffen"].add(key)
            else:
                b["_bedarf_bekannt"] = False
                b["_devices_betroffen"].add(key)  # ohne Bedarf konservativ als betroffen zaehlen

    rows = []
    for art, b in sorted(by_art.items()):
        mags = b.pop("_magnitudes")
        rows.append({
            "decision_type": b["decision_type"],
            "steuerungsart": b["steuerungsart"],
            "n_massnahmen": b["n_massnahmen"],
            "n_steuve_betroffen": len(b["_devices_betroffen"]),
            "n_steuve_gelistet": len(b["_devices_gelistet"]),
            "durchschnittl_reduzierte_leistung_kw": round(sum(mags) / len(mags), 3) if mags else 0.0,
            "gesamtdauer_min": b["gesamtdauer_min"],
            "gesamtdauer_h": round(b["gesamtdauer_min"] / 60.0, 2),
            "bedarf_bekannt": b["_bedarf_bekannt"],
        })

    return {
        "report": "vnbdigital_meldebogen_14a",
        "year": int(year),
        "month": int(month),
        "month_prefix": month_prefix,
        "utility": utility,
        "n_eingriffe_im_monat": len(in_month),
        "by_steuerungsart": rows,
        "chain_ok": bool(status.get("ok")),
        "chain_head_hash": status.get("head_hash"),
        "label": HONEST_LABEL,
        "caveat": (NOT_CERTIFIED + " Der Upload zu VNBdigital bleibt extern und manuell; die "
                   "Zuordnung zu den amtlichen Meldebogen-Kategorien verantwortet der Netzbetreiber."),
    }


def fairness_report(ledger_path: str, *, utility: str | None = None) -> dict:
    """W13: Diskriminierungsfreiheits-Prüfung über alle redispatch-Drosselungen.

    Aggregiert je steuVE über alle Eingriffe die absolute Kappung (Σ shed_kw) und — wenn der Bedarf
    im Ledger steht — die bedarfsnormalisierte mittlere Kappungsquote shed/Bedarf. Liefert je Sicht
    einen Gini-Koeffizienten und die Spannweite. Bedarfsnormalisiert ist die eigentliche
    Diskriminierungs-Aussage; ohne Bedarf wird nur die Gleichheit der gewährten Grenzen gemessen und
    explizit als eingeschränkt gelabelt.
    """
    payloads, status = _verified_payloads(ledger_path, decision_type="redispatch", utility=utility)

    # je Geraet sammeln
    dev: dict[tuple, dict] = {}
    bedarf_bekannt = True
    for p in payloads:
        asset = str(p.get("asset_id"))
        for a in p.get("affected") or []:
            key = (asset, a.get("device_index"))
            d = dev.setdefault(key, {"asset_id": asset, "device_index": a.get("device_index"),
                                     "n_eingriffe": 0, "summe_shed_kw": 0.0,
                                     "summe_limit_kw": 0.0, "quoten": [], "demand_kw": None})
            d["n_eingriffe"] += 1
            d["summe_limit_kw"] += _as_float(a.get("limit_kw"))
            if "shed_kw" in a and "demand_kw" in a:
                shed = _as_float(a.get("shed_kw"))
                demand = _as_float(a.get("demand_kw"))
                d["summe_shed_kw"] += shed
                d["demand_kw"] = demand
                if demand > 1e-9:
                    d["quoten"].append(shed / demand)
            else:
                bedarf_bekannt = False

    devices = []
    for key, d in sorted(dev.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] if kv[0][1] is not None else -1)):
        mean_quote = round(sum(d["quoten"]) / len(d["quoten"]), 4) if d["quoten"] else None
        devices.append({
            "asset_id": d["asset_id"],
            "device_index": d["device_index"],
            "n_eingriffe": d["n_eingriffe"],
            "summe_kappung_kwh": round(d["summe_shed_kw"], 3),
            "demand_kw": d["demand_kw"],
            "mittlere_kappungsquote": mean_quote,
        })

    n_dev = len(devices)
    abs_shed = [d["summe_kappung_kwh"] for d in devices]
    quoten = [d["mittlere_kappungsquote"] for d in devices if d["mittlere_kappungsquote"] is not None]

    primary = None
    if bedarf_bekannt and quoten:
        gini_q = _gini(quoten)
        primary = {
            "metric": "bedarfsnormalisierte mittlere Kappungsquote (shed/Bedarf)",
            "gini": gini_q,
            "min": round(min(quoten), 4),
            "max": round(max(quoten), 4),
            "spannweite": round(max(quoten) - min(quoten), 4),
            "interpretation": ("Gini 0 = alle steuVE relativ zu ihrem Bedarf gleich stark gedrosselt; "
                               "höhere Werte = ungleichere Lastverteilung der Drosselung."),
        }

    secondary = {
        "metric": "absolute Kappung je steuVE (Σ shed_kw)",
        "gini": _gini(abs_shed),
        "min": round(min(abs_shed), 3) if abs_shed else 0.0,
        "max": round(max(abs_shed), 3) if abs_shed else 0.0,
    }

    note = ("Das faire Water-Filling (`control/optimize.py`) setzt für alle gedrosselten steuVE "
            "dieselbe absolute Grenze und lässt schwache Verbraucher unangetastet — §14a verlangt eine "
            "gleiche garantierte Mindestleistung, NICHT eine gleiche relative Kappung. Dieser Bericht "
            "MISST die Verteilung und legt sie offen; die Bewertung als (nicht) diskriminierend bleibt "
            "Sache des Netzbetreibers / der Aufsicht.")
    if not bedarf_bekannt:
        note = ("ACHTUNG: Im Ledger fehlt der Bedarf je Gerät → nur die Gleichheit der GEWÄHRTEN "
                "Grenzen ist messbar, nicht die bedarfsnormalisierte Kappung. Eingeschränkte "
                "Aussagekraft. " + note)

    return {
        "report": "diskriminierungsfreiheit_14a",
        "utility": utility,
        "n_steuve": n_dev,
        "n_eingriffe": len(payloads),
        "bedarfsnormalisiert": bool(bedarf_bekannt and quoten),
        "primary_metric": primary,
        "secondary_metric": secondary,
        "devices": devices,
        "chain_ok": bool(status.get("ok")),
        "chain_head_hash": status.get("head_hash"),
        "label": HONEST_LABEL,
        "note": note,
        "caveat": NOT_CERTIFIED,
    }


# --------------------------------------------------------------------------- HTML-Renderer (Demo/Druck)

def _esc(v) -> str:
    return html.escape("" if v is None else str(v))


def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"><title>{_esc(title)}</title>
<style>
  @page {{ size:A4; margin:14mm; }}
  body {{ font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif; color:#1a2230; margin:0; }}
  .page {{ max-width:900px; margin:0 auto; padding:18px; }}
  h1 {{ font-size:20px; margin:0 0 2px; }} .sub {{ color:#5d6b7a; font-size:12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; margin:12px 0; }}
  th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid #edf1f5; }}
  th {{ color:#5d6b7a; text-transform:uppercase; font-size:10px; }} .num {{ text-align:right; }}
  .ok {{ color:#16735b; font-weight:700; }} .warn {{ color:#8a5a00; font-weight:700; }}
  .hash {{ font-family:ui-monospace,Consolas,monospace; font-size:10px; word-break:break-all; color:#5d6b7a; }}
  .box {{ border:1px solid #dce3ec; border-radius:8px; padding:10px 12px; margin:12px 0; }}
  button {{ background:#1f9c84; color:#fff; border:0; border-radius:8px; padding:8px 14px; font-weight:700; }}
  @media print {{ .noprint {{ display:none; }} }}
</style></head><body><div class="page">{body}
<p class="sub"><b>Ehrliche Grenze:</b> {_esc(NOT_CERTIFIED)}</p>
</div></body></html>"""


def render_meldebogen_html(ledger_path: str, year: int, month: int, *,
                           utility: str | None = None) -> str:
    m = monthly_meldebogen(ledger_path, year, month, utility=utility)
    chain = (f'<span class="ok">intakt</span>' if m["chain_ok"]
             else '<span class="warn">GEBROCHEN — Zahlen nicht vertrauenswürdig</span>')
    rows = "".join(
        "<tr>"
        f"<td>{_esc(r['steuerungsart'])}</td>"
        f"<td class='num'>{_esc(r['n_massnahmen'])}</td>"
        f"<td class='num'>{_esc(r['n_steuve_betroffen'])}</td>"
        f"<td class='num'>{_esc(r['durchschnittl_reduzierte_leistung_kw'])}</td>"
        f"<td class='num'>{_esc(r['gesamtdauer_h'])}</td>"
        "</tr>"
        for r in m["by_steuerungsart"]
    ) or "<tr><td colspan='5' class='sub'>Keine Eingriffe in diesem Monat.</td></tr>"
    body = f"""<h1>§14a-Meldebogen (Monatsbericht)</h1>
<div class="sub">Zeitraum {_esc(m['month_prefix'])} · {_esc(m.get('utility') or 'alle Mandanten')} · erzeugt {_esc(_utc_now())}</div>
<div class="box"><b>Hash-Kette:</b> {chain} &nbsp; · &nbsp; Eingriffe im Monat: {_esc(m['n_eingriffe_im_monat'])}
<div class="hash">{_esc(m['chain_head_hash'])}</div></div>
<table><thead><tr><th>Steuerungsart</th><th class="num">Maßnahmen</th><th class="num">betroffene steuVE</th>
<th class="num">Ø red. Leistung kW</th><th class="num">Gesamtdauer h</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="sub">{_esc(m['caveat'])}</p>
<div class="noprint"><button onclick="window.print()">Als PDF speichern / drucken</button></div>"""
    return _page("§14a-Meldebogen", body)


def render_fairness_html(ledger_path: str, *, utility: str | None = None) -> str:
    f = fairness_report(ledger_path, utility=utility)
    chain = ('<span class="ok">intakt</span>' if f["chain_ok"]
             else '<span class="warn">GEBROCHEN — Zahlen nicht vertrauenswürdig</span>')
    pm = f["primary_metric"]
    head = (f"<div class='box'><b>Diskriminierungsfreiheits-Maß (bedarfsnormalisiert):</b> "
            f"Gini {pm['gini']} · Spannweite {pm['spannweite']}<br><span class='sub'>{_esc(pm['interpretation'])}</span></div>"
            if pm else
            f"<div class='box warn'>Bedarf je Gerät fehlt im Ledger — nur eingeschränkte Sicht "
            f"(absolute Kappung), Gini {f['secondary_metric']['gini']}.</div>")
    rows = "".join(
        "<tr>"
        f"<td class='num'>{_esc(d['device_index'])}</td>"
        f"<td class='num'>{_esc(d['n_eingriffe'])}</td>"
        f"<td class='num'>{_esc(d['demand_kw'])}</td>"
        f"<td class='num'>{_esc(d['summe_kappung_kwh'])}</td>"
        f"<td class='num'>{_esc(d['mittlere_kappungsquote'])}</td>"
        "</tr>"
        for d in f["devices"]
    ) or "<tr><td colspan='5' class='sub'>Keine Drosselungen aufgezeichnet.</td></tr>"
    body = f"""<h1>§14a-Diskriminierungsfreiheits-Auditbericht</h1>
<div class="sub">{_esc(f.get('utility') or 'alle Mandanten')} · {_esc(f['n_steuve'])} steuVE · {_esc(f['n_eingriffe'])} Eingriffe · erzeugt {_esc(_utc_now())}</div>
<div class="box"><b>Hash-Kette:</b> {chain}<div class="hash">{_esc(f['chain_head_hash'])}</div></div>
{head}
<table><thead><tr><th class="num">steuVE-Index</th><th class="num">Eingriffe</th><th class="num">Bedarf kW</th>
<th class="num">Σ Kappung kWh</th><th class="num">Ø Kappungsquote</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="sub">{_esc(f['note'])}</p>
<div class="noprint"><button onclick="window.print()">Als PDF speichern / drucken</button></div>"""
    return _page("§14a-Diskriminierungsfreiheit", body)
