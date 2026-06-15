# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Tamper-evident Paragraph-14a audit ledger.

This module documents interventions in an append-only JSONL hash chain. It is a
manipulationssicherer Audit-Trail (Hash-Kette), not a legal certification. The
chain proves unchanged records since logging; legal conformity remains a legal
and operational assessment.
"""
from __future__ import annotations

import html
import hmac
import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from typing import Iterable


GENESIS_HASH = "0" * 64
LEDGER_SCHEMA_VERSION = "netzpilot-audit-ledger-v1"
DEFAULT_RULE_VERSION = "netzpilot-paragraph14a-v1"
HONEST_LABEL = "manipulationssicherer Audit-Trail (Hash-Kette)"


def canonical_json(payload: dict) -> str:
    """Return deterministic JSON for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def entry_hash(prev_hash: str, payload: dict) -> str:
    """Hash exactly prev_hash + canonical_json(payload)."""
    return sha256((str(prev_hash) + canonical_json(payload)).encode("utf-8")).hexdigest()


def sign_head_hash(head_hash: str, key: str | None) -> str | None:
    """Return an HMAC-SHA256 signature for the head hash, or None without a key."""
    if key is None or key == "":
        return None
    return hmac.new(str(key).encode("utf-8"), str(head_hash).encode("utf-8"), sha256).hexdigest()


def _read_entries(ledger_path: str) -> list[dict]:
    if not os.path.exists(ledger_path):
        return []
    out = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def verify_chain(ledger_path: str) -> dict:
    """Verify the append-only hash chain."""
    prev = GENESIS_HASH
    n = 0
    try:
        entries = _read_entries(ledger_path)
    except (OSError, json.JSONDecodeError):
        return {"ok": False, "n": 0, "broken_at": 0, "head_hash": None}

    for i, entry in enumerate(entries):
        payload = entry.get("payload")
        got_prev = entry.get("prev_hash")
        got_hash = entry.get("entry_hash")
        if not isinstance(payload, dict) or got_prev != prev:
            return {"ok": False, "n": n, "broken_at": i, "head_hash": prev}
        expected = entry_hash(prev, payload)
        if got_hash != expected:
            return {"ok": False, "n": n, "broken_at": i, "head_hash": prev}
        prev = expected
        n += 1
    return {"ok": True, "n": n, "broken_at": None, "head_hash": prev}


def append_entry(ledger_path: str, payload: dict) -> dict:
    """Append one payload to the ledger and return the stored ledger entry."""
    status = verify_chain(ledger_path)
    if not status["ok"]:
        raise ValueError(f"Audit ledger chain is broken at index {status['broken_at']}.")
    os.makedirs(os.path.dirname(os.path.abspath(ledger_path)), exist_ok=True)
    prev = status["head_hash"] or GENESIS_HASH
    stored = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "prev_hash": prev,
        "payload": payload,
        "entry_hash": entry_hash(prev, payload),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(canonical_json(stored) + "\n")
    return stored


def load_entries(ledger_path: str) -> list[dict]:
    """Load stored ledger entries. Invalid JSON is intentionally surfaced."""
    return _read_entries(ledger_path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hour_ts(forecast_date: str | None, hour: int) -> str:
    date = str(forecast_date or "1970-01-01")
    return f"{date}T{int(hour):02d}:00:00Z"


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _affected_from_limits(limits_kw: Iterable[float], demands_kw: Iterable[float] | None = None) -> list[dict]:
    """Betroffene steuVE je Eingriff.

    Mit bekanntem Bedarf (demands_kw) wird zusaetzlich die tatsaechliche Kappung shed_i =
    max(0, demand_i - limit_i) je Geraet mitgeschrieben — die Datenbasis fuer die
    bedarfsnormalisierte Diskriminierungsfreiheits-Pruefung (W13). Ohne Bedarf bleibt die
    Struktur exakt wie zuvor (nur device_index + limit_kw), rueckwaertskompatibel.
    """
    limits = list(limits_kw or [])
    demands = list(demands_kw or [])
    out = []
    for i, value in enumerate(limits):
        limit = round(_as_float(value), 6)
        rec = {"device_index": i, "limit_kw": limit}
        if i < len(demands):
            demand = round(_as_float(demands[i]), 6)
            rec["demand_kw"] = demand
            rec["shed_kw"] = round(max(0.0, demand - limit), 6)
        out.append(rec)
    return out


def build_intervention_payloads(
    result: dict,
    *,
    rule_version: str = DEFAULT_RULE_VERSION,
    recorded_at_utc: str | None = None,
    asset_id: str | None = None,
) -> list[dict]:
    """Build canonical intervention payloads from existing service decision fields."""
    recorded = recorded_at_utc or _utc_now()
    utility = str(result.get("utility") or "unknown")
    asset = asset_id or utility
    forecast_date = result.get("forecast_date")
    rating_kw = None
    if result.get("asset_limit"):
        rating_kw = result["asset_limit"].get("rating_kw")
    if rating_kw is None and result.get("congestion"):
        rating_kw = _as_float(result["congestion"].get("threshold_mw")) * 1000.0

    entries: list[dict] = []
    redispatch = result.get("redispatch") or {}
    device_demands = redispatch.get("device_demands_kw")
    for row in redispatch.get("hourly") or []:
        if not row.get("intervention"):
            continue
        hour = int(row.get("hour", 0))
        entries.append({
            "ts_utc": _hour_ts(forecast_date, hour),
            "recorded_at_utc": recorded,
            "asset_id": asset,
            "decision_type": "redispatch",
            "reason": {
                "trigger": "forecast_kw > threshold_kw",
                "basis": redispatch.get("basis"),
                "forecast_basis": redispatch.get("forecast_basis"),
                "forecast_kw": round(_as_float(row.get("forecast_kw")), 6),
                "threshold_kw": round(_as_float(redispatch.get("threshold_kw") or rating_kw), 6),
                "overload_kw": round(_as_float(row.get("overload_kw")), 6),
            },
            "magnitude_kw": round(_as_float(row.get("shed_kw")), 6),
            "duration_min": 60,
            "affected": _affected_from_limits(row.get("limits_kw"), device_demands),
            "rule_version": rule_version,
            "evidence_label": HONEST_LABEL,
        })

    dispatch = result.get("dispatch_plan") or {}
    for row in dispatch.get("hourly") or []:
        steuve_kw = _as_float(row.get("steuve_kw"))
        if steuve_kw <= 1e-9:
            continue
        hour = int(row.get("hour", 0))
        entries.append({
            "ts_utc": _hour_ts(forecast_date, hour),
            "recorded_at_utc": recorded,
            "asset_id": asset,
            "decision_type": "dispatch",
            "reason": {
                "trigger": "steuVE energy placement under network cap and imbalance objective",
                "cap_source": dispatch.get("cap_source"),
                "cap_kw": round(_as_float(row.get("cap_kw")), 6),
                "total_point_kw": round(_as_float(row.get("total_point_kw")), 6),
                "nomination_kw": round(_as_float(row.get("nomination_kw")), 6),
            },
            "magnitude_kw": round(steuve_kw, 6),
            "duration_min": 60,
            "affected": [{
                "asset": "steuve_dispatch",
                "scheduled_kw": round(steuve_kw, 6),
                "cap_kw": round(_as_float(row.get("cap_kw")), 6),
            }],
            "rule_version": rule_version,
            "evidence_label": HONEST_LABEL,
        })

    tariff = result.get("tariff_schedule") or {}
    schedule = list(tariff.get("schedule_kwh") or [])
    fees = list(tariff.get("fee_eur_per_kwh") or [])
    caps = tariff.get("cap_kw")
    dt_h = _as_float(tariff.get("dt_h"), 1.0) or 1.0
    for hour, scheduled_kwh in enumerate(schedule):
        energy = _as_float(scheduled_kwh)
        if energy <= 1e-9:
            continue
        kw = energy / dt_h
        cap = caps[hour] if isinstance(caps, list) and hour < len(caps) else None
        fee = fees[hour] if hour < len(fees) else None
        entries.append({
            "ts_utc": _hour_ts(forecast_date, hour),
            "recorded_at_utc": recorded,
            "asset_id": asset,
            "decision_type": "tariff",
            "reason": {
                "trigger": "module_3_grid_fee_schedule",
                "cap_source": tariff.get("cap_source"),
                "fee_eur_per_kwh": None if fee is None else round(_as_float(fee), 6),
                "cap_kw": None if cap is None else round(_as_float(cap), 6),
            },
            "magnitude_kw": round(kw, 6),
            "duration_min": int(round(dt_h * 60)),
            "affected": [{
                "asset": "tariff_flex_load",
                "scheduled_kwh": round(energy, 6),
                "scheduled_kw": round(kw, 6),
            }],
            "rule_version": rule_version,
            "evidence_label": HONEST_LABEL,
        })

    return entries


def append_forecast_audit(
    ledger_path: str,
    result: dict,
    *,
    signing_key: str | None = None,
    rule_version: str = DEFAULT_RULE_VERSION,
) -> dict:
    """Append all intervention entries from a service result and return audit metadata."""
    payloads = build_intervention_payloads(result, rule_version=rule_version)
    appended = [append_entry(ledger_path, payload) for payload in payloads]
    status = verify_chain(ledger_path)
    head = status["head_hash"]
    return {
        "status": "available" if appended else "no_entries",
        "label": HONEST_LABEL,
        "ledger_path": ledger_path,
        "head_hash": head,
        "head_signature_hmac_sha256": sign_head_hash(head, signing_key),
        "n_entries": status["n"],
        "entries_appended": len(appended),
        "chain_ok": bool(status["ok"]),
        "broken_at": status["broken_at"],
        "caveat": (
            "Hash-Kette beweist Unveraendertheit seit Aufzeichnung. "
            "Nicht BNetzA-zertifiziert und keine juristische Rechtssicherheitsbehauptung."
        ),
    }


def _escape(value) -> str:
    return html.escape("" if value is None else str(value))


def render_audit_report_html(
    ledger_path: str,
    *,
    signing_key: str | None = None,
    title: str = "Paragraph-14a Audit-Nachweis",
) -> str:
    """Render an A4 print-ready HTML evidence report for browser PDF export."""
    entries = load_entries(ledger_path)
    status = verify_chain(ledger_path)
    head = status.get("head_hash")
    signature = sign_head_hash(head, signing_key) if head else None
    rows = []
    for i, entry in enumerate(entries):
        p = entry.get("payload", {})
        reason = p.get("reason") if isinstance(p.get("reason"), dict) else {}
        affected = p.get("affected") or []
        rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{_escape(p.get('ts_utc'))}</td>"
            f"<td>{_escape(p.get('asset_id'))}</td>"
            f"<td>{_escape(p.get('decision_type'))}</td>"
            f"<td class='num'>{_escape(p.get('magnitude_kw'))}</td>"
            f"<td class='num'>{_escape(p.get('duration_min'))}</td>"
            f"<td>{_escape(reason.get('trigger'))}</td>"
            f"<td>{_escape(len(affected))}</td>"
            f"<td class='hash'>{_escape(entry.get('entry_hash'))}</td>"
            "</tr>"
        )
    generated = _utc_now()
    return f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<title>{_escape(title)}</title>
<style>
  @page {{ size: A4 landscape; margin: 12mm; }}
  body {{ font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif; color:#1a2230; margin:0; }}
  .page {{ max-width:1120px; margin:0 auto; padding:18px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }} .sub {{ color:#5d6b7a; font-size:12px; }}
  .box {{ border:1px solid #dce3ec; border-radius:8px; padding:10px; margin:12px 0; }}
  .hash {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:10px; word-break:break-all; }}
  table {{ width:100%; border-collapse:collapse; font-size:11px; }}
  th,td {{ text-align:left; padding:5px 6px; border-bottom:1px solid #edf1f5; vertical-align:top; }}
  th {{ color:#5d6b7a; text-transform:uppercase; font-size:10px; }} .num {{ text-align:right; }}
  .warn {{ color:#8a5a00; }} .ok {{ color:#16735b; font-weight:700; }}
  button {{ background:#1f9c84; color:#fff; border:0; border-radius:8px; padding:8px 14px; font-weight:700; }}
  @media print {{ .noprint {{ display:none; }} .page {{ max-width:none; }} }}
</style></head><body><div class="page">
<h1>{_escape(title)}</h1>
<div class="sub">Erzeugt {generated}. Nachweisart: {HONEST_LABEL}.</div>
<div class="noprint" style="margin-top:10px"><button onclick="window.print()">Als PDF speichern / drucken</button></div>
<div class="box">
  <div><b>Chain status:</b> <span class="{'ok' if status.get('ok') else 'warn'}">{_escape(status.get('ok'))}</span></div>
  <div><b>Eintraege:</b> {_escape(status.get('n'))}</div>
  <div><b>Ketten-Kopf-Hash:</b> <span class="hash">{_escape(head)}</span></div>
  <div><b>HMAC-SHA256-Signatur:</b> <span class="hash">{_escape(signature or 'nicht gesetzt')}</span></div>
</div>
<p class="sub"><b>Ehrliche Grenze:</b> Diese Hash-Kette beweist Unveraendertheit seit Aufzeichnung.
Sie ist nicht BNetzA-zertifiziert und keine Behauptung juristischer Rechtssicherheit.</p>
<table><thead><tr><th>#</th><th>Zeit UTC</th><th>Asset</th><th>Typ</th><th class="num">kW</th>
<th class="num">Min</th><th>Grund</th><th>Betroffene</th><th>Entry Hash</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</div></body></html>"""


def write_audit_report_html(
    ledger_path: str,
    out_path: str,
    *,
    signing_key: str | None = None,
    title: str = "Paragraph-14a Audit-Nachweis",
) -> str:
    """Write the print-ready audit evidence HTML and return its path."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_audit_report_html(ledger_path, signing_key=signing_key, title=title))
    return out_path
