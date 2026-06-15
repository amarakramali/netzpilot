"""Service-side drift monitoring glue for forecast residuals.

Keeps the verified drift math in netzpilot.eval.drift untouched. This module only
persists reference/recent residual windows and builds the additive service payload.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

from netzpilot.eval.drift import coverage_report, drift_report


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))
    return safe or "default"


def _utility_dir(base_dir: str, utility: str) -> str:
    path = os.path.join(base_dir, _safe_name(utility))
    os.makedirs(path, exist_ok=True)
    return path


def _finite_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_list(values) -> list[float]:
    out = []
    for value in list(values):
        v = _finite_float(value)
        if v is not None:
            out.append(v)
    return out


def _errors(actual, forecast) -> list[float]:
    out = []
    for a, f in zip(list(actual), list(forecast)):
        af = _finite_float(a)
        ff = _finite_float(f)
        if af is not None and ff is not None:
            out.append(af - ff)
    return out


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _coverage_payload(R: dict, start: int, end: int) -> dict | None:
    if not all(k in R for k in ("p10", "p90", "actual")):
        return None
    try:
        return coverage_report(
            list(R["p10"])[start:end],
            list(R["p90"])[start:end],
            list(R["actual"])[start:end],
            nominal=0.8,
        )
    except ValueError:
        return None


def save_drift_reference(base_dir: str, utility: str, reference_errors, *,
                         metadata: dict | None = None,
                         calibration: dict | None = None) -> dict:
    """Persist reference residuals with a versioned file plus latest pointer."""
    errors = _finite_list(reference_errors)
    if not errors:
        raise ValueError("empty drift reference")
    d = _utility_dir(base_dir, utility)
    stamp = _utc_stamp()
    versioned_path = os.path.join(d, f"reference_{stamp}.json")
    latest_path = os.path.join(d, "latest_reference.json")
    payload = {
        "version": 1,
        "utility": utility,
        "created_utc": _utc_iso(),
        "source": "rolling-origin residuals (actual - forecast)",
        "n_reference": len(errors),
        "reference_errors": errors,
        "calibration": calibration,
        "metadata": metadata or {},
        "versioned_path": versioned_path,
    }
    _write_json(versioned_path, payload)
    _write_json(latest_path, payload)
    return {"path": versioned_path, "latest_path": latest_path, "record": payload}


def load_latest_reference(base_dir: str, utility: str) -> dict | None:
    return _read_json(os.path.join(_utility_dir(base_dir, utility), "latest_reference.json"))


def save_recent_window(base_dir: str, utility: str, recent_errors, *,
                       metadata: dict | None = None) -> dict:
    """Persist the latest realized-error window for auditability."""
    errors = _finite_list(recent_errors)
    d = _utility_dir(base_dir, utility)
    stamp = _utc_stamp()
    versioned_path = os.path.join(d, f"recent_{stamp}.json")
    latest_path = os.path.join(d, "latest_recent.json")
    payload = {
        "version": 1,
        "utility": utility,
        "created_utc": _utc_iso(),
        "source": "rolling recent residuals (actual - forecast)",
        "n_recent": len(errors),
        "recent_errors": errors,
        "metadata": metadata or {},
        "versioned_path": versioned_path,
    }
    _write_json(versioned_path, payload)
    _write_json(latest_path, payload)
    return {"path": versioned_path, "latest_path": latest_path, "record": payload}


def _insufficient(reason: str, *, reference: dict | None = None,
                  recent: dict | None = None) -> dict:
    return {
        "status": "insufficient_data",
        "needs_recalibration": False,
        "reasons": [reason],
        "reference": reference,
        "recent": recent,
        "coverage": None,
        "action": "warn_only_no_auto_retraining",
        "caveat": "Drift monitoring is a warning signal, not causal proof.",
    }


def build_drift_payload(R: dict, *, utility: str, base_dir: str = "data_cache/drift",
                        reference_days: int = 28, recent_days: int = 14,
                        min_recent_days: int = 7,
                        metadata: dict | None = None) -> dict:
    """Build additive out['drift'] from rolling-origin residuals.

    If no reference exists yet, the early part of the supplied backtest becomes the
    reference distribution and is persisted versioned. The latest block becomes the
    recent live-error window. Existing references are reused and not overwritten.
    """
    actual = list(R.get("actual", []))
    model = list(R.get("model", []))
    n_total = min(len(actual), len(model))
    if n_total <= 0:
        return _insufficient("no backtest residuals available")

    recent_len = min(int(recent_days) * 24, n_total)
    min_recent = int(min_recent_days) * 24
    recent_start = n_total - recent_len
    recent_errors = _errors(actual[recent_start:n_total], model[recent_start:n_total])
    recent_meta = {
        "requested_recent_days": int(recent_days),
        "min_recent_days": int(min_recent_days),
        "n_recent_periods": len(recent_errors),
    }
    if metadata:
        recent_meta.update(metadata)
    recent_info = save_recent_window(base_dir, utility, recent_errors, metadata=recent_meta)
    recent_ref = {
        "path": recent_info["path"],
        "n_errors": len(recent_errors),
        "window_days": round(len(recent_errors) / 24.0, 3),
    }
    if len(recent_errors) < min_recent:
        return _insufficient(
            f"recent window too small ({len(recent_errors)} periods < {min_recent})",
            recent=recent_ref,
        )

    latest_reference = load_latest_reference(base_dir, utility)
    created_reference = False
    if latest_reference:
        reference_errors = _finite_list(latest_reference.get("reference_errors", []))
        reference_path = latest_reference.get("versioned_path")
        reference_created_utc = latest_reference.get("created_utc")
        reference_calibration = latest_reference.get("calibration")
    else:
        reference_len = min(int(reference_days) * 24, max(0, recent_start))
        reference_errors = _errors(actual[:reference_len], model[:reference_len])
        if len(reference_errors) < 10:
            return _insufficient(
                f"reference window too small ({len(reference_errors)} periods)",
                recent=recent_ref,
            )
        reference_calibration = _coverage_payload(R, 0, reference_len)
        reference_meta = {
            "requested_reference_days": int(reference_days),
            "n_reference_periods": len(reference_errors),
            "reference_window": "early backtest period before recent window",
        }
        if metadata:
            reference_meta.update(metadata)
        saved = save_drift_reference(
            base_dir,
            utility,
            reference_errors,
            metadata=reference_meta,
            calibration=reference_calibration,
        )
        latest_reference = saved["record"]
        reference_path = saved["path"]
        reference_created_utc = latest_reference.get("created_utc")
        created_reference = True

    reference_ref = {
        "path": reference_path,
        "created_utc": reference_created_utc,
        "n_errors": len(reference_errors),
        "created_this_run": created_reference,
        "calibration": reference_calibration,
    }
    if len(reference_errors) < 10:
        return _insufficient(
            f"reference window too small ({len(reference_errors)} periods)",
            reference=reference_ref,
            recent=recent_ref,
        )

    try:
        report = drift_report(reference_errors, recent_errors)
    except ValueError as e:
        return _insufficient(str(e), reference=reference_ref, recent=recent_ref)

    coverage = _coverage_payload(R, recent_start, n_total)
    combined_status = report["status"]
    reasons = list(report["reasons"])
    if coverage and coverage.get("status") == "drift":
        combined_status = "drift"
        reasons.append(
            f"DRIFT coverage={coverage['coverage']:.3f} "
            f"(nominal {coverage['nominal']:.1f}, gap {coverage['coverage_gap']:.3f})"
        )

    payload = dict(report)
    payload["distribution_status"] = report["status"]
    payload["status"] = combined_status
    payload["reasons"] = reasons
    payload["needs_recalibration"] = combined_status in {"watch", "drift"}
    payload["reference"] = reference_ref
    payload["recent"] = recent_ref
    payload["coverage"] = coverage
    payload["method"] = (
        "drift_report(reference residuals, recent residuals) plus optional "
        "P10/P90 coverage_report on the recent window"
    )
    payload["action"] = "warn_only_no_auto_retraining"
    payload["caveat"] = "Drift status means check/recalibrate; it is not causal proof."
    return payload
