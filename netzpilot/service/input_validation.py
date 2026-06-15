"""Service-side input validation glue for measured load series.

Keeps the verified validation math in netzpilot.data.validate untouched. This module only
adapts a loaded hourly pandas Series to the service contract: complete hourly index,
compact report, and optional use of cleaned replacement values.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from netzpilot.data.validate import validate_load


def _complete_hourly_index(hourly: pd.Series) -> pd.DatetimeIndex:
    if hourly.empty:
        raise ValueError("Leere Stundenreihe.")
    idx = hourly.sort_index().index
    return pd.date_range(idx.min(), idx.max(), freq="1h", tz=idx.tz)


def _with_timestamps(items: list[dict], index: pd.DatetimeIndex, max_items: int) -> list[dict]:
    out = []
    for item in items[:max_items]:
        row = dict(item)
        i = row.get("index")
        if isinstance(i, int) and 0 <= i < len(index):
            row["timestamp"] = index[i].isoformat()
        out.append(row)
    return out


def summarize_validation(report: dict, index: pd.DatetimeIndex, *, cleaned_values_used: bool,
                         would_apply_cleaned_values: bool, max_items: int = 20) -> dict:
    """Return a compact service payload without the full cleaned value vector."""
    issue_counts = Counter(item.get("type") for item in report.get("issues", []))
    issues = report.get("issues", [])
    replacements = report.get("replacements", [])
    return {
        "enabled": True,
        "values_unit": "MW",
        "period_per_day": 24,
        "n": int(report["n"]),
        "n_missing": int(report["n_missing"]),
        "n_outlier": int(report["n_outlier"]),
        "n_frozen": int(report["n_frozen"]),
        "n_negative": int(report["n_negative"]),
        "n_out_of_range": int(report["n_out_of_range"]),
        "n_replaced": int(report["n_replaced"]),
        "n_unreplaceable": int(report["n_unreplaceable"]),
        "quality_score": float(report["quality_score"]),
        "issue_type_counts": {str(k): int(v) for k, v in issue_counts.items() if k},
        "n_issues_total": len(issues),
        "n_replacements_total": len(replacements),
        "issues_sample": _with_timestamps(issues, index, max_items),
        "replacements_sample": _with_timestamps(replacements, index, max_items),
        "issues_truncated": max(0, len(issues) - max_items),
        "replacements_truncated": max(0, len(replacements) - max_items),
        "would_apply_cleaned_values": bool(would_apply_cleaned_values),
        "cleaned_values_used": bool(cleaned_values_used),
        "applied_cleaned_values": bool(cleaned_values_used),
        "original_preserved": True,
        "note": (
            report.get("note", "")
            + " Original CSV bleibt unangetastet; eingefrorene Phasen werden nur gemeldet."
        ),
    }


def validate_hourly_series(hourly: pd.Series, *, allow_negative: bool = False,
                           max_plausible=None, apply_cleaned: bool = True) -> tuple[pd.Series, dict]:
    """Validate a service input series and optionally return the cleaned replacement series.

    robust_load_csv may already have dropped raw NaNs. Reindexing to a complete hourly grid makes
    missing hours visible to validate_load before the forecast engine sees the data.
    """
    full_index = _complete_hourly_index(hourly)
    full = hourly.sort_index().reindex(full_index)
    report = validate_load(
        full.tolist(),
        period_per_day=24,
        allow_negative=allow_negative,
        max_plausible=max_plausible,
    )
    cleaned = pd.Series(report["cleaned"], index=full_index, dtype="float64")
    can_apply = (
        report["n_replaced"] > 0
        and report["n_unreplaceable"] == 0
        and bool(cleaned.notna().all())
    )
    cleaned_values_used = bool(apply_cleaned and can_apply)
    summary = summarize_validation(
        report,
        full_index,
        cleaned_values_used=cleaned_values_used,
        would_apply_cleaned_values=can_apply,
    )
    if cleaned_values_used:
        return cleaned, summary
    return hourly.sort_index(), summary
