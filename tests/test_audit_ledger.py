# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import json
import os
import tempfile

import pytest

from netzpilot.service.audit_ledger import (
    GENESIS_HASH,
    append_entry,
    append_forecast_audit,
    canonical_json,
    entry_hash,
    render_audit_report_html,
    sign_head_hash,
    verify_chain,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_CSV = os.path.join(ROOT, "data_cache", "real", "Netzumsatz-Lastgang-2025.csv")


def _payload(i=0):
    return {
        "ts_utc": f"2026-06-03T{8 + i:02d}:00:00Z",
        "asset_id": "asset-1",
        "decision_type": "redispatch",
        "reason": {"trigger": "forecast_kw > threshold_kw", "forecast_kw": 120.0 + i},
        "magnitude_kw": 10.0 + i,
        "duration_min": 60,
        "affected": [{"device_index": 0, "limit_kw": 4.2}],
        "rule_version": "test-v1",
    }


def test_canonical_json_and_hash_are_deterministic():
    a = {"b": 2, "a": {"z": 1, "y": [3, 2, 1]}}
    b = {"a": {"y": [3, 2, 1], "z": 1}, "b": 2}
    assert canonical_json(a) == canonical_json(b)
    assert entry_hash(GENESIS_HASH, a) == entry_hash(GENESIS_HASH, b)


def test_append_and_verify_chain_ok():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        e1 = append_entry(path, _payload(0))
        e2 = append_entry(path, _payload(1))
        status = verify_chain(path)
        assert status["ok"] is True
        assert status["n"] == 2
        assert status["broken_at"] is None
        assert status["head_hash"] == e2["entry_hash"]
        assert e2["prev_hash"] == e1["entry_hash"]


def test_verify_chain_detects_middle_tampering():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        for i in range(3):
            append_entry(path, _payload(i))
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        lines[1]["payload"]["magnitude_kw"] = 999.0
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(canonical_json(line) + "\n")
        status = verify_chain(path)
        assert status["ok"] is False
        assert status["broken_at"] == 1


def test_hmac_signature_is_key_dependent():
    head = "abc123"
    sig_a = sign_head_hash(head, "secret")
    sig_b = sign_head_hash(head, "secret")
    sig_wrong = sign_head_hash(head, "wrong")
    assert sig_a == sig_b
    assert sig_a != sig_wrong
    assert sign_head_hash(head, None) is None


def test_audit_report_contains_head_hash_signature_and_honest_label():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        append_entry(path, _payload(0))
        status = verify_chain(path)
        html = render_audit_report_html(path, signing_key="secret")
        assert status["head_hash"] in html
        assert sign_head_hash(status["head_hash"], "secret") in html
        assert "manipulationssicherer Audit-Trail (Hash-Kette)" in html
        assert "nicht BNetzA-zertifiziert" in html


def test_append_forecast_audit_extracts_redispatch_entries():
    result = {
        "utility": "AuditSW",
        "forecast_date": "2026-06-04",
        "redispatch": {
            "basis": "load",
            "forecast_basis": "day_ahead_p50_static",
            "threshold_kw": 100.0,
            "hourly": [
                {"hour": 0, "intervention": False, "forecast_kw": 90.0},
                {
                    "hour": 1,
                    "intervention": True,
                    "forecast_kw": 120.0,
                    "overload_kw": 20.0,
                    "shed_kw": 12.0,
                    "limits_kw": [4.2, 5.8],
                },
            ],
        },
    }
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        audit = append_forecast_audit(path, result, signing_key="secret", rule_version="rule-a")
        assert audit["status"] == "available"
        assert audit["entries_appended"] == 1
        assert audit["chain_ok"] is True
        assert audit["head_signature_hmac_sha256"] == sign_head_hash(audit["head_hash"], "secret")


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_runner_adds_audit_ledger_for_real_redispatch():
    from netzpilot.service.runner import run_forecast

    with tempfile.TemporaryDirectory() as td:
        ledger = os.path.join(td, "audit.jsonl")
        out = run_forecast(
            REAL_CSV,
            utility="AuditRunner",
            unit="kW",
            ts_col="Text",
            load_col="Reihe1",
            congestion_threshold_mw=33.0,
            steuve_demands_kw=[1000.0, 800.0, 600.0],
            rolling_redispatch=True,
            audit_ledger_path=ledger,
            audit_signing_key="secret",
        )
        assert out["audit"]["status"] == "available"
        assert out["audit"]["entries_appended"] > 0
        assert out["audit"]["chain_ok"] is True
        assert verify_chain(ledger)["ok"] is True
