#!/usr/bin/env python3
"""Build the T44 Paragraph-14a audit ledger demo."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from netzpilot.service.report import write_audit_report_html
from netzpilot.service.runner import run_forecast


REAL_CSV = os.path.join(ROOT, "data_cache", "real", "Netzumsatz-Lastgang-2025.csv")
OUT_LEDGER = os.path.join(ROOT, "data_cache", "benchmark", "audit_ledger_demo.jsonl")
OUT_HTML = os.path.join(ROOT, "data_cache", "benchmark", "audit_ledger_demo.html")


def main() -> None:
    os.makedirs(os.path.dirname(OUT_LEDGER), exist_ok=True)
    for path in (OUT_LEDGER, OUT_HTML):
        if os.path.exists(path):
            os.remove(path)
    result = run_forecast(
        REAL_CSV,
        utility="AuditDemo",
        unit="kW",
        ts_col="Text",
        load_col="Reihe1",
        congestion_threshold_mw=33.0,
        steuve_demands_kw=[1000.0, 800.0, 600.0],
        rolling_redispatch=True,
        audit_ledger_path=OUT_LEDGER,
        audit_signing_key="demo-signing-key",
    )
    write_audit_report_html(
        OUT_LEDGER,
        OUT_HTML,
        signing_key="demo-signing-key",
        title="NetzPilot Paragraph-14a Audit-Nachweis Demo",
    )
    audit = result["audit"]
    print(f"ledger={OUT_LEDGER}")
    print(f"report={OUT_HTML}")
    print(f"entries_appended={audit['entries_appended']} chain_ok={audit['chain_ok']}")
    print(f"head_hash={audit['head_hash']}")


if __name__ == "__main__":
    main()
