#!/usr/bin/env python3
"""Verify Forecast-Verlauf-Store (service/forecast_store.py). Deterministisch, schnell.
Aufruf: python scripts/verify_forecast_store.py
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.service.forecast_store import record_forecast, realized_track_record
from netzpilot.service.audit_ledger import canonical_json

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond
def raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True

def fp(date, p10, p50, p90):
    return {"date": date, "periods_per_day": len(p50),
            "hours": [{"hour": h, "p10": p10[h], "p50": p50[h], "p90": p90[h]} for h in range(len(p50))]}

with tempfile.TemporaryDirectory() as td:
    store = os.path.join(td, "forecasts.jsonl")
    # D1: bekannte Zahlen -> exakte Auswertung; D2: pending (keine Actuals)
    record_forecast(store, fp("2026-06-01", [8, 10], [10, 12], [12, 14]), "hilden")
    record_forecast(store, fp("2026-06-02", [8, 10], [10, 12], [12, 14]), "hilden")
    tr = realized_track_record(store, {"2026-06-01": [11, 11]})
    check("S1: chain_ok", tr["chain_ok"] is True)
    check("S1: 1 realisiert, D2 pending", len(tr["days"]) == 1 and tr["n_forecasts_stored"] == 2)
    d = tr["days"][0]
    check("S2: MAE exakt 1.0", abs(d["mae"] - 1.0) < 1e-12)
    check("S2: Coverage exakt 100", abs(d["coverage_p10_p90_pct"] - 100.0) < 1e-12)
    check("S2: Bias exakt 0.0", abs(d["bias"]) < 1e-12)
    check("S2: Residuum exakt [1,-1]", d["residual"] == [1.0, -1.0])
    check("S2: last_residual == Residuum des jüngsten realisierten Tags",
          tr["last_residual"] == [1.0, -1.0] and tr["last_residual_date"] == "2026-06-01")

    # S3: Duplikat (Neuausgabe für D1) -> letzter Eintrag gewinnt
    record_forecast(store, fp("2026-06-01", [9, 11], [11, 13], [13, 15]), "hilden")
    tr2 = realized_track_record(store, {"2026-06-01": [11, 11]})
    check("S3: Duplikat ersetzt (superseded=1)", tr2["n_duplicates_superseded"] == 1)
    check("S3: Auswertung gegen NEUESTE Ausgabe (MAE=1.0: |11-11|,|11-13|)",
          abs(tr2["days"][0]["mae"] - 1.0) < 1e-12 and tr2["days"][0]["residual"] == [0.0, -2.0])

    # S4: Perioden-Mismatch -> übersprungen, kein Crash
    tr3 = realized_track_record(store, {"2026-06-01": [11, 11, 11]})
    check("S4: Mismatch übersprungen", tr3["n_skipped_period_mismatch"] == 1 and len(tr3["days"]) == 0)

    # S5: Manipulation -> chain_ok False (durch die Schicht sichtbar)
    lines = open(store, encoding="utf-8").read().splitlines()
    rec = json.loads(lines[0]); rec["payload"]["p50"] = [99.0, 99.0]
    lines[0] = canonical_json(rec)
    with open(store, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    tr4 = realized_track_record(store, {"2026-06-01": [11, 11]})
    check("S5: Tamper -> chain_ok False", tr4["chain_ok"] is False)
    check("S5: Append auf manipulierte Kette -> ValueError",
          raises(lambda: record_forecast(store, fp("2026-06-03", [1], [2], [3]), "hilden")))

# S6: Validierung
check("S6: fp ohne hours -> ValueError",
      raises(lambda: record_forecast(os.path.join(tempfile.gettempdir(), "x.jsonl"), {"date": "d"}, "s")))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
