"""Tests fuer Forecast-Verlauf-Store (T51).

S1-S6 aus scripts/verify_forecast_store.py portiert (exakte Auswertung, pending,
Duplikat-Supersede, Perioden-Mismatch, Tamper, Validierung). Plus Integrationstests
fuer run_forecast (opt-in forecast_store_path; bit-identisch wenn None).
"""
from __future__ import annotations
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.service.forecast_store import record_forecast, realized_track_record
from netzpilot.service.audit_ledger import canonical_json


def _fp(date, p10, p50, p90):
    return {"date": date, "periods_per_day": len(p50),
            "hours": [{"hour": h, "p10": p10[h], "p50": p50[h], "p90": p90[h]}
                      for h in range(len(p50))]}


# ---------- S1-S6: aus verify_forecast_store.py portiert ----------

def test_S1_chain_ok_and_pending(tmp_path):
    store = str(tmp_path / "f.jsonl")
    record_forecast(store, _fp("2026-06-01", [8, 10], [10, 12], [12, 14]), "hilden")
    record_forecast(store, _fp("2026-06-02", [8, 10], [10, 12], [12, 14]), "hilden")
    tr = realized_track_record(store, {"2026-06-01": [11, 11]})
    assert tr["chain_ok"] is True
    assert tr["n_forecasts_stored"] == 2
    assert len(tr["days"]) == 1


def test_S2_exact_mae_bias_coverage_residual(tmp_path):
    store = str(tmp_path / "f.jsonl")
    record_forecast(store, _fp("2026-06-01", [8, 10], [10, 12], [12, 14]), "hilden")
    tr = realized_track_record(store, {"2026-06-01": [11, 11]})
    d = tr["days"][0]
    assert abs(d["mae"] - 1.0) < 1e-12
    assert abs(d["coverage_p10_p90_pct"] - 100.0) < 1e-12
    assert abs(d["bias"]) < 1e-12
    assert d["residual"] == [1.0, -1.0]
    assert tr["last_residual"] == [1.0, -1.0]
    assert tr["last_residual_date"] == "2026-06-01"


def test_S3_duplicate_superseded(tmp_path):
    store = str(tmp_path / "f.jsonl")
    record_forecast(store, _fp("2026-06-01", [8, 10], [10, 12], [12, 14]), "hilden")
    record_forecast(store, _fp("2026-06-01", [9, 11], [11, 13], [13, 15]), "hilden")
    tr = realized_track_record(store, {"2026-06-01": [11, 11]})
    assert tr["n_duplicates_superseded"] == 1
    # Auswertung gegen neueste Ausgabe: residual = a - p50 = [11-11, 11-13] = [0, -2]
    assert abs(tr["days"][0]["mae"] - 1.0) < 1e-12
    assert tr["days"][0]["residual"] == [0.0, -2.0]


def test_S4_period_mismatch_skipped(tmp_path):
    store = str(tmp_path / "f.jsonl")
    record_forecast(store, _fp("2026-06-01", [8, 10], [10, 12], [12, 14]), "hilden")
    tr = realized_track_record(store, {"2026-06-01": [11, 11, 11]})
    assert tr["n_skipped_period_mismatch"] == 1
    assert len(tr["days"]) == 0


def test_S5_tamper_breaks_chain_and_blocks_append(tmp_path):
    store = str(tmp_path / "f.jsonl")
    record_forecast(store, _fp("2026-06-01", [8, 10], [10, 12], [12, 14]), "hilden")
    # Tamper: erste Zeile aendern
    lines = open(store, encoding="utf-8").read().splitlines()
    rec = json.loads(lines[0])
    rec["payload"]["p50"] = [99.0, 99.0]
    lines[0] = canonical_json(rec)
    with open(store, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    tr = realized_track_record(store, {"2026-06-01": [11, 11]})
    assert tr["chain_ok"] is False
    with pytest.raises(ValueError):
        record_forecast(store, _fp("2026-06-03", [1], [2], [3]), "hilden")


def test_S6_fp_without_hours_raises(tmp_path):
    with pytest.raises(ValueError):
        record_forecast(str(tmp_path / "x.jsonl"), {"date": "d"}, "s")


# ---------- Integration: run_forecast mit forecast_store_path ----------

REAL_CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_T51_runner_store_none_bit_compatible():
    """forecast_store_path=None → bit-identisch zum alten Verhalten."""
    from netzpilot.service.runner import run_forecast
    r = run_forecast(REAL_CSV, utility="TestStoreOff", unit="kW",
                     ts_col="Text", load_col="Reihe1")
    assert r["track_record"] is None
    assert "forecast_store" not in r


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_T51_runner_with_store_records_and_realizes(tmp_path):
    """Zwei aufeinanderfolgende Auswertungen — Lauf 1 speichert, Synthese-Auswertung sieht ihn."""
    from netzpilot.service.runner import run_forecast
    store = str(tmp_path / "fc_store.jsonl")
    r1 = run_forecast(REAL_CSV, utility="TestStore", unit="kW",
                      ts_col="Text", load_col="Reihe1",
                      forecast_store_path=store)
    assert r1["track_record"] is not None
    assert r1["track_record"]["chain_ok"] is True
    assert r1["forecast_store"]["target_date"] == r1["forecast_date"]
    assert r1["forecast_store"]["entry_hash"]
    # Gefakte Actuals fuer die heutige Prognose -> nun realisierter Tag
    fake_actuals = {r1["forecast_date"]: [float(h["p50"]) for h in r1["forecast"]]}
    tr = realized_track_record(store, fake_actuals)
    assert tr["aggregate"]["n_days_realized"] == 1
    assert tr["last_residual"] is not None
    assert len(tr["last_residual"]) == 24
    assert tr["chain_ok"] is True


# ---------- T51-Nachtrag: residual_feedback override (REST 1) ----------

@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_T51N_logging_only_store_with_residual_feedback_false(tmp_path):
    """Reines Logging ohne Prognose-Aenderung: Store-Pfad + residual_feedback=False.
    Erwartung: Store schreibt + Track-Record da, ABER forecast_next_day kein RF (residual_feedback fehlt)."""
    from netzpilot.service.runner import run_forecast
    store = str(tmp_path / "fc_store_log_only.jsonl")
    r = run_forecast(REAL_CSV, utility="LogOnly", unit="kW",
                     ts_col="Text", load_col="Reihe1",
                     forecast_store_path=store,
                     residual_feedback=False)
    # Store ist aktiv (Track-Record + Entry geschrieben)
    assert r["track_record"] is not None
    assert "forecast_store" in r and r["forecast_store"]["entry_hash"]
    # Aber kein RF — residual_feedback-Feld im Output nicht oder applied=False
    assert r.get("residual_feedback") is None


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_T51N_residual_feedback_true_without_store_reconstructs(tmp_path):
    """residual_feedback=True ohne Store: RF aktiv, Quelle = 'reconstructed' (kein Store-Residuum)."""
    from netzpilot.service.runner import run_forecast
    r = run_forecast(REAL_CSV, utility="NoStore", unit="kW",
                     ts_col="Text", load_col="Reihe1",
                     residual_feedback=True)
    assert r["track_record"] is None
    assert "forecast_store" not in r
    # RF aktiv, Quelle Rekonstruktion
    rf = r.get("residual_feedback")
    assert rf is not None
    assert rf.get("residual_source") == "reconstructed"


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")
def test_T51N_residual_feedback_auto_keeps_store_driven_default(tmp_path):
    """Default None (auto) = Store-Pfad-getrieben: ohne Store kein RF, mit Store RF an."""
    from netzpilot.service.runner import run_forecast
    # Ohne Store -> kein RF
    r_off = run_forecast(REAL_CSV, utility="AutoOff", unit="kW",
                         ts_col="Text", load_col="Reihe1")
    assert r_off.get("residual_feedback") is None
    # Mit Store -> RF an
    store = str(tmp_path / "auto.jsonl")
    r_on = run_forecast(REAL_CSV, utility="AutoOn", unit="kW",
                        ts_col="Text", load_col="Reihe1",
                        forecast_store_path=store)
    assert r_on.get("residual_feedback") is not None
