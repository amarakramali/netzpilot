"""§14a-Compliance (W12 Meldebogen + W13 Diskriminierungsfreiheit). Fixture-frei (run_all_checks-Shim)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.service.audit_ledger import append_entry, append_forecast_audit, load_entries, canonical_json
from netzpilot.service.compliance_14a import (
    _gini,
    monthly_meldebogen,
    fairness_report,
    render_meldebogen_html,
    render_fairness_html,
)


def _redispatch_payload(ts, *, limits, demands, magnitude, duration=60, asset="SW"):
    affected = []
    for i, lim in enumerate(limits):
        rec = {"device_index": i, "limit_kw": float(lim)}
        if demands is not None:
            d = float(demands[i])
            rec["demand_kw"] = d
            rec["shed_kw"] = max(0.0, d - float(lim))
        affected.append(rec)
    return {
        "ts_utc": ts, "asset_id": asset, "decision_type": "redispatch",
        "reason": {"trigger": "forecast_kw > threshold_kw"},
        "magnitude_kw": float(magnitude), "duration_min": int(duration),
        "affected": affected, "rule_version": "test-v1",
    }


def test_gini_basic():
    assert _gini([]) == 0.0
    assert _gini([1.0, 1.0, 1.0]) == 0.0          # perfekt gleich
    assert _gini([0.0, 0.0, 0.0]) == 0.0          # niemand gedrosselt
    assert _gini([0.0, 0.4]) > 0.0                # einseitig
    assert _gini([-5.0, -5.0]) == 0.0             # negative geklemmt


def test_meldebogen_rollup_and_month_filter():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        # Juni: zwei redispatch-Eingriffe, je 2 steuVE, beide gedrosselt
        append_entry(path, _redispatch_payload("2026-06-01T18:00:00Z", limits=[6, 6], demands=[10, 10], magnitude=8))
        append_entry(path, _redispatch_payload("2026-06-02T19:00:00Z", limits=[7, 5], demands=[10, 10], magnitude=8))
        # Mai: darf NICHT mitzählen
        append_entry(path, _redispatch_payload("2026-05-15T18:00:00Z", limits=[6, 6], demands=[10, 10], magnitude=8))
        m = monthly_meldebogen(path, 2026, 6)
        assert m["chain_ok"] is True
        assert m["n_eingriffe_im_monat"] == 2            # Mai gefiltert
        row = m["by_steuerungsart"][0]
        assert row["decision_type"] == "redispatch"
        assert row["n_massnahmen"] == 2
        assert row["n_steuve_betroffen"] == 2            # 2 distinct Geräte, beide gedrosselt
        assert row["durchschnittl_reduzierte_leistung_kw"] == 8.0
        assert row["gesamtdauer_min"] == 120
        assert row["bedarf_bekannt"] is True


def test_fairness_demand_normalized_equal_is_low_gini():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        # beide steuVE identisch gedrosselt → faire (gleiche) relative Kappung → Gini 0
        append_entry(path, _redispatch_payload("2026-06-01T18:00:00Z", limits=[6, 6], demands=[10, 10], magnitude=8))
        append_entry(path, _redispatch_payload("2026-06-01T19:00:00Z", limits=[6, 6], demands=[10, 10], magnitude=8))
        f = fairness_report(path)
        assert f["bedarfsnormalisiert"] is True
        assert f["n_steuve"] == 2
        assert f["primary_metric"]["gini"] == 0.0
        assert f["primary_metric"]["spannweite"] == 0.0


def test_fairness_one_sided_is_higher_gini():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        # Gerät 0 stark gedrosselt, Gerät 1 gar nicht → ungleich → Gini > 0
        append_entry(path, _redispatch_payload("2026-06-01T18:00:00Z", limits=[4, 10], demands=[10, 10], magnitude=6))
        f = fairness_report(path)
        assert f["bedarfsnormalisiert"] is True
        assert f["primary_metric"]["gini"] > 0.0
        d0 = next(d for d in f["devices"] if d["device_index"] == 0)
        d1 = next(d for d in f["devices"] if d["device_index"] == 1)
        assert d0["mittlere_kappungsquote"] == 0.6        # (10-4)/10
        assert d1["mittlere_kappungsquote"] == 0.0


def test_fairness_without_demand_is_labeled_limited():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        append_entry(path, _redispatch_payload("2026-06-01T18:00:00Z", limits=[6, 6], demands=None, magnitude=8))
        f = fairness_report(path)
        assert f["bedarfsnormalisiert"] is False
        assert f["primary_metric"] is None
        assert "ACHTUNG" in f["note"]


def test_ledger_extension_roundtrip_and_backward_compat():
    # MIT device_demands_kw → affected bekommt demand_kw/shed_kw
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        result = {
            "utility": "SW", "forecast_date": "2026-06-04",
            "redispatch": {"basis": "load", "threshold_kw": 100.0,
                           "device_demands_kw": [10.0, 8.0],
                           "hourly": [{"hour": 1, "intervention": True, "forecast_kw": 120.0,
                                       "overload_kw": 20.0, "shed_kw": 12.0, "limits_kw": [4.0, 2.0]}]}}
        audit = append_forecast_audit(path, result)
        assert audit["entries_appended"] == 1
        aff = load_entries(path)[0]["payload"]["affected"]
        assert aff[0]["demand_kw"] == 10.0 and aff[0]["shed_kw"] == 6.0
        assert aff[1]["demand_kw"] == 8.0 and aff[1]["shed_kw"] == 6.0
    # OHNE device_demands_kw → exakt die alte Struktur (rückwärtskompatibel)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        result = {"utility": "SW", "forecast_date": "2026-06-04",
                  "redispatch": {"basis": "load", "threshold_kw": 100.0,
                                 "hourly": [{"hour": 1, "intervention": True, "forecast_kw": 120.0,
                                             "overload_kw": 20.0, "shed_kw": 12.0, "limits_kw": [4.2, 5.8]}]}}
        append_forecast_audit(path, result)
        aff = load_entries(path)[0]["payload"]["affected"]
        assert aff[0] == {"device_index": 0, "limit_kw": 4.2}
        assert "demand_kw" not in aff[1]


def test_broken_chain_is_flagged_not_hidden():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        append_entry(path, _redispatch_payload("2026-06-01T18:00:00Z", limits=[6, 6], demands=[10, 10], magnitude=8))
        append_entry(path, _redispatch_payload("2026-06-02T18:00:00Z", limits=[6, 6], demands=[10, 10], magnitude=8))
        # mittleren Eintrag manipulieren
        lines = load_entries(path)
        lines[0]["payload"]["magnitude_kw"] = 999.0
        with open(path, "w", encoding="utf-8") as fh:
            for ln in lines:
                fh.write(canonical_json(ln) + "\n")
        assert monthly_meldebogen(path, 2026, 6)["chain_ok"] is False
        assert fairness_report(path)["chain_ok"] is False


def test_determinism_and_html_is_honest():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        append_entry(path, _redispatch_payload("2026-06-01T18:00:00Z", limits=[4, 10], demands=[10, 10], magnitude=6))
        assert monthly_meldebogen(path, 2026, 6) == monthly_meldebogen(path, 2026, 6)
        assert fairness_report(path) == fairness_report(path)
        mh = render_meldebogen_html(path, 2026, 6)
        fh = render_fairness_html(path)
        assert "BNetzA" in mh and "BNetzA" in fh          # ehrliche Nicht-Zertifizierung sichtbar
        assert "Meldebogen" in mh and "Diskriminierungsfreiheit" in fh
