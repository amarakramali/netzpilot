# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Service-Tests: Runner, Persistenz, FastAPI-API — auf echter Hilden-CSV.

Überspringt sauber, wenn die echte Datei oder FastAPI/TestClient fehlen (CI-freundlich).
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REAL_CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"
pytestmark = pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="echte Hilden-CSV nicht vorhanden")


def test_runner_forecast_and_fahrplan():
    from netzpilot.service.runner import run_forecast
    r = run_forecast(REAL_CSV, utility="TestSW", unit="kW", ts_col="Text", load_col="Reihe1",
                     congestion_threshold_mw=33.0, steuve_malo="DE0001234567890")
    assert len(r["forecast"]) == 24
    for h in r["forecast"]:
        assert h["p10"] <= h["p50"] <= h["p90"]      # Monotonie der Quantile
    assert r["n_days_history"] > 300
    assert r["congestion"] is not None               # Schwelle 33 MW wird überschritten
    # §14a: Mindestleistung eingehalten
    assert all(sp["p_limit_kw"] >= 4.2 - 1e-9 for sp in r["fahrplan"]["setpoints"])


def test_runner_forecast_with_rebap_only_falls_back_to_upper_bound():
    """Ohne Spot: economics ist |reBAP|-basiert, klar als UPPER BOUND gelabelt."""
    from netzpilot.service.runner import run_forecast
    r = run_forecast(REAL_CSV, utility="TestSW", unit="kW", ts_col="Text", load_col="Reihe1",
                     rebap_prices=[-10.0, 20.0, 40.0])
    assert r["economics"]["eur_per_year_p25"] <= r["economics"]["eur_per_year_point_median"]
    assert r["economics"]["eur_per_year_point_median"] <= r["economics"]["eur_per_year_p75"]
    assert r["economics"]["rebap_spread_stats"]["n"] == 3
    assert "UEBERSCHAETZT" in r["economics"]["caveat"] or "UPPER BOUND" in r["economics"]["caveat"]
    assert r["economics_upper_bound"] is None


def test_runner_forecast_with_rebap_and_spot_uses_aufschlag():
    """Mit Spot: economics = Aufschlag (|reBAP-Spot|); economics_upper_bound = |reBAP| absolut."""
    from netzpilot.service.runner import run_forecast
    rebap = [-10.0, 20.0, 40.0, 80.0]
    spot = [5.0, 30.0, 50.0, 100.0]
    r = run_forecast(REAL_CSV, utility="TestSW", unit="kW", ts_col="Text", load_col="Reihe1",
                     rebap_prices=rebap, spot_prices=spot)
    assert "|reBAP - Spot|" in r["economics"]["basis"]
    assert r["economics"]["eur_per_year_p25"] <= r["economics"]["eur_per_year_point_median"]
    assert r["economics"]["eur_per_year_point_median"] <= r["economics"]["eur_per_year_p75"]
    assert r["economics_upper_bound"] is not None
    # Aufschlag-Headline muss kleiner-gleich |reBAP|-Upper-Bound sein (gespart wird nur der Aufschlag)
    assert r["economics"]["eur_per_year_point_median"] <= r["economics_upper_bound"]["eur_per_year_point_median"]


def test_runner_no_congestion_when_threshold_high():
    from netzpilot.service.runner import run_forecast
    r = run_forecast(REAL_CSV, utility="TestSW", unit="kW", ts_col="Text", load_col="Reihe1",
                     congestion_threshold_mw=10_000.0)
    assert r["congestion"] is None and r["fahrplan"] is None


def test_runner_aemt_roundtrip_on_congestion():
    """#16: §14a-Regelkreis end-to-end — Engpass→Fahrplan→aEMT-Quittung (ACCEPTED, mit ID)."""
    from netzpilot.service.runner import run_forecast
    r = run_forecast(REAL_CSV, utility="TestSW", unit="kW", ts_col="Text", load_col="Reihe1",
                     congestion_threshold_mw=33.0, steuve_malo="DE0001234567890",
                     submit_to_aemt=True)
    assert r["fahrplan"] is not None
    ack = r["aemt_ack"]
    assert ack is not None and ack["status"] == "ACCEPTED"
    assert ack["transmission_id"].startswith("aemt-")
    assert ack["fahrplan_fingerprint"] and ack["malo"] == "DE0001234567890"


def test_aemt_rejects_sub_minimum_setpoint():
    """aEMT lehnt empfängerseitig einen Fahrplan unter §14a-Mindestleistung ab (defense in depth)."""
    import copy
    from netzpilot.control.schema import make_fahrplan
    from netzpilot.control.aemt_adapter import MockAemt, AemtError
    fp = make_fahrplan("DE0001234567890",
                       [{"start_utc": "2026-01-01T10:00:00", "end_utc": "2026-01-01T12:00:00",
                         "p_limit_kw": 4.2}])
    fp_bad = copy.deepcopy(fp)
    fp_bad["setpoints"][0]["p_limit_kw"] = 1.0  # nach Validierung manipuliert
    try:
        MockAemt()._transmit(fp_bad)
        assert False, "aEMT hätte ablehnen müssen"
    except AemtError:
        pass


def test_runner_residual_load_uses_same_engine_and_drives_congestion(tmp_path):
    """#10: Erzeugungs-CSV aktiviert Residuallast-Prognose (gleiche Engine); §14a-Engpass
    laeuft dann auf der Residuallast (basis='residual'). Erzeugung = fester Anteil der echten
    Last -> garantiert ausgerichtete Stunden & Vorzeichen."""
    import csv as _csv
    from netzpilot.service.runner import run_forecast, _load2d_from_csv

    _l2d, _days, _meta, load_hourly = _load2d_from_csv(REAL_CSV, ts_col="Text", load_col="Reihe1", unit="kW")
    gen_csv = tmp_path / "gen.csv"
    with open(gen_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f, delimiter=";")
        w.writerow(["Zeit", "gen_MW"])
        for ts, mw in load_hourly.items():
            w.writerow([ts.isoformat(), f"{0.30 * float(mw):.6f}"])  # 30% Erzeugung in MW

    r = run_forecast(REAL_CSV, utility="TestSW", unit="kW", ts_col="Text", load_col="Reihe1",
                     generation_csv=str(gen_csv), generation_unit="MW",
                     congestion_threshold_mw=33.0, steuve_malo="DE0001234567890")
    rf = r["residual_forecast"]
    assert rf is not None and len(rf["forecast"]) == 24
    for h in rf["forecast"]:
        assert h["p10"] <= h["p50"] <= h["p90"]
    # Residuallast (Last - 30%) liegt im Schnitt unter der Last
    assert rf["recent_mean_residual_mw"] < r["recent_mean_load_mw"]
    assert rf["n_common_hours"] > 300 * 24
    if r["congestion"] is not None:
        assert r["congestion"]["basis"] == "residual"


def test_runner_without_generation_has_no_residual():
    """Gegencheck: ohne Erzeugung kein residual_forecast, Engpass-Basis bleibt 'load'."""
    from netzpilot.service.runner import run_forecast
    r = run_forecast(REAL_CSV, utility="TestSW", unit="kW", ts_col="Text", load_col="Reihe1",
                     congestion_threshold_mw=33.0)
    assert r["residual_forecast"] is None
    if r["congestion"] is not None:
        assert r["congestion"]["basis"] == "load"


def test_store_roundtrip(tmp_path):
    from netzpilot.service.store import ForecastStore
    s = ForecastStore(str(tmp_path))
    rec = {"utility": "X", "forecast_date": "2026-01-01", "forecast": [{"hour": 0, "p50": 1.0}]}
    s.save("X", rec)
    assert s.latest("X")["forecast_date"] == "2026-01-01"
    assert s.get("X", "2026-01-01")["utility"] == "X"
    assert "2026-01-01" in s.history("X")
    assert "X" in s.list_utilities()


def test_api_health_and_forecast(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    os.environ["NETZPILOT_STORE"] = str(tmp_path)
    import importlib
    import netzpilot.service.app as appmod
    importlib.reload(appmod)
    client = TestClient(appmod.app)

    assert client.get("/health").json()["status"] == "ok"

    with open(REAL_CSV, "rb") as f:
        resp = client.post("/forecast", files={"file": ("hilden.csv", f, "text/csv")},
                           data={"utility": "Hilden", "unit": "kW", "ts_col": "Text",
                                 "load_col": "Reihe1", "congestion_threshold_mw": "33"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["forecast"]) == 24 and body["utility"] == "Hilden"

    latest = client.get("/forecast/Hilden/latest").json()
    assert latest["forecast_date"] == body["forecast_date"]
    assert "Hilden" in client.get("/utilities").json()["utilities"]


def test_api_forecast_horizon_days_d2_d3_only_p50(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    os.environ["NETZPILOT_STORE"] = str(tmp_path)
    import importlib
    import netzpilot.service.app as appmod
    importlib.reload(appmod)
    client = TestClient(appmod.app)

    with open(REAL_CSV, "rb") as f:
        resp = client.post("/forecast", files={"file": ("hilden.csv", f, "text/csv")},
                           data={"utility": "Hilden", "unit": "kW", "ts_col": "Text",
                                 "load_col": "Reihe1", "horizon_days": "3"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    horizon = body["horizon"]
    assert len(body["forecast"]) == 24
    assert all({"p10", "p50", "p90"}.issubset(h.keys()) for h in body["forecast"])
    assert horizon is not None
    assert [d["horizon"] for d in horizon["days"]] == [2, 3]
    assert "D+1 produktiv bleibt" in horizon["note"]
    assert "bands_note" in horizon
    for day in horizon["days"]:
        assert len(day["hours"]) == 24
        assert all("p50" in h and "p10" not in h and "p90" not in h for h in day["hours"])

    bad = client.post("/forecast", data={"csv_path": REAL_CSV, "unit": "kW", "ts_col": "Text",
                                         "load_col": "Reihe1", "horizon_days": "8"})
    assert bad.status_code == 422


def test_api_forecast_horizon_bands_per_horizon(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    os.environ["NETZPILOT_STORE"] = str(tmp_path)
    import importlib
    import netzpilot.service.app as appmod
    importlib.reload(appmod)
    client = TestClient(appmod.app)

    with open(REAL_CSV, "rb") as f:
        resp = client.post("/forecast", files={"file": ("hilden.csv", f, "text/csv")},
                           data={"utility": "Hilden", "unit": "kW", "ts_col": "Text",
                                 "load_col": "Reihe1", "horizon_days": "3",
                                 "horizon_bands": "per_horizon"})
    assert resp.status_code == 200, resp.text
    horizon = resp.json()["horizon"]
    assert horizon["bands_mode"] == "per_horizon"
    assert "kalibriertem Band" in horizon["bands"]
    assert horizon["bands_note"] == horizon["bands"]
    assert [d["horizon"] for d in horizon["days"]] == [2, 3]
    for day in horizon["days"]:
        assert day["band"]["scale"] >= 1.0
        assert day["band"]["n_cal_days"] > 0
        assert "conf_c" in day["band"]
        assert all(h["p10"] <= h["p50"] <= h["p90"] for h in day["hours"])

    bad = client.post("/forecast", data={"csv_path": REAL_CSV, "unit": "kW", "ts_col": "Text",
                                         "load_col": "Reihe1", "horizon_days": "3",
                                         "horizon_bands": "x"})
    assert bad.status_code == 422
    bad_days = client.post("/forecast", data={"csv_path": REAL_CSV, "unit": "kW", "ts_col": "Text",
                                              "load_col": "Reihe1", "horizon_days": "1",
                                              "horizon_bands": "per_horizon"})
    assert bad_days.status_code == 422


def test_api_intraday_updates_view_without_touching_latest(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    os.environ["NETZPILOT_STORE"] = str(tmp_path)
    import importlib
    import netzpilot.service.app as appmod
    importlib.reload(appmod)
    client = TestClient(appmod.app)

    with open(REAL_CSV, "rb") as f:
        resp = client.post("/forecast", files={"file": ("hilden.csv", f, "text/csv")},
                           data={"utility": "HildenID", "unit": "kW", "ts_col": "Text",
                                 "load_col": "Reihe1"})
    assert resp.status_code == 200, resp.text
    latest_before = client.get("/forecast/HildenID/latest").json()

    upd = client.post("/intraday", data={"utility": "HildenID", "actuals": "21.0,22.5"})
    assert upd.status_code == 200, upd.text
    body = upd.json()
    assert body["utility"] == "HildenID"
    assert body["forecast_date"] == latest_before["forecast_date"]
    assert body["applied"] is True
    assert body["update_hour"] == 2
    assert body["n_hours_used"] == 2
    assert len(body["hours_rest"]) == 22
    assert abs(body["delta_mw"]) < 20.0
    assert "nicht jeder Tag gewinnt" in body["caveat"]

    latest_after = client.get("/forecast/HildenID/latest").json()
    assert latest_after == latest_before

    missing = client.post("/intraday", data={"utility": "Unbekannt", "actuals": "21.0,22.5"})
    assert missing.status_code == 404
    too_long = client.post("/intraday", data={"utility": "HildenID",
                                              "actuals": ",".join(["1"] * 24)})
    assert too_long.status_code == 422


def test_api_files_listing(tmp_path):
    """GET /files: Lastgang-Auswahl fürs Cockpit — nur data_cache/real, nur Dateinamen, nur Lastgang-Endungen."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    os.environ["NETZPILOT_STORE"] = str(tmp_path)
    import importlib
    import netzpilot.service.app as appmod
    importlib.reload(appmod)
    client = TestClient(appmod.app)

    j = client.get("/files").json()
    assert j["dir"] == "data_cache/real"
    assert "Netzumsatz-Lastgang-2025.csv" in j["files"]
    assert all("/" not in f and "\\" not in f for f in j["files"])   # kein Pfad, kein Traversal
    assert all(f.lower().endswith((".csv", ".xlsx", ".xls", ".xlsm")) for f in j["files"])
    assert j["files"] == sorted(j["files"])


def test_api_challenge_on_real_data(tmp_path):
    """POST /challenge: leakage-sicherer Sofort-Backtest auf hochgeladener Datei, ohne Persistierung."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    os.environ["NETZPILOT_STORE"] = str(tmp_path)
    import importlib
    import netzpilot.service.app as appmod
    importlib.reload(appmod)
    client = TestClient(appmod.app)

    with open(REAL_CSV, "rb") as f:
        resp = client.post("/challenge", files={"file": ("kunde.csv", f, "text/csv")},
                           data={"unit": "kW", "ts_col": "Text", "load_col": "Reihe1",
                                 "n_test": "28", "n_boot": "500"})
    assert resp.status_code == 200, resp.text
    c = resp.json()
    assert c["source_file"] == "kunde.csv" and c["n_test"] == 28
    sn = c["vs_snaive"]
    assert sn["ci95"][0] <= sn["skill_pct"] <= sn["ci95"][1]
    assert isinstance(sn["significant_5pct"], bool)
    assert c["mae_model_mw"] > 0 and c["mae_snaive_mw"] > 0
    # nichts persistiert: Challenge legt keinen Mandanten an
    assert client.get("/utilities").json()["utilities"] == []
    # zu wenig Testtage -> klare 422
    with open(REAL_CSV, "rb") as f:
        bad = client.post("/challenge", files={"file": ("k.csv", f, "text/csv")},
                          data={"unit": "kW", "ts_col": "Text", "load_col": "Reihe1",
                                "n_test": "400"})
    assert bad.status_code == 200  # Cap greift: 400 angefragt -> auf Datenlage gekappt
    assert bad.json()["n_test"] < 400
