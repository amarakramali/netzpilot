#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""NetzPilot Smoke-Test — EIN Befehl, der vor jedem Termin den ganzen Stack grün/rot prüft.

Zweck (zwei Fliegen, eine Klappe):
  1) Integrationstest: spielt die komplette Pipeline auf dem Beispiel-Lastgang durch — Last-Prognose,
     Residuallast (aus Wetter), §14a-Engpass→Fahrplan→aEMT-Quittung, Economics, Persistenz, und die
     FastAPI-Routen (/health, /inspect, /forecast, /report, /history).
  2) Termin-Absicherung: vor einem Pilot-Termin EINMAL laufen lassen — ist alles grün,
     funktioniert die Live-Demo.

Läuft auf dem HOST im venv (braucht fastapi/httpx; im Sandbox ohne diese werden API-Checks sauber
übersprungen). Reproduzierbar, ohne Internet. Aufruf:
    python scripts/smoke_test.py
Exit-Code 0 = alles grün, 1 = mindestens ein Check rot.
"""
from __future__ import annotations
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REAL_CSV = "data_cache/real/Netzumsatz-Lastgang-2025.csv"
PASS, FAIL, SKIP = [], [], []


def ok(name):
    PASS.append(name); print(f"  [PASS] {name}")


def bad(name, err):
    FAIL.append((name, err)); print(f"  [FAIL] {name}: {err}")


def skip(name, why):
    SKIP.append(name); print(f"  [skip] {name}: {why}")


def _make_weather_csv(load_hourly):
    import numpy as np, pandas as pd
    idx = load_hourly.index
    h = idx.hour.to_numpy()
    ghi = np.maximum(np.sin((h - 6) / 12 * np.pi), 0) * 700
    wdf = pd.DataFrame({
        "Zeit": idx.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shortwave_radiation": ghi, "direct_radiation": ghi * 0.7,
        "temperature_2m": 12 + ghi / 100, "wind_speed_10m": 4.0, "wind_speed_100m": 7.0,
    })
    p = os.path.join(tempfile.mkdtemp(), "weather.csv")
    wdf.to_csv(p, index=False)
    return p


def main():
    print("=== NetzPilot Smoke-Test ===\n")
    if not os.path.exists(REAL_CSV):
        print(f"FEHLT: {REAL_CSV} — Beispieldaten nicht vorhanden."); sys.exit(2)

    # --- 1. Engine-/Runner-Pipeline (dep-frei: numpy/pandas) ---
    print("Pipeline (echte Engine):")
    try:
        from netzpilot.service.runner import run_forecast, _load2d_from_csv
        r = run_forecast(REAL_CSV, utility="SmokeSW", unit="kW", ts_col="Text", load_col="Reihe1",
                         congestion_threshold_mw=33.0, steuve_malo="DE0001234567890",
                         steuve_demands_kw=[1000.0, 800.0, 600.0], rolling_redispatch=True,
                         rebap_csv="data_cache/real/rebap_2024.csv",
                         spot_csv="data_cache/real/spot_da_2024.csv",
                         submit_to_aemt=True)
        assert len(r["forecast"]) == 24, "keine 24h-Prognose"
        assert all(h["p10"] <= h["p50"] <= h["p90"] for h in r["forecast"]), "Quantile nicht monoton"
        ok("Last-Prognose 24h, P10<=P50<=P90")
        assert r["congestion"] is not None and r["fahrplan"] is not None, "kein Engpass/Fahrplan bei 33 MW"
        ok("§14a-Engpass erkannt + Fahrplan erzeugt")
        assert r["aemt_ack"] and r["aemt_ack"]["status"] == "ACCEPTED", "aEMT-Quittung fehlt"
        ok(f"aEMT-Quittung ACCEPTED ({r['aemt_ack']['transmission_id'][:24]}…)")
        assert r["redispatch"] and r["redispatch"]["forecast_basis"] == "day_ahead_p50_static"
        assert r["redispatch"]["total_shed_kwh"] <= r["redispatch"]["naive_shed_kwh"]
        ok("Rollierender Re-Dispatch-Feld erzeugt")
        assert r["economics_expected"] and r["economics_expected"]["expected_eur_per_year"] >= 0
        assert (r["economics_expected"]["expected_eur_per_year"]
                <= r["economics_upper_bound"]["eur_per_year_point_median"]), "Erwartung > Upper-Bound"
        ok("Economics: erwartet <= Risiko-Band <= |reBAP|-Rand")
    except Exception as e:
        bad("Last/§14a/Economics-Pipeline", repr(e))
        _load2d_from_csv = None

    # --- 2. Residuallast aus Wetter ---
    print("\nResiduallast (Wetter -> Erzeugung):")
    try:
        from netzpilot.service.runner import run_forecast, _load2d_from_csv
        _l, _d, _m, load_hourly = _load2d_from_csv(REAL_CSV, ts_col="Text", load_col="Reihe1", unit="kW")
        wcsv = _make_weather_csv(load_hourly)
        rr = run_forecast(REAL_CSV, utility="SmokeSW", unit="kW", ts_col="Text", load_col="Reihe1",
                          congestion_threshold_mw=33.0, weather_csv=wcsv,
                          pv_capacity_mw=8.0, wind_capacity_mw=5.0)
        rf = rr["residual_forecast"]
        assert rf and rf["generation_source"] == "weather", "Residuallast nicht aus Wetter"
        assert rf["recent_mean_residual_mw"] < rr["recent_mean_load_mw"], "Residual >= Last"
        ok(f"Residuallast aus Wetter ({rf['recent_mean_residual_mw']} < {rr['recent_mean_load_mw']} MW)")
    except Exception as e:
        bad("Residuallast-aus-Wetter", repr(e))

    # --- 3. Persistenz/Store ---
    print("\nPersistenz:")
    try:
        from netzpilot.service.store import ForecastStore
        d = tempfile.mkdtemp()
        s = ForecastStore(d)
        s.save("X", {"utility": "X", "forecast_date": "2026-01-01", "forecast": []})
        assert s.latest("X")["forecast_date"] == "2026-01-01"
        assert "2026-01-01" in s.history("X")
        ok("Store save/latest/history roundtrip")
    except Exception as e:
        bad("Store", repr(e))

    # --- 4. FastAPI-Routen (nur wenn fastapi/httpx vorhanden) ---
    print("\nFastAPI-Routen:")
    try:
        import fastapi, httpx  # noqa: F401
    except Exception:
        skip("API-Routen (/health,/inspect,/forecast,/report,/history)", "fastapi/httpx nicht installiert (im venv ausführen)")
    else:
        try:
            from fastapi.testclient import TestClient
            os.environ["NETZPILOT_STORE"] = tempfile.mkdtemp()
            import importlib, netzpilot.service.app as appmod
            importlib.reload(appmod)
            c = TestClient(appmod.app)
            assert c.get("/health").json()["status"] == "ok"
            ok("/health")
            with open(REAL_CSV, "rb") as f:
                ins = c.post("/inspect", files={"file": ("h.csv", f, "text/csv")}, data={"unit": "kW"})
            assert ins.status_code == 200 and ins.json()["ts_col"], "inspect liefert keine ts_col"
            ok(f"/inspect (erkennt ts_col={ins.json()['ts_col']}, load={ins.json()['suggested_load_col']})")
            with open(REAL_CSV, "rb") as f:
                fc = c.post("/forecast", files={"file": ("h.csv", f, "text/csv")},
                            data={"utility": "SmokeSW", "unit": "kW", "ts_col": "Text",
                                  "load_col": "Reihe1", "congestion_threshold_mw": "33"})
            assert fc.status_code == 200 and len(fc.json()["forecast"]) == 24
            ok("/forecast (24h)")
            rep = c.get("/report/SmokeSW/latest")
            assert rep.status_code == 200 and "<!DOCTYPE html>" in rep.text
            ok("/report/{utility}/latest (HTML)")
            hist = c.get("/history/SmokeSW").json()
            assert "dates" in hist
            ok("/history/{utility}")
        except Exception as e:
            bad("FastAPI-Routen", repr(e))

    # --- Zusammenfassung ---
    print("\n=== Ergebnis ===")
    print(f"  PASS: {len(PASS)}   FAIL: {len(FAIL)}   skip: {len(SKIP)}")
    if FAIL:
        print("\nROT — vor dem Termin beheben:")
        for n, e in FAIL:
            print(f"  - {n}: {e}")
        sys.exit(1)
    print("\nOK: ALLES GRUEN - Demo ist termin-bereit." + (f" ({len(SKIP)} API-Checks uebersprungen - im venv erneut laufen lassen)" if SKIP else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
