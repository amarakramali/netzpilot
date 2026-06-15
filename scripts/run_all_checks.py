# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Konsolidierter Check-Lauf (dep-freier Pfad): alle Tests + Integrations-Checks. Exit!=0 bei Fehler."""
import sys, os, types, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class _R:
    def __init__(s, e): s.e = e
    def __enter__(s): return s
    def __exit__(s, t, v, tb): assert t and issubclass(t, s.e); return True
class _Mark:
    @staticmethod
    def skipif(condition, reason=None):
        def deco(fn):
            if not condition:
                return fn
            def skipped(*args, **kwargs):
                return None
            return skipped
        return deco
sys.modules['pytest'] = types.SimpleNamespace(raises=lambda e: _R(e), mark=_Mark())
TEST_MODS = ["tests.test_metrics", "tests.test_data_integrity", "tests.test_leakage", "tests.test_control",
             "tests.test_features_weather", "tests.test_economics", "tests.test_robust", "tests.test_forecast",
             "tests.test_horizon", "tests.test_intraday", "tests.test_holiday_overrides",
             "tests.test_smallutility", "tests.test_cqr", "tests.test_pilot_in_a_box",
             "tests.test_v1_significance", "tests.test_results_json",
             "tests.test_t9_significance", "tests.test_t10_small_utility", "tests.test_t11_city_weather",
             "tests.test_t12_weather_lift", "tests.test_t13_weather_lift", "tests.test_t16_weather_lift",
             "tests.test_overload", "tests.test_validate", "tests.test_cvar", "tests.test_thermal",
             "tests.test_eebus_lpc", "tests.test_mehrmindermengen", "tests.test_vpp",
             "tests.test_reconcile", "tests.test_audit_ledger", "tests.test_compliance_14a",
             "tests.test_mscons",
             # 2026-06-04 nachgezogen (fixture-frei, shim-faehig):
             "tests.test_bilanzkreis", "tests.test_corpus_index", "tests.test_dispatch",
             "tests.test_generation_forecast", "tests.test_html_report", "tests.test_intl_benchmark",
             "tests.test_mc_savings", "tests.test_optimize_heterogen", "tests.test_redispatch",
             "tests.test_small_utility", "tests.test_tariff"]

# Fixture-basierte Module (neue Mechanismus-Tests + Service): brauchen ECHTES pytest.
# Im venv vorhanden -> volle Abdeckung; ohne pytest ehrlicher SKIP-Hinweis statt falscher FAILs.
PYTEST_MODS = ["tests/test_holiday_base.py", "tests/test_residual_feedback.py",
               "tests/test_coverage_calibration.py", "tests/test_asymmetric_calibration.py",
               "tests/test_forecast_store.py", "tests/test_drift.py", "tests/test_service.py"]
passed = failed = 0; fails = []
for m in TEST_MODS:
    try: mod = importlib.import_module(m)
    except Exception as e: failed += 1; fails.append((m, "IMPORT", repr(e)[:140])); continue
    for n in dir(mod):
        if n.startswith("test_"):
            try: getattr(mod, n)(); passed += 1
            except Exception as e: failed += 1; fails.append((m, n, repr(e)[:140]))
print(f"TESTS: {passed} passed, {failed} failed")
for f in fails: print("  FAIL", *f)
from netzpilot.data.smard import load_local_json
from netzpilot.features.build import to_daily, get_holidays
from netzpilot.eval.backtest import rolling_origin
from netzpilot.models.ridge_correction import RidgeCorrector
s = load_local_json("prognose_engine_v1/data/wk*.json"); load2d, days = to_daily(s)
hol = get_holidays(sorted({d.year for d in days}), "NW")
_, sm = rolling_origin(load2d, days, lambda: RidgeCorrector(10.0), holiday_set=hol)
mae = sm["metriken"]["model"]["MAE_MW"]
v1_ok = abs(mae - 1411.4) < 0.5
print(f"v1-Reproduktion MAE {mae} (Soll 1411.4):", "OK" if v1_ok else "ABWEICHUNG")

# --- Fixture-Tests via echtem pytest (Subprozess) ---
# WICHTIG: erst den Shim aus sys.modules entfernen — find_spec() stolpert sonst ueber das
# SimpleNamespace ohne __spec__ (ValueError, in venv beobachtet 2026-06-04).
sys.modules.pop("pytest", None)
import importlib.util, subprocess, shutil
pytest_rc = 0
try:
    _has_pytest = importlib.util.find_spec("pytest") is not None
except ValueError:
    _has_pytest = False
if _has_pytest:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", *PYTEST_MODS],
                       env={**os.environ, "PYTHONPATH": os.getcwd()})
    pytest_rc = r.returncode
    print(f"PYTEST (Fixture-Module): rc={pytest_rc}")
else:
    print(f"PYTEST: nicht installiert -> {len(PYTEST_MODS)} Fixture-Module UEBERSPRUNGEN "
          "(im venv laufen sie mit; pip install pytest)")

# --- Skript-Verifies (dep-frei, ohne Netz) ---
script_rc = 0
for sc in ["scripts/verify_fetch_smard.py"]:
    r = subprocess.run([sys.executable, sc])
    print(f"SCRIPT {sc}: rc={r.returncode}")
    script_rc |= r.returncode

# --- Cockpit-Node-Harness (falls node vorhanden) ---
node_rc = 0
if shutil.which("node"):
    r = subprocess.run(["node", "scripts/_cockpit_script_check.js"])
    node_rc = r.returncode
    print(f"NODE cockpit-harness: rc={node_rc}")
else:
    print("NODE: nicht installiert -> Cockpit-Harness UEBERSPRUNGEN")

print("== run_all_checks fertig ==")
sys.exit(0 if (failed == 0 and v1_ok and pytest_rc == 0 and script_rc == 0 and node_rc == 0) else 1)
