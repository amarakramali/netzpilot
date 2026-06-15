# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import csv
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from netzpilot.data.rebap import load_rebap
from netzpilot.eval.economics import (
    ausgleichsenergie_saving_eur,
    labor_saving_eur,
    rebap_spread_stats,
    saving_from_real_rebap,
    saving_from_rebap_spot,
    scenarios,
    spread_over_spot_stats,
)

def test_saving_basic():
    # 1 MW Dauerreduktion * 8760 h * 10 EUR/MWh = 87.600 EUR
    assert ausgleichsenergie_saving_eur(1.0, 10.0) == 87600.0

def test_scenarios_values():
    sc = scenarios(0.15)
    assert sc["spread_5_eur_mwh"] == round(0.15 * 8760 * 5)    # 6570
    assert sc["spread_15_eur_mwh"] == round(0.15 * 8760 * 15)  # 19710

def test_labor():
    assert labor_saving_eur(6, 65) == 20280.0

def test_negative_raises():
    with pytest.raises(ValueError):
        ausgleichsenergie_saving_eur(-0.1, 10.0)
    with pytest.raises(ValueError):
        ausgleichsenergie_saving_eur(0.1, -10.0)


def test_real_rebap_stats_drop_nan_and_are_monotone():
    stats = rebap_spread_stats([float("nan"), None, -10.0, 20.0, 40.0, float("inf"), "x"])
    assert stats["n"] == 3
    assert stats["p25_eur_mwh"] <= stats["median_abs_spread_eur_mwh"] <= stats["p75_eur_mwh"]
    saving = saving_from_real_rebap(0.5, [-10.0, 20.0, 40.0])
    assert saving["eur_per_year_p25"] <= saving["eur_per_year_point_median"] <= saving["eur_per_year_p75"]


def test_load_rebap_official_and_normalized_formats():
    with TemporaryDirectory() as tmp:
        official = Path(tmp) / "official.csv"
        official.write_text(
            "Datum;Zeitzone;von;bis;Einheit;reBAP unterdeckt;reBAP ueberdeckt\n"
            "01.01.2024;CET;00:00;00:15;EUR/MWh;75,15;75,15\n"
            "01.01.2024;CET;00:15;00:30;EUR/MWh;-1.234,56;-1.234,56\n",
            encoding="utf-8",
        )
        assert load_rebap(official) == [75.15, -1234.56]

        normalized = Path(tmp) / "normalized.csv"
        normalized.write_text("Zeit;reBAP_EUR_MWh\n2024-01-01T00:00:00+01:00;75.15\n", encoding="utf-8")
        assert load_rebap(normalized) == [75.15]


def test_load_rebap_real_2024_file_if_present():
    path = Path("data_cache/real/rebap_2024.csv")
    if not path.exists():
        return
    vals = load_rebap(path)
    assert 35000 <= len(vals) <= 35200
    assert any(v < 0 for v in vals)
    assert any(v > 0 for v in vals)
    assert any(abs(v) > 1000 for v in vals)


def test_spread_over_spot_stats_drops_nan_and_is_monotone():
    rebap = [float("nan"), None, 100.0, 50.0, -20.0, float("inf"), "x", 80.0]
    spot = [60.0, 60.0, 60.0, 40.0, 30.0, 60.0, 60.0, 100.0]
    # gueltige Paare: (100,60)|=40, (50,40)|=10, (-20,30)|=50, (80,100)|=20 -> n=4
    st = spread_over_spot_stats(rebap, spot)
    assert st["n"] == 4
    assert st["p25_eur_mwh"] <= st["median_abs_spread_over_spot_eur_mwh"] <= st["p75_eur_mwh"]
    assert st["median_abs_spread_over_spot_eur_mwh"] <= st["p95_eur_mwh"]


def test_spread_over_spot_length_mismatch_raises():
    with pytest.raises(ValueError):
        spread_over_spot_stats([1.0, 2.0, 3.0], [1.0, 2.0])


def test_spread_over_spot_all_nonfinite_raises():
    with pytest.raises(ValueError):
        spread_over_spot_stats([float("nan"), None], [1.0, 2.0])


def test_saving_from_rebap_spot_band_monotone():
    rebap = [100.0, 50.0, -20.0, 80.0]
    spot = [60.0, 40.0, 30.0, 100.0]
    saving = saving_from_rebap_spot(0.5, rebap, spot)
    assert saving["eur_per_year_p25"] <= saving["eur_per_year_point_median"] <= saving["eur_per_year_p75"]
    assert saving["delta_mae_mw"] == 0.5
    assert "spread_over_spot_stats" in saving


def test_real_2024_aufschlag_plausibility_if_files_present():
    """Sanity gegen die T20-Spec: |reBAP-Spot|-Median << |reBAP|-Median;
    Mean(|reBAP-Spot|) in plausibler Range fuer das volatile 2024-Jahr."""
    rebap_path = Path("data_cache/real/rebap_2024.csv")
    spot_path = Path("data_cache/real/spot_da_2024.csv")
    if not (rebap_path.exists() and spot_path.exists()):
        return
    rebap, spot = [], []
    with rebap_path.open(encoding="utf-8") as f:
        r = csv.reader(f, delimiter=";"); next(r)
        for row in r:
            try:
                rebap.append(float(row[1]))
            except (ValueError, IndexError):
                rebap.append(float("nan"))
    with spot_path.open(encoding="utf-8") as f:
        r = csv.reader(f, delimiter=";"); next(r)
        for row in r:
            spot.append(float(row[1]) if row[1] else float("nan"))
    assert len(rebap) == len(spot) >= 35000
    st_spread = spread_over_spot_stats(rebap, spot)
    # Echte 2024-Daten: Aufschlag < |reBAP| absolut (gespart wird nur der Aufschlag)
    finite_rebap = [v for v in rebap if v == v and abs(v) != float("inf")]
    st_abs = rebap_spread_stats(finite_rebap)
    assert st_spread["median_abs_spread_over_spot_eur_mwh"] < st_abs["median_abs_spread_eur_mwh"]
    # Volatiles 2024: |Spread|-Mean ~50-110 EUR/MWh; nie <2 (waere Marktphase ohne Stress)
    assert 20.0 <= st_spread["mean_abs_spread_over_spot_eur_mwh"] <= 120.0


def test_pilot_in_a_box_imports_aufschlag_function():
    """Sicherheitsnetz: pilot_in_a_box importiert die Aufschlag-Funktion (Headline-Pfad)."""
    import scripts.pilot_in_a_box as pib
    assert hasattr(pib, "saving_from_rebap_spot")
