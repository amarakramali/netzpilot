# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Checks for real DSO CSV ingestion used by Pilot-in-a-Box."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scripts.pilot_in_a_box import _to_num, inspect_load_columns, robust_load_csv


def test_pilot_loader_reads_hilden_style_weekday_csv():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "hilden.csv"
        rows = ["Text;X;Reihe1"]
        for h in range(8):
            rows.append(f"01.01.2025 {h:02d}:15 Mi;{h + 1};1.234,5")
        path.write_text("\n".join(rows), encoding="utf-8")

        hourly, ts_col, load_col = robust_load_csv(path, ts_col="Text", load_col="Reihe1", unit="kW")

    assert ts_col == "Text"
    assert load_col == "Reihe1"
    assert len(hourly) == 8
    assert np.isclose(float(hourly.iloc[0]), 1.2345)


def test_pilot_loader_reads_eam_metadata_csv_with_split_timestamp():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "eam.csv"
        rows = [
            "Netzbetreibername:;;EAM Netz GmbH;",
            "Einheit:;;kW;",
            ";;;",
            "Datum;von;bis;P (kW)",
        ]
        for h in range(8):
            rows.append(f"01.01.2024;{h:02d}:00:00;{h:02d}:15:00;24.147")
        path.write_text("\n".join(rows), encoding="utf-8")

        hourly, ts_col, load_col = robust_load_csv(path, ts_col="Datum+von", load_col="P (kW)", unit="kW")

    assert ts_col == "Datum+von"
    assert load_col == "P (kW)"
    assert len(hourly) == 8
    assert np.isclose(float(hourly.iloc[0]), 24.147)


def test_pilot_loader_parses_german_two_digit_year_dayfirst():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "dmy_yy.csv"
        rows = [
            "Netzbetreibername:;;Example;",
            "Einheit:;;kW;",
            ";;;",
            "Datum;von;bis;Wert",
            "1.2.22;00:00;00:15;1.000",
            "1.2.22;00:15;00:30;1.100",
            "2.1.22;00:00;00:15;2.000",
            "2.1.22;00:15;00:30;2.100",
        ]
        path.write_text("\n".join(rows), encoding="utf-8")

        hourly, ts_col, load_col = robust_load_csv(path, unit="kW")

    assert ts_col == "Datum+von"
    assert load_col == "Wert"
    local_index = hourly.index.tz_convert("Europe/Berlin")
    assert str(local_index.min().date()) == "2022-01-02"
    assert str(local_index.max().date()) == "2022-02-01"


def test_pilot_loader_reads_evdb_wert_header_csv():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "evdb.csv"
        rows = [
            "Netzbetreibername:;Energieversorgung Dahlenburg-Bleckede AG;;",
            "Einheit:;kW;;",
            ";;;",
            "Datum;von;bis;Wert",
        ]
        for h in range(8):
            rows.append(f"01.01.2024;{h:02d}:00;{h:02d}:15;1.234,{h}")
        path.write_text("\n".join(rows), encoding="utf-8")

        hourly, ts_col, load_col = robust_load_csv(path, unit="kW")

    assert ts_col == "Datum+von"
    assert load_col == "Wert"
    assert len(hourly) == 8
    assert np.isclose(float(hourly.iloc[0]), 1.234)


def test_pilot_loader_reads_herne_headerless_split_timestamp_csv():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "herne.csv"
        rows = [
            "§23c Abs. 3 Nr. 5 EnWG;;;;;;",
            "Bezug aus der vorgelagerten Ebene 2024 - Netzgebiet Herne;;;;;;",
            ";;;110/10kV;10kV;10/0,4kV;0,4kV",
            ";;Summe in kWh:;432.637.341;424.760.020;273.686.471;269.257.428",
            ";;;kW;kW;kW;kW",
        ]
        vals = [305, 444, 320, 611, 501, 487, 703, 599]
        for h, v in enumerate(vals):
            rows.append(
                f"01.01.2024;{h:02d}:00;{h:02d}:15;"
                f"39.{v:03d};38.{((436 + v) % 900) + 50:03d};"
                f"29.{((79 + v) % 900) + 50:03d};28.{((628 + v) % 900) + 50:03d}"
            )
        path.write_text("\n".join(rows), encoding="utf-8")

        hourly, ts_col, load_col = robust_load_csv(path, load_col="Load_1", unit="kW")

    assert ts_col == "Datum+von"
    assert load_col == "Load_1"
    assert len(hourly) == 8
    assert np.isclose(float(hourly.iloc[0]), 39.305)


def test_pilot_loader_rejects_ambiguous_multi_load_csv_without_load_col():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "multi.csv"
        rows = ["Datum;von;bis;A;B"]
        vals_a = [10, 12, 11, 13, 12, 14, 13, 15]
        vals_b = [20, 21, 19, 22, 20, 23, 21, 24]
        for h, (a, b) in enumerate(zip(vals_a, vals_b)):
            rows.append(f"01.01.2024;{h:02d}:00;{h:02d}:15;{a};{b}")
        path.write_text("\n".join(rows), encoding="utf-8")

        try:
            robust_load_csv(path, unit="MW")
        except SystemExit as exc:
            msg = str(exc)
        else:
            raise AssertionError("ambiguous multi-load CSV did not abort")

    assert "Mehrere plausible Lastspalten" in msg
    assert "--load-col" in msg
    assert "A:" in msg
    assert "B:" in msg


def test_pilot_loader_multi_load_csv_is_deterministic_with_load_col():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "multi.csv"
        rows = ["Datum;von;bis;A;B"]
        for h in range(8):
            rows.append(f"01.01.2024;{h:02d}:00;{h:02d}:15;{10 + h};{20 + h}")
        path.write_text("\n".join(rows), encoding="utf-8")

        hourly, ts_col, load_col = robust_load_csv(path, load_col="B", unit="MW")

    assert ts_col == "Datum+von"
    assert load_col == "B"
    assert len(hourly) == 8
    assert np.isclose(float(hourly.iloc[0]), 20.0)


def test_list_columns_reports_headerless_levels_and_units():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "herne.csv"
        rows = [
            "§23c Abs. 3 Nr. 5 EnWG;;;;;;",
            "Bezug aus der vorgelagerten Ebene 2024 - Netzgebiet Herne;;;;;;",
            ";;;110/10kV;10kV;10/0,4kV;0,4kV",
            ";;;kW;kW;kW;kW",
        ]
        vals = [30, 34, 31, 36, 33, 38, 35, 37]
        for h, v in enumerate(vals):
            rows.append(f"01.01.2024;{h:02d}:00;{h:02d}:15;{v};{20 + (v % 7)};{10 + (v % 5)};{5 + (v % 3)}")
        path.write_text("\n".join(rows), encoding="utf-8")

        info = inspect_load_columns(path, unit="kW")

    assert info["ts_col"] == "Datum+von"
    candidates = {c["name"]: c for c in info["load_candidates"]}
    assert candidates["Load_1"]["level"] == "110/10kV"
    assert candidates["Load_4"]["level"] == "0,4kV"
    assert candidates["Load_1"]["unit_hint"] == "kW"


def test_to_num_handles_decimal_comma_and_thousands_dot():
    vals = _to_num(pd.Series(["1.234,5", "-979,994", "24.147", "38"]))
    assert np.allclose(vals.to_numpy(dtype=float), [1234.5, -979.994, 24147.0, 38.0])
