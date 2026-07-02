# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

from pathlib import Path
import zipfile

from scripts.fetch_dso_corpus import core_specs, extended_specs, extract_csv


def test_core_catalogue_covers_existing_german_corpus_files():
    specs = core_specs()
    targets = {Path(s.target).name for s in specs}
    assert "Netzumsatz-Lastgang-2025.csv" in targets
    assert "herne_bezug_vorgelagerte_ebene_2024.csv" in targets
    assert "evdb_lastgang_ns_2024.csv" in targets
    assert "waren_2026_03_27_LGL_Strom_2025_Waren.csv" in targets
    assert "hilden_jhl_ms_2025.csv" not in targets
    assert len(targets) == len(specs)
    assert all(s.source_set == "core" for s in specs)


def test_extended_catalogue_is_unique_and_official_https():
    specs = extended_specs()
    targets = [s.target for s in specs]
    keys = [s.key for s in specs]
    assert len(targets) == len(set(targets))
    assert len(keys) == len(set(keys))
    assert all(s.url.startswith("https://") for s in specs)
    assert all(Path(s.target).is_relative_to(Path("data_cache/real")) for s in specs)


def test_extended_catalogue_contains_priority_sources():
    keys = {s.key for s in extended_specs()}
    assert "hilden_jhl_ms_2025" in keys
    assert "hilden_einspeisung_ns_2025" in keys
    assert "evdb_bezug_ms_2025" in keys
    assert "neuruppin_lgl_2020" in keys
    assert "bitterfeld_entnahme_ns_2024" in keys
    assert "ten_gesamtlast_2025" in keys
    assert "neusw_einspeisung_ms_2025" in keys
    assert "passau_jhl_ms_2025" in keys


def test_extract_csv_merges_quarter_archives(tmp_path):
    archive = tmp_path / "quarters.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for quarter, date in ((1, "01.01.2025"), (2, "01.04.2025")):
            zf.writestr(
                f"series-Q{quarter}.csv",
                "Netzbetreibername:;Test;;\n"
                f"Betrachtungszeitraum:;{date};bis;31.12.2025\n"
                "Datum;von;bis;Wert\n"
                f"{date};00:00;00:15;{quarter}\n",
            )
    target = tmp_path / "annual.csv"
    members = extract_csv(archive, target)
    text = target.read_text(encoding="utf-8")
    assert "series-Q1.csv" in members and "series-Q2.csv" in members
    assert text.count("Datum;von;bis;Wert") == 1
    assert "01.01.2025;00:00;00:15;1" in text
    assert "01.04.2025;00:00;00:15;2" in text
