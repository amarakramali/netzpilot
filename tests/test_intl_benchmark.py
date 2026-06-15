# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

import json
from pathlib import Path


def test_t30_intl_files_are_separate_from_real_corpus():
    intl = Path("data_cache/intl")
    if not intl.exists():
        return
    for code in ["de", "nl", "at", "ch", "fr"]:
        path = intl / f"entsoe_{code}_2024.csv"
        assert path.exists()
        assert path.read_text(encoding="utf-8").splitlines()[0].startswith("timestamp_utc,load_mw")
    assert not any(Path("data_cache/real").glob("entsoe_*_2024.csv"))


def test_t30_intl_benchmark_is_labelled_tso_not_dso():
    path = Path("data_cache/intl/intl_benchmark.json")
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["n_ok"] == 5
    assert result["n_signifikant_vs_snaive_5pct"] >= 1
    assert "national aggregate TSO load" in result["strict_scope"]
    assert all("distribution-network" in r["data_label"] for r in result["results"] if "error" not in r)


def test_t30_intl_sources_exist():
    path = Path("data_cache/intl/SOURCES.md")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    assert "ENTSO-E" in text
    assert "not distribution-network data" in text
