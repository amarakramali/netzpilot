#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Build the real public DSO corpus registry for T28.

The registry is the machine-readable source of truth for:
- which public DSO series are counted as real corpus rows,
- which column/voltage level is explicitly pinned,
- which rows are safe to use for the load-normalized pool prior.

It validates every row through the same robust loader as Pilot-in-a-Box before writing
``data_cache/real/corpus_index.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.features.build import to_daily_local
from scripts.pilot_in_a_box import robust_load_csv

REAL = "data_cache/real"
NETWORK_REPORT = "data_cache/benchmark/network_independence.md"
CORRELATION_THRESHOLD = 0.98
MIN_CORRELATION_OVERLAP_DAYS = 300
MIN_CORRELATION_OVERLAP_HOURS = MIN_CORRELATION_OVERLAP_DAYS * 24


def _p(name: str) -> str:
    return f"{REAL}/{name}"


def _entry(key, name, csv, col, unit, operator, series, year, source_url, legal_basis,
           ts=None, level=None, signed=False, kind="load", official_page=None,
           source_archive=None, include_in_benchmark=True, include_in_pool=None,
           country="DE", network_kind="dso_real"):
    pool_default = bool(
        (not signed)
        and network_kind == "dso_real"
        and kind in {"load", "bezug", "jhl"}
    )
    return {
        "key": key,
        "name": name,
        "path": csv,
        "csv": csv,
        "ts": ts,
        "col": col,
        "unit": unit,
        "operator": operator,
        "series": series,
        "year": year,
        "level": level,
        "legal_basis": legal_basis,
        "source_url": source_url,
        "official_page": official_page,
        "source_archive": source_archive,
        "country": country,
        "network_kind": network_kind,
        "signed": bool(signed),
        "kind": kind,
        "include_in_benchmark": bool(include_in_benchmark),
        "include_in_pool": bool(pool_default if include_in_pool is None else include_in_pool),
        "data_label": "signiert" if signed else ("profile_sum" if kind == "profile_sum" else "echte Messdaten"),
    }


HILDEN_PAGE = "https://stadtwerke-hilden.de/netzregulierung/veroeffentlichungspflichten-strom/"
EAM_PAGE = "https://www.eam-netz.de/ueber-uns/netzinformationen/veroeffentlichungspflichten/strom/-12-abs-3-stromnzv/"
HERNE_PAGE = "https://www.stadtwerke-herne.de/netze/stromnetz/veroeffentlichungspflichten"
EVDB_PAGE = "https://www.evdbag.de/netzbetrieb/veroeffentlichungen/"
NEURUPPIN_PAGE = "https://www.swn.de/pflichten-veroeffentlichungen-netze.html"
BITTERFELD_PAGE = "https://netz-bitterfeld-wolfen.de/veroeffentlichungspflichten-2023-copy/articles/veroeffentlichungspflichten-2024.html"
WAREN_PAGE = "https://stadtwerke-waren.de/stromnetz-veroeffentlichungspflicht/"
TEN_PAGE = "https://www.thueringer-energienetze.com/Ueber_uns/Veroeffentlichungspflichten/Netzdaten"
NEUSW_PAGE = "https://www.neu-sw.de/netze/unsere-netze/stromnetz"
PASSAU_PAGE = "https://netze.stadtwerke-passau.de/strom/netzveroeffentlichungen.html"
ENEDIS_PAGE = "https://opendata.enedis.fr/datasets/conso-inf36-region"
ECO2MIX_PAGE = "https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/"

CORPUS_SPECS = [
    _entry("hilden_slp_2025", "Stadtwerke Hilden - SLP-Summenlast 2025",
           _p("SLP-Summenlast-Lastgang-2025.csv"), "Reihe1", "kW", "Stadtwerke Hilden",
           "SLP-Summenlast", 2025,
           "https://stadtwerke-hilden.de/uploads/Netz/Ver%C3%B6ffentlichungspflichten-Strom/SLP-Summenlast-Lastgang-2025.csv",
           "EnWG / StromNZV Veroeffentlichungspflichten", ts="Text", kind="profile_sum",
           official_page=HILDEN_PAGE),
    _entry("hilden_netzumsatz_2025", "Stadtwerke Hilden - Netzumsatz 2025",
           _p("Netzumsatz-Lastgang-2025.csv"), "Reihe1", "kW", "Stadtwerke Hilden",
           "Netzumsatz", 2025,
           "https://stadtwerke-hilden.de/uploads/Netz/Ver%C3%B6ffentlichungspflichten-Strom/Netzumsatz-Lastgang-2025.csv",
           "EnWG / StromNZV Veroeffentlichungspflichten", ts="Text", official_page=HILDEN_PAGE),
    _entry("hilden_dba_2025", "Stadtwerke Hilden - DBA 2025",
           _p("DBA-Lastgang-2025.csv"), "Reihe1", "kW", "Stadtwerke Hilden",
           "DBA / Differenzbilanzkreis", 2025,
           "https://stadtwerke-hilden.de/uploads/Netz/Ver%C3%B6ffentlichungspflichten-Strom/DBA-Lastgang-2025.csv",
           "EnWG / StromNZV Veroeffentlichungspflichten", ts="Text", signed=True,
           kind="differenzbilanz", official_page=HILDEN_PAGE),
]

for bg in (1, 2, 3):
    CORPUS_SPECS.append(_entry(
        f"eam_bg{bg}_2024", f"EAM Netz - Differenzbilanzierung BG{bg} 2024",
        _p(f"StromNZV_Para_12_3_Nr3_Ergebnis_der_Differenzbilanzierung_BG_{bg}_2024.csv"),
        "P (kW)", "kW", "EAM Netz", f"Differenzbilanzierung BG {bg}", 2024,
        f"https://www.eam-netz.de/fileadmin/user_upload/Fuer_Partner/Netzinformationen/Veroeffentlichungspflichten/Strom/12Abs3StromNZV/StromNZV_Para_12_3_Nr3_Ergebnis_der_Differenzbilanzierung_BG_{bg}_2024.csv",
        "Paragraph 12 Abs. 3 Nr. 3 StromNZV", signed=True, kind="differenzbilanz",
        official_page=EAM_PAGE))

CORPUS_SPECS += [
    _entry("herne_bezug_110_10kv_2024", "Stadtwerke Herne - Bezug 110/10 kV 2024",
           _p("herne_bezug_vorgelagerte_ebene_2024.csv"), "Load_1", "kW", "Stadtwerke Herne",
           "Bezug vorgelagerte Ebene", 2024,
           "https://www.stadtwerke-herne.de/PDFs/Netze/stromnetz/veroeffentlichungspflichten/Lastgangver%C3%B6ffentlichung%202024/Bezug%20vorgelagerte%20Ebene_2024-Netzgebiet_Herne.csv",
           "Paragraph 23c Abs. 3 Nr. 5 EnWG", level="110/10 kV", kind="bezug",
           official_page=HERNE_PAGE),
    _entry("herne_bezug_10kv_2024", "Stadtwerke Herne - Bezug 10 kV 2024",
           _p("herne_bezug_vorgelagerte_ebene_2024.csv"), "Load_2", "kW", "Stadtwerke Herne",
           "Bezug vorgelagerte Ebene", 2024,
           "https://www.stadtwerke-herne.de/PDFs/Netze/stromnetz/veroeffentlichungspflichten/Lastgangver%C3%B6ffentlichung%202024/Bezug%20vorgelagerte%20Ebene_2024-Netzgebiet_Herne.csv",
           "Paragraph 23c Abs. 3 Nr. 5 EnWG", level="10 kV", kind="bezug", official_page=HERNE_PAGE),
    _entry("herne_bezug_10_04kv_2024", "Stadtwerke Herne - Bezug 10/0.4 kV 2024",
           _p("herne_bezug_vorgelagerte_ebene_2024.csv"), "Load_3", "kW", "Stadtwerke Herne",
           "Bezug vorgelagerte Ebene", 2024,
           "https://www.stadtwerke-herne.de/PDFs/Netze/stromnetz/veroeffentlichungspflichten/Lastgangver%C3%B6ffentlichung%202024/Bezug%20vorgelagerte%20Ebene_2024-Netzgebiet_Herne.csv",
           "Paragraph 23c Abs. 3 Nr. 5 EnWG", level="10/0.4 kV", kind="bezug", official_page=HERNE_PAGE),
    _entry("herne_bezug_04kv_2024", "Stadtwerke Herne - Bezug 0.4 kV 2024",
           _p("herne_bezug_vorgelagerte_ebene_2024.csv"), "Load_4", "kW", "Stadtwerke Herne",
           "Bezug vorgelagerte Ebene", 2024,
           "https://www.stadtwerke-herne.de/PDFs/Netze/stromnetz/veroeffentlichungspflichten/Lastgangver%C3%B6ffentlichung%202024/Bezug%20vorgelagerte%20Ebene_2024-Netzgebiet_Herne.csv",
           "Paragraph 23c Abs. 3 Nr. 5 EnWG", level="0.4 kV", kind="bezug", official_page=HERNE_PAGE),
    _entry("herne_differenz_2024", "Stadtwerke Herne - Differenzbilanzierung 2024",
           _p("herne_differenzbilanzierung_2024.csv"), "Load_1", "kW", "Stadtwerke Herne",
           "Differenzbilanzierung", 2024,
           "https://www.stadtwerke-herne.de/PDFs/Netze/stromnetz/veroeffentlichungspflichten/Lastgangver%C3%B6ffentlichung%202024/Differenzbilanzierung_2024-Netzgebiet_Herne.csv",
           "Paragraph 12 Abs. 3 Nr. 3 StromNZV", signed=True, kind="differenzbilanz",
           official_page=HERNE_PAGE),
    _entry("evdb_differenz_2024", "EVDB - Differenzbilanzkreis 2024",
           _p("evdb_differenzbilanzkreis_2024.csv"), "Wert", "kW", "Energieversorgung Dahlenburg-Bleckede AG",
           "Differenzbilanzkreis", 2024,
           "https://www.evdbag.de/wp-content/uploads/2025/05/lfd.-Nr.-10-StromNZV-%C2%A7-12-Abs.-3-Satz-3.csv",
           "Paragraph 12 Abs. 3 Satz 3 StromNZV", signed=True, kind="differenzbilanz",
           official_page=EVDB_PAGE),
    _entry("evdb_ms_2024", "EVDB - Lastgang MS 2024",
           _p("evdb_lastgang_ms_2024.csv"), "Wert", "kW", "Energieversorgung Dahlenburg-Bleckede AG",
           "Lastgang MS", 2024,
           "https://www.evdbag.de/wp-content/uploads/2025/05/lfd.-Nr.-22-EnWG-%C2%A7-23c-Abs.-3-Nr.-1-MS.csv",
           "Paragraph 23c Abs. 3 Nr. 1 EnWG", level="MS", kind="jhl", official_page=EVDB_PAGE),
    _entry("evdb_ns_2024", "EVDB - Lastgang NS 2024",
           _p("evdb_lastgang_ns_2024.csv"), "Wert", "kW", "Energieversorgung Dahlenburg-Bleckede AG",
           "Lastgang NS", 2024,
           "https://www.evdbag.de/wp-content/uploads/2025/05/lfd.-Nr.-22-EnWG-%C2%A7-23c-Abs.-3-Nr.-1-NS.csv",
           "Paragraph 23c Abs. 3 Nr. 1 EnWG", level="NS", kind="jhl", official_page=EVDB_PAGE),
    _entry("neuruppin_ms_2022", "Stadtwerke Neuruppin - Lastverlauf MS 2022",
           _p("neuruppin_lgl_strom_2022.csv"), "Wert.11", "kW", "Stadtwerke Neuruppin",
           "Lastverlauf Jahreshoechstlast MS", 2022,
           "https://www.swn.de/fileadmin/Dateien/strom/netze_downloads/Pflichten/Lastgangsdaten/2023_04_18_LGL_Strom_2022_Neuruppin.csv",
           "Paragraph 23c Abs. 3 EnWG", level="MS", kind="jhl", official_page=NEURUPPIN_PAGE),
    _entry("neuruppin_ns_2022", "Stadtwerke Neuruppin - Lastverlauf NS 2022",
           _p("neuruppin_lgl_strom_2022.csv"), "Wert.13", "kW", "Stadtwerke Neuruppin",
           "Lastverlauf Jahreshoechstlast NS", 2022,
           "https://www.swn.de/fileadmin/Dateien/strom/netze_downloads/Pflichten/Lastgangsdaten/2023_04_18_LGL_Strom_2022_Neuruppin.csv",
           "Paragraph 23c Abs. 3 EnWG", level="NS", kind="jhl", official_page=NEURUPPIN_PAGE),
]

for level, fname in [("MS", "bitterfeld_jhl_ms_2024.csv"), ("MS/NS", "bitterfeld_jhl_msns_2024.csv"), ("NS", "bitterfeld_jhl_ns_2024.csv")]:
    slug = level.lower().replace("/", "")
    CORPUS_SPECS.append(_entry(
        f"bitterfeld_{slug}_2024", f"NG Bitterfeld-Wolfen - Lastverlauf {level} 2024",
        _p(fname), "Wert", "kW", "NG Bitterfeld-Wolfen", f"Lastverlauf Jahreshoechstlast {level}",
        2024, f"https://netz-bitterfeld-wolfen.de/files/ngbw/nbStrom/Veroeffentlichungen/Veroeffentlichungspflichten_csv/2024/JHL_{level.replace('/', '')}.csv",
        "Paragraph 23c Abs. 3 Nr. 1 EnWG", level=level, kind="jhl", official_page=BITTERFELD_PAGE))

CORPUS_SPECS += [
    _entry("waren_bezug_ms_2025", "Stadtwerke Waren - Bezug vNB MS 2025",
           _p("waren_2026_03_27_LGL_Strom_2025_Waren.csv"), "Wert.2", "kW", "Stadtwerke Waren",
           "Bezug vorgelagerte Netzebene MS", 2025,
           "https://stadtwerke-waren.de/files_PDF/2026_03_27%20LGL%20Strom%202025%20Waren.zip",
           "Paragraph 23c Abs. 3 EnWG", level="MS", kind="bezug", official_page=WAREN_PAGE,
           source_archive=_p("waren_lgl_strom_2025_p23c.zip")),
    _entry("waren_jhl_ms_2025", "Stadtwerke Waren - Lastverlauf MS 2025",
           _p("waren_2026_03_27_LGL_Strom_2025_Waren.csv"), "Wert.10", "kW", "Stadtwerke Waren",
           "Lastverlauf Jahreshoechstlast MS", 2025,
           "https://stadtwerke-waren.de/files_PDF/2026_03_27%20LGL%20Strom%202025%20Waren.zip",
           "Paragraph 23c Abs. 3 EnWG", level="MS", kind="jhl", official_page=WAREN_PAGE,
           source_archive=_p("waren_lgl_strom_2025_p23c.zip")),
    _entry("waren_differenz_2025", "Stadtwerke Waren - Differenzbilanzierung 2025",
           _p("waren_2026_03_31_LGL-Strom_2025_Waren_p12.csv"), "Wert", "kW", "Stadtwerke Waren",
           "Differenzbilanzierung", 2025,
           "https://stadtwerke-waren.de/files_PDF/2026_03_31%20LGL-Strom_2025%20Waren_p12.zip",
           "Paragraph 12 Abs. 3 StromNZV", level="Netzgebiet", signed=True, kind="differenzbilanz",
           official_page=WAREN_PAGE, source_archive=_p("waren_lgl_strom_2025_p12.zip")),
]

for series_code, series_name, nr, kind in [
    ("jhl", "Lastverlauf Jahreshoechstlast", "1", "jhl"),
    ("bezug", "Bezug vorgelagerte Ebene", "5", "bezug"),
]:
    for level in ["HS", "HSU", "MS", "MSU", "NS"]:
        CORPUS_SPECS.append(_entry(
            f"ten_{series_code}_{level.lower()}_2025",
            f"TEN Thueringer Energienetze - {series_name} {level} 2025",
            _p(f"ten_{series_code}_{level.lower()}_2025.csv"), "Wert", "kW",
            "TEN Thueringer Energienetze", f"{series_name} {level}", 2025,
            f"https://www.thueringer-energienetze.com/Content/Documents/Ueber_uns/p_23c_3-{nr}_{level}_2025.zip",
            f"Paragraph 23c Abs. 3 Nr. {nr} EnWG", level=level, kind=kind,
            official_page=TEN_PAGE, source_archive=_p(f"ten_{series_code}_{level.lower()}_2025.zip")))

for series_code, series_name, nr, kind in [
    ("jhl", "Lastverlauf Jahreshoechstlast", "1", "jhl"),
    ("bezug", "Bezug vorgelagerte Ebene", "5", "bezug"),
]:
    for level in ["hsms", "msns", "ms", "ns"]:
        level_label = level.upper().replace("HSMS", "HS/MS").replace("MSNS", "MS/NS")
        CORPUS_SPECS.append(_entry(
            f"neusw_{series_code}_{level}_2025",
            f"neu.sw Neubrandenburg - {series_name} {level_label} 2025",
            _p(f"neusw_{series_code}_{level}_2025.csv"), "Unnamed: 3", "kW",
            "Neubrandenburger Stadtwerke", f"{series_name} {level_label}", 2025,
            f"https://www.neu-sw.de/downloads/netze/stromnetz/veroeffentlichungspflichten/netzstrukturdaten/enwg_23c_3_nr_{nr}_{level}.csv",
            f"Paragraph 23c Abs. 3 Nr. {nr} EnWG", level=level_label, kind=kind,
            official_page=NEUSW_PAGE))

PASSAU_BASE = "https://netze.stadtwerke-passau.de/files/dateien/netze/daten/strom/Netzdaten%20csv/Netzver%C3%B6ffentlichungen%202024"
for series_code, series_name, nr, kind in [
    ("jhl", "Lastverlauf Jahreshoechstlast", "1", "jhl"),
    ("bezug", "Bezug vorgelagerte Ebene", "5", "bezug"),
]:
    for level in ["HSMS", "MS", "MSNS", "NS"]:
        level_label = level.replace("HSMS", "HS/MS").replace("MSNS", "MS/NS")
        file_nr = "22" if nr == "1" else "26"
        CORPUS_SPECS.append(_entry(
            f"passau_{series_code}_{level.lower()}_2024",
            f"Stadtwerke Passau - {series_name} {level_label} 2024",
            _p(f"passau_{series_code}_{level.lower()}_2024.csv"), "Wert", "kW",
            "Stadtwerke Passau", f"{series_name} {level_label}", 2024,
            f"{PASSAU_BASE}/lfd.%20Nr.%20{file_nr}%20-%20EnWG%20%C2%A7%2023c%20Abs.%203%20Nr.%20{nr}%20{level}.csv",
            f"Paragraph 23c Abs. 3 Nr. {nr} EnWG", level=level_label, kind=kind,
            official_page=PASSAU_PAGE))

CORPUS_SPECS += [
    _entry("enedis_ile_de_france_2024", "Enedis FR - <=36kVA aggregate Ile-de-France 2024",
           _p("enedis_ile_de_france_2024.csv"), "load_mw", "MW", "Enedis",
           "conso-inf36-region P0 total <=36kVA Ile-de-France", 2024,
           "https://opendata.enedis.fr/api/explore/v2.1/catalog/datasets/conso-inf36-region/records",
           "Licence Ouverte / Open Licence version 2.0", ts="timestamp_utc", level="regional",
           kind="load", official_page=ENEDIS_PAGE, country="FR", network_kind="dso_real"),
    _entry("enedis_bourgogne_franche_comte_2024",
           "Enedis FR - <=36kVA aggregate Bourgogne-Franche-Comte 2024",
           _p("enedis_bourgogne_franche_comte_2024.csv"), "load_mw", "MW", "Enedis",
           "conso-inf36-region P0 total <=36kVA Bourgogne-Franche-Comte", 2024,
           "https://opendata.enedis.fr/api/explore/v2.1/catalog/datasets/conso-inf36-region/records",
           "Licence Ouverte / Open Licence version 2.0", ts="timestamp_utc", level="regional",
           kind="load", official_page=ENEDIS_PAGE, country="FR", network_kind="dso_real"),
    _entry("eco2mix_ile_de_france_2024", "RTE eco2mix - regional load Ile-de-France 2024",
           _p("eco2mix_ile_de_france_2024.csv"), "load_mw", "MW", "RTE / ODRE",
           "eco2mix regional consolidated/definitive load Ile-de-France", 2024,
           "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-regional-cons-def/records",
           "ODRE / RTE open data; TSO regional load, not DSO", ts="timestamp_utc", level="regional",
           kind="regional_load", official_page=ECO2MIX_PAGE, include_in_pool=False,
           country="FR", network_kind="tso_regional"),
]


def validate_entry(entry):
    hourly, ts_col, load_col, meta = robust_load_csv(
        entry["path"], ts_col=entry.get("ts"), load_col=entry["col"], unit=entry["unit"], return_meta=True)
    load2d, days, _ = to_daily_local(hourly)
    vals = load2d.reshape(-1)
    hash_vals = np.round(vals.astype(float), 6)
    value_hash = hashlib.sha256(hash_vals.tobytes()).hexdigest()
    mean_load = float(np.nanmean(vals))
    min_load = float(np.nanmin(vals))
    max_load = float(np.nanmax(vals))
    value_range = max_load - min_load
    mape_meaningless = bool(entry["signed"] or min_load <= 0.0 or (
        value_range > 0 and abs(mean_load) < 0.05 * value_range))
    out = dict(entry)
    n_samples_original = int(meta.get("load_candidate", {}).get("n_values", 0))
    hourly_samples = int(len(hourly))
    samples_per_hour = float(n_samples_original / hourly_samples) if hourly_samples else 0.0
    if samples_per_hour >= 3.0:
        min_samples_required = 34_000
    elif samples_per_hour >= 1.5:
        min_samples_required = 17_000
    else:
        min_samples_required = 8_000
    full_year_like = bool(n_samples_original >= min_samples_required and len(load2d) >= 300)
    out.update({
        "ts": ts_col,
        "col": load_col,
        "load_level_detected": meta.get("load_level"),
        "load_unit_hint": meta.get("load_unit_hint"),
        "n_samples_original": n_samples_original,
        "estimated_samples_per_hour": round(samples_per_hour, 3),
        "min_samples_required": min_samples_required,
        "n_days": int(len(load2d)),
        "n_hours": int(len(vals)),
        "value_hash": value_hash,
        "mean_load_mw": round(mean_load, 6),
        "min_load_mw": round(min_load, 6),
        "max_load_mw": round(max_load, 6),
        "mape_meaningless": mape_meaningless,
        "mape_note": "MAPE instabil (signiert, Null-/Rueckspeisung oder Mittel nahe Wertebereich)" if mape_meaningless else None,
        "first_day": str(days[0].date()) if len(days) else None,
        "last_day": str(days[-1].date()) if len(days) else None,
        "full_year_like": full_year_like,
        "validation": "ok" if full_year_like else "too_short_or_gappy",
    })
    # Pool prior is load-normalized; signed/near-zero series are intentionally excluded.
    out["include_in_benchmark"] = bool(out["include_in_benchmark"] and full_year_like)
    out["include_in_pool"] = bool(out["include_in_pool"] and full_year_like and mean_load > 1.0)
    return out


def _correlation_series(entry):
    hourly, _ts_col, _load_col, _meta = robust_load_csv(
        entry["path"], ts_col=entry.get("ts"), load_col=entry["col"],
        unit=entry.get("unit", "MW"), return_meta=True)
    s = hourly.astype(float).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    return s[~s.index.duplicated(keep="first")]


def _pair_correlation(a, b):
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    overlap_hours = int(len(joined))
    if overlap_hours < MIN_CORRELATION_OVERLAP_HOURS:
        return None, overlap_hours
    x = joined.iloc[:, 0].to_numpy(dtype=float)
    y = joined.iloc[:, 1].to_numpy(dtype=float)
    if np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
        return None, overlap_hours
    return float(np.corrcoef(x, y)[0, 1]), overlap_hours


def _find(parent, key):
    while parent[key] != key:
        parent[key] = parent[parent[key]]
        key = parent[key]
    return key


def _union(parent, a, b):
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def apply_network_independence(entries):
    """Mark benchmark rows that carry redundant information by load correlation.

    The benchmark still keeps every non-duplicate row. These flags only support the
    honest "independent network" headline and the audit report.
    """
    for e in entries:
        e["independent_network"] = False
        if not e.get("include_in_benchmark", False):
            if e.get("duplicate_of"):
                e["network_exclusion_note"] = "Exact duplicate excluded before correlation clustering."
            elif not e.get("full_year_like", False):
                e["network_exclusion_note"] = "Not full-year-like; excluded before correlation clustering."

    included = [e for e in entries if e.get("include_in_benchmark", False)]
    by_key = {e["key"]: e for e in included}
    series_by_key = {e["key"]: _correlation_series(e) for e in included}
    parent = {e["key"]: e["key"] for e in included}
    pair_stats = {}
    high_pairs = []

    for i, ea in enumerate(included):
        for eb in included[i + 1:]:
            ka = ea["key"]
            kb = eb["key"]
            corr, overlap_hours = _pair_correlation(series_by_key[ka], series_by_key[kb])
            if corr is None:
                continue
            pair_key = tuple(sorted((ka, kb)))
            stat = {
                "a": ka,
                "b": kb,
                "r": round(corr, 6),
                "overlap_hours": overlap_hours,
                "overlap_days": round(overlap_hours / 24.0, 1),
            }
            pair_stats[pair_key] = stat
            if corr >= CORRELATION_THRESHOLD:
                high_pairs.append(stat)
                _union(parent, ka, kb)

    groups = {}
    for e in included:
        groups.setdefault(_find(parent, e["key"]), []).append(e["key"])

    clusters = []
    for keys in groups.values():
        keys = sorted(keys)
        # Prefer best coverage first, then highest mean load, then deterministic key.
        representative = max(
            keys,
            key=lambda k: (
                int(by_key[k].get("n_hours", 0)),
                float(by_key[k].get("mean_load_mw", -np.inf)),
                k,
            ),
        )
        redundant = [k for k in keys if k != representative]
        pair_rs = [
            pair_stats[tuple(sorted((a, b)))]["r"]
            for i, a in enumerate(keys)
            for b in keys[i + 1:]
            if tuple(sorted((a, b))) in pair_stats
        ]
        clusters.append({
            "representative": representative,
            "representative_name": by_key[representative]["name"],
            "members": keys,
            "redundant": redundant,
            "size": len(keys),
            "operators": sorted({by_key[k]["operator"] for k in keys}),
            "min_pairwise_r": round(float(min(pair_rs)), 6) if pair_rs else None,
            "max_pairwise_r": round(float(max(pair_rs)), 6) if pair_rs else None,
        })

    clusters = sorted(clusters, key=lambda c: c["representative"])
    for idx, cluster in enumerate(clusters, 1):
        cluster_id = f"net_{idx:02d}"
        cluster["cluster_id"] = cluster_id
        for key in cluster["members"]:
            e = by_key[key]
            e["network_group"] = cluster_id
            e["network_representative"] = cluster["representative"]
            e["network_group_members"] = cluster["members"]
            e["network_group_size"] = cluster["size"]
            if key == cluster["representative"]:
                e["independent_network"] = True
                continue
            pair = pair_stats.get(tuple(sorted((key, cluster["representative"]))))
            e["independent_network"] = False
            e["redundant_of"] = cluster["representative"]
            e["corr"] = pair["r"] if pair else None
            e["corr_overlap_hours"] = pair["overlap_hours"] if pair else None
            e["redundancy_note"] = (
                f"Pearson-r >= {CORRELATION_THRESHOLD:.2f} within correlation cluster; "
                f"counted with {cluster['representative']} as one independent network/regional cluster."
            )

    return {
        "n_independent_networks": len(clusters),
        "n_correlation_redundant": len(included) - len(clusters),
        "independence_method": {
            "correlation": "Pearson r on hourly load values",
            "correlation_threshold": CORRELATION_THRESHOLD,
            "min_overlap_days": MIN_CORRELATION_OVERLAP_DAYS,
            "min_overlap_hours": MIN_CORRELATION_OVERLAP_HOURS,
            "join": "inner join on actual timestamps; only pairs with sufficient overlap are clustered",
            "representative_rule": "max coverage (n_hours), then highest mean_load_mw, then key",
        },
        "pool_correlation_policy": (
            "Pool-Prior logic unchanged: include_in_pool remains the only pool filter; "
            "correlation-redundant non-duplicates stay eligible unless explicitly changed later."
        ),
        "network_clusters": clusters,
        "high_correlation_pairs": sorted(high_pairs, key=lambda p: (-p["r"], p["a"], p["b"])),
    }


def write_network_report(summary, path):
    entries = {e["key"]: e for e in summary["entries"]}
    clusters = summary.get("network_clusters", [])
    high_pairs = summary.get("high_correlation_pairs", [])
    method = summary.get("independence_method", {})

    lines = [
        "# NetzPilot - Network Independence",
        "",
        f"*Generated {summary['generated_utc']}.*",
        "",
        (
            f"Summary: {summary['n_series']} unique benchmark series from "
            f"{summary['n_independent_networks']} independent network/regional clusters; "
            f"{summary['n_correlation_redundant']} series are correlation-redundant at "
            f"r >= {method.get('correlation_threshold', CORRELATION_THRESHOLD):.2f}."
        ),
        "",
        "Method: Pearson correlation on hourly load values after an inner join on actual timestamps. "
        f"Pairs need at least {method.get('min_overlap_days', MIN_CORRELATION_OVERLAP_DAYS)} days of overlap. "
        "Clusters use transitive union-find; one representative per cluster is counted as independent.",
        "",
        f"Pool policy: {summary.get('pool_correlation_policy')}",
        "",
        "## Clusters",
        "",
        "| Cluster | Representative | Redundant series | r to representative | Operators |",
        "|---|---|---|---:|---|",
    ]

    for c in sorted(clusters, key=lambda x: (-x["size"], x["representative"])):
        redundant = c.get("redundant", [])
        if redundant:
            redundant_text = "; ".join(f"{k} ({entries[k]['name']})" for k in redundant)
            corr_text = "; ".join(
                "-" if entries[k].get("corr") is None else f"{entries[k]['corr']:.6f}"
                for k in redundant
            )
        else:
            redundant_text = "-"
            corr_text = "-"
        rep = c["representative"]
        lines.append(
            f"| {c['cluster_id']} ({c['size']}) | {rep} ({entries[rep]['name']}) | "
            f"{redundant_text} | {corr_text} | {'; '.join(c.get('operators', []))} |"
        )

    lines += [
        "",
        "## High-Correlation Edges",
        "",
        "| A | B | Pearson r | Overlap hours |",
        "|---|---|---:|---:|",
    ]
    if high_pairs:
        for p in high_pairs:
            lines.append(f"| {p['a']} | {p['b']} | {p['r']:.6f} | {p['overlap_hours']} |")
    else:
        lines.append("| - | - | - | - |")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{REAL}/corpus_index.json")
    ap.add_argument("--network-report", default=NETWORK_REPORT)
    args = ap.parse_args()

    entries = [validate_entry(e) for e in CORPUS_SPECS]

    groups = {}
    for e in entries:
        if e["full_year_like"]:
            groups.setdefault(e["value_hash"], []).append(e)
    duplicate_groups = []
    for value_hash, group in groups.items():
        if len(group) <= 1:
            continue
        duplicate_groups.append([e["key"] for e in group])
        canonical = group[0]
        canonical["dedup_canonical"] = True
        canonical["dedup_group"] = [e["key"] for e in group]
        for dup in group[1:]:
            dup["duplicate_of"] = canonical["key"]
            dup["dedup_group"] = [e["key"] for e in group]
            dup["dedup_note"] = (
                f"Identischer Werte-Hash wie {canonical['key']}; aus Benchmark und Pool ausgeschlossen."
            )
            dup["include_in_benchmark"] = False
            dup["include_in_pool"] = False
            dup["validation"] = "duplicate"

    independence = apply_network_independence(entries)
    valid_unique = [e for e in entries if e["full_year_like"] and not e.get("duplicate_of")]
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_entries_total": len(entries),
        "n_series": len(valid_unique),
        "n_pool_series": sum(1 for e in entries if e["include_in_pool"]),
        "n_signed": sum(1 for e in entries if e["signed"]),
        "n_profile_sum": sum(1 for e in entries if e["kind"] == "profile_sum"),
        "n_duplicates_excluded": sum(1 for e in entries if e.get("duplicate_of")),
        "country_counts": {
            country: sum(1 for e in valid_unique if e.get("country") == country)
            for country in sorted({e.get("country") for e in valid_unique})
        },
        "network_kind_counts": {
            kind: sum(1 for e in valid_unique if e.get("network_kind") == kind)
            for kind in sorted({e.get("network_kind") for e in valid_unique})
        },
        "duplicate_groups": duplicate_groups,
        "entries": entries,
    }
    summary.update(independence)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_network_report(summary, args.network_report)
    print(f"corpus_index geschrieben: {args.out}")
    print(
        f"  series={summary['n_series']} independent_networks={summary['n_independent_networks']} "
        f"pool={summary['n_pool_series']} signed={summary['n_signed']}"
    )
    print(f"  network_report={args.network_report}")


if __name__ == "__main__":
    main()
