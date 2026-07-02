#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Download the public German DSO files used by NetzPilot.

Two source sets are available:

``core``
    Reconstructs the German part of the existing corpus declared in
    :mod:`scripts.build_corpus_index`.

``extended``
    Adds public series that were found during the July 2026 source audit:
    voltage-level load, feed-in, losses, SLP/forecast aggregates and newer
    publication years from the same official DSO pages.

All output stays below ``data_cache/real`` (gitignored).  A machine-readable
manifest records source URL, retrieval time, byte size and SHA-256.  The
statutory publication basis is not treated as a redistribution licence; keep
raw files out of the public repository unless the data owner grants one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REAL = Path("data_cache/real")
MANIFEST_NAME = "download_manifest.json"
USER_AGENT = "NetzPilot-public-data-fetcher/1.0 (+https://github.com/amarakramali/netzpilot)"
PUBLICATION_NOTE = (
    "Official statutory publication; no separate redistribution licence was asserted. "
    "Use locally and retain provenance; do not commit raw data without permission."
)


@dataclass(frozen=True)
class DownloadSpec:
    key: str
    operator: str
    year: int
    kind: str
    url: str
    target: str
    source_page: str
    source_set: str
    archive_target: str | None = None


def _spec(key, operator, year, kind, url, target, source_page, source_set="extended",
          archive_target=None):
    return DownloadSpec(
        key=key,
        operator=operator,
        year=int(year),
        kind=kind,
        url=url,
        target=str(REAL / target),
        source_page=source_page,
        source_set=source_set,
        archive_target=str(REAL / archive_target) if archive_target else None,
    )


def core_specs() -> list[DownloadSpec]:
    """Build download jobs from the existing corpus source of truth."""
    from scripts.build_corpus_index import CORPUS_SPECS

    jobs: dict[str, DownloadSpec] = {}
    for entry in CORPUS_SPECS:
        if entry.get("source_set", "core") != "core":
            continue
        if entry.get("country", "DE") != "DE":
            continue
        url = entry["source_url"]
        target = entry["path"]
        # API-derived French data is handled by fetch_fr_public_loads.py.
        if "/api/" in url:
            continue
        spec = DownloadSpec(
            key=entry["key"],
            operator=entry["operator"],
            year=int(entry["year"]),
            kind=entry["kind"],
            url=url,
            target=target,
            source_page=entry.get("official_page") or url,
            source_set="core",
            archive_target=entry.get("source_archive"),
        )
        previous = jobs.get(target)
        if previous and previous.url != spec.url:
            raise ValueError(f"Conflicting sources for {target}: {previous.url} / {spec.url}")
        jobs[target] = spec
    return sorted(jobs.values(), key=lambda x: (x.operator, x.target))


def extended_specs() -> list[DownloadSpec]:
    jobs: list[DownloadSpec] = []

    # Stadtwerke Hilden: the old corpus used only Netzumsatz, SLP sum and DBA.
    page = "https://stadtwerke-hilden.de/netzregulierung/veroeffentlichungspflichten-strom/"
    base = "https://stadtwerke-hilden.de/uploads/Netz/Ver%C3%B6ffentlichungspflichten-Strom"
    for level, label in (("5", "ms"), ("6", "msns"), ("7", "ns")):
        voltage = {"5": "Mittelspannung", "6": "Umspannung-MS_NS", "7": "Niederspannung"}[level]
        jobs.append(_spec(
            f"hilden_jhl_{label}_2025", "Stadtwerke Hilden", 2025, "jhl",
            f"{base}/NE-{level}-{voltage}-Lastverlauf-2025.csv",
            f"hilden_jhl_{label}_2025.csv", page))
        jobs.append(_spec(
            f"hilden_einspeisung_{label}_2025", "Stadtwerke Hilden", 2025, "generation",
            f"{base}/NE-{level}-{voltage}-dezentrale-Einspeisungen-2025.csv",
            f"hilden_einspeisung_{label}_2025.csv", page))
    jobs.append(_spec(
        "hilden_verlustenergie_2025", "Stadtwerke Hilden", 2025, "loss",
        f"{base}/Verlustenergie-Lastgang-2025.csv", "hilden_verlustenergie_2025.csv", page))

    # EVDB: current publication (uploaded June 2026, referring to the 2025 reporting year).
    page = "https://www.evdbag.de/netzbetrieb/veroeffentlichungen/"
    base = "https://www.evdbag.de/wp-content/uploads/2026/06"
    evdb_files = [
        ("dba", "differenzbilanz", "lfd.-Nr.-10-StromNZV-%C2%A7-12-Abs.-3-Satz-3.csv"),
        ("jhl_ms", "jhl", "lfd.-Nr.-22-EnWG-%C2%A7-23c-Abs.-3-Nr.-1-MS.csv.csv"),
        ("jhl_msns", "jhl", "lfd.-Nr.-22-EnWG-%C2%A7-23c-Abs.-3-Nr.-1-MSNS.csv"),
        ("jhl_ns", "jhl", "lfd.-Nr.-22-EnWG-%C2%A7-23c-Abs.-3-Nr.-1-NS.csv"),
        ("verlust_ms", "loss", "lfd.-Nr.-23-EnWG-%C2%A7-23c-Abs.-3-Nr.-2-MS.csv"),
        ("verlust_msns", "loss", "lfd.-Nr.-23-EnWG-%C2%A7-23c-Abs.-3-Nr.-2-MSNS.csv"),
        ("verlust_ns", "loss", "lfd.-Nr.-23-EnWG-%C2%A7-23c-Abs.-3-Nr.-2-NS.csv"),
        ("slp_ns", "profile_sum", "lfd.-Nr.-24-EnWG-%C2%A7-23c-Abs.-3-Nr.-3-NS.csv"),
        ("netzverluste_summe", "loss", "lfd.-Nr.-24-EnWG-%C2%A7-23c-Abs.-3-Nr.-3-Summe-NV.csv"),
        ("fahrplan", "forecast_sum", "lfd.-Nr.-25-EnWG-%C2%A7-23c-Abs.-3-Nr.-4-FP.csv"),
        ("rlk", "residual", "lfd.-Nr.-25-EnWG-%C2%A7-23c-Abs.-3-Nr.-4-RLK.csv"),
        ("bezug_ms", "bezug", "lfd.-Nr.-26-EnWG-%C2%A7-23c-Abs.-3-Nr.-5-MS.csv"),
        ("bezug_msns", "bezug", "lfd.-Nr.-26-EnWG-%C2%A7-23c-Abs.-3-Nr.-5-MSNS.csv"),
        ("bezug_ns", "bezug", "lfd.-Nr.-26-EnWG-%C2%A7-23c-Abs.-3-Nr.-5-NS.csv"),
    ]
    for direction in ("EIN", "RS"):
        for level in ("MS", "MSNS", "NS"):
            evdb_files.append((
                f"{direction.lower()}_{level.lower()}", "generation",
                f"lfd.-Nr.-27-EnWG-%C2%A7-23c-Abs.-3-Nr.-6-{direction}-{level}.csv"))
    for slug, kind, filename in evdb_files:
        jobs.append(_spec(
            f"evdb_{slug}_2025", "Energieversorgung Dahlenburg-Bleckede AG", 2025, kind,
            f"{base}/{filename}", f"evdb_{slug}_2025.csv", page))

    # Neuruppin: add two preceding years to the existing 2022 source.
    page = "https://www.swn.de/pflichten-veroeffentlichungen-netze.html"
    for year, filename in (
        (2020, "2021_04_21_LGL_Strom_2020_Neuruppin.csv"),
        (2021, "2022_05_17_LGL-Strom_2021_Neuruppin.csv"),
    ):
        jobs.append(_spec(
            f"neuruppin_lgl_{year}", "Stadtwerke Neuruppin", year, "multi_series",
            f"https://www.swn.de/fileadmin/Dateien/strom/netze_downloads/Pflichten/Lastgangsdaten/{filename}",
            f"neuruppin_lgl_strom_{year}.csv", page))

    # Bitterfeld-Wolfen: the old corpus used only the three JHL files.
    page = "https://netz-bitterfeld-wolfen.de/veroeffentlichungspflichten-2023-copy/articles/veroeffentlichungspflichten-2024.html"
    base = "https://netz-bitterfeld-wolfen.de/files/ngbw/nbStrom/Veroeffentlichungen/Veroeffentlichungspflichten_csv/2024"
    bitterfeld_files = [
        ("einspeisung_ms", "generation", "EIN%20MS.csv"),
        ("einspeisung_ns", "generation", "EIN%20NS.csv"),
        ("entnahme_ms", "load", "HL_Entn_MS.csv"),
        ("entnahme_msns", "load", "HL_Entn_MSNS.csv"),
        ("entnahme_ns", "load", "HL_Entn_NS.csv"),
        ("rueckspeisung_ms", "generation", "RS%20MS.csv"),
        ("rueckspeisung_msns", "generation", "RS%20MSNS.csv"),
        ("rueckspeisung_ns", "generation", "RS%20NS.csv"),
        ("netzverluste", "loss", "SummeNetzverluste.csv"),
        ("slp", "profile_sum", "SummenlastSLP.csv"),
    ]
    for slug, kind, filename in bitterfeld_files:
        jobs.append(_spec(
            f"bitterfeld_{slug}_2024", "NG Bitterfeld-Wolfen", 2024, kind,
            f"{base}/{filename}", f"bitterfeld_{slug}_2024.csv", page))

    # TEN: categories not present in the original JHL/Bezug corpus.
    page = "https://www.thueringer-energienetze.com/Ueber_uns/Veroeffentlichungspflichten/Netzdaten"
    base = "https://www.thueringer-energienetze.com/Content/Documents/Ueber_uns"
    ten_archives = [
        ("slp", "profile_sum", "p_23c_3-3a_SLP_2025.zip"),
        ("gesamtlast", "load", "p_23c_3-3b_gesamt_2025.zip"),
        ("fahrplan_ns", "forecast_sum", "p_23c_3-4_NS_FPP_2025.zip"),
    ]
    for level in ("HS", "HSU", "MS", "MSU", "NS"):
        ten_archives.append((f"einspeisung_{level.lower()}", "generation", f"p_23c_3-6_{level}_2025.zip"))
    for slug, kind, filename in ten_archives:
        jobs.append(_spec(
            f"ten_{slug}_2025", "TEN Thueringer Energienetze", 2025, kind,
            f"{base}/{filename}", f"ten_{slug}_2025.csv", page,
            archive_target=f"ten_{slug}_2025.zip"))
    for quarter in range(1, 5):
        filename = f"p12-3DBA-2025Q{quarter}.csv"
        jobs.append(_spec(
            f"ten_dba_2025_q{quarter}", "TEN Thueringer Energienetze", 2025,
            "differenzbilanz", f"{base}/{filename}", f"ten_dba_2025_q{quarter}.csv", page))

    # neu.sw: sums, forecast, feed-in and network-loss files beyond JHL/Bezug.
    page = "https://www.neu-sw.de/netze/unsere-netze/stromnetz"
    base = "https://www.neu-sw.de/downloads/netze/stromnetz/veroeffentlichungspflichten"
    neusw_files = [
        ("slp", "profile_sum", "netzstrukturdaten/enwg_23c_3_nr_3_1.csv"),
        ("netzverluste_summe", "loss", "netzstrukturdaten/enwg_23c_3_nr_3_2.csv"),
        ("fahrplan_ns", "forecast_sum", "netzstrukturdaten/enwg_23c_3_nr_4_ns.csv"),
        ("einspeisung_hsms", "generation", "netzstrukturdaten/enwg_23c_3_nr_6_hsms.csv"),
        ("einspeisung_msns", "generation", "netzstrukturdaten/enwg_23c_3_nr_6_msns.csv"),
        ("einspeisung_ms", "generation", "netzstrukturdaten/enwg_23c_3_nr_6_ms.csv"),
        ("einspeisung_ns", "generation", "netzstrukturdaten/enwg_23c_3_nr_6_ns.csv"),
        ("verlustlastgang", "loss", "netzverluste/lastgang_3_2_halbsatz.csv"),
    ]
    for slug, kind, path in neusw_files:
        jobs.append(_spec(
            f"neusw_{slug}_2025", "Neubrandenburger Stadtwerke", 2025, kind,
            f"{base}/{path}", f"neusw_{slug}_2025.csv", page))

    # Passau: replace the old 2024-only view with the complete 2025 publication.
    page = "https://netze.stadtwerke-passau.de/strom/netzveroeffentlichungen.html"
    base = "https://netze.stadtwerke-passau.de/files/dateien/netze/daten/strom/Netzdaten%20csv/Netzver%C3%B6ffentlichungen%202025"
    passau: list[tuple[str, str, str]] = []
    for nr, file_nr, kind, prefix, levels in (
        (1, 22, "jhl", "jhl", ("HSMS", "MS", "MSNS", "NS")),
        (2, 23, "loss", "verlust", ("HSMS", "MS", "MSNS", "NS")),
        (3, 24, "profile_sum", "summe", ("MS", "MSNS", "NS", "Summe NV", "Summe SLP")),
        (4, 25, "forecast_sum", "prognose", ("FP", "RLK")),
        (5, 26, "bezug", "bezug", ("HSMS", "MS", "MSNS", "NS")),
        (6, 27, "generation", "einspeisung", ("EIN MS", "EIN MSNS", "EIN NS")),
    ):
        for level in levels:
            filename = f"lfd. Nr. {file_nr} - EnWG § 23c Abs. 3 Nr. {nr} {level}.csv"
            # The publisher currently exposes this one file with a duplicated extension.
            if nr == 1 and level == "MS":
                filename += ".csv"
            slug = level.lower().replace(" ", "_").replace("/", "")
            passau.append((f"{prefix}_{slug}", kind, quote(filename, safe="")))
    for slug, kind, encoded_filename in passau:
        jobs.append(_spec(
            f"passau_{slug}_2025", "Stadtwerke Passau", 2025, kind,
            f"{base}/{encoded_filename}", f"passau_{slug}_2025.csv", page))

    targets = [job.target for job in jobs]
    if len(targets) != len(set(targets)):
        raise ValueError("Extended source catalogue contains duplicate targets")
    return sorted(jobs, key=lambda x: (x.operator, x.target))


def session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_replace(source: Path, destination: Path) -> None:
    """Replace a file, tolerating short-lived Windows scanner/indexer locks."""
    last_error = None
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.2 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def download_to(s: requests.Session, url: str, destination: Path, timeout: int = 180) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part",
                                     dir=destination.parent)
    os.close(fd)
    tmp = Path(temporary)
    try:
        with s.get(url, stream=True, timeout=(20, timeout)) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if tmp.stat().st_size == 0:
            raise RuntimeError(f"Empty response from {url}")
        atomic_replace(tmp, destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _decode_csv(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeError("Could not decode CSV archive member")


def extract_csv(archive: Path, destination: Path) -> str:
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir() and m.filename.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV in archive {archive}")
        merge_quarters = False
        if len(members) > 1:
            exact = [m for m in members if Path(m.filename).name.lower() == destination.name.lower()]
            if len(exact) == 1:
                member = exact[0]
            elif all("q" in Path(m.filename).stem.lower() for m in members):
                # TEN publishes one quarter per member.  Normalize those files to one
                # annual CSV with a single header so the existing robust loader can read it.
                members.sort(key=lambda m: m.filename.lower())
                member = members[0]
                merge_quarters = True
            else:
                names = ", ".join(m.filename for m in members)
                raise RuntimeError(f"Ambiguous CSV members in {archive}: {names}")
        else:
            member = members[0]
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part",
                                         dir=destination.parent)
        os.close(fd)
        tmp = Path(temporary)
        try:
            if merge_quarters:
                header = None
                data_rows = []
                for quarter in members:
                    lines = _decode_csv(zf.read(quarter)).splitlines()
                    header_index = next(
                        (i for i, line in enumerate(lines) if line.lower().startswith("datum;")), None)
                    if header_index is None:
                        raise RuntimeError(f"No Datum header in {quarter.filename}")
                    if header is None:
                        header = lines[:header_index + 1]
                        # Make the retained metadata accurately describe the merged year.
                        for i, line in enumerate(header):
                            if line.startswith("Betrachtungszeitraum:"):
                                header[i] = "Betrachtungszeitraum:;01.01.2025;bis;31.12.2025"
                    data_rows.extend(
                        line for line in lines[header_index + 1:]
                        if len(line) >= 11 and line[2:3] == "." and line[5:6] == "."
                    )
                tmp.write_text("\n".join((header or []) + data_rows) + "\n", encoding="utf-8")
            else:
                with zf.open(member) as source, tmp.open("wb") as target:
                    shutil.copyfileobj(source, target)
            atomic_replace(tmp, destination)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return ", ".join(m.filename for m in members) if merge_quarters else member.filename


def fetch_one(s: requests.Session, spec: DownloadSpec, force: bool = False) -> dict:
    target = Path(spec.target)
    archive_member = None
    status = "cached"
    if spec.archive_target:
        archive = Path(spec.archive_target)
        if force or not archive.exists():
            download_to(s, spec.url, archive)
            status = "downloaded"
        if force or not target.exists():
            archive_member = extract_csv(archive, target)
            status = "downloaded"
    elif force or not target.exists():
        download_to(s, spec.url, target)
        status = "downloaded"

    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"Target missing or empty after download: {target}")
    return {
        **asdict(spec),
        "status": status,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "archive_member": archive_member,
        "publication_note": PUBLICATION_NOTE,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("core", "extended", "all"), default="core",
                        help="Source set to fetch (default: core)")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument("--list", action="store_true", help="List jobs without downloading")
    parser.add_argument("--keep-going", action="store_true",
                        help="Continue after individual download failures")
    args = parser.parse_args()

    specs = []
    if args.set in {"core", "all"}:
        specs.extend(core_specs())
    if args.set in {"extended", "all"}:
        specs.extend(extended_specs())

    by_target: dict[str, DownloadSpec] = {}
    for spec in specs:
        old = by_target.get(spec.target)
        if old and old.url != spec.url:
            raise SystemExit(f"Conflicting download jobs for {spec.target}")
        by_target[spec.target] = spec
    specs = sorted(by_target.values(), key=lambda x: (x.source_set, x.operator, x.target))

    if args.list:
        for spec in specs:
            print(f"{spec.source_set:8} {spec.operator:40} {spec.target} <- {spec.url}")
        print(f"{len(specs)} physical download jobs")
        return

    REAL.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    s = session()
    for index, spec in enumerate(specs, 1):
        try:
            record = fetch_one(s, spec, force=args.force)
            records.append(record)
            print(f"[{index:03}/{len(specs):03}] {record['status']:10} "
                  f"{record['bytes']:>9} B  {spec.target}")
        except Exception as exc:
            failure = {**asdict(spec), "error": str(exc)}
            failures.append(failure)
            print(f"[{index:03}/{len(specs):03}] FAILED     {spec.target}: {exc}")
            if not args.keep_going:
                break

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_set": args.set,
        "n_jobs": len(specs),
        "n_ok": len(records),
        "n_failed": len(failures),
        "records": records,
        "failures": failures,
        "publication_note": PUBLICATION_NOTE,
    }
    manifest_path = REAL / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"manifest -> {manifest_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
