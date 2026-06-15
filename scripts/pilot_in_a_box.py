# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Pilot-in-a-Box — echte Stadtwerk-CSV rein, ehrlicher Nachweis raus.

Wirf den Lastgang eines Stadtwerks (CSV, beliebige Auflösung) hinein und erhalte:
- leakage-sicheren rolling-origin Backtest (ShrunkCorrector + CQR) gegen saisonal-naiv & Persistenz,
- MAE / MAPE / MASE / Skill / kalibrierte Coverage (80/90 %),
- eine transparente Ausgleichsenergie-€-Schätzung (klar als Annahme markiert),
- einen 1-Seiten-Report (Markdown) + metrics.json.

Nur numpy/pandas/stdlib -> läuft in der Sandbox UND auf dem Rechner des Stadtwerks ohne Extra-Installs.

Beispiel:
  python scripts/pilot_in_a_box.py --csv pfad/lastgang.csv --unit MW --region NW
"""
from __future__ import annotations
import argparse, json, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.features.build import get_holidays, to_daily_local
from netzpilot.eval.backtest import rolling_origin
from netzpilot.eval.conformal import rolling_origin_cqr
from netzpilot.eval.economics import saving_from_real_rebap, saving_from_rebap_spot
from netzpilot.data.rebap import load_rebap
from netzpilot.data.spot_da import load_rebap_spot_pairs
from netzpilot.models.robust_corrector import ShrunkCorrector

LOAD_HINTS = re.compile(r"(last|load|mw|kw|wirk|verbrauch|leistung|menge|summe|differenz|netz|dba|wert|value|p_?ges)", re.I)
WEEKDAY = re.compile(r"[\s,;]+[A-Za-zÄÖÜäöüß.]{2,3}\.?\s*$")  # dt. Wochentag am Ende, z.B. "01.01.2025 00:15 Mi"


def _parse_dt(s):
    s2 = s.astype(str).str.strip().str.replace(WEEKDAY, "", regex=True)
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%y %H:%M:%S",
        "%d.%m.%y %H:%M",
    ):
        parsed = pd.to_datetime(s2, errors="coerce", format=fmt)
        if parsed.notna().mean() > 0.7:
            return parsed
    a = pd.to_datetime(s2, errors="coerce")  # ISO YYYY-MM-DD zuerst (dayfirst würde ISO zerstören)
    if a.isna().mean() > 0.3:  # Fallback für deutsches DD.MM.YYYY
        b = pd.to_datetime(s2, errors="coerce", dayfirst=True)
        if b.notna().mean() > a.notna().mean():
            return b
    return a


def _looks_like_date(v):
    return bool(re.match(r"^\s*(\d{1,2}\.\d{1,2}\.(\d{2}|\d{4})|\d{4}-\d{2}-\d{2})\s*$", str(v)))


def _looks_like_time(v):
    return bool(re.match(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$", str(v)))


def _find_header_row(path, sep):
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            cols = [c.strip().lower() for c in line.strip().split(sep)]
            has_time = any(c in {"text", "datum", "timestamp", "zeit"} for c in cols)
            has_value = any(c == "reihe1" or c == "p (kw)" or c == "wert" or LOAD_HINTS.search(c) for c in cols)
            if has_time and {"datum", "von"}.issubset(set(cols)) and len(cols) > 3:
                return i, True
            if has_time and has_value:
                return i, True
            if len(cols) >= 4 and _looks_like_date(cols[0]) and _looks_like_time(cols[1]) and _looks_like_time(cols[2]):
                return i, False
    return 0, True


def _localize_load_index(idx):
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        return idx.tz_convert("UTC")
    try:
        return idx.tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward").tz_convert("UTC")
    except Exception:
        return idx.tz_localize("Europe/Berlin", ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")


def _to_num(s):
    s = s.astype(str).str.strip()
    if s.str.contains(",", na=False).mean() > 0.2:  # deutsches Dezimalkomma (Punkt = Tausender)
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif s.str.match(r"^-?\d{1,3}(\.\d{3})+$", na=False).mean() > 0.2:
        s = s.str.replace(".", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _is_index_like(v):
    x = v.dropna().to_numpy()
    if len(x) < 5:
        return False
    d = np.diff(x)
    return bool(np.allclose(d, d[0]) and abs(d[0]) >= 1)  # arithmetische Folge = Zeilenindex, keine Last


def robust_load_csv(path, ts_col=None, load_col=None, unit="MW"):
    """Robuster Loader für reale DSO-CSVs (Delimiter/Dezimalkomma/dt. Zeitstempel mit Wochentag,
    Index-/Leerspalten) -> stündliche UTC-Reihe in MW. Geprüft an Stadtwerke-Hilden-Format."""
    first = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                first = line
                break
    sep = ";" if (";" in first and first.count(";") >= first.count(",")) else ","
    header_row, has_header = _find_header_row(path, sep)
    df = pd.read_csv(path, sep=sep, engine="python", dtype=str, skiprows=header_row,
                     header=0 if has_header else None, encoding="utf-8", encoding_errors="replace")
    if not has_header:
        n_load = max(0, len(df.columns) - 3)
        df.columns = ["Datum", "von", "bis"] + [f"Load_{i}" for i in range(1, n_load + 1)]
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]

    if ts_col is None and {"Datum", "von"}.issubset(df.columns):
        ts = _parse_dt(df["Datum"].astype(str).str.strip() + " " + df["von"].astype(str).str.strip())
        ts_col = "Datum+von"
    elif ts_col == "Datum+von" and {"Datum", "von"}.issubset(df.columns):
        ts = _parse_dt(df["Datum"].astype(str).str.strip() + " " + df["von"].astype(str).str.strip())
    elif ts_col is None:  # Spalte mit der besten datetime-Parsebarkeit (nach Wochentag-Strip)
        scores = {c: _parse_dt(df[c]).notna().mean() for c in df.columns}
        ts_col = max(scores, key=scores.get)
        if scores[ts_col] < 0.5:
            raise SystemExit("Keine Zeitstempel-Spalte erkannt — bitte --ts-col angeben.")
        ts = _parse_dt(df[ts_col])
    else:
        ts = _parse_dt(df[ts_col])

    if load_col is None:  # echte Lastspalte: kein Zeilenindex, nicht überwiegend leer, nicht konstant
        num = {c: _to_num(df[c]) for c in df.columns if c != ts_col}
        num = {c: v for c, v in num.items()
               if v.notna().mean() > 0.5 and not _is_index_like(v) and float(np.nanstd(v.to_numpy())) > 0}
        if not num:
            raise SystemExit("Keine plausible Lastspalte gefunden — bitte --load-col angeben.")
        named = [c for c in num if LOAD_HINTS.search(c)]
        load_col = max(named or list(num), key=lambda c: int(num[c].nunique()))
        vals = num[load_col]
    else:
        vals = _to_num(df[load_col])

    s = pd.Series(vals.to_numpy(dtype=float), index=ts).dropna().sort_index()
    s = s[~s.index.duplicated(keep="first")]
    if s.index.tz is None:
        s.index = _localize_load_index(s.index)
    u = unit.lower()
    if u == "kw":
        s = s / 1000.0  # -> MW
    elif u == "w":
        s = s / 1e6
    hourly = s.resample("1h").mean().dropna()
    return hourly, ts_col, load_col


def _unit_factor_to_mw(unit):
    u = (unit or "MW").lower()
    if u == "kw":
        return 1.0 / 1000.0
    if u == "w":
        return 1.0 / 1_000_000.0
    return 1.0


def _split_rows(path, limit=80):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for _, line in zip(range(limit), f):
            rows.append(line.rstrip("\n\r"))
    return rows


def _infer_headerless_labels(rows, sep, columns):
    labels = {c: {} for c in columns}
    if not {"Datum", "von", "bis"}.issubset(columns):
        return labels
    load_cols = list(columns[3:])
    for line in rows:
        parts = [p.strip() for p in line.split(sep)]
        if len(parts) < 4:
            continue
        tail = parts[3:3 + len(load_cols)]
        nonempty = [p for p in tail if p]
        if len(nonempty) < max(1, len(load_cols) // 2):
            continue
        if all(re.search(r"\b[mk]?w\b", p, re.I) for p in nonempty):
            for col, value in zip(load_cols, tail):
                if value:
                    labels[col]["unit_hint"] = value
        elif any(re.search(r"kv|spannung|ebene|ns|ms", p, re.I) for p in nonempty):
            for col, value in zip(load_cols, tail):
                if value:
                    labels[col]["level"] = value
    return labels


def _is_excel(path):
    """Excel an Endung ODER Magic-Bytes erkennen (xlsx = ZIP 'PK', xls = OLE 'D0CF11E0')."""
    if str(path).lower().endswith((".xlsx", ".xls", ".xlsm")):
        return True
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        return head[:2] == b"PK" or head[:4] == b"\xd0\xcf\x11\xe0"
    except OSError:
        return False


def _read_excel_dso(path, nrows=None):
    """Excel-Lastgang lesen (alles als String, wie der CSV-Pfad). Erste Zeile = Kopf."""
    try:
        df = pd.read_excel(path, dtype=str, nrows=nrows)
    except ImportError:
        raise SystemExit("Excel-Datei erkannt, aber 'openpyxl' fehlt. Bitte als CSV speichern "
                         "(in Excel: Datei → Speichern unter → CSV) oder openpyxl installieren.")
    except Exception as e:
        raise SystemExit(f"Excel-Datei konnte nicht gelesen werden ({e}). Bitte als CSV speichern.")
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    labels = {c: {} for c in df.columns}
    return df, labels, "xlsx", 0, True


def _read_dso_csv(path, nrows=None):
    if _is_excel(path):
        return _read_excel_dso(path, nrows=nrows)
    first = ""
    rows = _split_rows(path)
    for line in rows:
        if line.strip():
            first = line
            break
    sep = ";" if (";" in first and first.count(";") >= first.count(",")) else ","
    header_row, has_header = _find_header_row(path, sep)
    df = pd.read_csv(path, sep=sep, engine="python", dtype=str, skiprows=header_row,
                     header=0 if has_header else None, encoding="utf-8", encoding_errors="replace",
                     nrows=nrows)
    if not has_header:
        n_load = max(0, len(df.columns) - 3)
        df.columns = ["Datum", "von", "bis"] + [f"Load_{i}" for i in range(1, n_load + 1)]
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    labels = _infer_headerless_labels(rows[:header_row], sep, df.columns) if not has_header else {c: {} for c in df.columns}
    return df, labels, sep, header_row, has_header


def _detect_timestamp(df, ts_col=None):
    if ts_col is None and {"Datum", "von"}.issubset(df.columns):
        ts = _parse_dt(df["Datum"].astype(str).str.strip() + " " + df["von"].astype(str).str.strip())
        ts_col = "Datum+von"
    elif ts_col == "Datum+von" and {"Datum", "von"}.issubset(df.columns):
        ts = _parse_dt(df["Datum"].astype(str).str.strip() + " " + df["von"].astype(str).str.strip())
    elif ts_col is None:
        scores = {c: _parse_dt(df[c]).notna().mean() for c in df.columns}
        ts_col = max(scores, key=scores.get)
        if scores[ts_col] < 0.5:
            raise SystemExit("Keine Zeitstempel-Spalte erkannt — bitte --ts-col angeben.")
        ts = _parse_dt(df[ts_col])
    else:
        ts = _parse_dt(df[ts_col])
    return ts, ts_col


def _candidate_dict(name, values, unit, labels):
    arr = values.dropna().to_numpy(dtype=float)
    factor = _unit_factor_to_mw(unit)
    meta = labels.get(name, {})
    return {
        "name": str(name),
        "mean_input_unit": round(float(np.mean(arr)), 3),
        "min_input_unit": round(float(np.min(arr)), 3),
        "max_input_unit": round(float(np.max(arr)), 3),
        "mean_MW": round(float(np.mean(arr) * factor), 3),
        "unit_hint": meta.get("unit_hint") or unit,
        "level": meta.get("level"),
        "n_values": int(len(arr)),
    }


def _plausible_load_candidates(df, ts_col, unit, labels):
    skip = {ts_col, "Datum", "von", "bis"}
    out = []
    for col in df.columns:
        if col in skip:
            continue
        vals = _to_num(df[col])
        if vals.notna().mean() <= 0.5:
            continue
        if _is_index_like(vals):
            continue
        if float(np.nanstd(vals.to_numpy(dtype=float))) <= 0:
            continue
        out.append((_candidate_dict(col, vals, unit, labels), vals))
    return out


def _format_candidates(candidates):
    lines = []
    for cand, _ in candidates:
        level = f", level={cand['level']}" if cand.get("level") else ""
        lines.append(
            f"- {cand['name']}: mean={cand['mean_input_unit']} {cand['unit_hint']} "
            f"({cand['mean_MW']} MW), min={cand['min_input_unit']}, max={cand['max_input_unit']}{level}"
        )
    return "\n".join(lines)


def inspect_load_columns(path, ts_col=None, unit="MW", sample_rows=2000):
    # Nur eine Stichprobe lesen: Spalten-/Format-/Zeit-Erkennung braucht nicht die ganze Jahresdatei.
    # Das drueckt die Erkennung von ~6 s (35k Zeilen, dateutil-Fallback je Spalte) auf Sekundenbruchteile.
    df, labels, sep, header_row, has_header = _read_dso_csv(path, nrows=sample_rows)
    ts, detected_ts_col = _detect_timestamp(df, ts_col)
    candidates = _plausible_load_candidates(df, detected_ts_col, unit, labels)
    return {
        "csv": str(path),
        "separator": sep,
        "header_row": int(header_row),
        "has_header": bool(has_header),
        "ts_col": detected_ts_col,
        "timestamp_parse_rate": round(float(pd.Series(ts).notna().mean()), 3),
        "load_candidates": [cand for cand, _ in candidates],
    }


def robust_load_csv(path, ts_col=None, load_col=None, unit="MW", return_meta=False):
    """Robuster Loader fuer reale DSO-CSVs. Bei mehreren plausiblen Lastspalten muss
    --load-col gesetzt werden, damit keine Netzebene still verwechselt wird.

    MSCONS (EDIFACT-Lastgang aus der Marktkommunikation) wird erkannt und ueber denselben
    Rueckgabevertrag in die stuendliche Leistungsreihe ueberfuehrt — read-only (siehe
    netzpilot.data.mscons). Danach laeuft die Engine identisch zum CSV-Pfad."""
    from netzpilot.data.mscons import looks_like_mscons, load_mscons_hourly
    if looks_like_mscons(path):
        hourly, ts_label, load_label, meta = load_mscons_hourly(path, unit=unit)
        if return_meta:
            return hourly, ts_label, load_label, meta
        return hourly, ts_label, load_label
    df, labels, _, _, _ = _read_dso_csv(path)
    ts, ts_col = _detect_timestamp(df, ts_col)

    if load_col is None:
        candidates = _plausible_load_candidates(df, ts_col, unit, labels)
        if not candidates:
            raise SystemExit("Keine plausible Lastspalte gefunden — bitte --load-col angeben.")
        if len(candidates) > 1:
            raise SystemExit(
                f"Mehrere plausible Lastspalten in {path} erkannt. Bitte explizit --load-col <Name> setzen.\n"
                + _format_candidates(candidates)
            )
        selected, vals = candidates[0]
        load_col = selected["name"]
    else:
        if load_col not in df.columns:
            candidates = _plausible_load_candidates(df, ts_col, unit, labels)
            raise SystemExit(f"Lastspalte {load_col!r} nicht gefunden. Kandidaten:\n{_format_candidates(candidates)}")
        vals = _to_num(df[load_col])
        if vals.notna().mean() <= 0.5:
            raise SystemExit(f"Lastspalte {load_col!r} ist nicht ausreichend numerisch.")
        selected = _candidate_dict(load_col, vals, unit, labels)

    s = pd.Series(vals.to_numpy(dtype=float), index=ts).dropna().sort_index()
    s = s[~s.index.duplicated(keep="first")]
    if s.index.tz is None:
        s.index = _localize_load_index(s.index)
    u = unit.lower()
    if u == "kw":
        s = s / 1000.0
    elif u == "w":
        s = s / 1e6
    hourly = s.resample("1h").mean().dropna()
    if return_meta:
        return hourly, ts_col, load_col, {
            "load_col": load_col,
            "load_level": selected.get("level"),
            "load_unit_hint": selected.get("unit_hint"),
            "load_candidate": selected,
        }
    return hourly, ts_col, load_col


def annual_eur(ae_sum_mwh, price):
    return ae_sum_mwh * price


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ts-col", default=None)
    ap.add_argument("--load-col", default=None)
    ap.add_argument("--unit", default="MW", choices=["MW", "kW", "W", "mw", "kw", "w"])
    ap.add_argument("--load-level", default=None, help="Explizite Netz-/Spannungsebene fuer Reporting")
    ap.add_argument("--region", default="NW", help="Bundesland-Code für Feiertage")
    ap.add_argument("--keep-days", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=14)
    ap.add_argument("--fast", action="store_true", help="CQR ueberspringen (schneller Smoke-Test)")
    ap.add_argument("--list-columns", action="store_true", help="Nur Zeit-/Lastspalten-Kandidaten inspizieren, kein Backtest")
    ap.add_argument("--rebap-csv", default=None, help="Optionale echte reBAP-Preisreihe fuer EUR-Band")
    ap.add_argument("--spot-csv", default=None,
                    help="Optionale Spot-DA-Preisreihe (QH-aligned an reBAP) — aktiviert Aufschlag-Headline")
    ap.add_argument("--name", default=None, help="Anzeigename des Stadtwerks/Datensatzes")
    ap.add_argument("--out", default="data_cache/pilot")
    args = ap.parse_args()

    if args.list_columns:
        print(json.dumps(inspect_load_columns(args.csv, args.ts_col, args.unit), indent=2, ensure_ascii=True))
        return

    hourly, ts_col, load_col, load_meta = robust_load_csv(
        args.csv, args.ts_col, args.load_col, args.unit, return_meta=True
    )
    load2d, days, good = to_daily_local(hourly)
    if len(load2d) < args.keep_days:
        args.keep_days = len(load2d)
    load2d, days = load2d[-args.keep_days:], days[-args.keep_days:]
    if len(load2d) < args.n_test + 30:
        raise SystemExit(f"Zu wenig vollständige Tage ({len(load2d)}). Mind. ~{args.n_test+30} nötig.")
    hol = get_holidays(sorted({d.year for d in days}), args.region)

    R, sm = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0), n_test=args.n_test, holiday_set=hol)
    if args.fast:
        c80 = c90 = {"coverage_%": None}
    else:
        cal = min(21, args.keep_days - args.n_test - 1)
        _, c80 = rolling_origin_cqr(load2d, days, lambda: ShrunkCorrector(10.0), alpha=0.2,
                                    cal_days=cal, n_test=args.n_test, holiday_set=hol)
        _, c90 = rolling_origin_cqr(load2d, days, lambda: ShrunkCorrector(10.0), alpha=0.1,
                                    cal_days=cal, n_test=args.n_test, holiday_set=hol)
    m = sm["metriken"]

    # Tägliche absolute Fehlersumme (MWh, da Stundenschritt) für €-Schätzung
    a = np.asarray(R["actual"], float)
    ae_model = float(np.abs(np.asarray(R["model"], float) - a).sum())
    ae_snaive = float(np.abs(np.asarray(R["snaive"], float) - a).sum())
    factor = 365.0 / args.n_test  # Annualisierung
    saved_mwh = (ae_snaive - ae_model) * factor
    prices = [30, 60, 120]  # €/MWh — reBAP ist volatil (2024 ruhig ~Spot, 2021/22 bis ~100-160)
    eur_saving = {p: round(saved_mwh * p) if np.isfinite(saved_mwh) else None for p in prices}
    eur_saving_fmt = {p: ("n/a" if eur_saving[p] is None else f"{eur_saving[p]:,}") for p in prices}
    delta_mae_mw = max(0.0, float(saved_mwh) / 8760.0) if np.isfinite(saved_mwh) else 0.0
    aufschlag_economics = None      # Headline: Aufschlag |reBAP - Spot|
    upper_bound_economics = None    # Upper Bound: |reBAP| absolut
    if args.rebap_csv and args.spot_csv:
        rebap_p, spot_p = load_rebap_spot_pairs(args.rebap_csv, args.spot_csv)
        aufschlag_economics = saving_from_rebap_spot(delta_mae_mw, rebap_p, spot_p)
        aufschlag_economics["rebap_source"] = args.rebap_csv
        aufschlag_economics["spot_source"] = args.spot_csv
        aufschlag_economics["caveat"] = (
            "reBAP-Nutzen ist Downside-Schutz, kein garantierter linearer Ertrag; "
            "die verteidigbarste Zahl bleibt die reale Bilanzkreis-Abrechnung des Stadtwerks."
        )
        upper_bound_economics = saving_from_real_rebap(delta_mae_mw, load_rebap(args.rebap_csv))
        upper_bound_economics["source_csv"] = args.rebap_csv
        upper_bound_economics["caveat"] = (
            "|reBAP|-Annahme ueberschaetzt den Nutzen (gespart wird nur der Aufschlag reBAP-Spot)."
        )
    elif args.rebap_csv:
        upper_bound_economics = saving_from_real_rebap(delta_mae_mw, load_rebap(args.rebap_csv))
        upper_bound_economics["source_csv"] = args.rebap_csv
        upper_bound_economics["caveat"] = (
            "Ohne Spot-DA-Reihe nur |reBAP|-Annahme verfuegbar — UPPER BOUND. "
            "Mit --spot-csv die belastbare Aufschlag-Headline aktivieren."
        )

    mase = round(m["model"]["MAE_MW"] / m["snaive"]["MAE_MW"], 3) if m["snaive"]["MAE_MW"] else None
    name = args.name or os.path.basename(args.csv)
    out = {
        "dataset": name, "ts_col": ts_col, "load_col": load_col, "unit_in": args.unit,
        "load_level": args.load_level or load_meta["load_level"],
        "load_unit_hint": load_meta["load_unit_hint"],
        "load_candidate": load_meta["load_candidate"],
        "n_days_used": int(len(load2d)), "n_test_days": args.n_test,
        "mean_load_MW": round(float(load2d.mean()), 3),
        "MAE_MW": m["model"]["MAE_MW"], "MAPE_%": m["model"]["MAPE_%"], "MASE_vs_snaive": mase,
        "skill_vs_snaive_%": m["model"]["Skill_vs_SaisonalNaiv_%"],
        "skill_vs_persistenz_%": m["model"]["Skill_vs_Persistenz_%"],
        "coverage80_%": c80["coverage_%"], "coverage90_%": c90["coverage_%"],
        "annual_abs_error_reduction_MWh": round(saved_mwh, 1) if np.isfinite(saved_mwh) else None,
        "annual_eur_saving_estimate": eur_saving,
        "aufschlag_economics": aufschlag_economics,
        "economics_upper_bound": upper_bound_economics,
        "eur_assumption": "Transparenter Proxy: Σ|Prognosefehler| × reBAP-Preis, annualisiert. "
                          "reBAP ist volatil (2024 ~Spot, 2021/22 bis ~100-160 €/MWh). Echte Zahl "
                          "braucht die reale Bilanzkreis-Abrechnung des Stadtwerks.",
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pilot_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    level_text = f", Ebene `{out['load_level']}`" if out["load_level"] else ""
    def _eur(v):
        return "n/a" if v is None else f"{int(round(v)):,}"
    real_rebap_md = ""
    if aufschlag_economics:
        st = aufschlag_economics["spread_over_spot_stats"]
        ub = upper_bound_economics
        real_rebap_md = f"""
## Belastbare Aufschlag-Headline (reBAP − Spot, korrekter Hebel)
Auf Basis der oeffentlichen reBAP- und Spot-DA-Viertelstundenreihen 2024:
**{_eur(aufschlag_economics['eur_per_year_point_median'])} EUR/Jahr** Median,
Band **{_eur(aufschlag_economics['eur_per_year_p25'])}-{_eur(aufschlag_economics['eur_per_year_p75'])} EUR/Jahr**.

| Basis | Wert |
|---|---:|
| dMAE gegen saisonal-naiv | {aufschlag_economics['delta_mae_mw']} MW |
| Median \\|reBAP - Spot\\| | {st['median_abs_spread_over_spot_eur_mwh']} EUR/MWh |
| P25-P75 Aufschlag | {st['p25_eur_mwh']} - {st['p75_eur_mwh']} EUR/MWh |
| Mean \\|reBAP - Spot\\| | {st['mean_abs_spread_over_spot_eur_mwh']} EUR/MWh |
| QH-Paare | {st['n']} |

> reBAP-Nutzen ist Downside-Schutz, kein garantierter linearer Ertrag. Gespart wird nur der
> **Aufschlag** reBAP − Spot, nicht der absolute reBAP-Preis. Die verteidigbarste Zahl bleibt die
> reale Bilanzkreis-Abrechnung des Stadtwerks.

### Oberer Rand |reBAP| (ueberschaetzt — nur als Vergleich)
**{_eur(ub['eur_per_year_point_median'])} EUR/Jahr** Median (Band **{_eur(ub['eur_per_year_p25'])}-{_eur(ub['eur_per_year_p75'])}**) —
nutzt den absoluten reBAP statt des Aufschlags und ueberschaetzt den realen Hebel.
"""
    elif upper_bound_economics:
        st = upper_bound_economics["rebap_spread_stats"]
        real_rebap_md = f"""
## Oberer Rand |reBAP| (ueberschaetzt — keine Spot-Reihe verfuegbar)
Auf Basis der oeffentlichen reBAP-2024-Viertelstundenreihe: **{_eur(upper_bound_economics['eur_per_year_point_median'])} EUR/Jahr**
Median, Band **{_eur(upper_bound_economics['eur_per_year_p25'])}-{_eur(upper_bound_economics['eur_per_year_p75'])} EUR/Jahr**.

| Basis | Wert |
|---|---:|
| dMAE gegen saisonal-naiv | {upper_bound_economics['delta_mae_mw']} MW |
| Median abs(reBAP) | {st['median_abs_spread_eur_mwh']} EUR/MWh |
| P25-P75 abs(reBAP) | {st['p25_eur_mwh']} - {st['p75_eur_mwh']} EUR/MWh |
| reBAP-Werte | {st['n']} Viertelstunden |

> **UPPER BOUND.** Diese Zahl ueberschaetzt den realen Hebel, weil gespart wird nur der
> Aufschlag reBAP − Spot (nicht der absolute reBAP-Preis). Mit `--spot-csv` die belastbare
> Aufschlag-Headline aktivieren.
"""
    md = f"""# NetzPilot — Pilot-Auswertung: {name}

*Leakage-sicherer Day-ahead-Backtest auf dem gelieferten Lastgang. Gepinnte Spalten:
Zeit = `{ts_col}`, Last = `{load_col}`{level_text} (Einheit {args.unit}). {len(load2d)} vollständige Tage,
{args.n_test} Testtage rollierend.*

## Genauigkeit
| Metrik | Wert |
|---|---|
| Ø Last | {out['mean_load_MW']} MW |
| MAE | {out['MAE_MW']} MW |
| MAPE | {out['MAPE_%']} % |
| MASE (vs. saisonal-naiv) | {mase} |
| **Skill vs. saisonal-naiv** | **{out['skill_vs_snaive_%']:+} %** |
| Skill vs. Persistenz | {out['skill_vs_persistenz_%']:+} % |

## Kalibrierte Unsicherheit (CQR)
| Soll | gemessene Coverage |
|---|---|
| 80 % | {out['coverage80_%']} % |
| 90 % | {out['coverage90_%']} % |

Belastbare P10/P50/P90-Bänder statt eines blanken Punktwerts.

## Wirtschaftlicher Hebel (transparente Schätzung)
Bessere Prognose → weniger Ausgleichsenergie. Reduktion der absoluten Prognosefehler-Energie
gegenüber saisonal-naiv: **≈ {out['annual_abs_error_reduction_MWh']} MWh/Jahr**.

| reBAP-Annahme | geschätzte Einsparung/Jahr |
|---|---|
| 30 €/MWh | {eur_saving_fmt[30]} € |
| 60 €/MWh | {eur_saving_fmt[60]} € |
| 120 €/MWh | {eur_saving_fmt[120]} € |

> {out['eur_assumption']}
{real_rebap_md}

## Ehrliche Hinweise
- Leakage-sicher: rollierender Backtest, Modell sieht nie die Zukunft; saisonal-naiv & Persistenz als Baselines.
- Der €-Wert ist ein **transparenter Proxy**; die belastbare Zahl folgt aus der **realen reBAP-/Bilanzkreis-
  Abrechnung** des Stadtwerks. Bitte 1 Jahr 15-min-Lastgang + reale Ausgleichsenergie-Kosten liefern.
- Reproduzierbar: `python scripts/pilot_in_a_box.py --csv <datei> --unit <MW|kW> --rebap-csv data_cache/real/rebap_2024.csv --spot-csv data_cache/real/spot_da_2024.csv`.
"""
    with open(os.path.join(args.out, "pilot_report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(json.dumps(out, indent=2, ensure_ascii=True))
    print(f"\n-> {args.out}/pilot_report.md  +  pilot_metrics.json")


if __name__ == "__main__":
    main()  # entrypoint
