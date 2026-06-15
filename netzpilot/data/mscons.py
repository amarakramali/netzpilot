"""MSCONS-Lese-Adapter (read-only) — EDIFACT-Lastgang → interne Stundenserie.

Stadtwerke bekommen ihre Lastgänge in der Marktkommunikation als EDIFACT-MSCONS-Nachricht
(edi@energy, BNetzA/BK6). Dieser Adapter liest den LASTGANG-TEIL einer solchen Nachricht und
liefert dieselbe stündliche Leistungsreihe wie der CSV-Loader — die Engine läuft danach unverändert.

BEWUSST NUR LESER (Scope-Grenze, „partner, don't rebuild"):
  - kein Senden, keine MaBiS-/EDM-Marktprozessabwicklung,
  - keine Quittungen (APERAK/CONTRL), keine Connect+/AS4-Übertragung,
  - keine Stammdaten-/Bilanzierungslogik — ausschließlich Messwert-Extraktion.

EHRLICHE REIFEGRAD-GRENZE: verifiziert an synthetischen, struktur-konformen MSCONS-Beispielen
(edi@energy-Segmentstruktur), NICHT an einem breiten Korpus echter Marktnachrichten. MSCONS ist
spröder als CSV (EDIFACT-Hüllen, OBIS-Varianten, DST, Intervall- vs. Registerwerte). Unklare
Konstrukte werden in meta['warnings'] ehrlich gemeldet statt still geraten. Erste echte
Stadtwerk-Nachricht im Pilot gegenprüfen.

Quellen zur Struktur: EDI@Energy MSCONS Anwendungshandbuch / MIG (BNetzA, BK6).
"""
from __future__ import annotations

import re
import pandas as pd

# EDIFACT-Standard-Trennzeichen (überschrieben durch ein UNA-Segment, falls vorhanden):
#   component (:)  element (+)  decimal (.)  release/escape (?)  reserved/space  segment (')
_DEFAULTS = {"comp": ":", "elem": "+", "dec": ".", "rel": "?", "seg": "'"}

# DTM-Qualifier für den Mess-/Intervallzeitraum (Beginn/Ende). 163/164 = Beginn/Ende Messperiode.
_DTM_START = {"163", "Z01", "200"}
_DTM_END = {"164", "Z02", "201"}
# QTY-Qualifier, die einen Messwert tragen (Verbrauch/Erzeugung/Bezug). Bewusst weit gefasst,
# unbekannte werden gemeldet, nicht verworfen.
_QTY_VALUE = {"220", "67", "167", "31", "Z01", "Z02", "Z18", "46", "201"}


def looks_like_mscons(path: str) -> bool:
    """Heuristik: beginnt der Datei-Anfang mit der EDIFACT-Hülle (UNA/UNB) oder enthält UNH+...MSCONS?"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(4096)
    except OSError:
        return False
    h = head.lstrip("﻿").lstrip()
    if h.startswith("UNA") or h.startswith("UNB"):
        return True
    return "MSCONS" in head and ("UNH" in head or "UNB" in head)


def _parse_una(text: str) -> tuple[dict, str]:
    """Liest ein optionales UNA-Segment (6 Zeichen nach 'UNA') und gibt (Trennzeichen, Rest-Text)."""
    sep = dict(_DEFAULTS)
    t = text.lstrip("﻿")
    if t.startswith("UNA") and len(t) >= 9:
        chars = t[3:9]
        sep = {"comp": chars[0], "elem": chars[1], "dec": chars[2], "rel": chars[3], "seg": chars[5]}
        t = t[9:]
    return sep, t


def _split(s: str, delim: str, rel: str, final: bool) -> list[str]:
    """Splittet s an delim mit EDIFACT-Release-Semantik.

    Der Release-Char (rel) hebt die Sonderbedeutung des FOLGENDEN Zeichens auf. Beim hierarchischen
    Parsen (Segmente→Elemente→Komponenten) muss er auf den oberen Ebenen ERHALTEN bleiben (final=False),
    damit die untere Ebene ihn noch sieht; erst auf der Komponenten-Ebene (final=True) wird er
    aufgelöst (entfernt, Folgezeichen literal). So bleibt z. B. ein escaptes Element-Trennzeichen im
    OBIS bis zur richtigen Ebene geschützt.
    """
    out, buf, i = [], [], 0
    while i < len(s):
        c = s[i]
        if c == rel and i + 1 < len(s):
            if final:
                buf.append(s[i + 1])            # auflösen: nur das Folgezeichen, literal
            else:
                buf.append(c); buf.append(s[i + 1])   # rel + Folgezeichen für die nächste Ebene behalten
            i += 2
            continue
        if c == delim:
            out.append("".join(buf)); buf = []
            i += 1
            continue
        buf.append(c); i += 1
    out.append("".join(buf))
    return out


def tokenize(text: str) -> tuple[list[tuple[str, list[list[str]]]], dict]:
    """EDIFACT-Text → Liste von Segmenten (tag, elemente[komponenten[]]) + Trennzeichen.

    Newlines/Whitespace zwischen Segmenten werden ignoriert (viele Sender brechen je Segment um).
    """
    sep, body = _parse_una(text)
    segments = []
    for raw in _split(body, sep["seg"], sep["rel"], final=False):
        seg = raw.strip("\r\n\t ")
        if not seg:
            continue
        elements = _split(seg, sep["elem"], sep["rel"], final=False)
        tag = elements[0].strip()
        if not tag:
            continue
        comps = [_split(el, sep["comp"], sep["rel"], final=True) for el in elements[1:]]
        segments.append((tag, comps))
    return segments, sep


def _parse_dtm_timestamp(value: str, fmt: str) -> pd.Timestamp | None:
    """DTM-Wert → tz-bewusster pd.Timestamp.

    Formate (edi@energy): 102=CCYYMMDD, 203=CCYYMMDDHHMM (lokal), 303=CCYYMMDDHHMM+ZZ (mit
    Zeitzonen-Offset; das '+'/'-' wurde im EDIFACT mit Release escaped und ist nach dem Tokenizing
    Teil des Werts). Ohne Offset wird Europe/Berlin angenommen (deutsche Marktnachricht) und nach
    UTC konvertiert — die interne Achse ist UTC, to_daily_local rechnet zurück nach Ortszeit.
    """
    m = re.match(r"^(\d{8,14})\s*([+-]\d{2,4})?$", value.strip())
    if not m:
        return None
    digits, off = m.group(1), m.group(2)
    try:
        if len(digits) >= 12:
            ts = pd.Timestamp(year=int(digits[0:4]), month=int(digits[4:6]), day=int(digits[6:8]),
                              hour=int(digits[8:10]), minute=int(digits[10:12]))
        else:                                              # nur Datum (102)
            ts = pd.Timestamp(year=int(digits[0:4]), month=int(digits[4:6]), day=int(digits[6:8]))
    except ValueError:
        return None
    if off:
        sign = 1 if off[0] == "+" else -1
        oh = int(off[1:3]); om = int(off[3:5]) if len(off) >= 5 else 0
        ts = ts.tz_localize("UTC") - sign * pd.Timedelta(hours=oh, minutes=om)
    else:
        # lokale deutsche Zeit; mehrdeutige/ungültige DST-Stunden tolerant abbilden
        ts = ts.tz_localize("Europe/Berlin", ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
    return ts


def _to_float(raw: str, dec: str) -> float | None:
    if raw is None or raw == "":
        return None
    s = raw.strip().replace(dec, ".") if dec != "." else raw.strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_mscons(text: str) -> dict:
    """Parst eine MSCONS-Nachricht und gibt {series, meta}.

    series: pd.Series (tz-aware UTC-Index → Wert in der Quelleinheit, vor Leistungs-/Einheiten-
            normalisierung). meta enthält Einheit, OBIS, erkanntes Muster, Intervallsekunden und
            ehrliche Warnungen.

    Block-Modell (deckt beide gängigen Muster ab): ein Zeitblock beginnt mit DTM(Beginn); die bis
    zum nächsten Beginn/LIN folgenden QTY-Werte werden gleichmäßig über [Beginn, Ende] verteilt.
    Ein Block mit genau einer QTY = Einzelintervall (Ende optional). Fehlt das Ende, wird es aus dem
    nächsten Blockbeginn bzw. der medianen Intervalldauer geschätzt.
    """
    segments, sep = tokenize(text)
    warnings: list[str] = []
    obis = None
    units: list[str] = []

    # Ereignisse in Auftrittsreihenfolge: QTY-Werte und Intervall-Beginn/-Ende. Die SG10-Gruppe
    # sendet je Messwert QTY+DTM (QTY zuerst); der Tagesblock sendet erst DTM(Beginn/Ende), dann
    # viele QTY. Beide Reihenfolgen werden über diese Ereignisliste sauber rekonstruiert.
    events: list[tuple] = []
    for tag, comps in segments:
        if tag == "PIA" and len(comps) >= 2 and comps[1] and comps[1][0]:
            obis = comps[1][0]
        elif tag == "DTM" and comps and comps[0]:
            q = comps[0][0] if comps[0] else ""
            val = comps[0][1] if len(comps[0]) > 1 else ""
            fmt = comps[0][2] if len(comps[0]) > 2 else "203"
            if q in _DTM_START or q in _DTM_END:
                ts = _parse_dtm_timestamp(val, fmt)
                if ts is None:
                    warnings.append(f"DTM nicht interpretierbar: {comps[0]}")
                else:
                    events.append(("s" if q in _DTM_START else "e", ts))
        elif tag == "QTY" and comps and comps[0]:
            qual = comps[0][0] if comps[0] else ""
            value = _to_float(comps[0][1] if len(comps[0]) > 1 else "", sep["dec"])
            if value is None:
                continue
            unit = comps[0][2] if len(comps[0]) > 2 else ""
            if qual not in _QTY_VALUE:
                warnings.append(f"QTY-Qualifier {qual!r} unbekannt — als Messwert behandelt.")
            if unit:
                units.append(unit.upper())
            events.append(("q", value))

    q_events = [e for e in events if e[0] == "q"]
    s_events = [e for e in events if e[0] == "s"]
    if not q_events:
        raise ValueError("Keine Lastgang-Messwerte (QTY) in der MSCONS-Nachricht gefunden.")

    median_step = 900.0      # Default-Intervall (15 min); im Periodenblock aus den Daten überschrieben
    records: list[tuple[pd.Timestamp, float]] = []
    if s_events and len(q_events) <= 1.5 * len(s_events):
        # Einzelintervall: ~ein Beginn-Zeitstempel je Messwert → Paarung in Reihenfolge.
        pattern = "single_interval"
        starts = [e[1] for e in s_events]
        for i, (_, v) in enumerate(q_events):
            if i < len(starts):
                records.append((starts[i], v))
            else:
                warnings.append("Mehr QTY als Zeitstempel — überzählige Werte verworfen.")
    else:
        # Periodenblock: Beginn öffnet einen Block, folgende QTY gleichmäßig über [Beginn, Ende].
        pattern = "period_block"
        blocks: list[dict] = []
        cur = None
        for kind, *payload in events:
            if kind == "s":
                if cur is not None and cur["qty"]:
                    blocks.append(cur)
                cur = {"start": payload[0], "end": None, "qty": []}
            elif kind == "e":
                if cur is None:
                    cur = {"start": None, "end": payload[0], "qty": []}
                else:
                    cur["end"] = payload[0]
            else:  # "q"
                if cur is None:
                    cur = {"start": None, "end": None, "qty": []}
                cur["qty"].append(payload[0])
        if cur is not None and cur["qty"]:
            blocks.append(cur)
        spans = [((b["end"] - b["start"]).total_seconds() / max(1, len(b["qty"])))
                 for b in blocks if b["start"] is not None and b["end"] is not None and b["qty"]]
        median_step = sorted(spans)[len(spans) // 2] if spans else 900.0   # Default 15 min
        for i, b in enumerate(blocks):
            n = len(b["qty"])
            if n == 0:
                continue
            start = b["start"]
            if start is None:
                warnings.append("QTY-Block ohne Beginn-Zeitstempel übersprungen.")
                continue
            end = b["end"]
            if end is None:
                nxt = blocks[i + 1]["start"] if i + 1 < len(blocks) else None
                end = nxt if nxt is not None else start + pd.Timedelta(seconds=median_step * n)
            step = (end - start).total_seconds() / n
            if step <= 0:
                step = median_step
            for j, v in enumerate(b["qty"]):
                records.append((start + pd.Timedelta(seconds=step * j), v))

    if not records:
        raise ValueError("Keine Lastgang-Messwerte mit Zeitbezug in der MSCONS-Nachricht gefunden.")

    records.sort(key=lambda r: r[0])
    idx = pd.DatetimeIndex([r[0] for r in records])
    series = pd.Series([r[1] for r in records], index=idx, dtype=float)
    series = series[~series.index.duplicated(keep="first")]

    unit = max(set(units), key=units.count) if units else ""
    step_s = int(round((series.index.to_series().diff().dropna().median().total_seconds())
                       if len(series) > 1 else median_step))
    meta = {
        "format": "MSCONS",
        "obis": obis,
        "qty_unit": unit,
        "n_values": int(len(series)),
        "interval_seconds": step_s,
        "n_timestamps": len(s_events),
        "pattern": pattern,
        "warnings": warnings,
        "scope_note": ("read-only Lastgang-Extraktion; kein Senden/MaBiS/APERAK. "
                       "An synthetischen struktur-konformen Beispielen verifiziert, nicht an echtem Marktkorpus."),
    }
    return {"series": series, "meta": meta}


# Energieeinheiten (Arbeit je Intervall) vs. Leistungseinheiten — für die Umrechnung in Leistung.
_ENERGY_UNITS = {"KWH", "MWH", "WH", "KWT"}     # KWT taucht real als kWh-Variante auf
_POWER_UNITS = {"KW", "MW", "W", "KWH/H"}


def load_mscons_hourly(path: str, unit: str = "MW"):
    """MSCONS-Datei → (stündliche Leistungsreihe in MW, ts_label, load_label, meta).

    Spiegelt den Rückgabevertrag von robust_load_csv(return_meta=True). Energie-QTY (kWh/MWh je
    Intervall) werden über die Intervalldauer in mittlere Leistung umgerechnet; Leistungs-QTY (kW/MW)
    direkt genommen. Ergebnis wie beim CSV-Pfad: stündliches Mittel in MW.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    parsed = parse_mscons(text)
    s = parsed["series"]
    meta = parsed["meta"]
    qty_unit = (meta.get("qty_unit") or "").upper()
    step_h = max(1e-9, (meta.get("interval_seconds") or 900) / 3600.0)

    note = ""
    if qty_unit in _ENERGY_UNITS:
        power = s / step_h                       # Arbeit je Intervall → mittlere Leistung
        scale = {"KWH": 1e-3, "KWT": 1e-3, "WH": 1e-6, "MWH": 1.0}.get(qty_unit, 1e-3)
        power_mw = power * scale
        note = f"Energie {qty_unit}/Intervall → Leistung (÷{step_h:.3f} h), in MW"
    elif qty_unit in _POWER_UNITS:
        scale = {"KW": 1e-3, "W": 1e-6, "MW": 1.0}.get(qty_unit, 1e-3)
        power_mw = s * scale
        note = f"Leistung {qty_unit} → MW"
    else:
        # Einheit unbekannt → den vom Aufrufer angegebenen unit-Hint anwenden (wie CSV-Pfad), ehrlich vermerken
        u = (unit or "MW").lower()
        scale = {"mw": 1.0, "kw": 1e-3, "w": 1e-6}.get(u, 1.0)
        power_mw = s * scale
        note = f"QTY-Einheit unbekannt ({qty_unit!r}); unit-Hint '{unit}' angewandt"
        meta.setdefault("warnings", []).append(note)

    if power_mw.index.tz is None:
        power_mw.index = power_mw.index.tz_localize("UTC")
    hourly = power_mw.resample("1h").mean().dropna()
    meta["power_conversion"] = note
    meta["load_col"] = f"MSCONS {meta.get('obis') or 'Lastgang'}"
    meta["load_level"] = None
    return hourly, "DTM", meta["load_col"], meta
