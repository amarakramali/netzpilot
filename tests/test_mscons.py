"""MSCONS-Lese-Adapter (W19): EDIFACT-Lastgang → interne Stundenserie. Fixture-frei (run_all_checks-Shim).

Verifiziert an synthetischen, struktur-konformen MSCONS-Nachrichten (edi@energy-Segmentstruktur).
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.data.mscons import (
    looks_like_mscons,
    parse_mscons,
    tokenize,
    load_mscons_hourly,
)

UNB = "UNB+UNOC:3+9900000000001:500+9900000000002:500+240101:1200+1'"
UNH = "UNH+1+MSCONS:D:04B:UN:2.4a'"


def _day_block(values, unit="KWH", day="20240101", end_day="20240102", obis="1-1?:1.29.0"):
    """Eine MSCONS-Nachricht: ein Tagesblock mit len(values) QTY über [day 00:00, end_day 00:00)."""
    segs = [UNB, UNH, "BGM+7+DOC1+9'", "DTM+137:202401011200?+01:303'",
            "LIN+1'", f"PIA+5+{obis}:SRW'",
            f"DTM+163:{day}0000?+01:303'", f"DTM+164:{end_day}0000?+01:303'"]
    for v in values:
        segs.append(f"QTY+220:{v}:{unit}'")
    segs += ["UNT+0+1'", "UNZ+1+1'"]
    return "\n".join(segs)


def _single_intervals(triples):
    """Einzelintervall-Muster: je Wert eine SG10 (QTY + DTM163 + DTM164). triples=[(val,startHHMM)]."""
    segs = [UNB, UNH, "BGM+7+DOC1+9'", "LIN+1'", "PIA+5+1-1?:1.29.0:SRW'"]
    for v, hhmm in triples:
        segs.append(f"QTY+220:{v}:KWH'")
        segs.append(f"DTM+163:20240101{hhmm}?+01:303'")
    return "\n".join(segs) + "\nUNT+0+1'\nUNZ+1+1'"


def test_looks_like_mscons_vs_csv():
    with tempfile.TemporaryDirectory() as td:
        m = os.path.join(td, "x.txt"); open(m, "w").write(UNB + "\n" + UNH)
        c = os.path.join(td, "y.csv"); open(c, "w").write("Datum;von;Last\n01.01.2024;00:00;12,3")
        una = os.path.join(td, "z.edi"); open(una, "w").write("UNA:+.? '" + UNB)
        assert looks_like_mscons(m) is True
        assert looks_like_mscons(una) is True
        assert looks_like_mscons(c) is False


def test_tokenize_release_and_default_separators():
    segs, sep = tokenize("UNB+A?+B:C'QTY+220:5.5:KWH'")
    assert sep["seg"] == "'"
    # Release ?+ → '+' bleibt literal im selben Element
    assert segs[0][0] == "UNB" and segs[0][1][0] == ["A+B", "C"]
    assert segs[1][0] == "QTY" and segs[1][1][0] == ["220", "5.5", "KWH"]


def test_una_overrides_separators():
    # UNA-Zeichen (Position): component=| element=# decimal=, release=? space=' ' segment=~
    text = "UNA|#,? ~UNB#A#B~QTY#220|7,5|KWH~"
    segs, sep = tokenize(text)
    assert sep["elem"] == "#" and sep["comp"] == "|" and sep["seg"] == "~"
    qty = [s for s in segs if s[0] == "QTY"][0]
    assert qty[1][0] == ["220", "7,5", "KWH"]      # Dezimal-Komma bleibt Rohwert (erst _to_float wandelt)


def test_day_block_96_quarter_hours():
    vals = [round(10.0 + i * 0.1, 2) for i in range(96)]
    parsed = parse_mscons(_day_block(vals))
    s = parsed["series"]
    assert len(s) == 96
    assert parsed["meta"]["pattern"] == "period_block"
    assert parsed["meta"]["interval_seconds"] == 900           # 15 min
    assert s.index.is_monotonic_increasing
    # Werte in Reihenfolge erhalten
    assert np.allclose(s.to_numpy(), np.array(vals))
    # gleichmäßiges 15-min-Raster
    deltas = s.index.to_series().diff().dropna().dt.total_seconds().unique()
    assert list(deltas) == [900.0]


def test_single_interval_pattern():
    parsed = parse_mscons(_single_intervals([(5.0, "0000"), (6.0, "0015"), (7.0, "0030")]))
    s = parsed["series"]
    assert len(s) == 3
    assert parsed["meta"]["pattern"] == "single_interval"
    assert np.allclose(s.to_numpy(), [5.0, 6.0, 7.0])


def test_energy_kwh_converted_to_mw_power():
    # 250 kWh je 15 min = 1000 kW = 1 MW mittlere Leistung -> Stundenmittel 1.0 MW
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "lg.mscons")
        open(p, "w").write(_day_block([250.0] * 96, unit="KWH"))
        hourly, ts_label, load_label, meta = load_mscons_hourly(p, unit="MW")
        assert len(hourly) == 24
        assert np.allclose(hourly.to_numpy(), 1.0)
        assert "Energie" in meta["power_conversion"]
        assert meta["obis"] == "1-1:1.29.0"            # Release im OBIS korrekt aufgelöst


def test_power_kw_taken_directly():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "lg.mscons")
        open(p, "w").write(_day_block([2000.0] * 96, unit="KW"))   # 2000 kW = 2 MW
        hourly, _, _, meta = load_mscons_hourly(p, unit="MW")
        assert np.allclose(hourly.to_numpy(), 2.0)
        assert "Leistung" in meta["power_conversion"]


def test_dst_long_day_100_values_does_not_crash():
    # Sommer->Winter: 100 Viertelstundenwerte an einem Tag; muss tolerant geparst werden
    parsed = parse_mscons(_day_block([100.0] * 100, day="20241027", end_day="20241028"))
    assert len(parsed["series"]) == 100
    assert parsed["series"].index.is_monotonic_increasing


def test_empty_message_raises():
    with pytest.raises(ValueError):
        parse_mscons(UNB + "\n" + UNH + "\nBGM+7+DOC1+9'\nUNT+0+1'")


def test_unknown_qty_qualifier_is_flagged_not_dropped():
    msg = "\n".join([UNB, UNH, "LIN+1'", "DTM+163:202401010000?+01:303'",
                     "DTM+164:202401010015?+01:303'", "QTY+998:42.0:KWH'", "UNT+0+1'"])
    parsed = parse_mscons(msg)
    assert len(parsed["series"]) == 1
    assert parsed["series"].iloc[0] == 42.0
    assert any("998" in w for w in parsed["meta"]["warnings"])
