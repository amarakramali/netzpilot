#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify the RESIDUAL-LOAD core logic independently (pure stdlib).

The runner builds Residuallast = Last - Erzeugung on the common hourly axis and then
forecasts it with the SAME engine. The forecasting engine is already verified elsewhere;
the NEW, error-prone part is the join + difference + unit handling. This script re-implements
ONLY that join in plain Python and checks it against hand-built fixtures, so the arithmetic
can be confirmed without numpy/pandas/the VM.

Run:
    cd "<repo root>"
    python scripts/verify_residual_logic.py
"""
import sys


def residual_join(load, gen):
    """Mirror of runner._residual_from_csvs join logic, dict-based.

    load, gen: {iso_hour: value_MW}. Returns (residual dict on common hours, n_common).
    Residuallast = Last - Erzeugung on the intersection of timestamps.
    """
    common = sorted(set(load) & set(gen))
    residual = {t: load[t] - gen[t] for t in common}
    return residual, len(common)


def to_mw(value, unit):
    u = (unit or "MW").lower()
    if u == "kw":
        return value / 1000.0
    if u == "w":
        return value / 1_000_000.0
    return value


def check(name, passed):
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return passed


def main():
    ok = True

    # 1) Basic difference on fully-overlapping hours
    load = {"h0": 10.0, "h1": 12.0, "h2": 8.0}
    gen = {"h0": 3.0, "h1": 5.0, "h2": 9.0}
    res, n = residual_join(load, gen)
    ok &= check("difference correct (10-3,12-5,8-9)",
                res == {"h0": 7.0, "h1": 7.0, "h2": -1.0})
    ok &= check("n_common counts all overlapping hours", n == 3)
    ok &= check("negative residual = Rueckspeisung (8-9=-1)", res["h2"] == -1.0)

    # 2) Partial overlap: only common timestamps survive (inner join)
    load2 = {"h0": 10.0, "h1": 12.0, "h2": 8.0, "h3": 5.0}
    gen2 = {"h1": 4.0, "h2": 2.0, "h9": 1.0}      # h0,h3 only in load; h9 only in gen
    res2, n2 = residual_join(load2, gen2)
    ok &= check("inner join drops non-common hours", set(res2) == {"h1", "h2"})
    ok &= check("inner join values correct (12-4,8-2)",
                res2 == {"h1": 8.0, "h2": 6.0})
    ok &= check("n_common = size of intersection", n2 == 2)

    # 3) Residual mean below load mean when generation is positive (the key sanity check)
    load3 = {f"h{i}": 20.0 for i in range(24)}
    gen3 = {f"h{i}": 6.0 for i in range(24)}      # 30% generation
    res3, _ = residual_join(load3, gen3)
    load_mean = sum(load3.values()) / len(load3)
    res_mean = sum(res3.values()) / len(res3)
    ok &= check("residual mean < load mean (gen>0)", res_mean < load_mean)
    ok &= check("residual mean exact (20-6=14)", abs(res_mean - 14.0) < 1e-9)

    # 4) Unit conversion to MW (loader applies this BEFORE the join)
    ok &= check("kW->MW factor (5000 kW = 5 MW)", abs(to_mw(5000.0, "kW") - 5.0) < 1e-9)
    ok &= check("W->MW factor (2e6 W = 2 MW)", abs(to_mw(2_000_000.0, "W") - 2.0) < 1e-9)
    ok &= check("MW stays MW", abs(to_mw(7.0, "MW") - 7.0) < 1e-9)

    # 5) Generation larger than load -> fully negative residual (net feed-in hour)
    load5 = {"h0": 4.0}
    gen5 = {"h0": 10.0}
    res5, _ = residual_join(load5, gen5)
    ok &= check("net feed-in: residual = 4-10 = -6", res5["h0"] == -6.0)

    print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG -- siehe FAIL oben")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
