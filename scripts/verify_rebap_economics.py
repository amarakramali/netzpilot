#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify the reBAP / Spot economics numbers on the REAL 2024 data.

Run:
    cd "<repo root>"            # the folder that contains the 'netzpilot' package
    python scripts/verify_rebap_economics.py

Pure stdlib (no numpy/pandas/internet needed). Does two independent things:

  1) Recomputes the three €-levers DIRECTLY from the CSVs with plain Python, so the
     numbers do not depend on the project code at all (independent check).
  2) Imports netzpilot.eval.economics and exercises the real functions
     (expected_saving_eur / saving_from_rebap_spot / saving_from_real_rebap),
     then checks that the module agrees with the independent computation.

Finally prints PASS/FAIL against the documented expectations:
    |reBAP| median        ~ 99   EUR/MWh   (overstated upper bound)
    |reBAP - Spot| median ~ 65   EUR/MWh   (volatility / downside band, still high)
    signed mean(reBAP-Spot) ~ 7  EUR/MWh   (honest EXPECTED lever)
    expected_saving(0.15 MW) ~ 9-10k EUR/yr  and  << spread-based  <<  |reBAP| upper bound
"""
import os
import sys
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)               # repo root = parent of scripts/
sys.path.insert(0, ROOT)

REBAP_CSV = os.path.join(ROOT, "data_cache", "real", "rebap_2024.csv")
SPOT_CSV = os.path.join(ROOT, "data_cache", "real", "spot_da_2024.csv")
DMAE_MW = 0.15                             # 5 MW EE-Portfolio, MAE 8% -> 5%


def load_last_column(path):
    """Read the last column of a CSV as floats.

    These files: delimiter ';', decimal POINT (e.g. 'Zeit;reBAP_EUR_MWh' -> '...;75.15').
    We also tolerate a comma-decimal fallback ('75,15') in case a column ever uses it.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        head = f.readline()
        delim = ";" if head.count(";") >= head.count(",") else ","
        f.seek(0)
        rdr = csv.reader(f, delimiter=delim)
        rows = list(rdr)
    vals = []
    for row in rows[1:]:                    # skip header
        if not row:
            continue
        raw = row[-1].strip()
        if not raw:
            continue
        try:
            v = float(raw)                 # files use dot decimals -> direct
        except ValueError:
            try:
                v = float(raw.replace(".", "").replace(",", "."))  # comma-decimal fallback
            except ValueError:
                continue
        vals.append(v)
    return vals


def _q(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    p = (len(sorted_vals) - 1) * q
    lo, hi = int(p), min(int(p) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (p - lo)


def main():
    for p in (REBAP_CSV, SPOT_CSV):
        if not os.path.exists(p):
            print(f"FEHLT: {p}")
            sys.exit(2)

    rebap = load_last_column(REBAP_CSV)
    spot = load_last_column(SPOT_CSV)
    print(f"geladen: reBAP n={len(rebap)}  Spot n={len(spot)}")
    n = min(len(rebap), len(spot))
    rebap, spot = rebap[:n], spot[:n]

    # --- 1) independent stdlib computation -------------------------------------
    abs_rebap = sorted(abs(x) for x in rebap)
    abs_spread = sorted(abs(r - s) for r, s in zip(rebap, spot))
    signed = [r - s for r, s in zip(rebap, spot)]
    med_rebap = _q(abs_rebap, 0.5)
    med_spread = _q(abs_spread, 0.5)
    signed_mean = sum(signed) / len(signed)

    print("\n--- unabhaengig (stdlib, direkt aus CSV) ---")
    print(f"  |reBAP| Median            = {med_rebap:8.2f} EUR/MWh   (Erwartung ~99)")
    print(f"  |reBAP-Spot| Median       = {med_spread:8.2f} EUR/MWh   (Erwartung ~65)")
    print(f"  signed mean(reBAP-Spot)   = {signed_mean:8.2f} EUR/MWh   (Erwartung ~7)")

    # --- 2) exercise the real module ------------------------------------------
    from netzpilot.eval.economics import (
        rebap_spread_stats, spread_over_spot_stats,
        expected_saving_eur, saving_from_rebap_spot, saving_from_real_rebap,
    )
    st_rebap = rebap_spread_stats(rebap)
    st_spread = spread_over_spot_stats(rebap, spot)
    exp = expected_saving_eur(DMAE_MW, rebap, spot)
    spread_save = saving_from_rebap_spot(DMAE_MW, rebap, spot)
    ub = saving_from_real_rebap(DMAE_MW, rebap)

    print("\n--- netzpilot.eval.economics ---")
    print(f"  rebap_spread_stats.median   = {st_rebap['median_abs_spread_eur_mwh']:8.2f} EUR/MWh")
    print(f"  spread_over_spot.median     = {st_spread['median_abs_spread_over_spot_eur_mwh']:8.2f} EUR/MWh")
    print(f"  expected_saving signed_mean = {exp['signed_mean_spread_eur_mwh']:8.2f} EUR/MWh")
    print(f"  => ERWARTET (signed mean)   = {exp['expected_eur_per_year']:>10,} EUR/Jahr".replace(",", "."))
    print(f"  => Aufschlag-Band |spread|  = {spread_save['eur_per_year_point_median']:>10,} EUR/Jahr "
          f"(P25 {spread_save['eur_per_year_p25']:,} .. P75 {spread_save['eur_per_year_p75']:,})".replace(",", "."))
    print(f"  => oberer Rand |reBAP|      = {ub['eur_per_year_point_median']:>10,} EUR/Jahr".replace(",", "."))

    # --- 3) checks -------------------------------------------------------------
    def close(a, b, tol):
        return abs(a - b) <= tol

    checks = [
        ("module |reBAP| == stdlib",
         close(st_rebap["median_abs_spread_eur_mwh"], med_rebap, 0.5)),
        ("module |reBAP-Spot| == stdlib",
         close(st_spread["median_abs_spread_over_spot_eur_mwh"], med_spread, 0.5)),
        ("module signed_mean == stdlib",
         close(exp["signed_mean_spread_eur_mwh"], signed_mean, 0.5)),
        ("|reBAP| median in [80,120]",      80 <= med_rebap <= 120),
        ("|reBAP-Spot| median in [40,90]",  40 <= med_spread <= 90),
        ("signed mean in [2,20]",           2 <= signed_mean <= 20),
        ("ERWARTET < Aufschlag-Band",
         exp["expected_eur_per_year"] < spread_save["eur_per_year_point_median"]),
        ("Aufschlag-Band <= oberer Rand",
         spread_save["eur_per_year_point_median"] <= ub["eur_per_year_point_median"]),
        ("ERWARTET in [5k,15k] EUR/Jahr",
         5_000 <= exp["expected_eur_per_year"] <= 15_000),
    ]
    print("\n--- Checks ---")
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed

    print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG -- siehe FAIL oben")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
