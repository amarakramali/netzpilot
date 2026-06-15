#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify Tarif-Optimierer (control/tariff.py) — reine stdlib, kein Internet.

Prueft Energieerhaltung, Caps/Fenster, Optimalitaet (Hand-Faelle + Zertifikat + Dominanz gegen
zufaellige zulaessige Fahrplaene), §14a-Engpass (cap=0), Infeasibilitaet, Validierung.

Aufruf: python scripts/verify_tariff.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.control.tariff import optimize_grid_fee_schedule

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

# --- S1: Hand-Fall, Optimum + Ersparnis ---
# fee=[0.10,0.05,0.20,0.05], p_max=10, dt=1 -> ceil=10/h; energy=15.
r = optimize_grid_fee_schedule([0.10, 0.05, 0.20, 0.05], 15.0, 10.0)
check("S1: Energie gedeckt (sum==15)", abs(r["scheduled_kwh"] - 15.0) < 1e-9 and r["feasible"])
check("S1: Kosten == 0,75 (15 kWh in 0,05-Stunden)", abs(r["total_cost_eur"] - 0.75) < 1e-9)
check("S1: schedule fuellt nur die guenstigen Stunden", r["schedule_kwh"][0] == 0.0 and r["schedule_kwh"][2] == 0.0)
check("S1: Ersparnis ggue. Sofort == 0,50", abs(r["saving_eur"] - 0.50) < 1e-9)

# --- S2: uniformer Tarif -> Kosten = fee*E, keine Ersparnis ---
r = optimize_grid_fee_schedule([0.15] * 24, 33.0, 11.0)
check("S2: uniform -> Kosten 4,95", abs(r["total_cost_eur"] - 4.95) < 1e-9)
check("S2: uniform -> Ersparnis 0", abs(r["saving_eur"]) < 1e-9)

# --- S3: §14a-Engpass cap=0 in der guenstigsten Stunde -> dort nichts ---
r = optimize_grid_fee_schedule([0.05, 0.20], 8.0, 10.0, cap_kw=[0.0, 10.0])
check("S3: cap=0 in Stunde0 -> kein Bezug dort", r["schedule_kwh"][0] == 0.0)
check("S3: Bedarf trotzdem gedeckt (8 in Stunde1)", abs(r["schedule_kwh"][1] - 8.0) < 1e-9 and r["feasible"])
check("S3: Kosten 8*0,20 == 1,60", abs(r["total_cost_eur"] - 1.60) < 1e-9)

# --- S4: Verfuegbarkeitsfenster respektiert ---
r = optimize_grid_fee_schedule([0.01, 0.10, 0.20, 0.01], 5.0, 10.0,
                               available=[False, True, True, False])
check("S4: gesperrte (guenstige) Stunden 0/3 leer", r["schedule_kwh"][0] == 0.0 and r["schedule_kwh"][3] == 0.0)
check("S4: Energie im Fenster gedeckt", abs(r["scheduled_kwh"] - 5.0) < 1e-9 and r["feasible"])

# --- S5: Infeasibilitaet ehrlich ---
r = optimize_grid_fee_schedule([0.1, 0.1], 15.0, 5.0)   # ceil 5+5=10 < 15
check("S5: infeasible -> feasible False", r["feasible"] is False)
check("S5: maximal platziert == 10", abs(r["scheduled_kwh"] - 10.0) < 1e-9)
check("S5: shortfall == 5", abs(r["shortfall_kwh"] - 5.0) < 1e-9)

# --- S6: Caps/Fenster nie verletzt + Energieerhaltung (zufaellig) ---
rng = random.Random(7)
viol = False
cons_ok = True
for _ in range(500):
    n = rng.randint(4, 24)
    fee = [round(rng.uniform(0.02, 0.40), 3) for _ in range(n)]
    pmax = rng.uniform(3, 15)
    caps = [rng.uniform(0, pmax) for _ in range(n)]
    avail = [rng.random() > 0.2 for _ in range(n)]
    ceil = [(min(pmax, caps[t]) if avail[t] else 0.0) for t in range(n)]
    Emax = sum(ceil)
    E = rng.uniform(0, Emax) if Emax > 0 else 0.0
    r = optimize_grid_fee_schedule(fee, E, pmax, cap_kw=caps, available=avail)
    for t in range(n):
        if r["schedule_kwh"][t] > ceil[t] + 1e-6:
            viol = True
    if abs(r["scheduled_kwh"] - E) > 1e-6:
        cons_ok = False
check("S6: kein Cap/Fenster verletzt (500 Zufallsfaelle)", not viol)
check("S6: Energieerhaltung in allen feasiblen Faellen", cons_ok)

# --- S7: Optimalitaets-ZERTIFIKAT: kein guenstigerer Slot ungenutzt, waehrend teurerer laeuft ---
# (KKT-Bedingung des LP; beweist Greedy-Optimalitaet ohne Solver)
rng = random.Random(11)
cert_ok = True
for _ in range(500):
    n = rng.randint(4, 24)
    fee = [round(rng.uniform(0.02, 0.40), 3) for _ in range(n)]
    pmax = rng.uniform(3, 15)
    caps = [rng.uniform(0, pmax) for _ in range(n)]
    ceil = [min(pmax, caps[t]) for t in range(n)]
    E = rng.uniform(0, sum(ceil))
    r = optimize_grid_fee_schedule(fee, E, pmax, cap_kw=caps)
    sch = r["schedule_kwh"]
    for t in range(n):
        if sch[t] > 1e-9:                       # Stunde t wird genutzt
            for s in range(n):
                if fee[s] < fee[t] - 1e-9 and sch[s] < ceil[s] - 1e-6:
                    cert_ok = False             # guenstigere Stunde s ungenutzt -> nicht optimal
check("S7: Optimalitaets-Zertifikat erfuellt (500 Faelle)", cert_ok)

# --- S8: Dominanz gegen zufaellige zulaessige Fahrplaene (Greedy nie teurer) ---
rng = random.Random(13)
dom_ok = True
for _ in range(300):
    n = rng.randint(4, 12)
    fee = [round(rng.uniform(0.02, 0.40), 3) for _ in range(n)]
    pmax = rng.uniform(5, 12)
    ceil = [pmax] * n
    E = rng.uniform(0, sum(ceil))
    r = optimize_grid_fee_schedule(fee, E, pmax)
    gcost = r["total_cost_eur"]
    # zufaelliger zulaessiger Fahrplan mit derselben Energie
    for _ in range(5):
        alloc = [0.0] * n
        rem = E
        for t in sorted(range(n), key=lambda i: rng.random()):
            take = min(pmax, rem)
            alloc[t] = take
            rem -= take
            if rem <= 1e-12:
                break
        rcost = sum(fee[t] * alloc[t] for t in range(n))
        if gcost > rcost + 1e-6:
            dom_ok = False
check("S8: Greedy <= jeder zufaellige zulaessige Fahrplan", dom_ok)

# --- S9: Validierung ---
def raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True
check("S9: leere fee -> ValueError", raises(lambda: optimize_grid_fee_schedule([], 5.0, 10.0)))
check("S9: negative Energie -> ValueError", raises(lambda: optimize_grid_fee_schedule([0.1], -1.0, 10.0)))
check("S9: cap-Laenge falsch -> ValueError", raises(lambda: optimize_grid_fee_schedule([0.1, 0.2], 5.0, 10.0, cap_kw=[1.0])))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
