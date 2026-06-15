#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify rollierender Re-Dispatch (control/redispatch.py) — reine stdlib, kein Internet.

Prüft die entscheidenden Eigenschaften:
  1) Kein Engpass -> kein Eingriff, voller Bedarf, 0 Abregelung.
  2) Engpass -> Summe der steuVE-Limits hält die Netzgrenze (Gesamtlast <= Schwelle), jede steuVE >= 4,2.
  3) Rollierend mit FRÜH überschätzter, später korrigierter Prognose regelt WENIGER ab als die
     pauschale Dauerdimmung (saved_vs_naive_kwh > 0) — der eigentliche Mehrwert.
  4) from_single_path: 24h-Bahn -> 24 Entscheidungsfenster, Stunde t sieht Rest des Tages.

Aufruf: python scripts/verify_redispatch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.control.redispatch import rolling_redispatch, from_single_path
from netzpilot.control.schema import MIN_GUARANTEED_KW

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond


# --- Szenario 1: kein Engpass (Last immer unter Schwelle) ---
demands = [11.0] * 50            # 50 Wärmepumpen à 11 kW = 550 kW steuerbar
threshold = 5000.0              # Netz weit über Last -> nie Engpass
base = 3000.0                   # Grundlast
path = [base + sum(demands)] * 24
r = rolling_redispatch(from_single_path(path), threshold, demands)
check("Szenario 1: keine Eingriffsstunde", r["intervention_hours"] == 0)
check("Szenario 1: 0 kWh abgeregelt", r["total_shed_kwh"] == 0.0)

# --- Szenario 2: echter Engpass in einzelnen Stunden, Constraints prüfen ---
# Grundlast 4600 kW + 550 kW steuVE = 5150 kW; Schwelle 5000 -> 150 kW müssen runter in Spitzenstunden
threshold = 5000.0
base = 4600.0
peak = base + sum(demands)      # 5150 kW
flat = base + 200.0            # 4800 kW (kein Engpass)
path = [flat]*8 + [peak]*4 + [flat]*12    # 4 Engpassstunden mittags
r = rolling_redispatch(from_single_path(path), threshold, demands)
interv = [h for h in r["hourly"] if h["intervention"]]
check("Szenario 2: genau 4 Eingriffsstunden", len(interv) == 4)
# In jeder Eingriffsstunde: Gesamtlast (base + Summe limits) <= Schwelle (+kleine Toleranz)
all_cap_ok = all((base + sum(h["limits_kw"])) <= threshold + 1e-6 for h in interv)
check("Szenario 2: Netzgrenze in jeder Eingriffsstunde gehalten", all_cap_ok)
all_floor_ok = all(all(l >= MIN_GUARANTEED_KW - 1e-9 for l in h["limits_kw"]) for h in interv)
check("Szenario 2: §14a-Mindestleistung 4,2 kW überall gewahrt", all_floor_ok)

# --- Szenario 3: rollierender Vorteil — frühe Prognose überschätzt, später Entwarnung ---
# Statisch (Vortagsprognose) sieht 6 Engpassstunden und würde alle dimmen.
# Rollierend: in 3 davon stellt sich kurzfristig heraus, dass die Last doch unter der Schwelle bleibt.
threshold = 5000.0
base = 4600.0
# Pro Entscheidungsstunde t eine eigene Bahn: die NÄCHSTE Stunde ist die aktuell beste Info.
# Wir bauen 24 Bahnen; in Stunden 10..15 sagt die Vortagsbahn "peak", aber die aktuelle (path[t][0])
# zeigt für 12,13,14 Entwarnung (flat) — dort darf NICHT abgeregelt werden.
flat = base + 200.0
peak = base + sum(demands)
actual_next = {10: peak, 11: peak, 12: flat, 13: flat, 14: flat, 15: peak}
forecasts = []
for t in range(24):
    nxt = actual_next.get(t, flat)
    # Bahn: erste Stunde = aktuelle (genaue) Info, Rest grob (egal, nur erste bindet)
    forecasts.append([nxt] + [peak]*(24 - t - 1))
r = rolling_redispatch(forecasts, threshold, demands)
check("Szenario 3: nur 3 echte Eingriffsstunden (12-14 entwarnt)", r["intervention_hours"] == 3)
# Pauschal hätte alle 6 "peak"-Stunden gedimmt -> rollierend spart Abregel-Energie
check("Szenario 3: rollierend spart ggü. pauschal (saved>0)", r["saved_vs_naive_kwh"] > 0)
print(f"     -> rollierend {r['total_shed_kwh']} kWh vs. pauschal {r['naive_shed_kwh']} kWh "
      f"(gespart {r['saved_vs_naive_kwh']} kWh)")

# --- Szenario 4: from_single_path-Struktur ---
fp = from_single_path([1.0, 2.0, 3.0])
check("Szenario 4: Fenster t=0 sieht ganzen Tag", fp[0] == [1.0, 2.0, 3.0])
check("Szenario 4: Fenster t=2 sieht nur Rest", fp[2] == [3.0])

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
