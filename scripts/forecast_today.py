# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Erzeugt aus der Lasthistorie den Day-ahead-Fahrplan fuer morgen (P10/P50/P90) als JSON."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.data.smard import load_local_json
from netzpilot.features.build import to_daily, get_holidays
from netzpilot.forecast import forecast_next_day
from netzpilot.models.robust_corrector import ShrunkCorrector

s = load_local_json(sys.argv[1] if len(sys.argv) > 1 else "prognose_engine_v1/data/wk*.json")
load2d, days = to_daily(s)
hol = get_holidays(sorted({d.year for d in days}), "NW")
fp = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0), holiday_set=hol)
os.makedirs("data_cache", exist_ok=True)
json.dump(fp, open("data_cache/forecast_next_day.json", "w"), indent=2, ensure_ascii=False)
print(f"Fahrplan fuer {fp['date']} (P10/P50/P90, MW):")
for r in fp["hours"]:
    if r["hour"] % 3 == 0:
        print(f"  {r['hour']:02d}:00  {r['p10']:8.0f} | {r['p50']:8.0f} | {r['p90']:8.0f}")
# Sanity
ok = all(h["p10"] <= h["p50"] <= h["p90"] for h in fp["hours"])
print("Monotonie P10<=P50<=P90:", ok, "| Werte plausibel:", all(20000 < h["p50"] < 90000 for h in fp["hours"]))
