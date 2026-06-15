"""Operative Ende-zu-Ende-Schleife (numpy + stdlib, lauffaehig):
Prognose (morgen, P90) -> Engpass-Erkennung an der Trafo-Grenze -> §14a-Fahrplan ->
Mock-aEMT -> HEMS drosselt Wallbox. Verbindet Prognose-Engine und §14a-Steuerkreis zum Produkt.
KEINE echten Netz-/SMGW-Daten — Simulation des logischen Ablaufs.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from netzpilot.data.smard import load_local_json
from netzpilot.data.synthetic_smallutility import make_small_utility_load
from netzpilot.features.build import to_daily, get_holidays
from netzpilot.forecast import forecast_next_day
from netzpilot.models.robust_corrector import ShrunkCorrector
from netzpilot.control.aemt_mock import AEMTMock
from netzpilot.control.hems_sim import Hems
from netzpilot.control.schema import make_fahrplan, MIN_GUARANTEED_KW

# 1) Prognose fuer eine kleine NS-Zelle (2 MW Peak)
cell = make_small_utility_load(load_local_json("prognose_engine_v1/data/wk*.json"), peak_mw=2.0, seed=0)
load2d, days = to_daily(cell); hol = get_holidays(sorted({d.year for d in days}), "NW")
fp = forecast_next_day(load2d, days, lambda: ShrunkCorrector(10.0), holiday_set=hol)
date = fp["date"]
p90 = {h["hour"]: h["p90"] for h in fp["hours"]}        # MW

# 2) Engpass-Erkennung an der Trafo-Grenze
limit_mw = round(0.92 * max(p90.values()), 2)
congestion = [h for h in range(24) if p90[h] > limit_mw]
print(f"Prognose {date}: Zellen-Peak (P90) {max(p90.values()):.2f} MW, Trafo-Grenze {limit_mw} MW")
print(f"Prognostizierte Engpass-Stunden: {congestion or 'keine'}")

# 3) §14a-Fahrplan fuer die Wallboxen der Zelle waehrend der Engpass-Stunden
aemt = AEMTMock().start()
malo = "DE0009998887776"
def post(obj):
    req = Request(aemt.base_url + "/fahrplan", data=json.dumps(obj).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=5) as r: return r.status
    except HTTPError as e: return e.code
if congestion:
    setpoints = [{"start_utc": f"{date}T{h:02d}:00:00+00:00", "end_utc": f"{date}T{h+1:02d}:00:00+00:00",
                  "p_limit_kw": MIN_GUARANTEED_KW} for h in congestion if h < 23]
    code = post(make_fahrplan(malo, setpoints, reason="forecast_congestion"))
    print(f"NetzPilot -> aEMT: §14a-Fahrplan ({len(setpoints)} Stunden, {MIN_GUARANTEED_KW} kW) -> HTTP {code}")

# 4) HEMS-Wallbox reagiert
hems = Hems(aemt.base_url, malo, device_nominal_kw=11.0)
curtailed = [h for h in range(24) if hems.applied_power_kw(f"{date}T{h:02d}:00:00+00:00")["curtailed"]]
print(f"HEMS: Wallbox in Stunden {curtailed or 'keine'} auf {MIN_GUARANTEED_KW} kW gedrosselt (statt 11 kW).")
print("Ablauf: Prognose -> Engpass -> §14a-Fahrplan -> aEMT -> HEMS -> steuVE. Produkt-Schleife geschlossen.")
aemt.stop()
