"""End-to-End-Demo des §14a-Steuerkreises (stdlib, lauffaehig):
NetzPilot (Prognose) -> Fahrplan -> Mock-aEMT (zertifizierter Gatekeeper) -> HEMS -> Wallbox.
Zeigt: netzdienliche Drosselung auf die §14a-Mindestleistung, und dass der aEMT illegale
Fahrplaene (unter 4,2 kW) ablehnt. KEIN echtes SMGW — reine Simulation des logischen Kreises.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from netzpilot.control.aemt_mock import AEMTMock
from netzpilot.control.hems_sim import Hems
from netzpilot.control.schema import make_fahrplan, MIN_GUARANTEED_KW


def post(url, obj):
    req = Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, json.loads(e.read())


aemt = AEMTMock().start()
print("Mock-aEMT laeuft auf", aemt.base_url)
malo = "DE0001234567890"
day = "2024-01-15"

fp = make_fahrplan(malo, [{"start_utc": f"{day}T17:00:00+00:00", "end_utc": f"{day}T20:00:00+00:00",
                           "p_limit_kw": MIN_GUARANTEED_KW}], reason="forecast_congestion")
print("NetzPilot -> aEMT  POST /fahrplan :", *post(aemt.base_url + "/fahrplan", fp))

bad = {**fp, "schedule_id": "bad", "setpoints": [{"start_utc": f"{day}T17:00:00+00:00",
        "end_utc": f"{day}T20:00:00+00:00", "p_limit_kw": 2.0}]}
print("Illegaler Fahrplan (2,0 kW)        :", *post(aemt.base_url + "/fahrplan", bad), "<- vom aEMT abgelehnt")

hems = Hems(aemt.base_url, malo, device_nominal_kw=11.0)
print("\nHEMS-Wallbox (Nennleistung 11 kW), Tag", day)
print("Std | §14a-Limit | Wallbox | Status")
curtailed_kwh = 0.0
for h in range(24):
    r = hems.applied_power_kw(f"{day}T{h:02d}:00:00+00:00")
    curtailed_kwh += max(0.0, r["desired_kw"] - r["applied_kw"])
    lim = "  -  " if r["limit_kw"] is None else f"{r['limit_kw']:.1f}kW"
    print(f" {h:02d} |   {lim:>6}  | {r['applied_kw']:>5.1f}kW | {'GEDROSSELT' if r['curtailed'] else 'frei'}")
print(f"\nDurch §14a-Drosselung nicht geladen: {curtailed_kwh:.1f} kWh  "
      f"(Mindestleistung {MIN_GUARANTEED_KW} kW stets eingehalten)")
print("Wertschoepfungskette: NetzPilot (Prognose/Berechnung) -> REST/JSON -> aEMT (zertifiziert) "
      "-> CLS/SMGW -> HEMS -> steuVE")
aemt.stop()
