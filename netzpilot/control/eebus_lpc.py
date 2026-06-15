"""EEBUS-LPC-Mapping für den §14a-Fahrplan.

EEBUS ist das von BNetzA/BSI für §14a vorgesehene Protokoll. Der einschlägige Use-Case ist
**LPC — „Limitation of Power Consumption"**: der Netzbetreiber/CEM gibt einer steuerbaren Einrichtung
eine Wirkleistungs-Bezugsgrenze vor, mit einem **Failsafe-Wert** (Rückfallgrenze bei Kommunikations-
ausfall) und einer Gültigkeitsdauer. NetzPilot erzeugt den §14a-Fahrplan (Wirkleistungsgrenzen je MaLo
und Zeitfenster, schema.make_fahrplan) und ÜBERSETZT ihn hier in die LPC-Datenstruktur. NetzPilot
sendet NICHT selbst über EEBUS SHIP/SPINE und berührt NIE das SMGW — der Transport ist Sache des
zertifizierten aEMT/CEM (Rollentrennung §14a).

Was hier ECHT + verifizierbar ist: die Daten-Abbildung Fahrplan → LPC (Einheiten kW→W, Failsafe =
§14a-Mindestleistung, Gültigkeitsdauer) + die §14a-Sicherheitsprüfung. Was BEWUSST extern bleibt: der
EEBUS-Transport (SHIP/SPINE), die Geräte-Kopplung — das implementiert der aEMT mit der echten Stack/HW.

Reine stdlib, additiv (aemt_adapter.py unverändert; nutzt dessen AemtAdapter-Basis).
"""
from __future__ import annotations

from datetime import datetime

from .schema import validate_fahrplan, MIN_GUARANTEED_KW
from .aemt_adapter import AemtAdapter


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def fahrplan_to_lpc(fahrplan: dict) -> dict:
    """Übersetzt einen §14a-Fahrplan in die EEBUS-LPC-Limitstruktur (reine Datenabbildung).

    Je setpoint ein LPC-Limit:
      - consumption_limit_w  = p_limit_kw · 1000        (aktive Bezugsleistungsgrenze [W])
      - failsafe_value_w     = floor_kw · 1000          (Rückfallgrenze = §14a-Mindestleistung)
      - duration_s           = end_utc − start_utc      [s]
      - is_limit_active      = True
    Top-Level failsafe_value_w = min über alle Limits (konservative garantierte Mindestleistung).
    Prüft §14a: kein Limit unter seinem Failsafe (sonst ValueError — nicht als gültiges LPC abbildbar).
    """
    validate_fahrplan(fahrplan)                          # senderseitige §14a-Prüfung (wirft bei Verstoß)
    limits = []
    failsafes = []
    for sp in sorted(fahrplan["setpoints"], key=lambda s: s["start_utc"]):
        p_kw = float(sp["p_limit_kw"])
        floor_kw = float(sp.get("floor_kw", MIN_GUARANTEED_KW))
        if p_kw < floor_kw - 1e-9:                       # defense in depth (validate_fahrplan deckt das auch)
            raise ValueError(f"LPC: p_limit {p_kw} kW < Failsafe {floor_kw} kW (§14a-Verstoß)")
        dur = (_parse(sp["end_utc"]) - _parse(sp["start_utc"])).total_seconds()
        failsafes.append(floor_kw * 1000.0)
        limits.append({
            "start_utc": sp["start_utc"],
            "end_utc": sp["end_utc"],
            "duration_s": dur,
            "consumption_limit_w": round(p_kw * 1000.0, 3),
            "failsafe_value_w": round(floor_kw * 1000.0, 3),
            "is_limit_active": True,
            "is_limit_changeable": True,
        })
    return {
        "use_case": "EEBUS LPC (Limitation of Power Consumption)",
        "malo": fahrplan.get("malo"),
        "schedule_id": fahrplan.get("schedule_id"),
        "reason": fahrplan.get("reason"),
        "failsafe_value_w": round(min(failsafes), 3) if failsafes else None,
        "n_limits": len(limits),
        "limits": limits,
        "transport": "stub_external",
        "note": "Datenabbildung Fahrplan→LPC. EEBUS-Transport (SHIP/SPINE) + SMGW-Kopplung sind EXTERN "
                "(zertifizierter aEMT/CEM); NetzPilot sendet nicht selbst und berührt das SMGW nicht.",
    }


class EebusLpcAdapter(AemtAdapter):
    """aEMT-Adapter, der den Fahrplan in die EEBUS-LPC-Struktur übersetzt.

    `submit` (geerbt) prüft den Fahrplan senderseitig und ruft `_transmit`. Hier wird die LPC-Payload
    GEBAUT und quittiert — der echte EEBUS-Versand bleibt bewusst extern (transport=stub_external).
    Ein produktiver Adapter würde an dieser Stelle die EEBUS-Bibliothek/Hardware ansprechen.
    """

    def __init__(self, operator: str = "EEBUS-LPC-Adapter (Mapping; Transport extern)"):
        self.operator = operator

    def _transmit(self, fahrplan: dict) -> dict:
        lpc = fahrplan_to_lpc(fahrplan)
        return {
            "status": "MAPPED",
            "operator": self.operator,
            "lpc": lpc,
            "note": "LPC-Payload erzeugt; Versand über EEBUS ist extern (aEMT/CEM). "
                    "Kein direkter SMGW-Zugriff durch NetzPilot.",
        }
