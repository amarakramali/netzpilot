# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""aEMT-Adapter: Übergabe eines §14a-Fahrplans an einen (zertifizierten) aktiven EMT.

Rollentrennung (wichtig für die regulatorische Glaubwürdigkeit):
  NetzPilot = Prognose-/Berechnungs-Engine. Es ERZEUGT den Fahrplan (Wirkleistungsgrenzen) und
  ÜBERGIBT ihn an einen zertifizierten White-Label-aEMT. NetzPilot steuert NIEMALS direkt das SMGW /
  die steuerbare Verbrauchseinrichtung — das ist gesetzlich dem aMSB/aEMT vorbehalten.

Dieses Modul definiert das Übergabe-Protokoll (abstrakt) + einen MOCK-aEMT für Demo/Test:
  - nimmt einen validierten Fahrplan an,
  - prüft §14a-Konformität erneut auf Empfängerseite (defense in depth),
  - quittiert mit Übertragungs-ID, Status und Zeitstempel (wie ein echter aEMT),
  - führt einen unveränderlichen Audit-Trail (für Nachweis/Revision).

Ein echter Adapter implementiert dasselbe Interface gegen die reale aEMT-API (REST/AS4/o. ä.).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from .schema import validate_fahrplan, MIN_GUARANTEED_KW, active_limit_kw


class AemtError(Exception):
    """Übergabe abgelehnt (z. B. §14a-Verletzung empfängerseitig erkannt)."""


class AemtAdapter:
    """Abstraktes Übergabe-Interface. Ein echter Adapter erbt und überschreibt `_transmit`."""

    def submit(self, fahrplan: dict) -> dict:
        validate_fahrplan(fahrplan)                       # senderseitige Prüfung
        return self._transmit(fahrplan)

    def _transmit(self, fahrplan: dict) -> dict:          # pragma: no cover - abstrakt
        raise NotImplementedError


def _fingerprint(fahrplan: dict) -> str:
    """Deterministischer Hash des Fahrplan-Inhalts — für Quittung & Manipulationsnachweis."""
    payload = json.dumps(fahrplan, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class MockAemt(AemtAdapter):
    """Simulierter zertifizierter aEMT — quittiert & protokolliert, steuert aber nichts real.

    Bildet ab, was beim echten Partner passiert: Empfangsprüfung, Annahme/Ablehnung, Quittung mit
    Übertragungs-ID. So lässt sich der §14a-Regelkreis end-to-end zeigen, ohne echte Anlagensteuerung.
    """

    def __init__(self, operator: str = "Mock-aEMT (zertifiziert, simuliert)"):
        self.operator = operator
        self.audit_log: list[dict] = []

    def _transmit(self, fahrplan: dict) -> dict:
        now = datetime.now(timezone.utc)
        # Empfängerseitige §14a-Kontrolle (defense in depth): nie unter Mindestleistung akzeptieren.
        for sp in fahrplan["setpoints"]:
            if float(sp["p_limit_kw"]) < MIN_GUARANTEED_KW - 1e-9:
                ack = {
                    "status": "REJECTED",
                    "reason": f"p_limit_kw < §14a-Mindestleistung {MIN_GUARANTEED_KW} kW",
                    "received_utc": now.isoformat(),
                    "operator": self.operator,
                    "schedule_id": fahrplan.get("schedule_id"),
                }
                self.audit_log.append(ack)
                raise AemtError(ack["reason"])

        ack = {
            "status": "ACCEPTED",
            "transmission_id": f"aemt-{now.strftime('%Y%m%dT%H%M%S')}-{_fingerprint(fahrplan)}",
            "schedule_id": fahrplan.get("schedule_id"),
            "malo": fahrplan.get("malo"),
            "n_setpoints": len(fahrplan["setpoints"]),
            "fahrplan_fingerprint": _fingerprint(fahrplan),
            "received_utc": now.isoformat(),
            "operator": self.operator,
            "note": "Simulierte Annahme. Ein echter aEMT würde den Fahrplan an das SMGW/die steuVE "
                    "weiterleiten; NetzPilot tut dies bewusst NICHT selbst (Rollentrennung §14a).",
        }
        self.audit_log.append(ack)
        return ack
