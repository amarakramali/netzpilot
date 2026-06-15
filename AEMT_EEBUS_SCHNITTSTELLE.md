# NetzPilot — §14a-Steuerschnittstelle (aEMT / EEBUS-LPC)

*Stand 2026-06-02. Spezifikation der Schnittstelle, über die NetzPilot §14a-Steuerung an die
Anlagenwelt übergibt. Wettbewerbs-Feature W5 (Vorbild: gridX „Grid Signal Processor"). Ehrlich
abgegrenzt: was NetzPilot tut und was bewusst dem zertifizierten aEMT/CEM überlassen bleibt.*

## 1. Rollengrenze (regulatorisch entscheidend)

NetzPilot ist die **Prognose- und Berechnungs-Engine**. Es erzeugt den §14a-**Fahrplan**
(Wirkleistungsgrenzen je Marktlokation und Zeitfenster) und **übergibt** ihn an einen zertifizierten,
white-label betriebenen **aEMT/CEM**. NetzPilot **sendet nicht selbst über EEBUS** (SHIP/SPINE) und
**berührt nie das Smart-Meter-Gateway (SMGW)** oder die steuerbare Verbrauchseinrichtung — das ist
gesetzlich dem aMSB/aEMT vorbehalten. Diese Trennung ist Absicht und Teil der Glaubwürdigkeit.

```
NetzPilot (Fahrplan/Optimierung)  →  aEMT/CEM (zertifiziert)  →  EEBUS (SHIP/SPINE)  →  SMGW/CLS  →  steuVE
   ^ implementiert + verifiziert         ^ Partner/extern          ^ extern             ^ extern
```

## 2. Was NetzPilot emittiert — der Fahrplan-Contract

Quelle: `netzpilot/control/schema.py:make_fahrplan` (validiert). Struktur:

- `malo` (Marktlokations-ID), `schedule_id`, `issued_by`, `reason`, `created_utc`
- `setpoints[]`, je Eintrag: `start_utc`, `end_utc`, `p_limit_kw` (aktive Wirkleistungsgrenze),
  `floor_kw` (garantierte Mindestleistung, Default 4,2 kW gem. §14a).

Harte Invarianten (sender- UND empfängerseitig geprüft, „defense in depth"): jede Grenze
`p_limit_kw ≥ floor_kw ≥ 0`; `end_utc > start_utc`; gültige MaLo. Eine Grenze unter der
§14a-Mindestleistung ist **nicht** abbildbar und wird abgelehnt.

## 3. EEBUS-LPC-Abbildung (der BNetzA/BSI-Use-Case)

EEBUS ist das für §14a vorgesehene Protokoll; der einschlägige Use-Case ist **LPC — „Limitation of
Power Consumption"**. `netzpilot/control/eebus_lpc.py:fahrplan_to_lpc` bildet den Fahrplan auf die
LPC-Datenstruktur ab (reine, verifizierte Datenabbildung — `scripts/verify_eebus_lpc.py`, 15 Checks grün):

| Fahrplan | EEBUS-LPC | Abbildung |
|---|---|---|
| `p_limit_kw` | `consumption_limit_w` | × 1000 (kW → W) |
| `floor_kw` (4,2) | `failsafe_value_w` | × 1000; Rückfallgrenze bei Kommunikationsausfall |
| `end_utc − start_utc` | `duration_s` | Gültigkeitsdauer der Begrenzung |
| aktiver Setpoint | `is_limit_active = true` | |

Top-Level `failsafe_value_w` = Minimum über alle Limits (konservative garantierte Mindestleistung).
`failsafe` ist das §14a-Sicherheitsnetz: fällt die Kommunikation aus, regelt die Anlage auf die
garantierte Mindestleistung, nicht ab.

## 4. Adapter-Interface

`netzpilot/control/aemt_adapter.py:AemtAdapter` (abstrakt) → `submit(fahrplan)` prüft senderseitig und
ruft `_transmit`. Implementierungen:

- **`MockAemt`** (aemt_adapter.py): simuliert den zertifizierten aEMT — Empfangsprüfung (§14a defense in
  depth), Annahme/Ablehnung, Quittung mit Übertragungs-ID + Fingerprint, unveränderlicher Audit-Trail.
  Für End-to-End-Demo des Regelkreises ohne echte Anlagensteuerung.
- **`EebusLpcAdapter`** (eebus_lpc.py): übersetzt den Fahrplan in die LPC-Payload und quittiert mit
  `status="MAPPED"`. Der EEBUS-Versand bleibt `transport="stub_external"` — ein produktiver Adapter
  spricht hier die echte EEBUS-Bibliothek/Hardware an.

Service-Hook: `run_forecast(..., submit_to_aemt=True, aemt_adapter="eebus_lpc")` nutzt diesen Adapter
und legt die erzeugte Payload additiv in `out["fahrplan_lpc"]` ab. Default bleibt
`aemt_adapter="mock"` fuer die bisherige Mock-aEMT-Quittung.

## 5. Implementiert vs. bewusst extern (ehrlich)

**Implementiert + verifiziert:** Fahrplan-Erzeugung + Validierung; §14a-Invarianten (4,2-kW-Floor) sender-
und empfängerseitig; Fahrplan→LPC-Datenabbildung inkl. Failsafe/Einheiten/Dauer; Mock-Quittung +
Audit-Trail.

**Bewusst extern (Partner/Hardware):** der EEBUS-Transport (SHIP-Pairing, SPINE-Datenmodell), die
SMGW-/CLS-Kopplung, die physische Anlagensteuerung, die Zertifizierung des aEMT. NetzPilot dupliziert
das nicht — es dockt sauber an.

**Annahmen, die ein Pilot klärt:** die konkreten Mindestanforderungen des jeweiligen Netzbetreibers an
die Schnittstelle, die reale MaLo-/Geräte-Zuordnung, die zertifizierte aEMT-API. Bis dahin ist die
Kette über `MockAemt` end-to-end demonstrierbar, die LPC-Abbildung über `fahrplan_to_lpc` verifizierbar.
