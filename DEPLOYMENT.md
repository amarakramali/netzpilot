# NetzPilot — Betrieb & Sicherheit (Deployment)

Drei Betriebsmodi, vom einfachsten zum produktivsten. Wähle den, der zur Situation passt.

## 1. Lokal (Ein-Klick, Entwicklung / kleiner Pilot)
`Start_NetzPilot.bat` (Windows) bzw. `start_netzpilot.sh` (Mac/Linux): legt venv an, installiert
Pakete, startet `uvicorn` auf `127.0.0.1:8000`, öffnet die UI. **Kein API-Key, kein TLS** — gedacht
für den lokalen Rechner im Stadtwerk oder hinter dem Firmen-VPN. Daten verlassen den Rechner nicht.

## 2. Container (On-Premise-Server, reproduzierbar)
```bash
cp .env.example .env            # optional: NETZPILOT_API_KEY setzen
docker compose up -d            # UI: http://localhost:8000/  ·  Doku: /docs
```
- Schlanker Container (`requirements-service.txt`: nur numpy/pandas/fastapi — kein lightgbm-Schwergewicht).
- Prognose-Historie wird über ein Volume persistiert (`data_cache/service_store`).
- Eingebauter Healthcheck gegen `/health`.

## 3. Internet-exponiert (echter Mehrkunden-Betrieb)
Drei Dinge sind dann Pflicht — ehrlich benannt, weil sie über reine „Demo" hinausgehen:

1. **API-Key** setzen (`NETZPILOT_API_KEY`): schützt alle Endpunkte außer `/health`, `/`, `/docs`
   per `X-API-Key`-Header. Eingebaut (siehe `netzpilot/service/security.py`).
2. **TLS**: NetzPilot spricht HTTP. Vor das Internet gehört ein Reverse-Proxy mit HTTPS, z. B.
   Caddy (automatisches Let's-Encrypt) oder nginx:
   ```
   # Caddyfile
   netzpilot.example.de {
       reverse_proxy localhost:8000
   }
   ```
3. **Mandanten-Trennung**: aktuell ein gemeinsamer Datastore je Instanz. Für echte Mehrkunden-Trennung
   pro Stadtwerk eine eigene Instanz/eigenen `NETZPILOT_STORE` betreiben (oder DB-Backend ergänzen).

## Logging / Monitoring
Jede Anfrage wird strukturiert geloggt (Methode, Pfad, Status, Dauer in ms) über den Logger
`netzpilot.service`. `/health` liefert Status + Version für externe Health-Checks (Uptime-Monitor,
Docker-Healthcheck, k8s-Probe).

## Ehrliche Sicherheits-Grenzen (nicht überverkaufen)
- Der API-Key ist ein einfacher gemeinsamer Schlüssel, kein OAuth/Nutzer-Management. Für einen Pilot
  ausreichend; für KRITIS/Mehrmandanten später echtes IdP/Token-Konzept ergänzen.
- Kein Rate-Limiting eingebaut (Reverse-Proxy übernimmt das, falls nötig).
- §14a-Fahrpläne gehen an einen **zertifizierten White-Label-aEMT** (Modul `control/aemt_adapter.py`,
  derzeit Mock) — NetzPilot steuert nie direkt das SMGW. Diese Rollentrennung ist bewusst und bleibt.
- Für produktiven §14a-Betrieb: ISO-27001-Hosting (DE/EU), DSGVO-AVV, NIS-2-Bewusstsein — im Pilot zu klären.
