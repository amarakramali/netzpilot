# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Mock-aEMT (stdlib http.server) — simuliert den BSI-zertifizierten White-Label-aEMT.

Rolle: nimmt Fahrplaene von NetzPilot per REST/JSON entgegen, VALIDIERT sie (erzwingt die
§14a-Mindestleistung — der Gatekeeper), speichert sie je MaLo und stellt die aktive
Wirkleistungsgrenze fuer das HEMS bereit (repraesentiert den Weg CLS -> SMGW -> HEMS).
KEIN echtes SMGW/Smart-Meter-Netz — reine Simulation des logischen Steuerkreises.
"""
from __future__ import annotations
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from .schema import validate_fahrplan, active_limit_kw


class AEMTMock:
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.schedules: dict[str, list[dict]] = {}
        self._log: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence default logging
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if urlparse(self.path).path != "/fahrplan":
                    return self._send(404, {"error": "not found"})
                n = int(self.headers.get("Content-Length", 0))
                try:
                    fp = json.loads(self.rfile.read(n))
                    validate_fahrplan(fp)                       # aEMT als zertifizierter Gatekeeper
                except (ValueError, KeyError, json.JSONDecodeError) as e:
                    return self._send(422, {"error": f"rejected: {e}"})
                outer.schedules.setdefault(fp["malo"], []).append(fp)
                outer._log.append(f"ACCEPT {fp['schedule_id']} malo={fp['malo']} setpoints={len(fp['setpoints'])}")
                self._send(202, {"status": "accepted", "schedule_id": fp["schedule_id"]})

            def do_GET(self):
                u = urlparse(self.path)
                parts = u.path.strip("/").split("/")
                if len(parts) != 2 or parts[0] != "fahrplan":
                    return self._send(404, {"error": "not found"})
                malo = parts[1]
                at = parse_qs(u.query).get("at", [None])[0]
                if at:
                    at = at.replace(" ", "+")
                limits = [active_limit_kw(fp, at) for fp in outer.schedules.get(malo, [])]
                limits = [x for x in limits if x is not None]
                self._send(200, {"malo": malo, "at": at,
                                 "p_limit_kw": (min(limits) if limits else None)})

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        h, p = self._server.server_address
        return f"http://{h}:{p}"

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
