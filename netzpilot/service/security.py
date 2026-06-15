# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Optionaler API-Key-Schutz + Request-Logging für den NetzPilot-Dienst.

Bewusst minimal und OPT-IN, passend zur Offline-/On-Premise-Philosophie:
- Ist die Umgebungsvariable NETZPILOT_API_KEY gesetzt, verlangen schreibende/abrufende Endpunkte
  den Header `X-API-Key`. Ist sie NICHT gesetzt (lokaler Ein-Klick-Betrieb), bleibt alles offen —
  der bestehende `Start_NetzPilot.bat`-Flow ändert sich nicht.
- `/health` und die UI (`/`) sind immer frei (Health-Checks / Login-Seite).
- Jede Anfrage wird mit Methode, Pfad, Status und Dauer geloggt (Audit/Monitoring-Grundlage).

Für echte Internet-Exposition zusätzlich TLS (Reverse-Proxy) — siehe DEPLOYMENT.md.
"""
from __future__ import annotations

import logging
import os
import time

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("netzpilot.service")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# Pfade, die ohne API-Key erreichbar bleiben (Health-Check, UI-Login-Seite, OpenAPI-Doku).
_PUBLIC_PREFIXES = ("/health", "/docs", "/openapi.json", "/redoc")


def _is_public(path: str) -> bool:
    return path == "/" or any(path.startswith(p) for p in _PUBLIC_PREFIXES)


async def api_key_and_logging_middleware(request: Request, call_next):
    """Starlette-Middleware: erst optionaler API-Key-Check, dann Logging mit Dauer."""
    start = time.monotonic()
    required_key = os.environ.get("NETZPILOT_API_KEY")

    if required_key and not _is_public(request.url.path):
        presented = request.headers.get("X-API-Key")
        if presented != required_key:
            logger.warning("401 %s %s (fehlender/falscher API-Key)", request.method, request.url.path)
            return JSONResponse({"detail": "Ungültiger oder fehlender X-API-Key."}, status_code=401)

    response = await call_next(request)
    dur_ms = (time.monotonic() - start) * 1000.0
    logger.info("%s %s -> %s (%.0f ms)", request.method, request.url.path,
                response.status_code, dur_ms)
    return response
