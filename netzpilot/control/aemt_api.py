# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Staged production-ish aEMT API (FastAPI) — NICHT in der Sandbox ausgefuehrt (kein FastAPI).
Identische Logik wie aemt_mock.py; fuer den Host-Lauf / echten White-Label-Partner-Anschluss.
Run (Host):  uvicorn netzpilot.control.aemt_api:app --port 8077
"""
from __future__ import annotations
try:
    from fastapi import FastAPI, HTTPException
except Exception:  # FastAPI optional/staged
    FastAPI = None

from .schema import validate_fahrplan, active_limit_kw

if FastAPI is not None:
    app = FastAPI(title="NetzPilot Mock-aEMT (staged)")
    _SCHEDULES: dict[str, list[dict]] = {}

    @app.post("/fahrplan", status_code=202)
    def post_fahrplan(fp: dict):
        try:
            validate_fahrplan(fp)
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=422, detail=str(e))
        _SCHEDULES.setdefault(fp["malo"], []).append(fp)
        return {"status": "accepted", "schedule_id": fp["schedule_id"]}

    @app.get("/fahrplan/{malo}")
    def get_fahrplan(malo: str, at: str):
        limits = [active_limit_kw(fp, at) for fp in _SCHEDULES.get(malo, [])]
        limits = [x for x in limits if x is not None]
        return {"malo": malo, "at": at, "p_limit_kw": (min(limits) if limits else None)}
