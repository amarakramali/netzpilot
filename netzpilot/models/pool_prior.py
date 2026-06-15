# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Persistierter Pool-Prior für den produktiven Multi-Mandanten-Effekt.

Der Pool-Prior ist der über viele Stadtwerke gemittelte Korrektur-Koeffizientenvektor — gelernt im
LAST-NORMALISIERTEN Raum (jedes Haus auf seine mittlere Last skaliert, sonst mittelt man
inkommensurable Niveaus, siehe Notizen/_pooling_befund.md). Er wird als JSON gespeichert und vom
Dienst geladen: hat ein neues Stadtwerk wenig Historie, startet es mit diesem Prior statt bei null.

WARUM HIER PERSISTIERT: Der Prior wächst mit der Kundenbasis. Ein neuer Pilot profitiert sofort vom
Wissen aller bisherigen — das ist der Datennetzwerkeffekt, produktiv gemacht. Geteilt werden nur
aggregierte Koeffizienten (kein Rohlastgang) → DSGVO-/pilot-fähig.
"""
from __future__ import annotations
import json
import os
import numpy as np

from ..features.build import build_features, resid_target
from .ridge_correction import RidgeCorrector

DEFAULT_PATH = "data_cache/pool/pool_prior.json"


def build_prior_from_series(series_list, lam: float = 10.0, first: int = 8):
    """Pool-Prior aus mehreren (load2d, days)-Reihen lernen — jede last-normalisiert.

    series_list: Liste von (load2d, days). Gibt dict mit w_pool (Liste) + Metadaten zurück.
    """
    import pandas as pd
    ws = []
    for load2d, days in series_list:
        if len(load2d) < first + 8:
            continue
        scale = float(np.mean(load2d[first:]))
        if scale <= 0:
            continue
        l2 = load2d / scale
        days = pd.DatetimeIndex(days)
        X = np.vstack([build_features(l2, days, t, None, None) for t in range(first, len(l2))])
        y = np.concatenate([resid_target(l2, t) for t in range(first, len(l2))])
        ws.append(RidgeCorrector(lam).fit(X, y).w)
    if not ws:
        raise ValueError("Keine ausreichend langen Reihen für den Pool-Prior.")
    W = np.vstack(ws)
    return {
        "w_pool": W.mean(axis=0).tolist(),
        "n_houses": int(len(ws)),
        "lam": lam,
        "space": "last-normalisiert (load/mean_load)",
        "n_features": int(W.shape[1]),
    }


def save_prior(prior: dict, path: str = DEFAULT_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prior, f, indent=2, ensure_ascii=False)
    return path


def load_prior(path: str = DEFAULT_PATH):
    """Lädt den Prior oder gibt None zurück, wenn keiner existiert (dann: normaler Korrektor)."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
