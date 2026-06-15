# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Repraesentativer KLEIN-Stadtwerk-Lastgang (Proxy bis echte Pilot-/OPSD-Daten vorliegen).

Zweck: den Forecaster dort testen, wo das Produkt wirklich laufen muss — auf kleiner, weniger
aggregierter (volatilerer) Last. KEIN Ersatz fuer echte Daten; ehrlich als Proxy gekennzeichnet.

Generatives Modell (transparent):
- Residential-Komponente: reale (nationale) Lastform, evening-peak leicht verschaerft.
- Industrie-Komponente: einige Grosskunden mit Werktags-Blockprofil (06-22 h), schwach am WE.
- Idiosynkratisches multiplikatives AR(1)-Rauschen: bildet die GERINGERE Aggregationsglaettung
  eines kleinen Versorgers ab (hoehere relative Volatilitaet).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_small_utility_load(national: pd.Series, peak_mw: float = 25.0,
                            residential_share: float = 0.55, n_large: int = 3,
                            ar_rho: float = 0.9, noise_sigma: float = 0.06, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = national.sort_index().index
    nat = national.sort_index().values.astype(float)
    natn = nat / nat.mean()
    loc = idx.tz_convert("Europe/Berlin") if idx.tz else idx.tz_localize("UTC").tz_convert("Europe/Berlin")
    hour = np.asarray(loc.hour); dow = np.asarray(loc.dayofweek)

    evening = np.exp(-((hour - 19) ** 2) / 8.0)
    resid = natn * (1 + 0.25 * evening); resid /= resid.mean()

    large = np.zeros_like(nat)
    for _ in range(n_large):
        block = ((hour >= 6) & (hour < 22) & (dow < 5)).astype(float)
        prof = np.where(block.astype(bool), rng.uniform(0.8, 1.0), rng.uniform(0.10, 0.25))
        large += rng.uniform(0.5, 1.5) * prof
    large /= (large.mean() + 1e-9)

    base = residential_share * resid + (1 - residential_share) * large
    base /= base.mean()

    eps = np.empty_like(nat); e = 0.0
    for i in range(len(nat)):
        e = ar_rho * e + rng.normal(0.0, noise_sigma); eps[i] = e
    load = base * np.exp(eps); load /= load.mean()
    load = load * (peak_mw / load.max())
    return pd.Series(load, index=idx, name="value")
