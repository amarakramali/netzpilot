# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Naive Baselines — bleiben in jeder Evaluation enthalten (Pflicht)."""
def persistence(load2d, d):     # gestern, gleiche Stunde
    return load2d[d - 1].copy()
def seasonal_naive(load2d, d):  # Vorwoche, gleiche Stunde
    return load2d[d - 7].copy()
