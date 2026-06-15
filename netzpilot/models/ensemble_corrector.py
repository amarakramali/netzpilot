# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Ensemble-Korrektor: mittelt mehrere Ridge-Staerken, jede mit Shrinkage-Sicherheitsnetz.

Motivation: Eine einzelne Ridge-Staerke (lam) ist ein Punkt-Bet auf die richtige Regularisierung.
Ein gleichgewichtetes Mittel mehrerer lam-Werte reduziert die VARIANZ der Korrektur (klassischer
Ensemble-Effekt), ohne neue Hyperparameter-Suche und ohne Overfit-Risiko — denn jeder Teil-Korrektor
behaelt sein eigenes Shrinkage Richtung Baseline (ShrunkCorrector). Schlaegt KEIN Mitglied die naive
Baseline, gehen alle s->0 und das Ensemble degeneriert sauber zur Baseline.

Gleiches fit/predict-Interface wie ShrunkCorrector -> als Drop-in in Backtest/Runner nutzbar.
"""
from __future__ import annotations
import numpy as np
from .robust_corrector import ShrunkCorrector


class EnsembleCorrector:
    """Mittelt die Vorhersagen mehrerer ShrunkCorrector mit verschiedenen Ridge-Staerken.

    lams: Tupel der Ridge-Staerken. Default deckt eine moderate Bandbreite ab (weicher..haerter).
    Jeder Teil-Korrektor waehlt seinen eigenen Shrinkage-Faktor auf einem Out-of-sample-Tail.
    """

    def __init__(self, lams=(3.0, 10.0, 30.0), tail_frac: float = 0.2):
        self.lams = tuple(lams)
        self.tail_frac = tail_frac

    def fit(self, X, y):
        self.members = [ShrunkCorrector(lam=l, tail_frac=self.tail_frac).fit(X, y) for l in self.lams]
        return self

    def predict(self, X):
        preds = np.vstack([m.predict(X) for m in self.members])
        return preds.mean(axis=0)
