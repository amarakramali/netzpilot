# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""T3: LightGBM-Quantilregression auf der Wochenabweichung (P10/P50/P90).
Lazy import von lightgbm, damit der Baseline-/Ridge-Pfad ohne lightgbm laeuft.
"""
from __future__ import annotations
import numpy as np
class LGBMQuantileCorrector:
    def __init__(self, alphas=(0.1, 0.5, 0.9), **params):
        self.alphas = tuple(alphas)
        self.params = dict(n_estimators=180, learning_rate=0.05, num_leaves=31,
                           min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                           random_state=42, n_jobs=1, verbosity=-1)
        self.params.update(params)
        self.models = {}
    def fit(self, X, y):
        import lightgbm as lgb           # erst hier importieren
        Xf = X[:, 1:]                     # Intercept-Spalte weglassen (Trees brauchen sie nicht)
        for a in self.alphas:
            self.models[a] = lgb.LGBMRegressor(objective="quantile", alpha=a, **self.params).fit(Xf, y)
        return self
    def predict(self, X):                # -> dict alpha -> vec
        Xf = X[:, 1:]
        return {a: m.predict(Xf) for a, m in self.models.items()}
    def predict_median(self, X):
        return self.predict(X)[0.5]
