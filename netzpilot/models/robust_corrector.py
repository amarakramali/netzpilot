"""Robuster Korrektor: Shrinkage der Korrektur Richtung Baseline (Saisonal-Naiv).

Motivation (T9-Befund): bei volatiler Klein-Last schlaegt eine ungedaempfte Ridge-Korrektur die
saisonal-naive Baseline nicht zuverlaessig. Der ShrunkCorrector waehlt pro Fit einen Shrinkage-
Faktor s in [0,1] auf einem vorgelagerten Hold-out-Tail (out-of-sample), der den MAE minimiert:
final = base + s * ridge.predict. Hilft die Korrektur nicht, geht s -> 0 (= Baseline). So wird das
Produkt nie deutlich schlechter als die triviale Baseline — wichtig fuer Vertrauen beim Kunden.
"""
from __future__ import annotations
import numpy as np
from .ridge_correction import RidgeCorrector


class ShrunkCorrector:
    def __init__(self, lam: float = 10.0, tail_frac: float = 0.2,
                 grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
        self.lam = lam; self.tail_frac = tail_frac; self.grid = tuple(grid)

    def fit(self, X, y):
        n = len(X)
        ntail = min(n - 24, max(24, int(n * self.tail_frac))) if n > 48 else 0
        if ntail >= 24:
            r = RidgeCorrector(self.lam).fit(X[:-ntail], y[:-ntail])
            pt = r.predict(X[-ntail:]); yt = y[-ntail:]
            self.s = min(self.grid, key=lambda s: float(np.mean(np.abs(yt - s * pt))))
        else:
            self.s = 1.0
        self.model = RidgeCorrector(self.lam).fit(X, y)   # Refit auf allen Daten
        return self

    def predict(self, X):
        return self.s * self.model.predict(X)


class ShrunkQuantileCorrector:
    """Shrink LightGBM residual quantiles toward the seasonal-naive baseline.

    The shrink factor is selected on the tail of the training window only. A
    value of 0 means "use the baseline residual of zero", while 1 keeps the
    quantile model unchanged.
    """

    def __init__(
        self,
        alphas=(0.1, 0.5, 0.9),
        tail_frac: float = 0.2,
        grid=(0.0, 0.25, 0.5, 0.6, 0.75, 1.0),
        **params,
    ):
        self.alphas = tuple(alphas)
        self.tail_frac = tail_frac
        self.grid = tuple(grid)
        self.params = params

    def fit(self, X, y):
        from .lgbm_quantile import LGBMQuantileCorrector

        n = len(X)
        ntail = min(n - 24, max(24, int(n * self.tail_frac))) if n > 96 else 0
        if ntail >= 24:
            tail_model = LGBMQuantileCorrector(alphas=self.alphas, **self.params).fit(X[:-ntail], y[:-ntail])
            tail_pred = tail_model.predict(X[-ntail:])
            median_alpha = min(tail_pred.keys(), key=lambda a: abs(a - 0.5))
            median = tail_pred[median_alpha]
            yt = y[-ntail:]
            self.s = min(self.grid, key=lambda s: float(np.mean(np.abs(yt - s * median))))
        else:
            self.s = 1.0
        self.model = LGBMQuantileCorrector(alphas=self.alphas, **self.params).fit(X, y)
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return {alpha: self.s * values for alpha, values in preds.items()}
