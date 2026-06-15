# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Fehler- und Skill-Metriken. Alle erwarten 1D-numpy-Arrays gleicher Laenge."""
import numpy as np
def mae(pred, actual):  return float(np.mean(np.abs(pred - actual)))
def rmse(pred, actual): return float(np.sqrt(np.mean((pred - actual) ** 2)))
def mape(pred, actual): return float(np.mean(np.abs((pred - actual) / actual)) * 100)
def mase(pred, actual, scale): return float(mae(pred, actual) / scale)
def skill(mae_model, mae_ref): return float((1 - mae_model / mae_ref) * 100)
def pinball(actual, qpred, tau):
    d = actual - qpred
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))
def crps_from_quantiles(levels, preds, actual):
    """Proxy-CRPS ueber ein Quantilraster: Mittel der Pinball-Verluste (feineres Raster -> besser)."""
    return float(np.mean([pinball(actual, preds[i], t) for i, t in enumerate(levels)]))
def coverage(actual, lo, hi):
    return float(np.mean((actual >= lo) & (actual <= hi)) * 100)
