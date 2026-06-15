"""Multi-Mandanten-Pooling: ein neues Stadtwerk profitiert vom Wissen ALLER anderen.

Das Problem: Ein kleines Stadtwerk mit nur wenigen Wochen Historie kann sein Korrekturmodell nicht
stabil schaetzen — zu wenig Daten, das Modell ueberfittet oder faellt auf die Baseline zurueck.

Die Loesung (Partial Pooling / Hierarchical Shrinkage): Wir lernen aus VIELEN Stadtwerken einen
gemeinsamen "Prior" — den mittleren Korrektur-Koeffizientenvektor ueber alle Haeuser. Das Modell
eines einzelnen Hauses wird dann Richtung dieses Pool-Mittels geschrumpft, GEWICHTET nach der eigenen
Datenmenge:

    w_haus = (n / (n + tau)) * w_eigen  +  (tau / (n + tau)) * w_pool

  - wenig eigene Tage (n klein)  -> stark Richtung Pool (borrowing strength),
  - viel eigene Historie (n gross) -> ueberwiegend eigenes Modell.

WARUM DAS EIN MOAT IST: w_pool entsteht aus der KUNDENBASIS. Je mehr Stadtwerke NetzPilot nutzen,
desto besser der Pool-Prior — und desto besser starten NEUE Kunden. Ein Nachbauer ohne diese Basis
hat keinen Pool. Selbstverstaerkender Datennetzwerkeffekt, nicht durch Code allein kopierbar.

Reine numpy-Implementierung, kompatibel zum bestehenden Feature-/Ridge-Stack (build_features,
RidgeCorrector). KEINE neue Abhaengigkeit.
"""
from __future__ import annotations
import numpy as np
from .ridge_correction import RidgeCorrector


def fit_pool_prior(houses_Xy, lam: float = 10.0):
    """Lerne den Pool-Prior: mittlerer (standardisierter) Ridge-Koeffizientenvektor ueber Haeuser.

    houses_Xy: Liste von (X, y) je Stadtwerk (Feature-Matrix + Residual-Ziel, wie im Backtest).
    Gibt (w_pool, mu_pool, sd_pool) zurueck — Koeffizienten + Feature-Standardisierung im Pool-Raum.

    Jedes Haus wird einzeln auf standardisierten Features gefittet; der Prior ist das Mittel der
    Koeffizienten. So zaehlt jedes Haus gleich (kein Dominieren grosser Haeuser).
    """
    ws = []
    for X, y in houses_Xy:
        r = RidgeCorrector(lam).fit(X, y)
        ws.append(r.w)
    W = np.vstack(ws)
    return W.mean(axis=0), W.std(axis=0)


class PooledCorrector:
    """Ridge-Korrektor mit Shrinkage Richtung eines vorab gelernten Pool-Priors.

    w_pool: aus fit_pool_prior() (ueber viele andere Stadtwerke). tau: Pool-Staerke (Pseudo-Tage).
    Bei n eigenen Trainingszeilen wird mit Gewicht n/(n+tau*24) zum eigenen Fit, sonst zum Pool gemischt.
    """

    def __init__(self, w_pool, lam: float = 10.0, tau_days: float = 30.0):
        self.w_pool = np.asarray(w_pool, dtype=float)
        self.lam = lam
        self.tau = tau_days * 24.0   # tau in Zeilen (Stunden), vergleichbar mit len(X)

    def fit(self, X, y):
        self._own = RidgeCorrector(self.lam).fit(X, y)
        n = len(X)
        alpha = n / (n + self.tau)                      # Gewicht des eigenen Modells
        # eigenes w und Pool-w leben im selben standardisierten Feature-Raum (gleiche Features)
        self.w = alpha * self._own.w + (1.0 - alpha) * self.w_pool
        self.alpha = alpha
        self.mu = self._own.mu
        self.sd = self._own.sd
        return self

    def predict(self, X):
        Xs = np.hstack([np.ones((len(X), 1)), (X[:, 1:] - self.mu) / self.sd])
        return Xs @ self.w
