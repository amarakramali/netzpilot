"""v1-Modell: Ridge-Regression auf der Wochenabweichung (reines numpy)."""
import numpy as np
class RidgeCorrector:
    def __init__(self, lam: float = 10.0):
        self.lam = lam
    def fit(self, X, y):
        self.mu = X[:, 1:].mean(0); self.sd = X[:, 1:].std(0); self.sd[self.sd == 0] = 1
        Xs = np.hstack([np.ones((len(X), 1)), (X[:, 1:] - self.mu) / self.sd])
        A = Xs.T @ Xs + self.lam * np.eye(Xs.shape[1]); A[0, 0] -= self.lam
        self.w = np.linalg.solve(A, Xs.T @ y)
        return self
    def predict(self, X):
        Xs = np.hstack([np.ones((len(X), 1)), (X[:, 1:] - self.mu) / self.sd])
        return Xs @ self.w
