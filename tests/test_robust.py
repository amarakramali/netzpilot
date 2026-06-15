import numpy as np
from netzpilot.models.robust_corrector import ShrunkCorrector
def _X(n, feat): return np.column_stack([np.ones(n), feat, np.zeros(n)])
def test_shrink_high_when_useful():
    rng = np.random.default_rng(0); n = 240; f = rng.normal(size=n)
    y = 3 * f + rng.normal(0, 0.1, n)
    m = ShrunkCorrector(0.001).fit(_X(n, f), y)
    assert m.s >= 0.75
def test_shrink_low_when_noise():
    rng = np.random.default_rng(1); n = 240; f = rng.normal(size=n)
    y = rng.normal(size=n)
    m = ShrunkCorrector(0.001).fit(_X(n, f), y)
    assert m.s <= 0.5
