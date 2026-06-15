import numpy as np
from netzpilot.eval import metrics as M
def test_mae_rmse():
    a = np.array([0., 0., 0.]); p = np.array([1., -1., 2.])
    assert abs(M.mae(p, a) - 4/3) < 1e-9
    assert abs(M.rmse(p, a) - np.sqrt(6/3)) < 1e-9
def test_pinball_median_half_mae():
    a = np.array([0., 0.]); p = np.array([2., -2.])
    assert abs(M.pinball(a, p, 0.5) - 0.5 * M.mae(p, a)) < 1e-9
def test_skill():
    assert abs(M.skill(50, 100) - 50.0) < 1e-9
