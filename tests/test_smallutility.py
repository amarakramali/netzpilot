from netzpilot.data.smard import load_local_json
from netzpilot.data.synthetic_smallutility import make_small_utility_load
def test_proxy_shape_and_positive():
    nat = load_local_json("prognose_engine_v1/data/wk*.json")
    s = make_small_utility_load(nat, peak_mw=25.0, seed=0)
    assert len(s) == len(nat) and (s.values > 0).all()
    assert 20.0 < s.max() <= 25.0 + 1e-6
