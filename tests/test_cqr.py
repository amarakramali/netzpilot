from netzpilot.data.smard import load_local_json
from netzpilot.features.build import to_daily, get_holidays
from netzpilot.eval.conformal import rolling_origin_conformal, rolling_origin_cqr
from netzpilot.models.ridge_correction import RidgeCorrector
def test_cqr_improves_calibration_at_90():
    s = load_local_json("prognose_engine_v1/data/wk*.json")
    load2d, days = to_daily(s); hol = get_holidays(sorted({d.year for d in days}), "NW")
    fac = lambda: RidgeCorrector(10.0)
    _, b = rolling_origin_conformal(load2d, days, fac, alpha=0.1, cal_days=28, holiday_set=hol, online=True, per_hour=False)
    _, q = rolling_origin_cqr(load2d, days, fac, alpha=0.1, cal_days=28, holiday_set=hol)
    assert abs(q["coverage_%"] - 90) <= abs(b["coverage_%"] - 90) + 1e-6
