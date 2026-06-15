"""T3 (HOST-LAUF, nicht in der Build-Sandbox ausfuehrbar: braucht lightgbm + pyarrow).

LightGBM-Quantil-Korrektur mit Wetter-Features auf dem 2-Jahres-Cache (T2).
Leakage-sicher: Wetter ist die Historical-FORECAST-Reihe (kein Reanalyse-Istwert).

Beispiel:
  python scripts/run_backtest_lgbm.py \
    --cache data_cache/t2_2022-01-01_2024-01-01 --n-test 90 --region NW
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from netzpilot.features.build import to_daily_local, frame_to_daily_local, get_holidays
from netzpilot.eval.backtest import rolling_origin_quantile
from netzpilot.report.report import write_report
from netzpilot.models.lgbm_quantile import LGBMQuantileCorrector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache/t2_2022-01-01_2024-01-01")
    ap.add_argument("--n-test", type=int, default=90)
    ap.add_argument("--region", default="NW")
    ap.add_argument("--out", default="data_cache/t3_lgbm")
    a = ap.parse_args()

    load = pd.read_parquet(os.path.join(a.cache, "smard_load_hour.parquet"))
    load_s = load.iloc[:, 0]; load_s.index = pd.to_datetime(load_s.index, utc=True)
    weather = pd.read_parquet(os.path.join(a.cache, "openmeteo_historical_forecast_hour.parquet"))
    weather.index = pd.to_datetime(weather.index, utc=True)

    load2d, days, good = to_daily_local(load_s)
    weather2d = frame_to_daily_local(weather, good)          # Forecast-Wetter, leakage-sicher
    hol = get_holidays(sorted({d.year for d in days}), a.region)

    factory = lambda: LGBMQuantileCorrector(alphas=(0.1, 0.5, 0.9))
    R, summary = rolling_origin_quantile(load2d, days, factory, first=8, n_test=a.n_test,
                                         weather2d=weather2d, holiday_set=hol)
    os.makedirs(a.out, exist_ok=True)
    write_report(summary, os.path.join(a.out, "report.md"), os.path.join(a.out, "results.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
