#!/usr/bin/env python3
"""Frische ECHTE Netzlast online ziehen (SMARD/Bundesnetzagentur, CC BY 4.0, kein API-Key)
und optional direkt durch die NetzPilot-Engine rechnen → Ergebnis-JSON im Service-Store.

Damit ist die Software jederzeit an AKTUELLEN echten Daten testbar — nicht nur am statischen
Korpus. Wiederverwendet den verifizierten Connector netzpilot/data/smard.py (Filter 410 = realisierter
Stromverbrauch/Netzlast; 4068/4067/1225 = PV/Wind-Erzeugung).

Ehrliche Grenzen:
- SMARD ist ÜBERTRAGUNGSNETZ-/Regelzonen-Ebene (GW-Skala), kein Stadtwerke-Lastgang. Für die
  Demo „Live-Daten → Prognose" geeignet; der Stadtwerke-Beweis bleibt der echte Korpus (Hilden & Co.).
- Feiertagskalender: je Regelzone wird EIN Bundesland als Näherung gewählt (Mehrländer-Zonen).

Beispiele:
  python scripts/fetch_smard.py --region Amprion --days 130 --run
  python scripts/fetch_smard.py --region DE --days 90 --with-generation --run
"""
from __future__ import annotations
import argparse, json, os, sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from netzpilot.data.smard import fetch_series, FILTERS, BASE

# Regelzone → Bundesland-Näherung für den Feiertagskalender (Mehrländer-Zonen: größtes Land).
HOLIDAY_REGION = {"DE": "NW", "Amprion": "NW", "TenneT": "NI", "50Hertz": "BB",
                  "TransnetBW": "BW", "AT": "BY", "LU": "NW"}


def fetch_to_csv(region: str, days: int, out_dir: str, resolution: str = "hour",
                 with_generation: bool = False) -> tuple[str, str | None, dict]:
    end = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    # cache_dir=None: kein Parquet-Cache (keine pyarrow-Abhängigkeit); Reihen sind klein.
    load = fetch_series(start, end, filter_id=FILTERS["load"], region=region,
                        resolution=resolution, cache_dir=None)
    if len(load) < 24 * 30:
        raise SystemExit(f"SMARD lieferte zu wenig Stunden ({len(load)}) für {region} {start}..{end}.")
    os.makedirs(out_dir, exist_ok=True)
    stamp = f"{load.index[0].strftime('%Y%m%d')}_{load.index[-1].strftime('%Y%m%d')}"
    csv_path = os.path.join(out_dir, f"smard_{region}_load_{stamp}.csv")
    load.rename("load_mw").to_frame().rename_axis("timestamp_utc").to_csv(csv_path)

    gen_path = None
    gen_parts = {}
    if with_generation:
        frames = {}
        for key in ("pv", "wind_onshore", "wind_offshore"):
            try:
                s = fetch_series(start, end, filter_id=FILTERS[key], region=region,
                                 resolution=resolution, cache_dir=None)
                if len(s):
                    frames[key] = s
                    gen_parts[key] = int(len(s))
            except Exception as e:  # einzelne fehlende Erzeugungsart ehrlich überspringen
                gen_parts[key] = f"nicht verfügbar: {e}"
        if frames:
            gen = pd.concat(frames, axis=1).sum(axis=1, min_count=1).dropna()
            gen = gen.loc[gen.index.intersection(load.index)]
            gen_path = os.path.join(out_dir, f"smard_{region}_gen_{stamp}.csv")
            gen.rename("generation_mw").to_frame().rename_axis("timestamp_utc").to_csv(gen_path)

    prov = {
        "source": "SMARD (Bundesnetzagentur), CC BY 4.0",
        "base_url": BASE, "filter_load": FILTERS["load"], "region": region,
        "resolution": resolution, "start": start, "end_exclusive": end,
        "n_hours_load": int(len(load)), "unit": "MW",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generation_parts": gen_parts or None,
        "level_note": "Übertragungsnetz-/Regelzonen-Ebene (GW-Skala), kein Stadtwerke-Lastgang.",
    }
    with open(csv_path.replace(".csv", "_source.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    return csv_path, gen_path, prov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="Amprion",
                    choices=["DE", "Amprion", "TenneT", "TransnetBW", "50Hertz", "AT", "LU"])
    ap.add_argument("--days", type=int, default=130)
    ap.add_argument("--resolution", default="hour", choices=["hour", "quarterhour"])
    ap.add_argument("--out-dir", default="data_cache/real")
    ap.add_argument("--with-generation", action="store_true",
                    help="zusätzlich PV+Wind ziehen → Residuallast-Prognose im --run")
    ap.add_argument("--run", action="store_true", help="direkt run_forecast + Service-Store")
    ap.add_argument("--utility", default=None)
    args = ap.parse_args()

    csv_path, gen_path, prov = fetch_to_csv(args.region, args.days, args.out_dir,
                                            resolution=args.resolution,
                                            with_generation=args.with_generation)
    print(f"Lastgang: {csv_path}  ({prov['n_hours_load']} h, Stand {prov['retrieved_utc']})")
    if gen_path:
        print(f"Erzeugung: {gen_path}")

    if args.run:
        from netzpilot.service.runner import run_forecast
        from netzpilot.service.store import ForecastStore
        utility = args.utility or f"SMARD {args.region} Netzlast (Regelzone, live)"
        out = run_forecast(
            csv_path, utility=utility, region=HOLIDAY_REGION.get(args.region, "NW"),
            unit="MW", ts_col="timestamp_utc", load_col="load_mw",
            generation_csv=gen_path, generation_ts_col="timestamp_utc",
            generation_load_col="generation_mw",
            drift_monitoring=True,
            forecast_store_path=f"data_cache/forecast_store/smard_{args.region}.jsonl",
        )
        fc = out.get("forecast") or []
        assert len(fc) == 24 and all(f["p10"] <= f["p50"] <= f["p90"] for f in fc)
        ForecastStore().save(utility, out)
        peak = max(f["p50"] for f in fc)
        print(f"Forecast {out['forecast_date']}: peak P50 = {peak:.0f} MW"
              + (f", Residuallast aktiv ({out['residual_forecast']['n_common_hours']} h gemeinsam)"
                 if out.get("residual_forecast") else "")
              + f" → Mandant „{utility}“ im Service-Store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
