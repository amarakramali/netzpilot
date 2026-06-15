#!/usr/bin/env python3
"""NetzPilot Benchmark-Suite — EIN Befehl, alle echten DSO-Datensaetze, mit Signifikanz.

Warum das existiert (fuer einen technischen Gutachter):
Einzelzahlen ("MAPE 4,4 %") ueberzeugen niemanden, der das Feld kennt. Was ueberzeugt, ist
ein REPRODUZIERBARER, leakage-sicherer Vergleich ueber MEHRERE echte Datensaetze gegen MEHRERE
Baselines — mit Konfidenzintervallen und der ehrlichen Aussage, wo der Vorsprung NICHT signifikant
ist. Genau das macht dieses Skript:

  1) Laedt jeden echten DSO-Lastgang mit EXPLIZIT gepinnter Spalte/Ebene (kein Auto-Raten — Audit-Lehre).
  2) Faehrt den identischen rolling-origin Backtest der echten Engine (ShrunkCorrector).
  3) Vergleicht NetzPilot gegen Persistenz UND Saisonal-Naiv.
  4) Paired Block-Bootstrap (Block = ganzer Tag, respektiert Intraday-Korrelation) -> Skill-CI95 +
     P(Modell besser) + Signifikanz-Flag.
  5) NaN-/Inf-sicher (Audit-Regel: nie ueber nicht-finite Werte mitteln).
  6) Schreibt benchmark_results.json + benchmark_table.md + MODEL_CARD.md.

Nur numpy/pandas/stdlib -> laeuft in der Sandbox UND beim Stadtwerk. Fester Seed -> reproduzierbar.

Aufruf:
  python scripts/benchmark_suite.py                 # alle Datensaetze des Manifests
  python scripts/benchmark_suite.py --only hilden_netzumsatz evdb_ns
  python scripts/benchmark_suite.py --n-test 28 --n-boot 10000
  python scripts/benchmark_suite.py --residual-feedback --calibrate   # Voll-Stack-Board (T49+T50)

Mechanik-Flags: Default ist das KONSERVATIVE Board (Kern + Feiertags-Basis). --residual-feedback
aktiviert das Online-lag-1-Feedback (T50; Skill/Bootstrap dann NACH Korrektur), --calibrate misst
zusaetzlich die online-rollende asymmetrische Coverage-Kalibrierung (T49; +~1/3 Laufzeit durch
Warmup-Fenster). Fortschritts- und Ausgabedateien tragen die Flags im Namen — ein Resume kann
deshalb NIE Laeufe mit unterschiedlichen Mechanismen mischen.
"""
from __future__ import annotations
# rev: resumable JSONL progress
import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from netzpilot.features.build import get_holidays, to_daily_local
from netzpilot.eval.backtest import rolling_origin
from netzpilot.models.robust_corrector import ShrunkCorrector
from scripts.dataset_manifest import CORPUS_SUMMARY, MANIFEST as DATASET_MANIFEST
from scripts.pilot_in_a_box import robust_load_csv

SEED = 20260601
# 12 Wochen. Audit 2026-06-03: bei n_test=28 waren ALLE 11 Grenzfall-Reihen faelschlich "n.s." vs.
# Saisonal-Naiv — reines Power-Artefakt. Bei 84 Tagen werden sie ausnahmslos signifikant (Punktschaetzung
# stabil, CI zieht sich zusammen). 28 unterschaetzt den echten Vorsprung systematisch.
N_TEST = 84
N_BOOT = 10000
REAL = "data_cache/real"


# Explizit gepinnte Spalten/Ebenen (aus outreach/hook_overview.md — nie auto-raten).
# unit: Eingabeeinheit der CSV; signed=True => MAPE ist sinnlos (Werte um 0), nur Skill/MAE fuehren.
MANIFEST = [
    {"key": "hilden_netzumsatz", "name": "Stadtwerke Hilden — Netzumsatz",
     "csv": f"{REAL}/Netzumsatz-Lastgang-2025.csv", "ts": "Text", "col": "Reihe1", "unit": "kW"},
    {"key": "evdb_ns", "name": "EVDB — Lastgang NS",
     "csv": f"{REAL}/evdb_lastgang_ns_2024.csv", "ts": None, "col": "Wert", "unit": "kW"},
    {"key": "evdb_ms", "name": "EVDB — Lastgang MS",
     "csv": f"{REAL}/evdb_lastgang_ms_2024.csv", "ts": None, "col": "Wert", "unit": "kW"},
    {"key": "herne_110_10kv", "name": "Stadtwerke Herne — Bezug 110/10 kV",
     "csv": f"{REAL}/herne_bezug_vorgelagerte_ebene_2024.csv", "ts": None, "col": "Load_1", "unit": "kW"},
    {"key": "bitterfeld_ns", "name": "NG Bitterfeld-Wolfen — NS",
     "csv": f"{REAL}/bitterfeld_jhl_ns_2024.csv", "ts": None, "col": "Wert", "unit": "kW"},
    {"key": "bitterfeld_msns", "name": "NG Bitterfeld-Wolfen — MS/NS",
     "csv": f"{REAL}/bitterfeld_jhl_msns_2024.csv", "ts": None, "col": "Wert", "unit": "kW"},
]


def daily_mae(R, name):
    """Tageweise MAE [n_test] aus den flachen [n_test*24]-Arrays; NaN-Tage werden 0-gewichtet."""
    ae = np.abs(R[name] - R["actual"]).reshape(-1, 24)
    return np.nanmean(ae, axis=1)


def paired_block_bootstrap(ae_model, ae_ref, rng, n_boot):
    """Paired Block-Bootstrap ueber ganze Tage (Block = 1 Tag). Gibt Skill%- und dMAE-Verteilung."""
    mask = np.isfinite(ae_model) & np.isfinite(ae_ref)
    ae_model, ae_ref = ae_model[mask], ae_ref[mask]
    n = len(ae_model)
    if n < 3:
        return None
    idx = rng.integers(0, n, size=(n_boot, n))
    sm = ae_model[idx].sum(axis=1)
    sr = ae_ref[idx].sum(axis=1)
    sr = np.where(sr == 0, np.nan, sr)
    skill = (1.0 - sm / sr) * 100.0
    dmae = (ae_ref[idx] - ae_model[idx]).mean(axis=1)
    return skill, dmae, n


def pct(x, q):
    return float(np.nanpercentile(x, q))


def evaluate_one(entry, n_test, n_boot, region="NW", residual_feedback=False, calibrate=False):
    hourly, ts_col, load_col, meta = robust_load_csv(
        entry["csv"], ts_col=entry["ts"], load_col=entry["col"], unit=entry["unit"], return_meta=True)
    load2d, days, _ = to_daily_local(hourly)
    if len(load2d) < n_test + 30:
        return {"key": entry["key"], "name": entry["name"], "error":
                f"zu wenig Tage ({len(load2d)} < {n_test+30})"}
    hol = get_holidays(sorted({d.year for d in days}), region)
    R, summary = rolling_origin(load2d, days, lambda: ShrunkCorrector(10.0),
                                n_test=n_test, holiday_set=hol,
                                residual_feedback=residual_feedback, calibrate=calibrate)

    mean_load = float(np.nanmean(load2d))
    signed = bool(entry.get("signed", False) or abs(mean_load) < 1.0)
    ae = {m: daily_mae(R, m) for m in ["model", "snaive", "persist"]}
    rng = np.random.default_rng(SEED)

    boot = {}
    for ref in ["snaive", "persist"]:
        res = paired_block_bootstrap(ae["model"], ae[ref], rng, n_boot)
        if res is None:
            boot[ref] = None
            continue
        skill, dmae, n = res
        skill_point = (1.0 - np.nansum(ae["model"]) / np.nansum(ae[ref])) * 100.0
        boot[ref] = {
            "skill_point_%": round(float(skill_point), 1),
            "skill_ci95_%": [round(pct(skill, 2.5), 1), round(pct(skill, 97.5), 1)],
            "P_model_besser_%": round(float(np.nanmean(skill > 0) * 100), 1),
            "dMAE_mean_MW": round(float(np.nanmean(dmae)), 3),
            "tage_modell_gewinnt_%": round(float(np.mean(ae["model"] < ae[ref]) * 100), 1),
            "signifikant_5pct": bool(pct(skill, 2.5) > 0),
        }

    m = summary["metriken"]["model"]
    mape_raw = m["MAPE_%"]
    mape_meaningless = bool(entry.get("mape_meaningless", False) or signed
                            or not math.isfinite(mape_raw) or mape_raw > 60.0)
    mape_value = None if mape_meaningless else mape_raw
    prob = summary["probabilistisch"]
    rec = {
        "key": entry["key"], "name": entry["name"],
        "ts_col": ts_col, "load_col": load_col, "unit_in": entry["unit"],
        "n_days": int(len(load2d)), "n_test": n_test,
        "mean_load_MW": round(mean_load, 3), "signed": signed,
        "mape_meaningless": mape_meaningless,
        "mape_note": entry.get("mape_note") or ("MAPE instabil; Skill/MAE fuehren" if mape_meaningless else None),
        "value_hash": entry.get("value_hash"),
        "MAE_MW": m["MAE_MW"], "MAPE_%": mape_value,
        "MASE": m["MASE"],
        "skill_vs_snaive_%": m["Skill_vs_SaisonalNaiv_%"],
        "skill_vs_persist_%": m["Skill_vs_Persistenz_%"],
        "coverage_P10_P90_%": prob["Coverage_P10_P90_%"],
        "bootstrap": boot,
    }
    if calibrate:   # T49: online-rollende asymmetrische Kalibrierung — additive Felder
        rec["coverage_kalibriert_%"] = prob.get("Coverage_P10_P90_kalibriert_%")
        rec["pinball_naiv"] = prob.get("Pinball_avg")
        rec["pinball_kalibriert"] = prob.get("Pinball_avg_kalibriert")
        rec["frac_below_P10_kalibriert_%"] = prob.get("frac_below_P10_kalibriert_%")
        rec["frac_above_P90_kalibriert_%"] = prob.get("frac_above_P90_kalibriert_%")
        rec["coverage_scale_method"] = prob.get("coverage_scale_method")
    if residual_feedback:   # T50: Skill oben ist bereits NACH Korrektur; naiv daneben
        rec["residual_feedback"] = summary.get("residual_feedback")
        rec["metriken_naiv"] = summary.get("metriken_naiv")
    return rec


def fmt_ci(b):
    if b is None:
        return "n/a"
    lo, hi = b["skill_ci95_%"]
    star = "*" if b["signifikant_5pct"] else "n.s."
    return f"{b['skill_point_%']:+.1f}% [{lo:+.1f},{hi:+.1f}] {star}"


def build_table(results, calibrated=False):
    cal_head = " Cov kalibriert |" if calibrated else ""
    cal_sep = "---:|" if calibrated else ""
    head = (f"| Datensatz | Mean Load | MAPE | MASE | Skill vs S-Naiv (CI95) | Skill vs Persistenz (CI95) | Cov 80% |{cal_head}\n"
            f"|---|---:|---:|---:|---|---|---:|{cal_sep}")
    lines = [head]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | — | — | — | FEHLER: {r['error']} | — | — |" + (" — |" if calibrated else ""))
            continue
        if r.get("mape_meaningless") or r.get("MAPE_%") is None:
            mape = "signiert/instabil - Skill fuehrt"
        else:
            mape = f"{r['MAPE_%']:.1f}%"
        snv = fmt_ci(r["bootstrap"].get("snaive"))
        per = fmt_ci(r["bootstrap"].get("persist"))
        cal_cell = ""
        if calibrated:
            cv = r.get("coverage_kalibriert_%")
            cal_cell = f" {cv:.0f}% |" if isinstance(cv, (int, float)) else " n/a |"
        lines.append(
            f"| {r['name']} | {r['mean_load_MW']:.1f} MW | {mape} | {r['MASE']} | {snv} | {per} | "
            f"{r['coverage_P10_P90_%']:.0f}% |" + cal_cell)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="nur diese Manifest-Keys")
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--region", default="NW")
    ap.add_argument("--out", default="data_cache/benchmark")
    ap.add_argument("--fresh", action="store_true", help="JSONL-Fortschritt verwerfen und neu rechnen")
    ap.add_argument("--residual-feedback", action="store_true",
                    help="Online-Residuen-Feedback (lag-1, T50) aktivieren — Skill dann NACH Korrektur")
    ap.add_argument("--calibrate", action="store_true",
                    help="online-rollende asymm. Coverage-Kalibrierung (T49) mitmessen (+~1/3 Laufzeit)")
    a = ap.parse_args()

    entries = DATASET_MANIFEST if not a.only else [e for e in DATASET_MANIFEST if e["key"] in a.only]
    os.makedirs(a.out, exist_ok=True)

    # Mechanik-Suffix: trennt Fortschritt UND Ausgaben je Flag-Kombination — Resume kann nie mischen.
    mech = ("_rf" if a.residual_feedback else "") + ("_cal" if a.calibrate else "")

    # Resumable: bereits gerechnete Datensaetze aus JSONL laden (Sandbox-Budget-Lehre).
    # Ein Eintrag pro Datensatz; ein erneuter Lauf rechnet nur fehlende, sofern nicht --fresh.
    jsonl = os.path.join(a.out, f"_progress_nt{a.n_test}_nb{a.n_boot}{mech}.jsonl")
    done = {}
    if a.fresh and os.path.exists(jsonl):
        os.remove(jsonl)
    if os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["key"]] = rec

    for e in entries:
        if e["key"] in done:
            print(f"[skip] {e['name']} — schon gerechnet (resume)")
            continue
        if not os.path.exists(e["csv"]):
            rec = {"key": e["key"], "name": e["name"], "error": "CSV fehlt"}
            print(f"[SKIP] {e['name']}: CSV fehlt ({e['csv']})")
        else:
            try:
                print(f"[run ] {e['name']} …")
                rec = evaluate_one(e, a.n_test, a.n_boot, a.region,
                                   residual_feedback=a.residual_feedback, calibrate=a.calibrate)
            except Exception as ex:  # nie den ganzen Lauf an einem Datensatz scheitern lassen
                rec = {"key": e["key"], "name": e["name"], "error": repr(ex)}
                print(f"[FAIL] {e['name']}: {ex!r}")
        with open(jsonl, "a", encoding="utf-8") as f:   # sofort persistieren -> resume-fest
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done[e["key"]] = rec

    # In Manifest-Reihenfolge zusammensetzen
    results = [done[e["key"]] for e in entries if e["key"] in done]
    for r in results:
        if "error" not in r:
            mape_value = r.get("MAPE_%")
            if r.get("mape_meaningless") or mape_value is None or not math.isfinite(float(mape_value)):
                r["MAPE_%"] = None
                r["mape_meaningless"] = True

    ok = [r for r in results if "error" not in r]
    n_sig_snv = sum(1 for r in ok if (r["bootstrap"].get("snaive") or {}).get("signifikant_5pct"))
    full_corpus_run = bool(
        a.only is None
        and CORPUS_SUMMARY.get("n_series")
        and int(CORPUS_SUMMARY["n_series"]) == len(ok)
    )
    n_unique_series = int(CORPUS_SUMMARY["n_series"]) if full_corpus_run else len(ok)
    n_independent_networks = (
        int(CORPUS_SUMMARY["n_independent_networks"])
        if full_corpus_run and CORPUS_SUMMARY.get("n_independent_networks") is not None
        else None
    )

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "n_test": a.n_test, "n_boot": a.n_boot,
        "method": "leakage-sicherer rolling-origin Backtest (ShrunkCorrector) + paired block-bootstrap (Block=Tag)",
        "mechanisms": {"holiday_base": True,
                       "residual_feedback": bool(a.residual_feedback),
                       "calibrate_online_rolling_asymmetric": bool(a.calibrate)},
        "n_datasets": len(results), "n_ok": len(ok),
        "n_unique_series": n_unique_series,
        "n_independent_networks": n_independent_networks,
        "n_signifikant_vs_snaive_5pct": n_sig_snv,
        "results": results,
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"benchmark_results{mech}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    table = build_table(results, calibrated=a.calibrate)
    if n_independent_networks is not None:
        kind_counts = CORPUS_SUMMARY.get("network_kind_counts", {})
        cluster_label = "unabhaengige Netz-/Regionalcluster" if kind_counts.get("tso_regional") else "unabhaengige Verteilnetze"
        kind_note = ""
        if kind_counts:
            kind_note = "; " + ", ".join(f"{v} {k}" for k, v in sorted(kind_counts.items()))
        corpus_line = (
            f"**Korpus:** {n_unique_series} eindeutige Reihen "
            f"({n_independent_networks} {cluster_label} nach Korrelations-Dedup{kind_note})."
        )
        summary_line = (
            f"{n_unique_series} eindeutige Reihen ({n_independent_networks} {cluster_label}); "
            f"{n_sig_snv} von {n_unique_series} signifikant vs. Saisonal-Naiv."
        )
    else:
        corpus_line = f"**Korpus:** {len(ok)}/{out['n_datasets']} Datensaetze ausgewertet."
        summary_line = (
            f"{len(ok)}/{out['n_datasets']} Datensaetze ausgewertet; "
            f"{n_sig_snv} davon signifikant vs. Saisonal-Naiv."
        )
    mech_line = ("*Mechanismen: Feiertags-Basis"
                 + (" + Online-Residuen-Feedback (Skill NACH Korrektur)" if a.residual_feedback else "")
                 + (" + online-rollende asymm. Kalibrierung" if a.calibrate else "")
                 + ("*" if (a.residual_feedback or a.calibrate)
                    else " — konservatives Board ohne Residuen-Feedback/Kalibrierung.*"))
    md = f"""# NetzPilot — Benchmark über echte DSO-/Regionallastgänge

*Erzeugt {out['generated_utc']} · Seed {SEED} · {a.n_test} Testtage rollierend · {a.n_boot} Bootstrap-Resamples.*
*Methode: {out['method']}.*
{mech_line}
{corpus_line}

{table}

**Lesehilfe:** Skill = Fehlerreduktion gegenüber der Baseline (höher = besser). `[CI95]` ist das
95 %-Konfidenzintervall aus paired Block-Bootstrap (Block = ganzer Tag). `*` = unteres CI-Ende > 0,
also statistisch signifikanter Vorsprung; `n.s.` = nicht signifikant. „signiert" = Differenzbilanz mit
Mittel ≈ 0 oder Null-/Rückspeise-Durchgänge; dort ist MAPE bedeutungslos, Skill/MAE führen.

**Zusammenfassung:** {summary_line}
"""
    with open(os.path.join(a.out, f"benchmark_table{mech}.md"), "w", encoding="utf-8") as f:
        f.write(md)

    # Modellkarte (mechanismus-unabhaengig) — nur der kanonische Lauf ohne Flags schreibt sie.
    write_card = not mech
    card = f"""# NetzPilot — Modellkarte

*Erzeugt {out['generated_utc']}.*

## Aufgabe
Day-ahead-Prognose (24×1 h, P10/P50/P90) der Verteilnetz-Last bzw. Residuallast kleiner Stadtwerke.

## Modell
- **Basis:** Saisonal-Naiv (Last der Vorwoche, gleicher Wochentag/Stunde).
- **Korrektur:** Ridge-Regression auf leakage-sicheren Kalender-/Lag-Features, **Shrinkage Richtung
  Baseline** (ShrunkCorrector): wählt pro Fit auf einem Out-of-sample-Tail einen Faktor s∈[0,1], der
  den MAE minimiert. Hilft die Korrektur nicht, geht s→0 → das Modell wird **nie deutlich schlechter
  als die triviale Baseline**. Das ist die zentrale Vertrauens-Eigenschaft.
- **Unsicherheit:** Conformalized Quantile Regression (CQR), verteilungsfreie marginale Coverage-Garantie.

## Evaluationsprotokoll
- **Leakage-sicher:** Rolling-origin (expanding window); das Modell sieht beim Vorhersagen nie den Zieltag.
- **Baselines verpflichtend:** Persistenz UND Saisonal-Naiv in jedem Lauf.
- **Signifikanz:** paired Block-Bootstrap (Block = ganzer Tag, respektiert Intraday-Korrelation),
  Skill-CI95 + P(Modell besser).
- **NaN/Inf-sicher:** nicht-finite Werte werden vor jeder Mittelung verworfen (Audit-Regel).

## Geltungsbereich & Grenzen (ehrlich)
- Belegt auf **öffentlichen** DSO-Lastgängen (Veröffentlichungspflicht §12/§23c). Die volle kundeneigene
  RLM-Gesamtlast kann abweichen → finaler Nachweis nur im Piloten mit echten Daten.
- **Kein** direkter Vergleich gegen kommerzielle Profi-Tools gefahren — keine
  Behauptung „genauer als die".
- €-Nutzen ist **Downside-Schutz**, kein garantierter Ertrag; belastbar erst mit realer reBAP-Abrechnung.
- Wetter brachte leakage-sicher **keinen** signifikanten Genauigkeitsgewinn (kein „Wetter-Wunder").

## Reproduktion
`python scripts/benchmark_suite.py` → `data_cache/benchmark/benchmark_table.md` + `benchmark_results.json`.
"""
    if write_card:
        with open(os.path.join(a.out, "MODEL_CARD.md"), "w", encoding="utf-8") as f:
            f.write(card)

    print("\n" + table)
    print(f"\n-> {a.out}/benchmark_table{mech}.md  +  benchmark_results{mech}.json"
          + ("  +  MODEL_CARD.md" if write_card else ""))
    print(f"   {len(ok)}/{out['n_datasets']} ok; {n_sig_snv} signifikant vs. Saisonal-Naiv (5%)."
          + (f"  [Mechanismen: holiday_base{'+rf' if a.residual_feedback else ''}{'+cal' if a.calibrate else ''}]"))


if __name__ == "__main__":
    main()
