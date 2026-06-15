#!/usr/bin/env python3
"""Verify Monte-Carlo-€-Band (eval/mc_savings.py) — reine stdlib, kein Internet.

Prueft die statistischen Garantien des Tages-Block-Bootstraps:
  - Determinismus (Seed), observed == Summe der Beitraege,
  - Unverzerrtheit (Bootstrap-Mittel ~ observed, ueber den Standardfehler geprueft),
  - Varianz-Korrektheit (Bootstrap-Std ~ sqrt(B)*popstd der Bloecke),
  - prob_positive-Semantik + Grenzfaelle (konstant positiv -> 1, null -> 0, symmetrisch -> ~0,5),
  - Bindung an B.1 (observed == compare_forecasts_eur-Einsparung),
  - partieller Letztblock, Eingabe-Validierung.

Aufruf: python scripts/verify_mc_savings.py
"""
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.eval.mc_savings import block_bootstrap_band
from netzpilot.eval.bilanzkreis import savings_contrib_per_qh, compare_forecasts_eur

ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond


# --- S1: Determinismus ---
contrib = [math.sin(i) * 10 + (i % 7 - 3) for i in range(96 * 30)]   # 30 Tage
r1 = block_bootstrap_band(contrib, seed=123, n_resamples=1500)
r2 = block_bootstrap_band(contrib, seed=123, n_resamples=1500)
check("S1: gleicher Seed -> identisches Ergebnis", r1 == r2)
r3 = block_bootstrap_band(contrib, seed=999, n_resamples=1500)
check("S1: anderer Seed -> gleiche observed_total", r3["observed_total_eur"] == r1["observed_total_eur"])

# --- S2: observed_total == Summe der Beitraege ---
check("S2: observed == sum(contrib)", abs(r1["observed_total_eur"] - round(sum(contrib), 2)) < 1e-6)

# --- S3: konstant positiv -> entartetes Band (alle Bloecke gleich) ---
const = [2.0] * (96 * 10)        # 10 volle Tage, jede QH 2 EUR
r = block_bootstrap_band(const, block_len=96, n_resamples=500)
check("S3: konstant -> p5 == p95 == observed", r["p5_eur"] == r["p95_eur"] == r["observed_total_eur"])
check("S3: konstant -> std 0, prob_positive 1", r["std_eur"] == 0.0 and r["prob_positive"] == 1.0)
check("S3: konstant -> mean == observed", r["mean_eur"] == r["observed_total_eur"])

# --- S4: null -> Band null, prob_positive 0 ---
r = block_bootstrap_band([0.0] * (96 * 5), n_resamples=300)
check("S4: null -> alles 0", r["observed_total_eur"] == 0.0 and r["p95_eur"] == 0.0 and r["std_eur"] == 0.0)
check("S4: null -> prob_positive 0", r["prob_positive"] == 0.0)

# --- S5: Unverzerrtheit: |mean - observed| < 4*Standardfehler (= std/sqrt(N)) ---
N = 6000
r = block_bootstrap_band(contrib, seed=7, n_resamples=N)
se = r["std_eur"] / math.sqrt(N)
check("S5: Bootstrap-Mittel ~ observed (im 4*SE-Band)",
      abs(r["mean_eur"] - r["observed_total_eur"]) < 4 * se + 1e-9)

# --- S6: Varianz-Korrektheit bei block_len=1: std ~ sqrt(B)*popstd(Bloecke) ---
import random as _rnd
_rnd.seed(1)
vals = [_rnd.gauss(0.5, 3.0) for _ in range(300)]
r = block_bootstrap_band(vals, block_len=1, n_resamples=8000, seed=3)
expected_std = math.sqrt(len(vals)) * statistics.pstdev(vals)
rel = abs(r["std_eur"] - expected_std) / expected_std
check(f"S6: Bootstrap-Std ~ sqrt(B)*popstd (rel.Abw {rel:.3f} < 0,1)", rel < 0.10)
check("S6: n_blocks == n_periods bei block_len=1", r["n_blocks"] == len(vals))

# --- S7: symmetrisch GEPAARTE Tagesbloecke (Gesamtsumme 0, aber nicht entartet) -> prob_positive ~ 0,5 ---
# prob_positive ist um observed zentriert; bei observed~0 und symmetrischem Block-Pool muss es ~0,5 sein.
sym = []
for _ in range(100):                       # 100 Paare = 200 Tage
    v = abs(_rnd.gauss(0, 5)) + 0.5        # nicht-entartet
    sym += [v / 96] * 96                   # Tag mit Summe +v
    sym += [-v / 96] * 96                  # Tag mit Summe -v
r = block_bootstrap_band(sym, block_len=96, n_resamples=4000, seed=5)
check("S7: observed ~ 0 (symmetrisch)", abs(r["observed_total_eur"]) < 1e-6)
check("S7: prob_positive in [0,4 ; 0,6]", 0.4 <= r["prob_positive"] <= 0.6)

# --- S8: Bindung an B.1: observed == compare_forecasts_eur-Einsparung ---
actual = [10.0 + math.sin(i / 5) for i in range(96 * 12)]
sa = [a + math.cos(i / 4) * 0.8 for i, a in enumerate(actual)]      # Prognose A
sb = [a + math.cos(i / 4) * 0.2 for i, a in enumerate(actual)]      # Prognose B (besser)
reb = [50.0 + 80 * math.sin(i / 9) for i in range(len(actual))]
spot = [40.0 + 5 * math.cos(i / 11) for i in range(len(actual))]
c = savings_contrib_per_qh(actual, sa, sb, reb, spot)
cmp = compare_forecasts_eur(actual, sa, sb, reb, spot)
r = block_bootstrap_band(c, n_resamples=500)
check("S8: observed == compare_forecasts savings",
      abs(r["observed_total_eur"] - cmp["savings_b_vs_a_eur"]) < 0.02)

# --- S9: partieller Letztblock korrekt gezaehlt + observed erhalten ---
part = [1.0] * (96 * 4 + 37)               # 4 volle Tage + 37 QH
r = block_bootstrap_band(part, block_len=96, n_resamples=200)
check("S9: n_blocks == ceil(n/block_len) = 5", r["n_blocks"] == 5)
check("S9: observed == sum trotz Teilblock", abs(r["observed_total_eur"] - round(sum(part), 2)) < 1e-6)

# --- S10: Eingabe-Validierung ---
def raises(fn):
    try:
        fn(); return False
    except ValueError:
        return True
check("S10: leere Eingabe -> ValueError", raises(lambda: block_bootstrap_band([])))
check("S10: block_len 0 -> ValueError", raises(lambda: block_bootstrap_band([1.0], block_len=0)))
check("S10: n_resamples 0 -> ValueError", raises(lambda: block_bootstrap_band([1.0], n_resamples=0)))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
