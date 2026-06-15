#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Verify MinT-Reconciliation (models/reconcile.py) — numpy, kein Internet.

Aufruf: python scripts/verify_reconcile.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from netzpilot.models.reconcile import (build_summing_matrix, build_temporal_summing_matrix,
                                         reconcile, coherence_error, _w_inv)

ok = True
def check(name, cond):
    global ok
    safe_name = str(name).encode("ascii", "replace").decode("ascii")
    print(f"  [{'PASS' if cond else 'FAIL'}] {safe_name}")
    ok = ok and cond

# Hierarchie: total = a + b + c (4 Knoten, 3 Blätter)
S, names = build_summing_matrix(["a", "b", "c"], {"total": ["a", "b", "c"]})
check("Aufbau: node_names + S-Form", names == ["total", "a", "b", "c"] and S.shape == (4, 3))

# --- S1: Kohärenz (inkohärente Eingabe -> kohärente Ausgabe) ---
rng = np.random.default_rng(0)
for method in ("ols", "wls_struct"):
    base = np.array([100.0, 30.0, 30.0, 30.0])    # 30+30+30=90 != 100 -> inkohärent
    rec = reconcile(base, S, method=method)
    check(f"S1 [{method}]: reconciled kohärent (Fehler ~0)", coherence_error(rec, S) < 1e-9)

# --- S2: Projektion — idempotent + kohärente Eingabe unverändert ---
base = np.array([100.0, 30.0, 30.0, 30.0])
r1 = reconcile(base, S, "ols")
r2 = reconcile(r1, S, "ols")
check("S2: idempotent (reconcile(reconcile)=reconcile)", np.allclose(r1, r2, atol=1e-9))
coherent = np.array([45.0, 20.0, 15.0, 10.0])      # 20+15+10=45 kohärent
check("S2: kohärente Eingabe bleibt unveraendert", np.allclose(reconcile(coherent, S, "ols"), coherent, atol=1e-9))

# --- S3: OLS-Handwert (total=a+b, base=[10,4,5] -> [9.667,4.333,5.333]) ---
S2m, _ = build_summing_matrix(["a", "b"], {"total": ["a", "b"]})
rec = reconcile(np.array([10.0, 4.0, 5.0]), S2m, "ols")
check("S3: OLS-Handwert [9.667,4.333,5.333]",
      np.allclose(rec, [29/3, 13/3, 16/3], atol=1e-6))
check("S3: kohärent (total == a+b)", abs(rec[0] - (rec[1] + rec[2])) < 1e-9)

# --- S4: Genauigkeitsgewinn. MinT/OLS = Projektion auf den kohaerenten Raum. Liegt die Wahrheit
#         im kohaerenten Raum, ist die ORTHOGONALE Projektion (OLS) per Pythagoras NIE weiter von der
#         Wahrheit weg als die Basis — ueber die GESAMTE Hierarchie (Trace), nicht je Einzelknoten. ---
rng = np.random.default_rng(42)
n_draw = 4000
worse = 0
sse_base = sse_rec = 0.0
for _ in range(n_draw):
    bottoms = rng.uniform(5.0, 15.0, 3)
    truth = np.array([bottoms.sum(), *bottoms])    # liegt exakt im kohaerenten Raum
    base = truth + rng.normal(0.0, 1.0, 4)         # unabhaengiges Rauschen auf allen Knoten
    rec = reconcile(base, S, "ols")
    eb = float(np.sum((base - truth) ** 2)); er = float(np.sum((rec - truth) ** 2))
    sse_base += eb; sse_rec += er
    if er > eb + 1e-9:
        worse += 1
check("S4: OLS-Projektion nie schlechter (Pythagoras, alle Knoten)", worse == 0)
check(f"S4: mittlerer Gesamt-SSE sinkt ({sse_rec/n_draw:.3f} < {sse_base/n_draw:.3f})", sse_rec < sse_base)

# --- S4b: wls_struct (DEFAULT) senkt den mittleren Gesamt-SSE ueber die Hierarchie ---
rng = np.random.default_rng(7)
sb = sr = 0.0
for _ in range(n_draw):
    bottoms = rng.uniform(5.0, 15.0, 3)
    truth = np.array([bottoms.sum(), *bottoms])
    base = truth + rng.normal(0.0, 1.0, 4)
    rec = reconcile(base, S, "wls_struct")
    sb += float(np.sum((base - truth) ** 2)); sr += float(np.sum((rec - truth) ** 2))
check(f"S4b: wls_struct mittlerer Gesamt-SSE sinkt ({sr/n_draw:.3f} < {sb/n_draw:.3f})", sr < sb)

# --- S5: WLS-Struct-Gewichte = Zeilensummen von S ---
Winv = _w_inv("wls_struct", S, None, None)
check("S5: wls_struct W^-1 diag == [1/3,1,1,1]", np.allclose(np.diag(Winv), [1/3, 1, 1, 1]))

# --- S6: wls_var / mint_shrink mit Residuen -> kohärent ---
resid = rng.normal(0, 1, (4, 200))
for method in ("wls_var", "mint_shrink"):
    rec = reconcile(np.array([100.0, 30.0, 30.0, 30.0]), S, method=method, residuals=resid)
    check(f"S6 [{method}]: kohärent", coherence_error(rec, S) < 1e-9)

# --- S7: Horizont-Matrix (n × H) wird spaltenweise reconciled ---
baseH = np.array([[100.0, 90.0], [30.0, 30.0], [30.0, 30.0], [30.0, 30.0]])  # (4 × 2)
recH = reconcile(baseH, S, "ols")
check("S7: (n×H) kohärent in jeder Spalte", coherence_error(recH, S) < 1e-9 and recH.shape == (4, 2))

# --- S8: Validierung ---
def raises(fn):
    try:
        fn(); return False
    except (ValueError, np.linalg.LinAlgError):
        return True
check("S8: unbekanntes Blatt -> ValueError", raises(lambda: build_summing_matrix(["a"], {"t": ["x"]})))
check("S8: wls_var ohne residuals -> ValueError", raises(lambda: reconcile(np.array([1.0, 1.0]), build_summing_matrix(["a"], {"t": ["a"]})[0], "wls_var")))
check("S8: base falsche Laenge -> ValueError", raises(lambda: reconcile(np.array([1.0, 2.0]), S, "ols")))

# --- S9: Temporale Hierarchie (eine Reihe): Tagessumme + Stundensummen, exakte Constraint ---
St, namest = build_temporal_summing_matrix(12, [12, 4])   # 1 Tag (12) + 3 "Stunden" (je 4) + 12 Blätter
check("S9: temporale S-Form (16 × 12), Blätter zuletzt",
      St.shape == (16, 12) and namest[-12:] == [f"q{i}" for i in range(12)])
rng = np.random.default_rng(3)
baset = rng.uniform(1.0, 5.0, 16)                          # inkohärente Basisprognosen aller Knoten
rect = reconcile(baset, St, "ols")
check("S9: temporal reconciled kohärent (Fehler ~0)", coherence_error(rect, St) < 1e-9)
check("S9: Tagessumme == Σ aller 12 Blätter", abs(rect[0] - rect[-12:].sum()) < 1e-9)
check("S9: ungültige Blockgröße -> ValueError",
      raises(lambda: build_temporal_summing_matrix(12, [5])))

print("\nERGEBNIS:", "ALLE CHECKS GRUEN" if ok else "ABWEICHUNG — siehe FAIL")
sys.exit(0 if ok else 1)
