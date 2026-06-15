"""Hierarchische Prognose-Reconciliation (MinT) — kohärente Prognosen über Spannungsebenen.

Tiefe-Achse: ein mehrstufiges Verteilnetz hat eine NATÜRLICHE Hierarchie — die Last auf der oberen
Ebene (z.B. 110/10 kV) ist die Summe der unteren Stränge (10 kV → 0,4 kV …). Prognostiziert man jede
Ebene einzeln, sind die Bahnen i.d.R. INKOHÄRENT (die Strang-Prognosen summieren sich nicht exakt zur
Ebenen-Prognose). MinT (Minimum Trace, Wickramasuriya/Athanasopoulos/Hyndman 2019) projiziert die
Basisprognosen auf den kohärenten Unterraum UND minimiert dabei die Spur der Reconciliation-Fehler-
Kovarianz — das stellt Kohärenz her UND verbessert i.d.R. die Genauigkeit (Information wird zwischen den
Ebenen geteilt).

Kern (lineare Algebra, exakt):
    reconciled = S · P · base_forecast,   P = (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹
    S = Summenmatrix (n Knoten × m Blattknoten); W = Kovarianz der Basis-Prognosefehler.
    SP ist eine PROJEKTION auf den kohärenten Raum: SP·S = S (kohärente Prognosen bleiben unverändert),
    und reconciled ist per Konstruktion kohärent (Aggregate = Summe ihrer Blätter).

Gewichtungsvarianten für W (von robust→datenhungrig):
    ols          W = I                              (kein Fehlerwissen nötig)
    wls_struct   W = diag(Zeilensummen von S)        (Struktur-Skalierung; DEFAULT, keine Historie nötig)
    wls_var      W = diag(Varianz der Basisfehler)   (braucht Residuen-Stichprobe je Knoten)
    mint_shrink  W = schrumpf(Stichproben-Kovarianz)  (voll, Schäfer-Strimmer-Shrinkage)

Ehrlich (CLAUDE.md): MinT braucht die Basisprognosen ALLER Knoten + (für var/shrink) leakage-sichere
Basis-Residuen. Der Nutzen entsteht nur bei einer ECHTEN Hierarchie (mehrstufige DSO-Reihen wie
Herne/TEN). Reine numpy, additiv — ändert die Prognose-Engine nicht.
"""
from __future__ import annotations

import numpy as np


def build_summing_matrix(bottom, aggregates):
    """Summenmatrix S für eine Summen-Hierarchie.

    bottom:     Liste der Blatt-Knoten-Namen (m Stück, die feinste Ebene).
    aggregates: dict {Aggregat-Name: [Blatt-Namen, die es summiert]} (Reihenfolge = Ausgabereihenfolge).

    Rückgabe (S, node_names): S hat Form (n × m) mit n = len(aggregates)+m. Knotenreihenfolge =
    [Aggregate in dict-Reihenfolge] + [Blätter in bottom-Reihenfolge]; die letzten m Zeilen sind die
    Identität (jedes Blatt auf sich selbst).
    """
    bottom = list(bottom)
    idx = {name: j for j, name in enumerate(bottom)}
    m = len(bottom)
    if m == 0:
        raise ValueError("bottom (Blattknoten) ist leer.")
    rows, names = [], []
    for agg_name, leaves in aggregates.items():
        row = [0.0] * m
        for lf in leaves:
            if lf not in idx:
                raise ValueError(f"Aggregat '{agg_name}' referenziert unbekanntes Blatt '{lf}'.")
            row[idx[lf]] = 1.0
        rows.append(row); names.append(agg_name)
    for name in bottom:                       # Identitätszeilen der Blätter
        row = [0.0] * m
        row[idx[name]] = 1.0
        rows.append(row); names.append(name)
    return np.asarray(rows, dtype=float), names


def build_temporal_summing_matrix(n_bottom, block_sizes):
    """Summenmatrix S für eine TEMPORALE Hierarchie EINER Reihe (Athanasopoulos et al. 2017).

    Die Summen-Constraint ist hier EXAKT per Konstruktion: die Tagesenergie = Σ der Stunden =
    Σ der Viertelstunden derselben beobachteten Reihe — keine spannungs-/topologische Annahme nötig.
    Damit sind 15-min-Nominierungen kohärent zum Stunden-/Tagesfahrplan, den der Bilanzkreis bilanziert.

    n_bottom:    Anzahl der feinsten Perioden (z.B. 96 Viertelstunden eines Tages).
    block_sizes: Liste von Blockgrößen in Bottom-Perioden, die je eine Aggregationsebene bilden;
                 jede muss n_bottom teilen. Beispiel [96, 4] -> Tagessumme (1 Knoten) + 24 Stundensummen.
                 (Größere Blöcke zuerst angeben → erscheinen oben in node_names.)

    Rückgabe (S, node_names) wie build_summing_matrix; Blätter q0..q{n-1} stehen am Ende.
    """
    if n_bottom <= 0:
        raise ValueError("n_bottom muss > 0 sein.")
    bottom = [f"q{i}" for i in range(n_bottom)]
    aggregates = {}
    for b in block_sizes:
        if n_bottom % b != 0:
            raise ValueError(f"Blockgröße {b} teilt n_bottom={n_bottom} nicht.")
        for g in range(n_bottom // b):
            aggregates[f"agg{b}_{g}"] = bottom[g * b:(g + 1) * b]
    return build_summing_matrix(bottom, aggregates)


def _w_inv(method, S, residuals, shrink_lambda):
    n = S.shape[0]
    if method == "ols":
        return np.eye(n)
    if method == "wls_struct":
        w = S.sum(axis=1)                     # Zeilensummen (Anzahl Blätter je Knoten)
        return np.diag(1.0 / np.where(w > 0, w, 1.0))
    if method in ("wls_var", "mint_shrink"):
        if residuals is None:
            raise ValueError(f"method='{method}' braucht residuals (n × Stichproben).")
        R = np.asarray(residuals, dtype=float)
        if R.shape[0] != n:
            raise ValueError(f"residuals hat {R.shape[0]} Zeilen, erwartet {n}.")
        if method == "wls_var":
            v = R.var(axis=1, ddof=1)
            v = np.where(v > 1e-12, v, 1e-12)
            return np.diag(1.0 / v)
        # mint_shrink: Schrumpfung der Stichproben-Kovarianz Richtung Diagonale
        cov = np.cov(R, ddof=1)
        d = np.diag(np.diag(cov))
        lam = 0.2 if shrink_lambda is None else float(shrink_lambda)
        W = lam * d + (1.0 - lam) * cov
        return np.linalg.pinv(W)
    raise ValueError(f"unbekannte method '{method}'.")


def reconcile(base_forecasts, S, method="wls_struct", residuals=None, shrink_lambda=None):
    """Reconcile Basisprognosen aller Knoten zu KOHÄRENTEN Prognosen (MinT).

    base_forecasts: Array (n,) für eine Periode oder (n, H) über einen Horizont; Reihenfolge = node_names
                    aus build_summing_matrix (Aggregate zuerst, dann Blätter).
    S:              Summenmatrix (n × m).
    Rückgabe: reconciled (gleiche Form wie base_forecasts), garantiert kohärent.
    """
    S = np.asarray(S, dtype=float)
    base = np.asarray(base_forecasts, dtype=float)
    n, m = S.shape
    if base.shape[0] != n:
        raise ValueError(f"base_forecasts hat {base.shape[0]} Knoten, S erwartet {n}.")
    Winv = _w_inv(method, S, residuals, shrink_lambda)
    P = np.linalg.solve(S.T @ Winv @ S, S.T @ Winv)     # (m × n), = (SᵀW⁻¹S)⁻¹ SᵀW⁻¹
    SP = S @ P                                          # (n × n) Projektion auf den kohärenten Raum
    return SP @ base


def coherence_error(values, S):
    """Maximale Verletzung der Summen-Constraints: max |Aggregat − Σ seiner Blätter|.

    values: Array (n,) oder (n, H) in node_names-Reihenfolge. S: (n × m). 0 = perfekt kohärent.
    (Die letzten m Knoten sind die Blätter; Aggregat-Zeile i muss = Σ S[i,j]·Blatt_j sein.)
    """
    S = np.asarray(S, dtype=float)
    v = np.asarray(values, dtype=float)
    n, m = S.shape
    bottom = v[n - m:]                                  # die m Blattwerte
    implied = S @ bottom                                # was die Knoten laut S sein müssten
    return float(np.max(np.abs(v - implied)))
