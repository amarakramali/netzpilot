# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Feiertags-bewusster Saisonal-Naiv-Baseline (deterministische Baseline-Reparatur).

Befund 2026-06-03 (gemessen): die Standard-Basis `load2d[d-7]` (Vorwoche, gleicher Wochentag) ist
SYSTEMATISCH VERZERRT, wenn d-7 ein Feiertag war — dann liegt die Referenz ~17–20 % zu niedrig (auf
holiday-sensitiver Aggregat-Last), und der Korrektor muss das nur teils auffangen. Ein gefittetes
„Vorwoche-war-Feiertag"-Flag half die Aggregat-Reihen, schadete aber industrieller Last katastrophal
(Overfit auf ~1 Tag/Jahr). Die DETERMINISTISCHE Reparatur — bei d-7-Feiertag die nächste NICHT-Feiertags-
Referenz gleichen Wochentags (d-14, d-21, …) nehmen — ist parameterfrei (kein Overfit) und gemessen:

    Hilden Netzumsatz +2,74 % / Herne +3,15 % (Aggregat, n_test=84; lw-Tag +51,6 % / +69,3 %),
    Bitterfeld MS −0,31 % (Industrie ~neutral — dort ist d-7-Feiertag ≈ d-14, also faktisch No-Op).

Leakage-sicher: die Referenz liegt IMMER strikt vor d (nur Vergangenheit). Rückwärtskompatibel: ohne
`days`/`holiday_set` identisch zu `load2d[d-7]` (altes Verhalten). Reine stdlib/numpy.
"""
from __future__ import annotations


def holiday_aware_base(load2d, d, days=None, holiday_set=None):
    """Saisonal-Naiv-Basis für Tag d, feiertagsbereinigt.

    Standard: `load2d[d-7]` (Vorwoche, gleicher Wochentag). Sind `days` (DatetimeIndex) UND `holiday_set`
    gegeben und ist d-7 ein Feiertag, wird in 7-Tage-Schritten (gleicher Wochentag) zur nächsten
    NICHT-Feiertags-Referenz zurückgegangen (d-14, d-21, …), solange der Index >= 0 bleibt.
    Rückgabe: Kopie der Referenzzeile (Form wie load2d[·]).
    """
    ref = d - 7
    if ref < 0:
        raise ValueError(f"d={d}: keine Vorwochen-Referenz (d-7<0).")
    if days is not None and holiday_set:
        k = d - 7
        # zurück bis Nicht-Feiertag, aber nur solange eine weitere Vorwochen-Referenz existiert
        while k - 7 >= 0 and days[k].date() in holiday_set:
            k -= 7
        ref = k
    return load2d[ref].copy()


def holiday_aware_resid_target(load2d, d, days=None, holiday_set=None):
    """Zielgröße des Korrekturmodells, konsistent zur feiertagsbereinigten Basis: load2d[d] − base."""
    return load2d[d] - holiday_aware_base(load2d, d, days, holiday_set)
