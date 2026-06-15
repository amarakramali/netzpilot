# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Feiertags-Overrides: Nutzer markiert Tage explizit als (Nicht-)Feiertag — statt dass die
Software rät. Fixture-frei (läuft im run_all_checks-Shim)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netzpilot.features.build import apply_holiday_overrides, get_holidays
from netzpilot.forecast import forecast_next_day
from netzpilot.models.robust_corrector import ShrunkCorrector


def _synthetic():
    """Synthetik MIT Feiertagen in der Historie — sonst kann das Modell den Feiertags-
    Koeffizienten nicht lernen und ein Zieltag-Override hätte (korrekt!) keine Wirkung."""
    rng = np.random.default_rng(13)
    nd, hours = 120, 24
    base = 20 + 5 * np.sin(np.arange(hours) / 24 * 2 * np.pi)
    week = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.8])
    load2d = np.array([
        base * week[d % 7] * (1 + 0.04 * np.sin(d / 25)) + rng.normal(0, 0.3, hours)
        for d in range(nd)
    ])
    days = pd.date_range("2025-09-01", periods=nd, freq="D")   # Montag-Start
    hist_hols = {days[i].date() for i in (20, 48, 76, 104)}    # 4 „Feiertage" in der Historie
    for i in (20, 48, 76, 104):
        load2d[i] *= 0.75                                       # Feiertag = deutlich weniger Last
    return load2d, days, lambda: ShrunkCorrector(10.0), hist_hols


def test_apply_overrides_add_remove_and_validation():
    base_set = {pd.Timestamp("2026-01-01").date()}
    out = apply_holiday_overrides(base_set, ["2026-05-15", pd.Timestamp("2026-12-24")],
                                  ["2026-01-01"])
    assert pd.Timestamp("2026-05-15").date() in out
    assert pd.Timestamp("2026-12-24").date() in out
    assert pd.Timestamp("2026-01-01").date() not in out      # remove gewinnt
    assert apply_holiday_overrides(None, None, None) == set()
    assert apply_holiday_overrides(base_set) == base_set     # No-op kopiert
    with pytest.raises(ValueError):
        apply_holiday_overrides(base_set, ["kein-datum"])


def test_override_changes_target_flag_and_anchor():
    load2d, days, factory, hist_hols = _synthetic()
    target = pd.Timestamp(days[-1]) + pd.Timedelta(days=1)

    # (a) Zieltag als Feiertag markieren -> gelerntes Feiertags-Merkmal kippt -> Prognose ändert sich.
    #     Beide Läufe nutzen DENSELBEN Historie-Kalender (Modell hat Feiertage gelernt);
    #     nur der Zieltag-Override unterscheidet sie.
    f_plain = forecast_next_day(load2d, days, factory, holiday_set=set(hist_hols),
                                round_digits=None)
    f_hol = forecast_next_day(load2d, days, factory,
                              holiday_set=apply_holiday_overrides(hist_hols, [target]),
                              round_digits=None)
    p_plain = np.array([h["p50"] for h in f_plain["hours"]])
    p_hol = np.array([h["p50"] for h in f_hol["hours"]])
    assert not np.allclose(p_plain, p_hol), "Override des Zieltags muss die Prognose ändern"
    assert p_hol.mean() < p_plain.mean(), "gelernter Feiertagseffekt drückt die Prognose nach unten"

    # (b) ANKER (Vorwoche des Zieltags) als Feiertag markieren -> Basis springt auf d-14
    anchor = target - pd.Timedelta(days=7)
    f_anchor = forecast_next_day(load2d, days, factory,
                                 holiday_set=apply_holiday_overrides(hist_hols, [anchor]),
                                 round_digits=None)
    p_anchor = np.array([h["p50"] for h in f_anchor["hours"]])
    assert not np.allclose(p_plain, p_anchor), "Anker-Override muss den Vorwochen-Anker verschieben"

    # (c) Determinismus mit Overrides
    f_hol2 = forecast_next_day(load2d, days, factory,
                               holiday_set=apply_holiday_overrides(hist_hols, [target]),
                               round_digits=None)
    assert f_hol == f_hol2

    # (d) Ehrliches Verhalten OHNE gelernte Feiertage: Override des Zieltags allein ändert nichts
    #     (Koeffizient 0 mangels Trainingsvarianz) — der Anker-Mechanismus wirkt trotzdem.
    f0 = forecast_next_day(load2d, days, factory, holiday_set=set(), round_digits=None)
    f0_hol = forecast_next_day(load2d, days, factory,
                               holiday_set=apply_holiday_overrides(set(), [target]),
                               round_digits=None)
    assert np.allclose([h["p50"] for h in f0["hours"]],
                       [h["p50"] for h in f0_hol["hours"]])


def test_remove_neutralizes_calendar_holiday():
    hol = get_holidays([2026], "NW")
    neujahr = pd.Timestamp("2026-01-01").date()
    assert neujahr in hol
    out = apply_holiday_overrides(hol, None, ["2026-01-01"])
    assert neujahr not in out
    assert len(out) == len(hol) - 1                          # nur dieser eine Tag entfernt
