# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""Kostenoptimaler Lastfahrplan gegen ein zeitvariables Netzentgelt (§14a EnWG Modul 3) — Achse A.3.

Hintergrund: §14a EnWG erlaubt seit 2025 als **Modul 3** ein ZEITVARIABLES Netzentgelt — der
Netzbetreiber setzt je Tageszeit unterschiedliche Netzentgelt-Stufen (Hoch-/Standard-/Niedrigtarif).
Eine steuerbare Verbrauchseinrichtung (E-Auto, gepufferte Waermepumpe) hat einen Tages-Energiebedarf,
kann aber waehlen, WANN sie ihn deckt. Dieses Modul rechnet den kostenoptimalen Fahrplan: die benoetigte
Energie in die guenstigsten Stunden schieben — UNTER Einhaltung von Leistungsgrenze, Verfuegbarkeits-
fenster und der §14a-Engpass-Caps (aus control/optimize bzw. redispatch).

Modell (transparent, ein gegebenes Tarifprofil — KEINE erfundenen Einsparungen):
    minimiere  sum_t fee_t * x_t
    u.d.N.     sum_t x_t = energy_kwh                (Tagesbedarf muss gedeckt werden)
               0 <= x_t <= ceil_t                    (ceil_t = min(p_max, cap_t) * dt, 0 wenn nicht verfuegbar)
    x_t = Energie [kWh], die in Stunde t bezogen wird; fee_t = Netzentgelt [EUR/kWh] in Stunde t.

Loesung: GREEDY nach Preis — fuelle die guenstigsten Stunden zuerst bis zu ihrer Obergrenze ceil_t,
bis der Energiebedarf gedeckt ist. Fuer dieses Ein-Constraint-Problem (eine Gleichheit + Box-Schranken)
ist das BEWEISBAR optimal (Austausch-Argument: jede Umschichtung von einer guenstigeren in eine teurere
Stunde erhoeht die Kosten). Exakt, reine stdlib, KEIN Solver.

Verzahnung mit §14a-Engpass: cap_kw[t] ist die in Stunde t maximal zulaessige steuVE-Leistung (z.B. aus
optimize_setpoints / rolling_redispatch). In einer Engpassstunde mit cap=0 wird dort NICHTS geplant —
die Netzsicherheit dominiert die Kostenoptimierung. So ergaenzen sich A.1/A.2 (raeumlich faire Kappung)
und A.3 (zeitlich guenstigste Platzierung INNERHALB der Kappung).

Ehrliche Grenzen (CLAUDE.md):
  - Optimiert nur die NETZENTGELT-Komponente gegen ein vom Netzbetreiber GESETZTES Profil; die Ersparnis
    ist gegenueber ungeplanter Platzierung (z.B. Sofortladen) auf DEMSELBEN Tarif — kein Markt-/Spot-Spiel.
  - Verschiebbares Energiebudget-Modell: KEINE thermische Dynamik / kein Ladezustands-Komfortmodell
    (eine WP mit begrenztem Pufferspeicher kann nicht beliebig verschoben werden). v1-Grenze, klar benannt.
  - Modul-3-Tariffenster sind netzbetreiberspezifisch und werden als EINGABE uebergeben (nicht erfunden).
"""
from __future__ import annotations

MIN_GUARANTEED_KW = 4.2  # §14a-Mindestleistung (nur als Default-Bezug; A.3 cappt, garantiert nicht)


def optimize_grid_fee_schedule(fee_eur_per_kwh, energy_kwh, p_max_kw,
                               cap_kw=None, available=None, dt_h: float = 1.0) -> dict:
    """Kostenoptimaler Bezugsfahrplan einer steuerbaren Last gegen ein zeitvariables Netzentgelt.

    fee_eur_per_kwh: Liste der Netzentgelt-Stufen je Periode [EUR/kWh] (Laenge = Horizont, z.B. 24).
    energy_kwh:      ueber den Horizont zu beziehende Gesamtenergie [kWh] (Tagesbedarf der Anlage).
    p_max_kw:        maximale Anschlussleistung der Anlage [kW].
    cap_kw:          optionale je-Periode-Obergrenze der zulaessigen Leistung [kW] (§14a-Engpass-Cap);
                     None => kein zusaetzliches Limit (nur p_max). cap_kw[t]=0 => Stunde gesperrt.
    available:       optionale Bool-Liste je Periode (False => Anlage nicht am Netz, x_t=0).
    dt_h:            Periodenlaenge in Stunden (1.0 = Stundenraster; 0.25 = Viertelstunde).

    Rueckgabe dict:
      schedule_kwh:      Liste der bezogenen Energie je Periode [kWh] (Summe = energy_kwh, wenn feasible),
      power_kw:          schedule_kwh / dt_h (zur Anschauung),
      total_cost_eur:    sum fee_t * x_t,
      naive_cost_eur:    Kosten der ungeplanten Sofort-Platzierung (frueheste verfuegbare Stunden zuerst),
      saving_eur:        naive_cost - total_cost (>= 0),
      feasible:          False, wenn der Bedarf die Summe der Obergrenzen sprengt,
      scheduled_kwh:     tatsaechlich platzierte Energie (== energy_kwh wenn feasible, sonst Maximum),
      shortfall_kwh:     ungedeckter Rest (0 wenn feasible).
    """
    fee = [float(f) for f in fee_eur_per_kwh]
    n = len(fee)
    if n == 0:
        raise ValueError("fee_eur_per_kwh ist leer.")
    E = float(energy_kwh)
    if E < 0:
        raise ValueError("energy_kwh muss >= 0 sein.")
    if p_max_kw < 0:
        raise ValueError("p_max_kw muss >= 0 sein.")
    if dt_h <= 0:
        raise ValueError("dt_h muss > 0 sein.")

    if cap_kw is None:
        caps = [float(p_max_kw)] * n
    else:
        if len(cap_kw) != n:
            raise ValueError(f"cap_kw-Laenge {len(cap_kw)} != Horizont {n}.")
        caps = [min(float(p_max_kw), float(c)) for c in cap_kw]
    if available is None:
        avail = [True] * n
    else:
        if len(available) != n:
            raise ValueError(f"available-Laenge {len(available)} != Horizont {n}.")
        avail = [bool(a) for a in available]

    # Obergrenze je Periode [kWh]: Leistung*dt, 0 wenn nicht verfuegbar.
    ceil_kwh = [max(0.0, caps[t]) * dt_h if avail[t] else 0.0 for t in range(n)]
    total_ceiling = sum(ceil_kwh)

    # --- Greedy nach Preis: guenstigste Stunden zuerst fuellen ---
    schedule = [0.0] * n
    remaining = min(E, total_ceiling)        # bei Infeasibilitaet so viel wie moeglich
    for t in sorted(range(n), key=lambda i: fee[i]):
        if remaining <= 1e-12:
            break
        take = min(ceil_kwh[t], remaining)
        schedule[t] = take
        remaining -= take

    scheduled = sum(schedule)
    feasible = scheduled >= E - 1e-9
    total_cost = sum(fee[t] * schedule[t] for t in range(n))

    # --- Naive Baseline: Sofort-Platzierung (frueheste verfuegbare Stunden zuerst), gleiche Energiemenge ---
    naive = [0.0] * n
    rem = scheduled                          # gleiche platzierte Menge -> fairer Kostenvergleich
    for t in range(n):
        if rem <= 1e-12:
            break
        take = min(ceil_kwh[t], rem)
        naive[t] = take
        rem -= take
    naive_cost = sum(fee[t] * naive[t] for t in range(n))

    return {
        "schedule_kwh": [round(x, 6) for x in schedule],
        "power_kw": [round(x / dt_h, 6) for x in schedule],
        "total_cost_eur": round(total_cost, 6),
        "naive_cost_eur": round(naive_cost, 6),
        "saving_eur": round(naive_cost - total_cost, 6),
        "feasible": feasible,
        "scheduled_kwh": round(scheduled, 6),
        "shortfall_kwh": round(max(0.0, E - scheduled), 6),
    }
