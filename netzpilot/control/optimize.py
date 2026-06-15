# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Amar Akram

"""§14a-Fahrplan-OPTIMIERUNG: minimale, faire Abregelung statt pauschalem Dimmen.

Bisher (schema.make_fahrplan): bei Engpass werden ALLE steuerbaren Verbrauchseinrichtungen pauschal
auf die 4,2-kW-Mindestleistung gedimmt. Das regelt typisch VIEL mehr ab als noetig und ist unfair.

Hier: gegeben die prognostizierte Ueberlast (wie viel kW muss die Summe der steuVE-Leistung in einer
Stunde runter, damit die Netzgrenze haelt) und die je-steuVE angeforderte Leistung — berechne die
Leistungsgrenzen, die

  (1) die Netzgrenze exakt einhalten (Summe <= cap),
  (2) jeder steuVE >= MIN_GUARANTEED_KW garantieren (§14a),
  (3) die GESAMTE Abregelung minimieren,
  (4) FAIR verteilen: gleiche absolute Grenze fuer alle, die ueberhaupt gedimmt werden
      (progressive Water-Filling-Loesung — wer wenig anfordert, wird gar nicht erst gedimmt).

Das ist die EXAKTE Loesung des Problems "min Summe Abregelung s.t. Summe<=cap, p_i>=floor, p_i<=demand_i"
bei gleichverteilter Dimm-Last — analytisch via Water-Filling, KEIN externer Solver noetig (reine
stdlib/numpy). Reproduzierbar und beweisbar optimal.

Diese domaenenspezifische Optimierung (faire, minimale §14a-Abregelung) ist der eigentliche Mehrwert
gegenueber einem trivialen "alle auf 4,2 kW"-Nachbau.
"""
from __future__ import annotations

from .schema import MIN_GUARANTEED_KW


def optimize_setpoints(demands_kw, cap_kw, floor_kw: float = MIN_GUARANTEED_KW):
    """Faire, minimale Abregelung mehrerer steuVE unter einer Summen-Kappung.

    demands_kw: Liste der angeforderten Leistungen je steuerbarer Verbrauchseinrichtung [kW].
    cap_kw:     erlaubte Summenleistung aller steuVE in dieser Stunde [kW] (= Netzentlastungsziel).
    floor_kw:   garantierte Mindestleistung je steuVE (§14a: 4,2 kW).

    Rueckgabe: dict mit
      - limits_kw:        Liste der optimalen Leistungsgrenzen je steuVE (gleiche Reihenfolge),
      - total_shed_kw:    gesamte abgeregelte Leistung (Summe demand - Summe limit),
      - feasible:         False, wenn schon die Summe der Mindestgarantien die Kappung sprengt
                          (dann ist §14a NICHT mit der Netzgrenze vereinbar -> Eskalation noetig),
      - binding_floor:    True, wenn mindestens eine steuVE auf dem 4,2-kW-Floor sitzt.

    Algorithmus (Water-Filling): wir suchen einen einheitlichen Deckel `level`, sodass
        sum_i min(demand_i, max(level, floor)) == cap.
    steuVE unter `level` bleiben unangetastet (fair: wer wenig zieht, wird nicht gedimmt),
    der Rest wird gleichmaessig auf `level` (>= floor) gekappt. Monotone Funktion in `level`
    -> exakte Bisektion.
    """
    d = [float(x) for x in demands_kw]
    n = len(d)
    if n == 0:
        return {"limits_kw": [], "total_shed_kw": 0.0, "feasible": True, "binding_floor": False}

    total_demand = sum(d)
    min_possible = n * floor_kw                       # alle auf Mindestleistung

    # Kein Engpass: Kappung >= Bedarf -> nichts abregeln.
    if cap_kw >= total_demand:
        return {"limits_kw": list(d), "total_shed_kw": 0.0, "feasible": True, "binding_floor": False}

    # §14a vs. Netz: selbst alle auf 4,2 kW sprengt die Kappung -> nicht erfuellbar.
    if cap_kw < min_possible - 1e-9:
        limits = [floor_kw] * n
        return {"limits_kw": limits,
                "total_shed_kw": round(total_demand - sum(limits), 3),
                "feasible": False, "binding_floor": True}

    # Water-Filling: einheitliches `level` so dass Summe der gekappten Leistungen == cap.
    def alloc(level):
        return [min(di, max(level, floor_kw)) for di in d]

    lo, hi = floor_kw, max(d)
    for _ in range(100):                              # Bisektion, 100 Schritte = << 1e-9 Genauigkeit
        mid = 0.5 * (lo + hi)
        if sum(alloc(mid)) > cap_kw:
            hi = mid
        else:
            lo = mid
    limits = alloc(0.5 * (lo + hi))
    # numerische Korrektur: exakt auf cap skalieren, ohne floor zu verletzen
    s = sum(limits)
    if s > cap_kw:
        over = s - cap_kw
        dimmable = [i for i, (li, di) in enumerate(zip(limits, d)) if li > floor_kw + 1e-9]
        if dimmable:
            per = over / len(dimmable)
            for i in dimmable:
                limits[i] = max(floor_kw, limits[i] - per)

    return {
        "limits_kw": [round(x, 3) for x in limits],
        "total_shed_kw": round(total_demand - sum(limits), 3),
        "feasible": True,
        "binding_floor": any(abs(x - floor_kw) < 1e-6 for x in limits),
    }


def naive_shed_kw(demands_kw, floor_kw: float = MIN_GUARANTEED_KW) -> float:
    """Abregelung der bisherigen pauschalen 'alle auf 4,2 kW'-Strategie — als Vergleichsmassstab."""
    return round(sum(float(x) - floor_kw for x in demands_kw if float(x) > floor_kw), 3)


def optimize_setpoints_heterogen(devices, cap_kw):
    """Faire, minimale Abregelung einer HETEROGENEN steuVE-Flotte (WP/Wallbox/Speicher gemischt).

    Verallgemeinert optimize_setpoints: jede Einrichtung hat ihren EIGENEN §14a-Mindestwert und ein
    Gewicht, das steuert, wie stark sie an der Abregelung beteiligt wird. So lassen sich Anlagen mit
    unterschiedlichen Mindestleistungen (Wallbox 4,2 kW, große WP höher) und unterschiedlicher
    Dimm-Bereitschaft (ein Speicher, der gut puffern kann, traegt mehr; eine kritische Last weniger)
    fair zusammen behandeln. optimize_setpoints (homogen) bleibt unveraendert die Basis fuer den
    einfachen Fall.

    devices: Liste von dicts je steuVE:
        {"demand_kw": float, "floor_kw": float (optional, default 4,2), "weight": float (optional,
         default 1,0; HOEHER = nimmt mehr Abregelung auf)}
    cap_kw: erlaubte Summenleistung aller steuVE [kW].

    Modell: gekappte Leistung p_i = clip(demand_i - weight_i * x, floor_i, demand_i), wobei x >= 0 der
    gemeinsame „Abregel-Druck" ist. Wer mehr Gewicht hat, gibt pro x mehr ab; jeder respektiert seinen
    eigenen Floor; wer wenig zieht, wird zuerst gar nicht angetastet. sum(p_i)=cap via Bisektion in x
    (monoton fallend in x) — exakt, reine stdlib, kein Solver. Reduziert sich fuer gleiche floor/weight
    NICHT exakt auf das homogene Water-Filling (anderes Fairness-Kriterium: gleiche absolute Reduktion
    pro Gewichtseinheit statt gleicher Deckel), ist aber ebenso minimal in der Summe (Summe == cap, wenn
    erfuellbar) und §14a-sicher.

    Rueckgabe wie optimize_setpoints: limits_kw, total_shed_kw, feasible, binding_floor.
    """
    d = [float(x["demand_kw"]) for x in devices]
    floors = [float(x.get("floor_kw", MIN_GUARANTEED_KW)) for x in devices]
    weights = [max(1e-9, float(x.get("weight", 1.0))) for x in devices]
    n = len(d)
    if n == 0:
        return {"limits_kw": [], "total_shed_kw": 0.0, "feasible": True, "binding_floor": False}

    total_demand = sum(d)
    min_possible = sum(floors)

    if cap_kw >= total_demand:                       # kein Engpass
        return {"limits_kw": [round(x, 3) for x in d], "total_shed_kw": 0.0,
                "feasible": True, "binding_floor": False}

    if cap_kw < min_possible - 1e-9:                 # §14a vs. Netz unvereinbar -> alle auf ihren Floor
        return {"limits_kw": [round(f, 3) for f in floors],
                "total_shed_kw": round(total_demand - min_possible, 3),
                "feasible": False, "binding_floor": True}

    def alloc(x):
        return [min(di, max(fi, di - wi * x)) for di, fi, wi in zip(d, floors, weights)]

    # hi gross genug, dass JEDE steuVE auf ihrem Floor landet: x_i = (demand_i - floor_i)/weight_i.
    # (max(d) reicht NICHT, sobald ein Gewicht < 1 ist -> sonst konvergiert die Bisektion gegen den Rand.)
    spreads = [(di - fi) / wi for di, fi, wi in zip(d, floors, weights) if di > fi]
    lo, hi = 0.0, (max(spreads) if spreads else 1.0)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if sum(alloc(mid)) > cap_kw:
            lo = mid                                 # mehr Druck noetig
        else:
            hi = mid
    limits = alloc(0.5 * (lo + hi))
    # numerische Feinkorrektur exakt auf cap, ohne Floor zu verletzen
    s = sum(limits)
    if s > cap_kw:
        over = s - cap_kw
        adj = [i for i, (li, fi) in enumerate(zip(limits, floors)) if li > fi + 1e-9]
        wsum = sum(weights[i] for i in adj)
        if wsum > 0:
            for i in adj:
                limits[i] = max(floors[i], limits[i] - over * weights[i] / wsum)

    return {
        "limits_kw": [round(x, 3) for x in limits],
        "total_shed_kw": round(total_demand - sum(limits), 3),
        "feasible": True,
        "binding_floor": any(abs(li - fi) < 1e-6 for li, fi in zip(limits, floors)),
    }
