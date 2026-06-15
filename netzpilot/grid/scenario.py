"""Statistischer Flexibilitäts-Zwilling — Was-wäre-wenn-Szenarien für den Netzausbau.

Idee: ein voller Lastfluss-„digitaler Zwilling" ist für kleine
Stadtwerke zu groß. Aber NetzPilot hat bereits eine PROBABILISTISCHE Last-/Residuallast-Prognose je Asset
(grid/overload.py). Daraus lässt sich ohne neue Modellklasse ein STATISTISCHER Zwilling bauen: man addiert
geplante DER-Flotten (Wärmepumpen, Wallboxen, PV) mit ihrem Gleichzeitigkeitsfaktor auf die Prognose und
liest direkt ab, wie sich Überlast-Wahrscheinlichkeit und Hosting-Capacity an diesem Anschlusspunkt ändern.
→ NetzPilot wird vom Prognose- zum INVESTITIONSPLANUNGS-Tool ("Verträgt Trafo A 50 neue Wärmepumpen?").

EHRLICHER SCOPE (CLAUDE.md):
- Komponiert NUR die bereits verifizierte Einzelasset-Engine overload.py (unverändert). KEIN Netzlastfluss,
  keine Topologie — gilt für EINEN Anschlusspunkt/Trafo/Strang gegen seine Bemessungsgrenze.
- Eine DER-Flotte wird als ERWARTETE koinzidente Zusatzlast modelliert: added = Σ count·rated_kw·GLF·Profil.
  Die Prognose-Unsicherheit (Residuen) bleibt voll erhalten; die Flotte verschiebt den Punkt. Das addiert
  NICHT die eigene stochastische Streuung der Flotte (konservativ bzgl. Timing, optimistisch bzgl. des
  Gleichzeitigkeits-Tails) — daher ist `coincidence` der zentrale Planungs-Hebel und wird als Band gefahren.
- Die GLF-/kW-Defaults unten sind ILLUSTRATIVE Planungsannahmen (Größenordnung VDE-AR-N 4100 / BDEW-
  Gleichzeitigkeit) und MÜSSEN durch netzspezifische Werte ersetzt werden. Nichts hier ist eine Messung.

Reine stdlib. Additiv.
"""
from __future__ import annotations

from .overload import overload_forecast, hosting_capacity_kw

# Illustrative Planungsannahmen je EINHEIT — Größenordnung VDE-AR-N 4100 / BDEW-Gleichzeitigkeit.
# NICHT autoritativ: vom Anwender durch eigene Gleichzeitigkeitsfaktoren (GLF) zu ersetzen.
BEISPIEL_DER = {
    "waermepumpe":  {"rated_kw": 4.0,  "coincidence": 0.8,  "sign": +1},   # WP-Heizlast, hohe Gleichzeitigkeit (Kälte)
    "wallbox_11kw": {"rated_kw": 11.0, "coincidence": 0.3,  "sign": +1},   # private Wallbox, GLF sinkt mit Anzahl
    "pv_10kwp":     {"rated_kw": 10.0, "coincidence": 0.85, "sign": -1},   # PV speist ein → entlastet (Vorzeichen −)
}


def der_added_load_kw(fleets, horizon, profiles=None):
    """Erwartete koinzidente Zusatzlast je Periode [kW] aus einer Liste von DER-Flotten.

    fleets: Liste von dicts {name, count, rated_kw, coincidence (0..1), sign (+1 Last/−1 Erzeugung),
            profile (optional: Länge-`horizon`-Gewichte 0..1, Default überall 1.0)}.
    Rückgabe: Liste der Länge `horizon` mit signierter Zusatzlast (Last positiv, Erzeugung negativ).
    """
    if horizon <= 0:
        raise ValueError("horizon muss > 0 sein.")
    added = [0.0] * horizon
    for f in fleets:
        count = float(f.get("count", 0))
        rated = float(f["rated_kw"])
        glf = float(f.get("coincidence", 1.0))
        sign = float(f.get("sign", +1))
        if count < 0:
            raise ValueError(f"Flotte '{f.get('name')}': count < 0.")
        if not (0.0 <= glf <= 1.0):
            raise ValueError(f"Flotte '{f.get('name')}': coincidence {glf} nicht in [0,1].")
        prof = f.get("profile")
        if prof is None:
            prof = [1.0] * horizon
        if len(prof) != horizon:
            raise ValueError(f"Flotte '{f.get('name')}': profile-Länge {len(prof)} != horizon {horizon}.")
        per_unit_peak = rated * glf
        for h in range(horizon):
            added[h] += sign * count * per_unit_peak * float(prof[h])
    return added


def simulate_scenario(point_kw, residuals_kw, rating_kw, fleets, dt_h: float = 1.0,
                      risk_alpha: float = 0.05, profiles=None) -> dict:
    """Was-wäre-wenn: DER-Flotten auf die Prognose addieren, Überlast-Risiko + Hosting-Capacity vor/nach.

    point_kw[H], residuals_kw[H][·], rating_kw, dt_h, risk_alpha: wie overload_forecast.
    fleets: siehe der_added_load_kw.
    Rückgabe dict: base (overload_forecast ohne Flotten), scenario (mit), delta (Risiko-Änderungen),
    added_peak_kw, hosting_capacity_base/scenario, note.
    """
    H = len(point_kw)
    added = der_added_load_kw(fleets, H, profiles)
    scen_point = [float(point_kw[h]) + added[h] for h in range(H)]

    base = overload_forecast(point_kw, residuals_kw, rating_kw, dt_h, risk_alpha)
    scen = overload_forecast(scen_point, residuals_kw, rating_kw, dt_h, risk_alpha)
    hc_base = hosting_capacity_kw(point_kw, residuals_kw, rating_kw, risk_alpha)
    hc_scen = hosting_capacity_kw(scen_point, residuals_kw, rating_kw, risk_alpha)

    return {
        "base": base,
        "scenario": scen,
        "added_peak_kw": round(max((abs(a) for a in added), default=0.0), 3),
        "delta": {
            "max_exceedance_prob": round(scen["max_exceedance_prob"] - base["max_exceedance_prob"], 4),
            "hours_at_risk": scen["hours_at_risk"] - base["hours_at_risk"],
            "expected_overload_kwh_total": round(
                scen["expected_overload_kwh_total"] - base["expected_overload_kwh_total"], 4),
        },
        "hosting_capacity_base_kw": hc_base["hosting_capacity_kw"],
        "hosting_capacity_scenario_kw": hc_scen["hosting_capacity_kw"],
        "note": "Statistischer Zwilling: erwartete koinzidente DER-Zusatzlast auf die Prognoseverteilung "
                "(Residuen unverändert). Einzelasset, KEIN Netzlastfluss. GLF/kW sind Planungsannahmen "
                "(VDE-AR-N 4100/BDEW-Größenordnung), durch netzspezifische Werte ersetzen. "
                "Flotten-Eigenstreuung nicht modelliert → coincidence als Band fahren.",
    }


def coincidence_band(point_kw, residuals_kw, rating_kw, fleets, factors=(0.5, 1.0, 1.5),
                     dt_h: float = 1.0, risk_alpha: float = 0.05) -> dict:
    """Sensitivität: das Szenario über mehrere GLF-Skalierungen fahren (low/erwartet/high).

    Multipliziert jede Flotten-coincidence mit jedem Faktor (auf [0,1] geklemmt) und liefert je Faktor
    die Überlast-Spitzen-Wkt + verbleibende Hosting-Capacity — macht die Abhängigkeit von der wichtigsten
    Planungsannahme transparent statt sie zu verstecken.
    """
    rows = []
    for s in factors:
        scaled = []
        for f in fleets:
            g = min(1.0, max(0.0, float(f.get("coincidence", 1.0)) * s))
            scaled.append({**f, "coincidence": g})
        out = simulate_scenario(point_kw, residuals_kw, rating_kw, scaled, dt_h, risk_alpha)
        rows.append({
            "coincidence_factor": s,
            "added_peak_kw": out["added_peak_kw"],
            "scenario_max_exceedance_prob": out["scenario"]["max_exceedance_prob"],
            "scenario_hours_at_risk": out["scenario"]["hours_at_risk"],
            "hosting_capacity_scenario_kw": out["hosting_capacity_scenario_kw"],
        })
    return {"band": rows, "note": "GLF-Sensitivität; coincidence je Flotte × Faktor (auf [0,1] geklemmt)."}
