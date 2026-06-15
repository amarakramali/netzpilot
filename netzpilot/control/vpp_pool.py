"""VPP-/Pool-Dispatch — mehrere steuVE hinter einer gemeinsamen Netzgrenze.

Mehrere steuerbare Anlagen als EINE koordinierte Ressource (virtuelles Kraftwerk).
Hier der §14a-relevante Kern: N steuerbare Verbrauchseinrichtungen sitzen hinter einem gemeinsamen
Netzknoten/Trafo mit EINER Kapazitätsgrenze. Übersteigt die Pool-Summe die Grenze, wird die nötige
Abregelung FAIR und MINIMAL auf die Anlagen verteilt — mit der bereits verifizierten heterogenen
Wasserfüllung (control.optimize.optimize_setpoints_heterogen, je Anlage eigener Floor + Gewicht). Dieses
Modul orchestriert das über den Tag und aggregiert die Pool-Sicht (Pool-Last, Pool-Flexibilitätsband,
je-Anlage-Limits).

EHRLICH (CLAUDE.md): die faire Aufteilung je Periode ist die verifizierte A.2-Optimierung; dieses Modul
ist die AGGREGATIONS-/Orchestrierungs-Schicht (Pool-Band + je-Periode-Kappung + Reporting), KEINE neue
zeitübergreifende Gesamtoptimierung (kein Speicher-/SOC-Modell, keine zeitliche Verschiebung über
Perioden — das wäre der stochastische MILP, bewusst nicht hier). Reine stdlib, additiv.
"""
from __future__ import annotations

from .optimize import optimize_setpoints_heterogen
from .schema import MIN_GUARANTEED_KW


def pool_dispatch(assets, shared_cap_kw, dt_h: float = 1.0) -> dict:
    """Koordinierter Pool-Dispatch unter einer gemeinsamen Netzgrenze je Periode.

    assets: Liste je steuVE mit
        {"demand_kw": [H Werte], optional "floor_kw" (Default 4,2), "weight" (Default 1), "id"}.
    shared_cap_kw: [H] gemeinsame Netzkapazität des Pools je Periode [kW].

    Pro Periode h: faire, minimale Kappung der Pool-Anlagen auf die Summe shared_cap_kw[h]
    (optimize_setpoints_heterogen). Rückgabe dict:
      hourly: [{hour, pool_demand_kw, pool_limit_kw, cap_kw, pool_shed_kw, feasible, asset_limits_kw[]}],
      per_asset: [{id, demand_kwh, granted_kwh, shed_kwh}],
      pool_band: [{hour, min_kw (Σ floor), max_kw (Σ demand)}],
      pool_demand_kwh, pool_granted_kwh, pool_shed_kwh, grid_safe, all_feasible.
    """
    n = len(assets)
    if n == 0:
        raise ValueError("Keine Assets im Pool.")
    H = len(shared_cap_kw)
    if H == 0:
        raise ValueError("Leerer Horizont (shared_cap_kw).")
    floors, weights, demands, ids = [], [], [], []
    for i, a in enumerate(assets):
        d = list(a["demand_kw"])
        if len(d) != H:
            raise ValueError(f"asset[{i}].demand_kw Länge {len(d)} != Horizont {H}.")
        demands.append([float(x) for x in d])
        floors.append(float(a.get("floor_kw", MIN_GUARANTEED_KW)))
        weights.append(max(1e-9, float(a.get("weight", 1.0))))
        ids.append(a.get("id", f"asset{i}"))

    hourly, pool_band = [], []
    asset_granted = [0.0] * n
    asset_demand = [0.0] * n
    pool_demand_kwh = pool_granted_kwh = pool_shed_kwh = 0.0
    grid_safe = True
    all_feasible = True

    for h in range(H):
        cap = float(shared_cap_kw[h])
        devices = [{"demand_kw": demands[i][h], "floor_kw": floors[i], "weight": weights[i]}
                   for i in range(n)]
        res = optimize_setpoints_heterogen(devices, cap)
        limits = res["limits_kw"]
        pool_demand = sum(demands[i][h] for i in range(n))
        pool_limit = sum(limits)
        if pool_limit > cap + 1e-6:
            grid_safe = False
        if not res["feasible"]:
            all_feasible = False
        for i in range(n):
            asset_granted[i] += limits[i] * dt_h
            asset_demand[i] += demands[i][h] * dt_h
        pool_demand_kwh += pool_demand * dt_h
        pool_granted_kwh += pool_limit * dt_h
        pool_shed_kwh += res["total_shed_kw"] * dt_h
        hourly.append({
            "hour": h,
            "pool_demand_kw": round(pool_demand, 3),
            "pool_limit_kw": round(pool_limit, 3),
            "cap_kw": round(cap, 3),
            "pool_shed_kw": round(res["total_shed_kw"], 3),
            "feasible": res["feasible"],
            "asset_limits_kw": limits,
        })
        pool_band.append({
            "hour": h,
            "min_kw": round(sum(floors), 3),                 # Σ garantierte Mindestleistung
            "max_kw": round(pool_demand, 3),                 # Σ angeforderte Leistung
        })

    return {
        "hourly": hourly,
        "pool_band": pool_band,
        "per_asset": [{"id": ids[i], "demand_kwh": round(asset_demand[i], 3),
                       "granted_kwh": round(asset_granted[i], 3),
                       "shed_kwh": round(asset_demand[i] - asset_granted[i], 3)} for i in range(n)],
        "n_assets": n,
        "pool_demand_kwh": round(pool_demand_kwh, 3),
        "pool_granted_kwh": round(pool_granted_kwh, 3),
        "pool_shed_kwh": round(pool_shed_kwh, 3),
        "grid_safe": grid_safe,
        "all_feasible": all_feasible,
        "basis": "Pool-Aggregation; faire minimale Kappung je Periode via verifizierter A.2-Wasserfüllung. "
                 "Keine zeitübergreifende Optimierung/kein Speicher (bewusst v1).",
    }
