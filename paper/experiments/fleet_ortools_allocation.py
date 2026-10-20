"""OR-Tools multi-depot allocation baseline run through the fleet mission planner.

Base sites remain the production sites. OR-Tools assigns the same coverable
drops with a 32-drop per-vehicle payload-count capacity and routed-distance
arc costs. Individual vehicle/drop assignments are restricted by the
production single-drop battery reach test. The production downstream sorter,
sortie planner, crossing repair and WPL exporter then run unmodified. This is
an allocation baseline; OR-Tools tour order is not used as the flown route.
"""
import json
import math
import sys
from pathlib import Path
from unittest import mock

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

import multi_uav_comparison as cmp
sys.path.insert(0, cmp.REPO)
import src.mission.multi_base as bases

OUT = Path(cmp.HERE) / "results" / "fleet_ortools"
original = bases.select_bases


def ortools_allocation(candidates, drops, distance_fn, obstacle_mask, **kwargs):
    result = original(candidates, drops, distance_fn, obstacle_mask, **kwargs)
    home = result["bases"]
    indexes = sorted(result["assignment"])
    k = len(home)
    if k < 2 or not indexes:
        return result
    reach = bases.single_drop_reach(home, [drops[i] for i in indexes], distance_fn,
        headwind_mps=kwargs["headwind_mps"], per_drop_kg=kwargs["per_drop_kg"],
        reserve=kwargs["reserve"])
    n = len(indexes)
    sites = [drops[i] for i in indexes] + home
    manager = pywrapcp.RoutingIndexManager(n + k, k, list(range(n, n + k)), list(range(n, n + k)))
    routing = pywrapcp.RoutingModel(manager)
    def distance(a, b):
        pa, pb = sites[manager.IndexToNode(a)], sites[manager.IndexToNode(b)]
        d = distance_fn(pa, pb)
        return min(10**9, int(round(d * 100))) if math.isfinite(d) else 10**9
    arc = routing.RegisterTransitCallback(distance)
    routing.SetArcCostEvaluatorOfAllVehicles(arc)
    capacity = max(1, int(kwargs["payload_capacity_kg"] // kwargs["per_drop_kg"]))
    demand = routing.RegisterUnaryTransitCallback(lambda index: int(manager.IndexToNode(index) < n))
    routing.AddDimensionWithVehicleCapacity(demand, 0, [capacity] * k, True, "PayloadCount")
    for pos in range(n):
        variable = routing.VehicleVar(manager.NodeToIndex(pos))
        for vehicle in range(k):
            if pos not in reach[vehicle]:
                variable.RemoveValue(vehicle)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 3
    solution = routing.SolveWithParameters(params)
    if solution is None:
        return result
    assigned = {}
    for pos, drop_index in enumerate(indexes):
        assigned[drop_index] = int(solution.Value(routing.VehicleVar(manager.NodeToIndex(pos))))
    result["assignment"] = assigned
    return result


def compact(r):
    if "uavs" not in r:
        return {"map": r["map"], "weather": r["weather"], "outcome": r["outcome"],
                "visited": 0}
    loads = [u["route_length_m"] for u in r["uavs"]]
    n = len(loads)
    gini = sum(abs(a - b) for a in loads for b in loads) / (2 * n * sum(loads)) if sum(loads) else 0
    return {"map": r["map"], "weather": r["weather"], "outcome": r["outcome"],
            "visited": r["visited"], "drop_points": r["drop_points"], "n_uavs": r["n_uavs"],
            "makespan_cruise_min": r["est_completion_min"], "route_gini": gini,
            "total_route_km": r["total_route_km"], "mission_valid": r["all_missions_valid"],
            "overlaps_remaining": r["overlaps_remaining"],
            "max_sortie_battery": r["max_sortie_battery"]}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmp.OUT = str(OUT)
    rows = []
    with mock.patch.object(bases, "select_bases", ortools_allocation):
        for map_name in ("varanasi", "kanpur"):
            for weather in ("moderate", "dummy"):
                row = compact(cmp.run_config(map_name, weather, 4))
                rows.append(row)
                print(row, flush=True)
    (OUT / "comparison.json").write_text(json.dumps(rows, indent=2))
