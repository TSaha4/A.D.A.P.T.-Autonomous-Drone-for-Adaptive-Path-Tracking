"""Compare production nearest-neighbour ordering with local search and OR-Tools.

All methods use the same closed Euclidean safe-point tour objective used by the
production ordering step. These are ordering-only results, before obstacle routing.
"""
import json
from pathlib import Path

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from tsp_optimality import held_karp
from src.routing.pathfinding import nearest_neighbor_tsp

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "paper/experiments/results/e2e"
OUT = ROOT / "paper/experiments/results/ordering"


def length(tour, d):
    return float(sum(d[a, b] for a, b in zip(tour, tour[1:])))


def two_opt(tour, d):
    best = list(tour)
    while True:
        chosen = None
        old = length(best, d)
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cost = length(cand, d)
                if cost < old - 1e-9 and (chosen is None or cost < chosen[0]):
                    chosen = (cost, cand)
        if chosen is None:
            return best
        best = chosen[1]


def or_opt(tour, d):
    best = list(tour)
    while True:
        chosen = None
        old = length(best, d)
        core = best[1:-1]
        for span in (1, 2, 3):
            for i in range(len(core) - span + 1):
                block = core[i:i + span]
                rest = core[:i] + core[i + span:]
                for j in range(len(rest) + 1):
                    cand = [best[0]] + rest[:j] + block + rest[j:] + [best[0]]
                    cost = length(cand, d)
                    if cost < old - 1e-9 and (chosen is None or cost < chosen[0]):
                        chosen = (cost, cand)
        if chosen is None:
            return best
        best = chosen[1]


def ortools_tour(d):
    n = len(d)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    model = pywrapcp.RoutingModel(manager)
    def cost(a, b):
        return int(round(d[manager.IndexToNode(a), manager.IndexToNode(b)] * 1000))
    idx = model.RegisterTransitCallback(cost)
    model.SetArcCostEvaluatorOfAllVehicles(idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 2
    solution = model.SolveWithParameters(params)
    if solution is None:
        return None
    route, cursor = [], model.Start(0)
    while not model.IsEnd(cursor):
        route.append(manager.IndexToNode(cursor))
        cursor = solution.Value(model.NextVar(cursor))
    return route + [0]


def evaluate(name):
    record = json.loads((IN / f"{name}.json").read_text())
    points = [tuple(p) for p in record["safe_points"]]
    home = tuple(record["home"])
    start = points.index(home)
    pts = [points[start]] + [p for i, p in enumerate(points) if i != start]
    p = np.asarray(pts, float)
    d = np.linalg.norm(p[:, None] - p[None, :], axis=2)
    _, ordered = nearest_neighbor_tsp(points, home)
    nn = [pts.index(tuple(x)) for x in ordered]
    routes = {"nearest_neighbor": nn,
              "two_opt": two_opt(nn, d),
              "or_opt": or_opt(nn, d),
              "two_opt_then_or_opt": or_opt(two_opt(nn, d), d),
              "or_tools": ortools_tour(d)}
    result = {"map": name, "points": len(pts), "objective": "closed Euclidean tour in display pixels",
              "length_px": {k: length(v, d) for k, v in routes.items() if v is not None}}
    if len(pts) <= 16:
        result["held_karp_optimal_px"] = held_karp(pts, 0)
    return result


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [evaluate(m) for m in ("varanasi", "kanpur")]
    (OUT / "comparison.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
