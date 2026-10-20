"""E5 - Per-leg production D* Lite vs a plain grid A* (EVALUATION BASELINE ONLY; not part of A.D.A.P.T.).

Both planners receive exactly the legs the pipeline plans in E1: the same 5x-downsampled obstacle grid
(cv2.INTER_NEAREST), the same endpoint mapping and the same production snapping. A* uses the same
8-connected graph, the same Euclidean step costs, the same obstacle semantics (an edge is blocked if either
end cell is an obstacle) and the same Chebyshev heuristic, so both should return equal optimal costs.
Reported per map: leg count, agreement on reachability, max |cost difference|, total runtime, and work
counters (A* node expansions; D* Lite vertex pops and vertex updates, counted via an evaluation-side
subclass that does not change the production algorithm).
Output: results/planner/planner_baseline.json and planner_legs.csv
"""
import csv
import heapq
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from src.routing.dstarlite import DStarLite  # noqa: E402
from src.routing.pathfinding import snap_to_nearest_free_cell  # noqa: E402

OUT = os.path.join(common.RESULTS, "planner")
FACTOR = 5


class CountingDStarLite(DStarLite):
    def __init__(self, *a, **k):
        self.pops = self.updates = 0
        super().__init__(*a, **k)

    def pop(self):
        self.pops += 1
        return super().pop()

    def update_vertex(self, u):
        self.updates += 1
        return super().update_vertex(u)


def astar(shape, start, goal, obstacles):
    rows, cols = shape
    h = lambda p: max(abs(p[0] - goal[0]), abs(p[1] - goal[1]))
    if start in obstacles or goal in obstacles:
        return None, 0
    g = {start: 0.0}
    parent = {}
    heap = [(h(start), 0.0, start)]
    closed = set()
    expansions = 0
    while heap:
        f, gc, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        expansions += 1
        if u == goal:
            path = [u]
            while u in parent:
                u = parent[u]
                path.append(u)
            return path[::-1], expansions
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                v = (u[0] + dx, u[1] + dy)
                if not (0 <= v[0] < cols and 0 <= v[1] < rows) or v in obstacles or v in closed:
                    continue
                ng = gc + math.sqrt(dx * dx + dy * dy)
                if ng < g.get(v, math.inf):
                    g[v] = ng
                    parent[v] = u
                    heapq.heappush(heap, (ng + h(v), ng, v))
    return None, expansions


def path_cost(path):
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def main():
    os.makedirs(OUT, exist_ok=True)
    summary, legs_rows = {}, []
    for m in common.MAPS:
        rec = json.load(open(os.path.join(common.RESULTS, "e2e", f"{m}.json")))
        mask = np.load(os.path.join(common.RESULTS, "e2e", f"{m}.npz"))["obstacle_mask"]
        h, w = mask.shape
        nw, nh = w // FACTOR, h // FACTOR
        small = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        yi, xi = np.where(small > 0)
        obstacles = set(zip(xi.tolist(), yi.tolist()))
        pts = rec["ordered_points"]
        s = {"legs": 0, "both_found": 0, "both_none": 0, "disagree": 0, "max_abs_cost_diff": 0.0,
             "dstar_s": 0.0, "astar_s": 0.0, "dstar_pops": 0, "dstar_updates": 0, "astar_expansions": 0,
             "grid": [nw, nh], "obstacle_cells": len(obstacles)}
        for i in range(len(pts) - 1):
            sg = (min(max(0, int(pts[i][0] // FACTOR)), nw - 1), min(max(0, int(pts[i][1] // FACTOR)), nh - 1))
            gg = (min(max(0, int(pts[i + 1][0] // FACTOR)), nw - 1), min(max(0, int(pts[i + 1][1] // FACTOR)), nh - 1))
            if sg in obstacles:
                sg, _ = snap_to_nearest_free_cell(sg, obstacles, nw, nh)
            if gg in obstacles:
                gg, _ = snap_to_nearest_free_cell(gg, obstacles, nw, nh)
            if sg == gg:
                continue
            t = time.perf_counter()
            d = CountingDStarLite((nh, nw), sg, gg)
            dp = d.plan_path(obstacles)
            td = time.perf_counter() - t
            t = time.perf_counter()
            ap, exp = astar((nh, nw), sg, gg, obstacles)
            ta = time.perf_counter() - t
            dfound, afound = bool(dp), ap is not None
            dc = path_cost(dp) if dfound else None
            ac = path_cost(ap) if afound else None
            s["legs"] += 1
            s["dstar_s"] += td
            s["astar_s"] += ta
            s["dstar_pops"] += d.pops
            s["dstar_updates"] += d.updates
            s["astar_expansions"] += exp
            if dfound and afound:
                s["both_found"] += 1
                s["max_abs_cost_diff"] = max(s["max_abs_cost_diff"], abs(dc - ac))
            elif not dfound and not afound:
                s["both_none"] += 1
            else:
                s["disagree"] += 1
            legs_rows.append({"map": m, "leg": i, "start": sg, "goal": gg, "dstar_found": dfound, "astar_found": afound,
                              "dstar_cost": dc, "astar_cost": ac, "dstar_s": td, "astar_s": ta,
                              "dstar_pops": d.pops, "dstar_updates": d.updates, "astar_expansions": exp})
        summary[m] = s
        print(m, s)
    common.save_json(summary, "planner", "planner_baseline.json")
    with open(os.path.join(OUT, "planner_legs.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(legs_rows[0].keys()))
        wr.writeheader()
        wr.writerows(legs_rows)


if __name__ == "__main__":
    main()
