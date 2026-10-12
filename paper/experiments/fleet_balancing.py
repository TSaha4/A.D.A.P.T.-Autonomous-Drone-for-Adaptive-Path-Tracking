"""Evaluate an assignment-only workload balancing alternative in the full fleet planner.

The alternative keeps the production base selection and single-drop reach test,
then moves coverable drops between selected bases when this reduces the largest
estimated round-trip load. The unchanged downstream planner validates routes,
battery, mission encoding and crossing repair. No change is made to production.
"""
import json
import sys
from pathlib import Path
from unittest import mock

import multi_uav_comparison as cmp
sys.path.insert(0, cmp.REPO)
import src.mission.multi_base as bases

OUT = Path(cmp.HERE) / "results" / "fleet_balancing"


def balancing_assign(chosen, reach, candidates, drops, distance_fn):
    assignment = bases_original(chosen, reach, candidates, drops, distance_fn)
    if len(chosen) < 2:
        return assignment
    costs = {(i, b): 2.0 * distance_fn(candidates[c], drop)
             for i, drop in enumerate(drops) for b, c in enumerate(chosen) if i in reach[c]}
    def objective(a):
        loads = [0.0] * len(chosen)
        for i, b in a.items():
            loads[b] += costs[i, b]
        return max(loads), sum(loads)
    while True:
        before = objective(assignment)
        best = None
        for i, old in sorted(assignment.items()):
            for b in range(len(chosen)):
                if b == old or (i, b) not in costs:
                    continue
                trial = dict(assignment)
                trial[i] = b
                obj = objective(trial)
                if obj < before and (best is None or obj < best[0]):
                    best = obj, trial
        if best is None:
            return assignment
        assignment = best[1]


def gini(xs):
    if not xs or sum(xs) == 0:
        return 0.0
    n = len(xs)
    return sum(abs(a - b) for a in xs for b in xs) / (2 * n * sum(xs))


def compact(r, method):
    if "uavs" not in r:
        return {"map": r["map"], "weather": r["weather"], "method": method,
                "outcome": r["outcome"], "visited": 0}
    loads = [u["route_length_m"] for u in r["uavs"]]
    return {"map": r["map"], "weather": r["weather"], "method": method,
            "outcome": r["outcome"], "visited": r["visited"], "drop_points": r["drop_points"],
            "n_uavs": r["n_uavs"], "makespan_cruise_min": r["est_completion_min"],
            "route_gini": gini(loads), "total_route_km": r["total_route_km"],
            "mission_valid": r["all_missions_valid"],
            "overlaps_remaining": r["overlaps_remaining"],
            "max_sortie_battery": r["max_sortie_battery"]}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    bases_original = bases._assign
    rows = []
    for map_name in ("varanasi", "kanpur"):
        for weather in ("moderate", "dummy"):
            cmp.OUT = str(OUT / "nearest")
            baseline = cmp.run_config(map_name, weather, 4)
            rows.append(compact(baseline, "nearest_feasible"))
            cmp.OUT = str(OUT / "balanced")
            with mock.patch.object(bases, "_assign", balancing_assign):
                balanced = cmp.run_config(map_name, weather, 4)
            rows.append(compact(balanced, "balanced_roundtrip_proxy"))
            print(map_name, weather, rows[-2:], flush=True)
    (OUT / "comparison.json").write_text(json.dumps(rows, indent=2))
