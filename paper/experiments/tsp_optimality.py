"""E4 - Route ordering: production nearest-neighbour (NN) heuristic vs exact optimum.

The objective is the one NN itself uses: closed Euclidean tour length (display px) over the safe points,
starting and ending at HOME (HOME is itself one of the safe points in A.D.A.P.T.).
  * Random instances (uniform points in a 750 x 750 px square): exact optimum by Held-Karp DP [Held & Karp 1962].
  * Varanasi (16 safe points): exact optimum by Held-Karp.
  * Kanpur (33 safe points): exact DP infeasible; the minimum-spanning-tree weight is used as a LOWER BOUND
    on the optimum, so NN/MST is an UPPER bound on NN's optimality ratio.
Output: results/tsp/tsp_random.csv, results/tsp/tsp_maps.json
"""
import csv
import json
import os
import random
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from src.routing.pathfinding import nearest_neighbor_tsp  # noqa: E402

OUT = os.path.join(common.RESULTS, "tsp")


def tour_length(points):
    p = np.array(points, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def held_karp(points, start=0):
    n = len(points)
    if n <= 2:
        return 2 * (np.linalg.norm(np.subtract(points[0], points[-1])) if n == 2 else 0.0)
    d = np.linalg.norm(np.array(points)[:, None, :] - np.array(points)[None, :, :], axis=2)
    others = [i for i in range(n) if i != start]
    idx = {v: k for k, v in enumerate(others)}
    m = len(others)
    INF = float("inf")
    dp = np.full((1 << m, m), INF)
    for k, v in enumerate(others):
        dp[1 << k, k] = d[start, v]
    for mask in range(1, 1 << m):
        row = dp[mask]
        for k in range(m):
            if not (mask >> k) & 1 or row[k] == INF:
                continue
            base = row[k]
            vk = others[k]
            for j in range(m):
                if (mask >> j) & 1:
                    continue
                nm = mask | (1 << j)
                c = base + d[vk, others[j]]
                if c < dp[nm, j]:
                    dp[nm, j] = c
    full = (1 << m) - 1
    return float(min(dp[full, k] + d[others[k], start] for k in range(m)))


def mst_weight(points):
    p = np.array(points, dtype=float)
    n = len(p)
    in_tree = np.zeros(n, bool)
    best = np.full(n, np.inf)
    best[0] = 0.0
    total = 0.0
    for _ in range(n):
        u = int(np.argmin(np.where(in_tree, np.inf, best)))
        in_tree[u] = True
        total += best[u]
        best = np.minimum(best, np.linalg.norm(p - p[u], axis=1))
    return float(total)


def nn_length(points, home):
    _, ordered = nearest_neighbor_tsp([tuple(p) for p in points], home=tuple(home))
    return tour_length(ordered)


def random_instances():
    rng = random.Random(2026)
    rows = []
    for n, count in ((5, 300), (6, 300), (8, 300), (10, 200), (12, 100)):
        ratios = []
        for _ in range(count):
            pts = [(rng.uniform(0, 750), rng.uniform(0, 750)) for _ in range(n)]
            home = pts[0]
            opt = held_karp(pts, 0)
            ratios.append(nn_length(pts, home) / opt)
        rows.append({"n_points": n, "instances": count, "mean_ratio": statistics.fmean(ratios),
                     "median_ratio": statistics.median(ratios), "max_ratio": max(ratios),
                     "share_optimal": sum(r < 1 + 1e-9 for r in ratios) / count})
        print(rows[-1])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tsp_random.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def real_maps():
    res = {}
    for m in common.MAPS:
        rec = json.load(open(os.path.join(common.RESULTS, "e2e", f"{m}.json")))
        pts = [tuple(p) for p in rec["safe_points"]]
        home = tuple(rec["home"])
        nn = nn_length(pts, home)
        entry = {"safe_points": len(pts), "nn_length_px": nn}
        if 2 <= len(pts) <= 16:
            start = pts.index(home)
            opt = held_karp(pts, start)
            entry.update(optimal_length_px=opt, nn_over_optimal=nn / opt)
        elif len(pts) > 16:
            lb = mst_weight(pts)
            entry.update(mst_lower_bound_px=lb, nn_over_mst_upper_bound_on_ratio=nn / lb)
        res[m] = entry
        print(m, entry)
    common.save_json(res, "tsp", "tsp_maps.json")


if __name__ == "__main__":
    random_instances()
    real_maps()
