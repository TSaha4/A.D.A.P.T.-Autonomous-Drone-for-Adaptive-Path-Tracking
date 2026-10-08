"""
How the multi-base mode divides flood regions among drones (evaluation only; reads existing results).

For every feasible run of multi_uav_comparison.py, the flood regions are recomputed exactly as the run computed
them (same map, simulated operator clicks, production segmentation and contour filter), each drop point is
attributed to the region it belongs to (the contour nearest to it; drop points lie within 20 px of their region),
and the assignment in run_summary.json is used to count, per run:
  regions_served       regions with at least one drop point assigned to a drone,
  regions_split        regions whose drop points were assigned to more than one drone,
  regions_per_drone    number of regions each drone serves.
Output: results/multi_uav/clusters.json and a printed summary.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [REPO, HERE]
from common import simulated_clicks  # noqa: E402
from src.vision.image_processing import ImageProcessor  # noqa: E402


def regions(map_name):
    img = cv2.imread(os.path.join(REPO, "data", "input", f"{map_name}.png"))
    if img.shape[1] > 750:
        s = 750 / img.shape[1]
        img = cv2.resize(img, (750, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    p = ImageProcessor()
    p.image = img.copy()
    p.sample_points = simulated_clicks(img, 8)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        p.compute_dynamic_hsv()
        mask = p.mask_flood_areas(img)
        return p.find_filtered_contours(mask, min_area=200)


def main():
    records = json.load(open(os.path.join(HERE, "results", "multi_uav", "comparison.json")))
    cache, out = {}, []
    for r in records:
        if "uavs" not in r:
            continue
        contours = cache.setdefault(r["map"], regions(r["map"]))
        summary = json.load(open(os.path.join(HERE, r["session"], "run_summary.json")))
        drops = [tuple(p) for p in summary["drop_points"]]
        owner = {int(i): b for i, b in summary["assignment"].items()}
        region_of = [int(np.argmax([cv2.pointPolygonTest(c, (float(x), float(y)), True) for c in contours]))
                     for x, y in drops]
        bases_of_region = {}
        for i, b in owner.items():
            bases_of_region.setdefault(region_of[i], set()).add(b)
        per_drone = {}
        for reg, bs in bases_of_region.items():
            for b in bs:
                per_drone[b] = per_drone.get(b, 0) + 1
        rec = {"map": r["map"], "weather": r["weather"], "cap": r["cap"], "drones": r["n_uavs"],
               "regions": len(contours), "regions_served": len(bases_of_region),
               "regions_split": sum(len(bs) > 1 for bs in bases_of_region.values()),
               "regions_per_drone": [per_drone[b] for b in sorted(per_drone)]}
        out.append(rec)
        print(rec)
    json.dump(out, open(os.path.join(HERE, "results", "multi_uav", "clusters.json"), "w"), indent=1)
    multi = [o for o in out if o["drones"] > 1]
    served = sum(o["regions_served"] for o in multi)
    split = sum(o["regions_split"] for o in multi)
    print(f"runs with >1 drone: {len(multi)}; regions served {served}; split between drones {split}")


if __name__ == "__main__":
    main()
