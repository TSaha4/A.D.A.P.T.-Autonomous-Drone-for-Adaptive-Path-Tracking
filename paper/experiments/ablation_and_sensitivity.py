"""E2 - Ablation: current-only vs current+predicted flood obstacles.
E3 - Sensitivity: HSV margins, number of (simulated) operator clicks, METERS_PER_PIXEL.

All runs use the unmodified pipeline through common.run(). Results: results/ablation.json,
results/sensitivity.csv, results/mpp_sweep.csv. Every outcome is reported, favourable or not.
"""
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from src.mission.coordinates import pixel_to_latlon  # noqa: E402

OUT = os.path.join(common.RESULTS, "sweeps")


def exposure(route, region_mask, step=0.5):
    """Fraction of the route polyline length (display px) that lies on pixels of region_mask."""
    pts = np.array(route, dtype=float)
    inside = total = 0.0
    h, w = region_mask.shape
    for a, b in zip(pts, pts[1:]):
        seg = np.linalg.norm(b - a)
        n = max(1, int(math.ceil(seg / step)))
        ts = (np.arange(n) + 0.5) / n
        xy = a + np.outer(ts, b - a)
        xi = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
        inside += seg * np.mean(region_mask[yi, xi] > 0)
        total += seg
    return inside / total if total else 0.0


def ablation():
    res = {}
    for m in common.MAPS:
        full = common.run(m, out_dir=OUT, tag=f"{m}_pred", predict=True, keep_arrays=True)
        cur = common.run(m, out_dir=OUT, tag=f"{m}_nopred", predict=False, keep_arrays=True)
        fa, ca = full.pop("_arrays"), cur.pop("_arrays")
        predicted_only = ((fa["pred_mask"] > 0) & (fa["mask"] == 0)).astype(np.uint8)
        row = {}
        for name, r in (("current+predicted", full), ("current-only", cur)):
            row[name] = {k: r.get(k) for k in ("obstacle_fraction", "planned_legs", "no_path_legs",
                                                "snapped_endpoints", "unsnappable_endpoints", "route_points",
                                                "route_length_px", "mission_items", "total_s")}
            if r.get("full_path"):
                row[name]["route_fraction_in_predicted_only_area"] = exposure(r["full_path"], predicted_only)
                row[name]["route_fraction_in_current_flood"] = exposure(r["full_path"], fa["mask"])
        row["predicted_only_area_fraction"] = float(predicted_only.mean())
        res[m] = row
        print(m, json.dumps(row, indent=None, default=float)[:600])
    common.save_json(res, "sweeps", "ablation.json")


def sensitivity():
    rows = []
    configs = [dict(hue_margin=h, sv_margin=s, n_clicks=8) for h in (10, 20, 30) for s in (15, 30, 45)]
    configs += [dict(hue_margin=20, sv_margin=30, n_clicks=c) for c in (4, 16)]
    for m in common.MAPS:
        for cfg in configs:
            tag = f"{m}_h{cfg['hue_margin']}_s{cfg['sv_margin']}_c{cfg['n_clicks']}"
            r = common.run(m, out_dir=OUT, tag=tag, **cfg)
            row = {"map": m, **cfg}
            row.update({k: r.get(k) for k in ("hsv_lower", "hsv_upper", "mask_fraction", "regions", "planned_legs",
                                               "no_path_legs", "unsnappable_endpoints", "route_points",
                                               "route_length_px", "mission_items", "exit", "total_s")})
            rows.append(row)
            print(tag, {k: row[k] for k in ("mask_fraction", "regions", "no_path_legs", "route_points")})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def mpp_sweep():
    """The scale only enters at geographic conversion: convert the E1 Varanasi route with several values."""
    side = json.load(open(os.path.join(common.RESULTS, "e2e", "varanasi.sidecar.json")))
    rows = []
    for mpp_orig in (0.5, 1.0, 2.0, 5.0, 10.0, 93.0):
        k = mpp_orig / 0.7716049382716049  # display scale for Varanasi (750/972)
        pts = [pixel_to_latlon(x, y, side["image_center_px"], side["geo_center"], k) for x, y in side["full_path"]]
        length = 0.0
        for (a, b), (c, d) in zip(pts, pts[1:]):
            p1, p2 = math.radians(a), math.radians(c)
            hv = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
            length += 2 * 6371008.8 * math.asin(min(1.0, math.sqrt(hv)))
        lats, lons = [p[0] for p in pts], [p[1] for p in pts]
        rows.append({"meters_per_pixel_original": mpp_orig, "route_length_m": length,
                     "extent_ns_m": (max(lats) - min(lats)) * 111320.0,
                     "extent_ew_m": (max(lons) - min(lons)) * 111320.0 * math.cos(math.radians(side["geo_center"][0]))})
    with open(os.path.join(OUT, "mpp_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    which = sys.argv[1:] or ["ablation", "sensitivity", "mpp"]
    if "ablation" in which:
        ablation()
    if "mpp" in which:
        mpp_sweep()
    if "sensitivity" in which:
        sensitivity()
