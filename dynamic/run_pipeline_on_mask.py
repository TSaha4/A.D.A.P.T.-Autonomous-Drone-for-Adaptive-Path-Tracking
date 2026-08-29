#!/usr/bin/env python3
"""Mask-direct A.D.A.P.T. pipeline: consumes a binary flood mask (SAR GeoTIFF
derivative or any mask PNG) + geo.json — NO color segmentation at all.
Runs: contours -> clusters -> safe drops -> NN-TSP -> D* Lite -> georeferenced WPL.

Usage:
  python run_pipeline_on_mask.py --dir out/nepal2026 --lat 28.05 --lon 85.15
"""
import argparse, json, os, sys

_here = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if os.path.isdir(os.path.join(_c, "src")):
        sys.path.insert(0, _c)
        break
else:
    sys.exit("repo root with src/ not found — run from inside the A.D.A.P.T. repo")
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.vision.clustering import cluster_contours
from src.routing.pathfinding import nearest_neighbor_tsp, compute_full_path
from src.mission.safe_dropzone import find_safe_drop_points
from src.mission.mission_output import generate_mission_file
from src.weather.flood_spread import predict_spread

WEATHER = {"precipitation": 10.0, "wind_speed_10m": 30.0, "wind_direction_10m": 270.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder containing flood_mask.png + geo.json")
    ap.add_argument("--min-area", type=int, default=150, help="min blob area in px (10m px -> 150px = 1.5 ha)")
    ap.add_argument("--max-clusters", type=int, default=30, help="cap TSP stops (drop smallest beyond this)")
    ap.add_argument("--weather", default="dummy", choices=["dummy", "api"])
    args = ap.parse_args()

    mask = cv2.imread(os.path.join(args.dir, "flood_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        sys.exit("flood_mask.png not found — run fetch_s1_cdse.py or fetch_gibs.py first")
    geo = json.load(open(os.path.join(args.dir, "geo.json")))
    lat, lon, mpp = geo["lat_center"], geo["lon_center"], geo["meters_per_pixel"]

    # weather for the spread predictor (dummy = deterministic offline)
    weather = WEATHER

    pred = predict_spread(mask, weather, 2.0)
    combined = cv2.bitwise_or(mask, pred)

    contours = [c for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if cv2.contourArea(c) >= args.min_area]
    if len(contours) < 2:
        sys.exit(f"only {len(contours)} blobs >= {args.min_area}px — not enough drop zones")
    # cap stops: keep the biggest clusters
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:args.max_clusters]

    centers, edges = cluster_contours(contours)
    points = [(int(x), int(y)) for x, y in centers]
    print(f"{len(points)} drop zones detected ({(mask>0).sum()} px @ {mpp:.0f} m/px)")

    safe = find_safe_drop_points(edges, list(range(len(points))), points)
    home = min(safe, key=lambda p: np.hypot(p[0] - mask.shape[1] / 2, p[1] - mask.shape[0] / 2))
    order, ordered = nearest_neighbor_tsp(safe, home=home)
    full_path, drop_idx = compute_full_path(ordered, combined)

    wp = os.path.join(args.dir, "mission.waypoints")
    generate_mission_file(full_path, drop_idx, home,
                          (mask.shape[1] // 2, mask.shape[0] // 2), (lat, lon), mpp,
                          filename=wp)

    # annotated map: overlay mission on the OSM style map when available
    style_path = os.path.join(args.dir, "style_map.png")
    if not os.path.exists(style_path):
        parent = os.path.dirname(os.path.abspath(args.dir).rstrip("/"))
        cand = os.path.join(parent, "style_map.png")
        if os.path.exists(cand):
            style_path = cand
    if os.path.exists(style_path):
        vis = cv2.imread(style_path)
        vis = cv2.resize(vis, (mask.shape[1], mask.shape[0]))
    else:
        vis = cv2.cvtColor(cv2.bitwise_or(mask, cv2.bitwise_and(pred, mask)), cv2.COLOR_GRAY2BGR)

    # green: detected flood contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 1)
    # orange: predicted spread contours
    pc, _ = cv2.findContours(pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, pc, -1, (0, 165, 255), 1)
    k = max(vis.shape[1], vis.shape[0]) / 1400.0   # proportional scale
    th = max(1, int(round(2 * k)))
    for i in drop_idx:
        if i < len(full_path):
            cv2.circle(vis, tuple(map(int, full_path[i])), int(7 * k) + 2, (255, 0, 0), -1)
    cv2.circle(vis, tuple(map(int, home)), int(9 * k) + 2, (0, 0, 0), -1)
    for i in range(len(full_path) - 1):
        cv2.line(vis, tuple(map(int, full_path[i])), tuple(map(int, full_path[i + 1])), (0, 0, 255), th)
    legend = [("DETECTED flood", (0, 255, 0)), ("PREDICTED spread", (0, 165, 255)),
              ("drop zone", (255, 0, 0)), ("HOME", (0, 0, 0)), ("path", (0, 0, 255))]
    fs = 0.5 * max(k, 0.6)
    row, box = int(26 * max(k, 0.8)), int(13 * max(k, 0.8))
    for j, (txt, col) in enumerate(legend):
        y = int(26 * k) + j * row + box
        cv2.rectangle(vis, (10, y - box), (10 + 2 * box, y + box), col, -1)
        cv2.putText(vis, txt, (16 + 3 * box, y + box // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(1, int(1.2 * k)), cv2.LINE_AA)
    out_png = os.path.join(args.dir, "mission_map.png")
    plt.figure(figsize=(12, 12 * vis.shape[0] / vis.shape[1]))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"A.D.A.P.T. mission: {len(points)} zones, {len(full_path)} path pts")
    plt.axis("off")
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print("mission ->", wp, "| map ->", out_png)


if __name__ == "__main__":
    main()
