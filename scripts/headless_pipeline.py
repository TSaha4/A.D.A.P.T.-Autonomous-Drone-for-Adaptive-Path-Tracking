#!/usr/bin/env python3
"""Headless A.D.A.P.T. pipeline runner: auto-seeds HSV samples from the
annotation color band (replaces the interactive click GUI), then runs the
repo's real pipeline: mask -> contours -> clusters -> safe drops -> TSP ->
D* Lite -> WPL mission + annotated map. Saves per-image results."""
import os, sys, json, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib
import matplotlib.pyplot as plt

from src.vision.image_processing import ImageProcessor
from src.vision.clustering import cluster_contours
from src.routing.pathfinding import nearest_neighbor_tsp, compute_full_path
from src.mission.safe_dropzone import find_safe_drop_points
from src.mission.mission_output import generate_mission_file
from src.weather.flood_spread import predict_spread

logging.disable(logging.INFO)

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "input", "test")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "output", "test_runs")
WEATHER = {"precipitation": 10.0, "wind_speed_10m": 30.0, "wind_direction_10m": 270.0}

# name: (center_lat, center_lon, meters_per_pixel, seed_band)
SEED_BANDS = {
    "red":   lambda h: (h <= 10) | (h >= 170),
    "blue":  lambda h: (h >= 100) & (h <= 130),
    "green": lambda h: (h >= 35) & (h <= 85),
}
IMAGES = {
    "varanasi.png":           (25.3176, 82.9739, 110, "red"),
    "kanpur.png":             (26.4499, 80.3319, 110, "red"),
    "assam.png":              (26.35,   91.50,   380, "red"),
    "it2025_india_floods.png":(23.18,   79.95,    90, "red"),
    "it2025_map_3x4.png":     (23.18,   79.95,    90, "red"),
    "it2025_map_1x1.png":     (25.50,   81.00,   120, "red"),
    "bihar2019_map1.jpg":     (25.60,   85.10,   350, "red"),
    "punjab2025.png":         (31.00,   75.30,   250, "red"),
    "assam2026_satmap.jpg":   (26.90,   94.40,   120, "red"),
    "varanasi_blue.png":      (25.3176, 82.9739, 110, "blue"),
    "varanasi_green.png":     (25.3176, 82.9739, 110, "green"),
}


def seed_points(image, band_fn, max_pts=40, min_dist=15):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = band_fn(h) & (s > 90) & (v > 60)
    ys, xs = np.nonzero(m)
    if len(xs) < 30:
        return [], float(m.mean() * 100)
    pts, d2min = [], min_dist ** 2
    rng = np.random.default_rng(42)
    idx = rng.choice(len(xs), size=min(len(xs), 6000), replace=False)
    for i in idx:
        x, y = int(xs[i]), int(ys[i])
        if all((x - px) ** 2 + (y - py) ** 2 >= d2min for px, py in pts):
            pts.append((x, y))
        if len(pts) >= max_pts:
            break
    return pts, float(m.mean() * 100)


def visualize(image, contours, pred_mask, full_path, drop_indices, home, out_png):
    vis = image.copy()
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    if pred_mask is not None:
        pc, _ = cv2.findContours(pred_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, pc, -1, (0, 165, 255), 1)
    for idx in drop_indices:
        if idx < len(full_path):
            cv2.circle(vis, tuple(map(int, full_path[idx])), 5, (255, 0, 0), -1)
    if home:
        cv2.circle(vis, tuple(map(int, home)), 9, (0, 0, 0), -1)
    for i in range(len(full_path) - 1):
        cv2.line(vis, tuple(map(int, full_path[i])), tuple(map(int, full_path[i + 1])), (0, 0, 255), 2)
    plt.figure(figsize=(10, 10 * vis.shape[0] / vis.shape[1]))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("A.D.A.P.T. headless run: green=mask contours, orange=predicted spread, blue=drops, black=home, red=path")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()


def run_one(name, lat, lon, mpp, band):
    out = os.path.join(OUT_DIR, os.path.splitext(name)[0])
    os.makedirs(out, exist_ok=True)
    img = cv2.imread(os.path.join(TEST_DIR, name))
    if img is None:
        return {"image": name, "status": "UNREADABLE"}
    h, w = img.shape[:2]
    if w > 750:
        img = cv2.resize(img, (750, int(h * 750 / w)), interpolation=cv2.INTER_AREA)
    res = {"image": name, "status": "ok"}

    pts, red_pct = seed_points(img, SEED_BANDS[band])
    res["annotated_fraction_pct"] = round(red_pct, 2)
    res["seed_points"] = len(pts)
    if not pts:
        res["status"] = "FAIL: no annotation-colored pixels found"
        cv2.imwrite(os.path.join(out, "mask.png"), img)
        return res

    proc = ImageProcessor()
    proc.image = img.copy()
    proc.sample_points = pts
    proc.compute_dynamic_hsv()
    mask = proc.mask_flood_areas(img)
    cv2.imwrite(os.path.join(out, "mask.png"), mask)

    pred = predict_spread(mask, WEATHER, 2.0)
    combined = cv2.bitwise_or(mask, pred)
    contours = proc.find_filtered_contours(mask, min_area=200)
    if not contours:
        res["status"] = "FAIL: no contours >= 200 px (segmentation empty)"
        return res
    res["contours"] = len(contours)

    centers, edges = cluster_contours(contours)
    points = [(int(x), int(y)) for x, y in centers]
    res["clusters"] = len(points)
    if len(points) < 2:
        res["status"] = "FAIL: <2 clusters, TSP meaningless"
        return res

    safe = find_safe_drop_points(edges, list(range(len(points))), points)
    res["safe_drops"] = len(safe)
    home = min(safe, key=lambda p: np.hypot(p[0] - img.shape[1] / 2, p[1] - img.shape[0] / 2))
    order, ordered = nearest_neighbor_tsp(safe, home=home)
    full_path, drop_idx = compute_full_path(ordered, combined)
    res["path_points"] = len(full_path)
    res["tsp_order"] = order

    generate_mission_file(full_path, drop_idx, home,
                          (img.shape[1] // 2, img.shape[0] // 2), (lat, lon), mpp,
                          filename=os.path.join(out, "mission.waypoints"))
    with open(os.path.join(out, "mission.waypoints")) as f:
        res["wpl_lines"] = sum(1 for _ in f)
    visualize(img, contours, pred, full_path, drop_idx, home, os.path.join(out, "mission_map.png"))
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for name, (lat, lon, mpp, band) in IMAGES.items():
        print(f"=== {name} ===", flush=True)
        try:
            r = run_one(name, lat, lon, mpp, band)
        except Exception as e:
            r = {"image": name, "status": f"ERROR: {type(e).__name__}: {e}"}
        print(json.dumps(r), flush=True)
        results.append(r)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nsummary saved to", os.path.join(OUT_DIR, "results.json"))


if __name__ == "__main__":
    main()
