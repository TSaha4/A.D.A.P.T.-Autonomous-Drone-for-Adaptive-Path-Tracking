import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import logging
import cv2
import numpy as np
from pathlib import Path

from config import settings
from src.vision.image_processing import ImageProcessor
from src.weather.flood_spread import predict_spread
from src.mission.safe_dropzone import find_safe_drop_points, filter_drops_by_home_connectivity
from src.mission.constraints import score_home_candidates, plan_mission_stops, wind_kmh_to_mps
from src.routing.pathfinding import nearest_neighbor_tsp, compute_full_path

logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

def analyze_weather(weather_file, weather_label):
    print("=" * 80)
    print(f"DIAGNOSTIC: {weather_label}")
    print("=" * 80)

    with open(weather_file, "r") as f:
        weather = json.load(f)
    print(f"Weather: {weather}")

    img_path = "data/input/varanasi.png"
    image = cv2.imread(img_path)
    DISPLAY_WIDTH = 750
    h, w = image.shape[:2]
    if w > DISPLAY_WIDTH:
        new_h = int(round(h * (DISPLAY_WIDTH / float(w))))
        image = cv2.resize(image, (DISPLAY_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    proc = ImageProcessor()
    proc.sample_points = [(400, 340), (462, 264), (546, 314)]
    proc.image = image.copy()
    proc.compute_dynamic_hsv()
    mask = proc.mask_flood_areas(image)
    if mask.ndim == 3 and mask.shape[2] > 1:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    forecasts = predict_spread(mask, weather, 2.0)
    pred_mask = forecasts[max(forecasts)]
    combined_obstacle_mask = cv2.bitwise_or(mask, pred_mask)

    contours = proc.find_filtered_contours(mask, min_area=100.0)
    print(f"Total flood contours detected: {len(contours)}")

    from src.vision.clustering import cluster_contours
    try:
        cluster_centers, cluster_edges = cluster_contours(contours)
    except Exception as e:
        print(f"cluster_contours failed: {e}")
        return

    points = []
    arr = np.asarray(cluster_centers)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    for row in arr:
        points.append((int(row[0]), int(row[1])))

    safe_points = find_safe_drop_points(cluster_edges, list(range(len(points))), points,
                                        obstacle_mask=mask, raw_contours=contours)
    print(f"Total safe drop points generated: {len(safe_points)}")

    west_drops = [(idx, pt) for idx, pt in enumerate(safe_points) if pt[0] < 300]
    east_drops = [(idx, pt) for idx, pt in enumerate(safe_points) if pt[0] >= 300]
    print(f"Western drops (X < 300): {len(west_drops)} -> {[pt for _, pt in west_drops]}")
    print(f"Eastern drops (X >= 300): {len(east_drops)} -> {[pt for _, pt in east_drops]}")

    # Connected components analysis
    downsample_factor = 5
    dil_kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(combined_obstacle_mask, dil_kernel)
    gw, gh = image.shape[1] // downsample_factor, image.shape[0] // downsample_factor
    grid_mask = cv2.resize(dilated_mask, (gw, gh), interpolation=cv2.INTER_NEAREST)
    free = (grid_mask == 0).astype(np.uint8)
    num_comp, labels = cv2.connectedComponents(free, connectivity=8)
    print(f"\nD* Lite grid free-space components: {num_comp}")

    print("\n--- DROP ZONE COMPONENTS ---")
    for idx, pt in west_drops:
        cell = (int(pt[0] // downsample_factor), int(pt[1] // downsample_factor))
        c = labels[cell[1], cell[0]] if (0 <= cell[0] < gw and 0 <= cell[1] < gh) else -1
        print(f"  [WEST] Drop {idx:02d} at {pt} -> Component {c}")
    for idx, pt in east_drops:
        cell = (int(pt[0] // downsample_factor), int(pt[1] // downsample_factor))
        c = labels[cell[1], cell[0]] if (0 <= cell[0] < gw and 0 <= cell[1] < gh) else -1
        print(f"  [EAST] Drop {idx:02d} at {pt} -> Component {c}")

    # Home candidate evaluation
    headwind_mps = wind_kmh_to_mps(weather.get("wind_speed_10m", 0.0))
    wind_from_deg = weather.get("wind_direction_10m")

    dry_y, dry_x = np.where(combined_obstacle_mask == 0)
    stride = max(1, int(max(image.shape[:2]) / 40))
    candidates = [(int(x), int(y)) for y, x in zip(dry_y[::stride], dry_x[::stride])]

    home_eval = score_home_candidates(
        candidates, safe_points, combined_obstacle_mask,
        settings.METERS_PER_PIXEL,
        headwind_mps=headwind_mps,
        wind_from_deg=wind_from_deg,
        payload_kg=0.0,
        per_drop_payload_kg=settings.PAYLOAD_PER_DROP_KG,
        return_details=True
    )
    print(f"\nHome Evaluation: Evaluated={home_eval['evaluated']}, Feasible={len(home_eval['scored'])}, Rejected={len(home_eval['rejected'])}")

    if home_eval["scored"]:
        home = home_eval["scored"][0][1]
        print(f"Selected Feasible Home: {home}")
    else:
        metrics_by_cand = {m["candidate"]: m for m in home_eval.get("sortie_metrics", [])}
        best_candidate_pt = None
        max_reachable_count = 0
        best_cand_metric = None
        for cost, cand, hours, battery, *extra in home_eval["rejected"]:
            metric = metrics_by_cand.get(cand)
            if metric and metric.get("reachable_count", 0) > max_reachable_count:
                max_reachable_count = metric["reachable_count"]
                best_candidate_pt = cand
                best_cand_metric = metric
        home = best_candidate_pt
        print(f"Fallback Home (max reachable): {home} ({max_reachable_count}/{len(safe_points)} reachable)")
        if best_cand_metric:
            print(f"  Unreachable from fallback: {best_cand_metric['unreachable_drops']}")

    # Connectivity filter for chosen home
    kept, excluded = filter_drops_by_home_connectivity(safe_points, home, combined_obstacle_mask, downsample_factor=5)
    print(f"\nConnectivity filter for HOME {home}: Kept={len(kept)}, Excluded={len(excluded)}")
    for ex_idx in excluded:
        pt = safe_points[ex_idx]
        region = "WEST" if pt[0] < 300 else "EAST"
        print(f"  Excluded drop {ex_idx} at {pt} ({region})")

    # TSP + mission simulation for chosen home
    try:
        order, _ = nearest_neighbor_tsp(kept, home=home)
        ord_drops = [kept[i] for i in order]
        sim_route = plan_mission_stops(home, ord_drops, forecasts, mask,
                                       meters_per_pixel=settings.METERS_PER_PIXEL,
                                       headwind_mps=headwind_mps,
                                       initial_battery=1.0)
        visited = [ord_drops[i] for i in sim_route.drop_indices if i < len(ord_drops)]
        visited_west = [p for p in visited if p[0] < 300]
        visited_east = [p for p in visited if p[0] >= 300]
        unreached = [p for p in ord_drops if p not in visited]
        unreached_west = [p for p in unreached if p[0] < 300]
        unreached_east = [p for p in unreached if p[0] >= 300]
        print(f"\nMission Simulation from HOME {home}:")
        print(f"  Visited: {len(visited)} (West={len(visited_west)}, East={len(visited_east)})")
        print(f"  Unreached: {len(unreached)} (West={len(unreached_west)}, East={len(unreached_east)})")
        print(f"  Sorties: {len(sim_route.reload_indices) + 1}")
        if unreached:
            print(f"  Unreached drops: {[('W' if p[0]<300 else 'E', p) for p in unreached]}")
        if sim_route.deviations:
            print(f"  Deviations ({len(sim_route.deviations)}):")
            for d in sim_route.deviations[:10]:
                print(f"    {d}")
    except Exception as e:
        print(f"  Simulation failed: {e}")
        import traceback; traceback.print_exc()

    print()

if __name__ == "__main__":
    analyze_weather("data/input/dummy_weather.json", "EXTREME (Precip=10mm/h, Wind=30km/h)")
    print("\n")
    analyze_weather("data/input/dummy_weather_moderate.json", "MODERATE (Precip=3mm/h, Wind=12km/h)")
