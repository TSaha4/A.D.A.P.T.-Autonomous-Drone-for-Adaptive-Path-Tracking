#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from typing import List, Tuple, Any

import cv2
import numpy as np

from src.vision.image_processing import ImageProcessor
from src.vision.clustering import cluster_contours
from src.mission.safe_dropzone import find_safe_drop_points
from src.mission.mission_output import generate_mission_file, display_path_on_map
from src.weather.weather_api import get_weather_data
from src.weather.flood_spread import predict_spread_sequence
from src.mission.constraints import DroneSpec
from src.routing.mission_planner import MissionPlanner, DropZone
from config import settings

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def show_image_safe(title: str, img: np.ndarray, wait: bool = True) -> None:
    #Safe cv2.imshow wrapper.
    if img is None:
        logging.debug("Requested to show an empty image.")
        return
    try:
        disp = img
        if disp.dtype == np.bool_:
            disp = (disp.astype("uint8") * 255)
        elif disp.dtype in (np.float32, np.float64):
            m, M = disp.min(), disp.max()
            if M - m > 1e-8:
                disp = ((disp - m) / (M - m) * 255.0).astype("uint8")
            else:
                disp = (disp * 255.0).astype("uint8")
        elif disp.dtype != np.uint8:
            disp = disp.astype("uint8")

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, disp)
        if wait:
            cv2.waitKey(0)
            cv2.destroyWindow(title)
    except cv2.error as e:
        logging.warning(f"Could not show image window ('{title}'): {e}. Continuing without GUI.")


def to_point_list(cluster_centers: Any) -> List[Tuple[int, int]]:
    #Convert cluster_centers to list of (x,y) int tuples.
    if cluster_centers is None:
        return []
    arr = np.asarray(cluster_centers)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("cluster_centers must be convertible to Nx2 array")
    pts = [(int(round(float(x))), int(round(float(y)))) for x, y in arr[:, :2]]
    return pts


def snap_to_dry(point: Tuple[int, int], flood_mask: np.ndarray, max_radius: int = 8):
    """
    Contour/hull-derived "safe" points can still sit on the flood mask boundary
    (a water pixel).  Snap such a point to the nearest *dry* pixel so the drop
    actually lands on land.  Returns (dry_point, distance_px); the original
    point unchanged if it is already dry.
    """
    from collections import deque
    x0, y0 = int(point[0]), int(point[1])
    h, w = flood_mask.shape[:2]
    if 0 <= y0 < h and 0 <= x0 < w and flood_mask[y0, x0] == 0:
        return (x0, y0), 0

    queue = deque([(x0, y0, 0)])
    seen = {(x0, y0)}
    while queue:
        x, y, d = queue.popleft()
        if d > max_radius:
            break
        if 0 <= y < h and 0 <= x < w and flood_mask[y, x] == 0:
            return (x, y), d
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nx, ny = x + dx, y + dy
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append((nx, ny, d + 1))
    return (x0, y0), max_radius


def main():
    parser = argparse.ArgumentParser(description="Generate drone mission from flood-map image.")
    parser.add_argument("image_path", nargs="?", default="data/input/varanasi.png", help="Input map image path")
    parser.add_argument("--display-width", "-w", type=int, default=750, help="Resize display width")
    parser.add_argument("--min-area", type=int, default=200, help="Minimum contour area to keep")
    parser.add_argument("--lat", type=float, default=25.3176, help="Latitude for weather API")
    parser.add_argument("--lon", type=float, default=82.9739, help="Longitude for weather API")
    parser.add_argument("--horizon", type=float, default=0.0,
                        help="(legacy) Forecast horizon in hours appended to --forecast-min")
    parser.add_argument("--forecast-min", type=str, default=",".join(map(str, settings.DEFAULT_FORECAST_MIN)),
                        help="Comma-separated future timestamps (minutes) to predict flood contours at")
    parser.add_argument("--flood-backend", choices=["auto", "dl", "ca"], default=settings.FLOOD_BACKEND,
                        help="Flood-forecast backend (dl=ConvLSTM when weights exist, ca=physics baseline)")
    parser.add_argument("--package-kg", type=float, default=settings.DEFAULT_PACKAGE_KG,
                        help="Payload weight of each relief package (kg)")
    parser.add_argument("--home-candidates", type=int, default=settings.HOME_CANDIDATES,
                        help="Max home/base candidate pixels to evaluate")
    parser.add_argument("--path-downsample", type=int, default=settings.PATH_DOWNSAMPLE,
                        help="Downsample factor for the D* Lite planning grid")
    parser.add_argument("--dummy-weather", action="store_true", help="Use dummy extreme weather data to force visible flood spread")
    args = parser.parse_args()

    DISPLAY_WIDTH = args.display_width

    # 1. Fetch or use dummy weather data
    if args.dummy_weather:
        dummy_file = "data/input/dummy_weather.json"
        logging.info(f"Using dummy weather data from {dummy_file}...")
        try:
            import json
            with open(dummy_file, "r") as f:
                weather = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load dummy weather: {e}. Falling back to default.")
            weather = {"precipitation": 20.0, "wind_speed_10m": 30.0, "wind_direction_10m": 270.0}
    else:
        logging.info(f"Fetching weather data for lat={args.lat}, lon={args.lon}...")
        weather = get_weather_data(args.lat, args.lon)
    
    logging.info(f"Weather: {weather}")
    
    # Check safety thresholds
    if weather["wind_speed_10m"] > settings.MAX_SAFE_WIND_SPEED:
        logging.warning("HIGH WIND WARNING: Drone operations may be unsafe.")
    if weather["precipitation"] > settings.MAX_SAFE_PRECIPITATION:
        logging.warning("HEAVY RAIN WARNING: Drone operations may be unsafe.")

    logging.info(f"Loading image from '{args.image_path}' ...")
    image = cv2.imread(args.image_path)
    if image is None:
        logging.error("Error loading image file. Please check the path.")
        sys.exit(1)

    h, w = image.shape[:2]
    if w > DISPLAY_WIDTH:
        scale_factor = DISPLAY_WIDTH / float(w)
        new_h = int(round(h * scale_factor))
        image = cv2.resize(image, (DISPLAY_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        logging.info(f"Resized image to {DISPLAY_WIDTH}x{new_h}")

    proc = ImageProcessor()
    try:
        proc.select_sample_points(image)
    except Exception as e:
        logging.error(f"Point selection failed: {e}")
        sys.exit(1)

    mask = proc.mask_flood_areas(image)
    if mask is None:
        logging.error("mask_flood_areas returned None.")
        sys.exit(1)

    if mask.ndim == 3 and mask.shape[2] > 1:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        if mask.dtype == np.bool_:
            mask = (mask.astype("uint8") * 255)
        else:
            m, M = mask.min(), mask.max()
            if M - m > 1e-8:
                mask = ((mask - m) / (M - m) * 255.0).astype("uint8")
            else:
                mask = (mask * 255.0).astype("uint8")

    show_image_safe("Flood Mask", mask, wait=True)

    # 2. Time-indexed flood forecast (Task 1) - each future timestamp gets its
    #    own predicted contour so routing can query the flood at each leg ETA.
    forecast_min = [int(float(t)) for t in args.forecast_min.split(",") if t.strip()]
    if args.horizon and int(args.horizon * 60) not in forecast_min:
        forecast_min.append(int(args.horizon * 60))
    logging.info(f"Predicting flood contours at t = {forecast_min} min (backend={args.flood_backend})...")
    timeline = predict_spread_sequence(mask, weather, forecast_min, backend=args.flood_backend)
    logging.info(f"Flood forecast produced {len(timeline)} frames "
                 f"({timeline.source}); last horizon t={timeline.times_min[-1]:.0f} min")
    pred_mask = timeline.masks[-1]

    contours = proc.find_filtered_contours(mask, min_area=args.min_area)
    if not contours:
        logging.info("No significant flood areas detected.")
        sys.exit(0)

    try:
        cluster_centers, cluster_edges = cluster_contours(contours)
    except Exception as e:
        logging.error(f"cluster_contours failed: {e}")
        sys.exit(1)

    points = to_point_list(cluster_centers)
    if len(points) == 0:
        logging.info("No cluster centers found.")
        sys.exit(0)

    # Safe points from cluster edges (dry land at the flood boundary)
    try:
        safe_points = find_safe_drop_points(cluster_edges, list(range(len(points))), points)
    except Exception as e:
        logging.error(f"find_safe_drop_points failed: {e}")
        safe_points = []

    if not safe_points:
        logging.warning("No safe drop points computed.")
        sys.exit(0)

    # Each safe point is one relief drop carrying args.package_kg.  Contour/hull
    # points can sit ON the flood mask, so first snap each to dry land.
    zones = []
    current_flood = timeline.obstacle_mask_at(0.0)
    for i, s in enumerate(safe_points):
        px = (int(round(s[0])), int(round(s[1])))
        dry_px, dist = snap_to_dry(px, current_flood)
        if dist > 0:
            logging.info(f"Safe point {px} on flood boundary -> snapped to dry "
                         f"pixel {dry_px} (offset {dist} px).")
        zones.append(DropZone(zone_id=f"Z{i}", pixel=dry_px,
                              weight_kg=args.package_kg))
    logging.info(f"{len(zones)} drop zones from safe points (each {args.package_kg} kg package).")

    spec = DroneSpec(
        battery_wh=settings.DRONE_BATTERY_WH,
        hover_endurance_min=settings.DRONE_ENDURANCE_MIN,
        cruise_airspeed_mps=settings.DRONE_CRUISE_MPS,
        payload_capacity_kg=settings.DRONE_PAYLOAD_CAPACITY_KG,
    )
    planner = MissionPlanner(
        spec=spec, weather=weather, timeline=timeline,
        meters_per_pixel=settings.METERS_PER_PIXEL,
        downsample_factor=args.path_downsample,
    )

    # 4. Optimise the home/base location (Task 4) over safe candidate pixels.
    logging.info("Optimising home/base location over safe candidate pixels...")
    home, plan, _ = planner.optimize_home(zones, max_candidates=args.home_candidates)
    if home is None:
        logging.error("Mission infeasible: no safe home candidate can reach any drop point.")
        sys.exit(1)

    logging.info(f"Mission home = {home}: status={plan.status}, "
                 f"{len(plan.served_ids)}/{len(zones)} zones served, "
                 f"{len(plan.sorties)} sortie(s), est. total "
                 f"{plan.total_duration_min:.1f} min, {plan.total_energy_wh:.1f} Wh.")

    for note in plan.notes:
        logging.info(f"[plan] {note}")
    for z, reason in plan.skipped:
        logging.warning(f"Zone {z.zone_id} {z.pixel} NOT served: {reason}")

    if plan.status == "infeasible":
        logging.error("Mission infeasible: no drop point could be reached within "
                      "battery/payload limits. Review map, weights, or drone spec.")
        sys.exit(1)

    # 5. Emit QGC WPL 110 mission file(s) -- one per sortie, same schema as before.
    #    A reload/battery swap requires landing at base, hence a fresh sortie file.
    os.makedirs("data/output", exist_ok=True)
    image_center_px = (image.shape[1] // 2, image.shape[0] // 2)
    geo_center = (args.lat, args.lon)
    try:
        written = []
        if len(plan.sorties) == 1:
            output_file = "data/output/enriched_drone_mission.waypoints"
            s = plan.sorties[0]
            generate_mission_file(
                [(int(x), int(y)) for x, y in s.path], s.drop_indices, home,
                image_center_px, geo_center, settings.METERS_PER_PIXEL,
                filename=output_file)
            written.append(output_file)
        else:
            for s in plan.sorties:
                output_file = f"data/output/sortie_{s.number:02d}.waypoints"
                generate_mission_file(
                    [(int(x), int(y)) for x, y in s.path], s.drop_indices, home,
                    image_center_px, geo_center, settings.METERS_PER_PIXEL,
                    filename=output_file)
                written.append(output_file)
        for f in written:
            logging.info(f"Mission file '{f}' generated successfully "
                         f"(QGC WPL 110).")
        if len(written) > 1:
            logging.info(f"{len(written)} sortie files written (multi-sortie "
                         "mission: reload / battery swap between sorties).")
    except Exception as e:
        logging.error(f"generate_mission_file failed: {e}")

    # Display / save the annotated map for each sortie's route
    try:
        os.makedirs("data/output", exist_ok=True)
        for s in plan.sorties:
            save_map = ("data/output/mission_map.png" if len(plan.sorties) == 1
                        else f"data/output/mission_map_sortie_{s.number:02d}.png")
            out = display_path_on_map(
                image.copy(), contours, pred_mask, list(s.path), s.drop_indices,
                home, tsp_path=list(s.path),
                save_path=save_map,
                title=f"Sortie {s.number} - Drone Path with Safe Drop Zones, "
                      "Obstacles & Home")
            if out:
                logging.info(f"Annotated mission map saved to '{out}'.")
    except Exception as e:
        logging.warning(f"display_path_on_map failed: {e}")

    logging.info("Done.")

if __name__ == "__main__":
    main()
