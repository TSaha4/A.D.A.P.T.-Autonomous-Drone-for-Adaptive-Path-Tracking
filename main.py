#!/usr/bin/env python3
import sys
import argparse
import logging
from typing import List, Tuple, Any

import cv2
import numpy as np

from src.vision.image_processing import ImageProcessor
from src.vision.clustering import cluster_contours
from src.routing.pathfinding import nearest_neighbor_tsp, compute_full_path
from src.mission.safe_dropzone import find_safe_drop_points
from src.mission.mission_output import generate_mission_file, display_path_on_map
from src.weather.weather_api import get_weather_data
from src.weather.flood_spread import predict_spread
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


def main():
    parser = argparse.ArgumentParser(description="Generate drone mission from flood-map image.")
    parser.add_argument("image_path", nargs="?", default="data/input/varanasi.png", help="Input map image path")
    parser.add_argument("--display-width", "-w", type=int, default=750, help="Resize display width")
    parser.add_argument("--min-area", type=int, default=200, help="Minimum contour area to keep")
    parser.add_argument("--lat", type=float, default=25.3176, help="Latitude for weather API")
    parser.add_argument("--lon", type=float, default=82.9739, help="Longitude for weather API")
    parser.add_argument("--horizon", type=float, default=2.0, help="Hours for flood spread prediction")
    args = parser.parse_args()

    DISPLAY_WIDTH = args.display_width

    # 1. Fetch weather data
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

    # 2. Predict flood spread
    logging.info(f"Predicting flood spread for horizon={args.horizon} hours...")
    pred_mask = predict_spread(mask, weather, args.horizon)
    
    # Combine masks for obstacle avoidance
    combined_obstacle_mask = cv2.bitwise_or(mask, pred_mask)

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

    # Safe points from cluster edges
    try:
        safe_points = find_safe_drop_points(cluster_edges, list(range(len(points))), points)
    except Exception as e:
        logging.error(f"find_safe_drop_points failed: {e}")
        safe_points = []

    if not safe_points:
        logging.warning("No safe drop points computed.")
        sys.exit(0)

    # Select HOME from safe points closest to map center
    safe_points_arr = np.array(safe_points)
    center = np.array([image.shape[1] // 2, image.shape[0] // 2])
    distances = np.linalg.norm(safe_points_arr - center, axis=1)
    home_index = int(np.argmin(distances))
    home = tuple(safe_points[home_index])
    logging.info(f"Selected HOME at safe point {home}")

    # Compute path with nearest-neighbor TSP
    try:
        path_order, ordered_points = nearest_neighbor_tsp(safe_points, home=home)
    except Exception as e:
        logging.error(f"nearest_neighbor_tsp failed: {e}")
        sys.exit(1)

    logging.info(f"TSP Path order: {path_order}")
    
    # 3. Compute D* Lite full path avoiding obstacles
    try:
        logging.info("Computing full obstacle-avoiding path...")
        full_path, drop_indices = compute_full_path(ordered_points, combined_obstacle_mask)
    except Exception as e:
        logging.error(f"compute_full_path failed: {e}")
        sys.exit(1)

    # Generate mission file with takeoff from HOME, waypoints, drops, and landing at HOME
    try:
        output_file = "data/output/enriched_drone_mission.waypoints"
        generate_mission_file(full_path, drop_indices, home, output_file)
        logging.info(f"Mission file '{output_file}' generated successfully.")
    except Exception as e:
        logging.error(f"generate_mission_file failed: {e}")

    # Display path and HOME
    try:
        display_path_on_map(image.copy(), contours, pred_mask, full_path, drop_indices, home)
    except Exception as e:
        logging.warning(f"display_path_on_map failed: {e}")

    logging.info("Done.")

if __name__ == "__main__":
    main()
