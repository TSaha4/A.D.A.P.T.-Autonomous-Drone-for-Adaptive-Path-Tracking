#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import math
import json
from pathlib import Path

import cv2
import numpy as np

from src.vision.image_processing import ImageProcessor
from src.vision.clustering import cluster_contours
from src.weather.weather_api import get_weather_data, sanitize_weather
from src.weather.flood_spread import predict_spread, forecast_frames
from src.mission.constraints import plan_mission_stops
from src.mission.coordinates import pixel_to_latlon
from config import settings

# ---------------------------------------------------------------------------
# Re-export every symbol that tests import from ``main``.
# The implementations now live in src.pipeline.common / multi_base_planner /
# single_base_planner, but backward compatibility is non-negotiable.
# ---------------------------------------------------------------------------
from src.pipeline.common import (                    # noqa: F401
    drop_index_by_point,
    path_length_px,
    show_image_safe,
    to_point_list,
    map_routed_drop_distances,
    create_session_dir,
    write_summary,
    refine_route,
    remove_mission_file,
)


from src.pipeline.multi_base_planner import run_multi_base   # noqa: F401
from src.pipeline.single_base_planner import run_single_base  # noqa: F401

# Re-export symbols that tests mock on ``main_module`` via patch.object.
from src.mission.mission_output import (                      # noqa: F401
    generate_mission_file, display_path_on_map,
)
from src.routing.pathfinding import polyline_intersections     # noqa: F401

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Generate drone mission from flood-map image.")
    parser.add_argument("image_positional", nargs="?", default=None, help="Input map image path")
    parser.add_argument("--image", dest="image_path", default=None, help="Input map image path")
    parser.add_argument("--display-width", "-w", type=int, default=750, help="Resize display width")
    parser.add_argument("--min-area", type=int, default=200, help="Minimum contour area to keep")
    parser.add_argument("--max-mask-fraction", type=float, default=None,
                        help="Reject an implausibly broad flood mask before planning (for example 0.8)")
    parser.add_argument("--lat", type=float, default=None,
                        help="Latitude of the image centre (weather + waypoints); defaults to the image region preset")
    parser.add_argument("--lon", type=float, default=None,
                        help="Longitude of the image centre (weather + waypoints); defaults to the image region preset")
    parser.add_argument("--horizon", type=float, default=2.0, help="Hours for flood spread prediction")
    parser.add_argument("--dummy-weather", action="store_true", help="Use dummy extreme weather data (precip=10mm, wind=30km/h)")
    parser.add_argument("--dummy-weather-moderate", action="store_true", help="Use realistic moderate dummy weather preset (precip=3mm, wind=12km/h)")
    parser.add_argument("--mode", choices=("single", "multi"), default=None,
                        help="single: one UAV; multi: base selection and one mission per base")
    parser.add_argument("--max-bases", type=int, default=None,
                        help="--mode multi only: maximum number of bases/UAVs (default MAX_BASES from settings)")
    parser.add_argument("--session-root", default=None,
                        help="--mode multi only: folder in which the session folder is created (default data/output)")
    parser.add_argument("--no-gui", action="store_true", help="Run without graphical windows (for automated/headless runs)")
    parser.add_argument("--auto-sample", action="store_true", help="Use default flood sample points without manual clicking")
    parser.add_argument("--sample-points", type=str, default=None, help="Semicolon-separated sample points x,y;x,y")
    parser.add_argument("--geotiff-reference", help="GeoTIFF with the same original pixel dimensions as the input image")
    args = parser.parse_args()

    image_path = args.image_path or args.image_positional

    # Automatic mode resolution: always default to multi-base planning unless
    # the user explicitly passed --mode single. Multi-base planning places bases
    # across the map to cover all reachable flood zones; single-base is only
    # useful for constrained testing or compatibility.
    if args.mode is None:
        args.mode = "multi"

    if args.mode == "single" and (args.max_bases is not None or args.session_root is not None):
        parser.error("--max-bases and --session-root apply only to --mode multi")
    max_bases = settings.MAX_BASES if args.max_bases is None else args.max_bases
    if args.mode == "multi" and max_bases < 1:
        parser.error("the maximum number of bases (--max-bases / MAX_BASES) must be at least 1")
    if not math.isfinite(args.horizon):
        parser.error("--horizon must be a finite number of hours")
    if args.max_mask_fraction is not None and not 0 < args.max_mask_fraction < 1:
        parser.error("--max-mask-fraction must be between 0 and 1")

    session_root = args.session_root or "data/output"
    if args.mode == "multi" and os.path.exists(session_root) and not os.path.isdir(session_root):
        parser.error(f"--session-root {session_root!r} exists and is not a folder")

    DISPLAY_WIDTH = args.display_width
    if DISPLAY_WIDTH <= 0:
        parser.error(f"--display-width must be a positive integer, got {DISPLAY_WIDTH}")

    if not image_path:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            image_path = filedialog.askopenfilename(title="Select flood map image",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")])
            root.destroy()
        except Exception as exc:
            parser.error(f"--image is required when the file picker is unavailable: {exc}")
    if not image_path:
        parser.error("--image is required when the file picker is unavailable")

    # Geographic anchor
    preset = settings.REGION_PRESETS.get(Path(image_path).stem.lower())
    if args.lat is None or args.lon is None:
        if preset is None:
            args.lat = 25.3176 if args.lat is None else args.lat
            args.lon = 82.9739 if args.lon is None else args.lon
        else:
            args.lat = preset["lat"] if args.lat is None else args.lat
            args.lon = preset["lon"] if args.lon is None else args.lon
            logging.info("Geo anchor (image centre): lat=%.4f lon=%.4f from region preset '%s' (%s).",
                         args.lat, args.lon, Path(image_path).stem.lower(), preset["region"])
    else:
        logging.info("Geo anchor (image centre): lat=%.4f lon=%.4f from CLI.", args.lat, args.lon)

    # Reject unusable geographic reference or scale BEFORE any work is done (or previous mission touched)
    try:
        pixel_to_latlon(0, 0, (0, 0), (args.lat, args.lon), settings.METERS_PER_PIXEL)
    except ValueError as e:
        logging.error(f"Invalid --lat/--lon or METERS_PER_PIXEL: {e}")
        sys.exit(2)

    if preset is not None:
        logging.warning("GEOREFERENCE SCALE: waypoints use METERS_PER_PIXEL=%.1f, but this map's estimated "
                        "true scale is ~%.0f m/px; waypoint offsets from the image centre are compressed "
                        "~%.0fx (documented limitation, georeferencing not autonomous).",
                        settings.METERS_PER_PIXEL, preset["approx_m_per_px"],
                        preset["approx_m_per_px"] / settings.METERS_PER_PIXEL)

    # 1. Fetch or use dummy weather data
    if args.dummy_weather_moderate:
        dummy_file = "data/input/dummy_weather_moderate.json"
        logging.info(f"Using moderate dummy weather data from {dummy_file}...")
        try:
            with open(dummy_file, "r") as f:
                weather = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load moderate dummy weather: {e}. Falling back to default.")
            weather = {"precipitation": 3.0, "wind_speed_10m": 12.0, "wind_direction_10m": 270.0}
    elif args.dummy_weather:
        dummy_file = "data/input/dummy_weather.json"
        logging.info(f"Using dummy weather data from {dummy_file}...")
        try:
            with open(dummy_file, "r") as f:
                weather = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load dummy weather: {e}. Falling back to default.")
            weather = {"precipitation": 10.0, "wind_speed_10m": 30.0, "wind_direction_10m": 270.0}
    else:
        logging.info(f"Fetching weather data for lat={args.lat}, lon={args.lon}...")
        weather = get_weather_data(args.lat, args.lon)
        if weather.get("status") == "fallback":
            logging.warning("LIVE WEATHER UNAVAILABLE: planning with fallback calm weather (0 mm/h, 0 km/h); "
                            "battery drain and flood growth are under-estimated if conditions are worse.")
    weather = sanitize_weather(weather, "Dummy weather file" if (args.dummy_weather or args.dummy_weather_moderate) else "Weather API")

    logging.info(f"Weather: {weather}")
    
    # Check safety thresholds
    if weather["wind_speed_10m"] > settings.MAX_SAFE_WIND_SPEED:
        logging.warning("HIGH WIND WARNING: Drone operations may be unsafe.")
    if weather["precipitation"] > settings.MAX_SAFE_PRECIPITATION:
        logging.warning("HEAVY RAIN WARNING: Drone operations may be unsafe.")

    logging.info(f"Loading image from '{image_path}' ...")
    image = cv2.imread(image_path)
    if image is None and os.path.isfile(image_path):
        try:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            image = None
    if image is None:
        logging.error("Error loading image file. Please check the path.")
        sys.exit(1)

    orig_h, orig_w = image.shape[:2]
    scale_factor = 1.0
    if orig_w > DISPLAY_WIDTH:
        scale_factor = DISPLAY_WIDTH / float(orig_w)
        new_h = int(round(orig_h * scale_factor))
        image = cv2.resize(image, (DISPLAY_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    output_file = os.path.join("data", "output", "enriched_drone_mission.waypoints")
    if args.mode == "single" and os.path.lexists(output_file):
        if not remove_mission_file(output_file):
            logging.error("Refusing to continue: the earlier mission could be mistaken for this run's output.")
            sys.exit(1)
        logging.info(f"Removed mission file from a previous run: '{output_file}'")

    # Optional GeoTIFF coordinate reference
    coordinate_fn = None
    if args.geotiff_reference:
        from src.mission.geotiff import GeoTiffReference
        try:
            geo_reference = GeoTiffReference.from_file(args.geotiff_reference)
            if geo_reference.width != orig_w or geo_reference.height != orig_h:
                parser.error("--geotiff-reference dimensions must match the original input image")
            coordinate_fn = lambda x, y: geo_reference.display_to_latlon(x, y, scale_factor)
            args.lat, args.lon = geo_reference.centre
            settings.METERS_PER_PIXEL = geo_reference.local_mpp
            logging.info("Using GeoTIFF reference: %s (CRS: %s)", args.geotiff_reference, geo_reference.crs)
        except Exception as e:
            parser.error(f"invalid --geotiff-reference: {e}")

    # Process image with ImageProcessor
    proc = ImageProcessor()
    if args.auto_sample or args.no_gui or args.sample_points:
        if args.sample_points:
            pts = []
            for item in args.sample_points.split(";"):
                if item.strip():
                    px, py = map(int, item.strip().split(","))
                    pts.append((px, py))
            proc.sample_points = pts
        else:
            try:
                proc.sample_points = proc.auto_sample_points(image)
            except ValueError as e:
                logging.error(f"Automatic flood sampling failed: {e}")
                sys.exit(1)
            logging.info("Auto flood sample points (%d): %s ...", len(proc.sample_points), proc.sample_points[:5])
        proc.image = image.copy()
        proc.compute_dynamic_hsv()
    else:
        try:
            proc.select_sample_points(image)
        except Exception as e:
            logging.error(f"Point selection failed: {e}")
            sys.exit(1)

    mask = proc.mask_flood_areas(image)

    if mask is None:
        logging.error("Failed to create flood mask.")
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

    # Mask sanity guard
    coverage = float(np.count_nonzero(mask) / mask.size)
    if args.max_mask_fraction is not None and coverage > args.max_mask_fraction:
        logging.error("Flood mask covers %.1f%% of the image, above the %.1f%% review gate; no mission exported.",
                      100 * coverage, 100 * args.max_mask_fraction)
        sys.exit(2)

    # Handle empty mask (no flood) as a valid case
    if coverage == 0.0:
        logging.warning("NO FLOOD DETECTED - nothing planned")
        # No session folder is created; exit cleanly with code 0.
        sys.exit(0)

    if not (settings.MASK_MIN_COVERAGE_FRACTION <= coverage <= settings.MASK_MAX_COVERAGE_FRACTION):
        logging.error("MASK SANITY FAILED: flood mask covers %.2f%% of the image (bounds: %.2f%% - %.1f%%). Aborting run; check HSV sampling.",
                      coverage * 100,
                      settings.MASK_MIN_COVERAGE_FRACTION * 100, settings.MASK_MAX_COVERAGE_FRACTION * 100)
        sys.exit(2)
    logging.info("MASK SANITY: flood mask covers %.2f%% of the image (within sane range %.2f%% - %.1f%%).",
                 coverage * 100, settings.MASK_MIN_COVERAGE_FRACTION * 100,
                 settings.MASK_MAX_COVERAGE_FRACTION * 100)

    if not args.no_gui:
        show_image_safe("Flood Mask", mask, wait=True)

    # 3. Predict flood spread
    logging.info(f"Predicting flood spread over {args.horizon} hours...")
    forecasts = predict_spread(mask, weather, args.horizon)
    if not forecasts:
        logging.error("No future flood forecast frames were generated.")
        sys.exit(1)
    pred_mask = forecasts[max(forecasts)]
    combined_obstacle_mask = cv2.bitwise_or(mask, pred_mask)

    raw_contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = proc.find_filtered_contours(mask, min_area=args.min_area)
    logging.info("Drop zones detected: %d (of %d raw contours)", len(contours), len(raw_contours))
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

    # Dispatch to multi-base or single mode
    if args.mode == "multi":
        run_multi_base(image, mask, forecasts, contours, cluster_edges, points, weather, (args.lat, args.lon),
                       scale_factor, max_bases, session_root=session_root, show=(not args.no_gui),
                       coordinate_fn=coordinate_fn, raw_contours=raw_contours)
        return

    # Single-UAV mode
    run_single_base(image, mask, forecasts, contours, cluster_edges, points,
                    (args.lat, args.lon), scale_factor, combined_obstacle_mask, pred_mask,
                    output_file=output_file, show=(not args.no_gui), coordinate_fn=coordinate_fn)


if __name__ == "__main__":
    main()
