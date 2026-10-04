#!/usr/bin/env python3
import sys
import argparse
import logging
import re
from typing import List, Tuple, Any

import cv2
import numpy as np

from src.vision.image_processing import ImageProcessor
from src.vision.clustering import cluster_contours
from src.routing.pathfinding import (nearest_neighbor_tsp, compute_full_path, polyline_flood_pixels,
                                     build_routing_grid, routed_distance_function, polyline_intersections,
                                     grid_distance_field, locate_cell)
from src.mission.safe_dropzone import (find_safe_drop_points,
                                       filter_drops_by_home_connectivity)
from src.mission.mission_output import (generate_mission_file, display_path_on_map,
                                        validate_waypoints_file, basemap_sanity, split_sorties)
from src.weather.weather_api import get_weather_data
from src.weather.flood_spread import predict_spread
from src.mission.constraints import (plan_mission_stops, log_deviation,
                                     wind_kmh_to_mps, sortie_battery_report)
from src.mission.multi_base import select_bases, single_drop_reach
from src.mission.overlap_repair import execute_overlap_repairs
from pathlib import Path
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


def map_routed_drop_distances(route, ordered_drops, leg_distances, home):
    """Associate D* Lite segment lengths with semantic drop IDs.

    ``leg_distances`` is indexed by route waypoint legs, while the resource
    planner processes original TSP drop IDs. Keeping that mapping explicit
    avoids shifting distances after reload or flood-rejected stops.
    """
    mapped = {}
    for leg_index, (a, b) in enumerate(zip(route.points, route.points[1:])):
        if leg_index < len(leg_distances):
            mapped[("pair", tuple(map(int, a)), tuple(map(int, b)))] = float(leg_distances[leg_index])
    waiting = {}
    for drop_id, point in enumerate(ordered_drops):
        waiting.setdefault(tuple(map(int, point)), []).append(drop_id)
    for route_point_index in route.drop_indices:
        point = tuple(map(int, route.points[route_point_index]))
        available = waiting.get(point, [])
        if not available:
            continue
        drop_id = available.pop(0)
        leg_index = route_point_index - 1
        if 0 <= leg_index < len(leg_distances):
            distance = float(leg_distances[leg_index])
            mapped[drop_id] = distance
            if tuple(map(int, route.points[route_point_index - 1])) == tuple(map(int, home)):
                mapped[("home", drop_id)] = distance
    return mapped


def create_session_dir(root="data/output"):
    """New ``<root>/session_<timestamp>`` folder for this run; never reuses an existing one."""
    from datetime import datetime
    base = Path(root) / f"session_{datetime.now():%Y%m%d_%H%M%S}"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}_{n}")
    path.mkdir(parents=True)
    return path


def write_summary(summary, session_dir):
    import json
    try:
        with open(session_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
    except OSError as e:
        logging.warning(f"Could not write run summary: {e}")


def refine_route(plan, routed_distances, max_iterations):
    """Alternate resource planning and D* Lite routing until the stop list settles.

    ``plan(distances)`` returns a PlannedRoute; ``routed_distances(route)``
    returns routed leg distances for it. Routed distances accumulate across
    iterations (keyed by exact leg). Returns ``(route, status, iterations)``,
    ``status`` being ``converged`` (stop list unchanged), ``cycle`` (it repeats
    an earlier one) or ``iteration_cap``.
    """
    route = plan(None)
    history = [tuple(route.points)]
    known = {}
    for iteration in range(1, max_iterations + 1):
        known.update(routed_distances(route))
        refined = plan(dict(known))
        key = tuple(refined.points)
        if key == history[-1]:
            return refined, "converged", iteration
        if key in history:
            return refined, "cycle", iteration
        history.append(key)
        route = refined
    return route, "iteration_cap", max_iterations


def main():
    parser = argparse.ArgumentParser(description="Generate drone mission from flood-map image.")
    parser.add_argument("--image", dest="image_path", help="Input map image path")
    parser.add_argument("--display-width", "-w", type=int, default=750, help="Resize display width")
    parser.add_argument("--min-area", type=int, default=200, help="Minimum contour area to keep")
    parser.add_argument("--lat", type=float, default=None,
                        help="Latitude of the image centre (weather + waypoints); defaults to the image region preset")
    parser.add_argument("--lon", type=float, default=None,
                        help="Longitude of the image centre (weather + waypoints); defaults to the image region preset")
    parser.add_argument("--horizon", type=float, default=2.0, help="Hours for flood spread prediction")
    parser.add_argument("--dummy-weather", action="store_true", help="Use dummy extreme weather data (precip=10mm, wind=30km/h)")
    parser.add_argument("--dummy-weather-moderate", action="store_true", help="Use realistic moderate dummy weather preset (precip=3mm, wind=12km/h)")
    parser.add_argument("--no-gui", action="store_true", help="Run without graphical windows (for automated/headless runs)")
    parser.add_argument("--auto-sample", action="store_true", help="Use default flood sample points without manual clicking")
    parser.add_argument("--sample-points", type=str, default=None, help="Semicolon-separated sample points x,y;x,y")
    args = parser.parse_args()

    DISPLAY_WIDTH = args.display_width

    if not args.image_path:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            args.image_path = filedialog.askopenfilename(title="Select flood map image",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")])
            root.destroy()
        except Exception as exc:
            parser.error(f"--image is required when the file picker is unavailable: {exc}")
    if not args.image_path or not Path(args.image_path).is_file():
        parser.error(f"Image does not exist: {args.image_path}")

    # Geographic anchor for weather and waypoint export: explicit CLI values,
    # else the bundled region preset for this image. Never a silent default city.
    preset = settings.REGION_PRESETS.get(Path(args.image_path).stem.lower())
    if args.lat is None or args.lon is None:
        if preset is None:
            parser.error(f"No --lat/--lon given and no region preset for '{Path(args.image_path).name}'. "
                         "Pass the image-centre latitude/longitude explicitly.")
        args.lat = preset["lat"] if args.lat is None else args.lat
        args.lon = preset["lon"] if args.lon is None else args.lon
        logging.info("Geo anchor (image centre): lat=%.4f lon=%.4f from region preset '%s' (%s).",
                     args.lat, args.lon, Path(args.image_path).stem.lower(), preset["region"])
    else:
        logging.info("Geo anchor (image centre): lat=%.4f lon=%.4f from CLI.", args.lat, args.lon)
    if preset is not None:
        logging.warning("GEOREFERENCE SCALE: waypoints use METERS_PER_PIXEL=%.1f, but this map's estimated "
                        "true scale is ~%.0f m/px; waypoint offsets from the image centre are compressed "
                        "~%.0fx (documented limitation, georeferencing not autonomous).",
                        settings.METERS_PER_PIXEL, preset["approx_m_per_px"],
                        preset["approx_m_per_px"] / settings.METERS_PER_PIXEL)
    summary = {"image": str(args.image_path), "geo_anchor": [args.lat, args.lon]}
    session_dir = create_session_dir()
    summary["session_dir"] = str(session_dir)
    logging.info("SESSION OUTPUT: %s", session_dir)

    # 1. Fetch or use dummy weather data
    if args.dummy_weather_moderate:
        dummy_file = "data/input/dummy_weather_moderate.json"
        logging.info(f"Using moderate dummy weather data from {dummy_file}...")
        try:
            import json
            with open(dummy_file, "r") as f:
                weather = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load moderate dummy weather: {e}. Falling back to default.")
            weather = {"precipitation": 3.0, "wind_speed_10m": 12.0, "wind_direction_10m": 270.0}
    elif args.dummy_weather:
        dummy_file = "data/input/dummy_weather.json"
        logging.info(f"Using dummy weather data from {dummy_file}...")
        try:
            import json
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
    summary["weather"] = dict(weather)

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
    if args.auto_sample or args.no_gui or args.sample_points:
        if args.sample_points:
            pts = []
            for item in args.sample_points.split(";"):
                if item.strip():
                    px, py = map(int, item.strip().split(","))
                    pts.append((px, py))
            proc.sample_points = pts
        else:
            # Sample the dominant saturated overlay colour (the flood annotation).
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

    coverage = np.count_nonzero(mask) / float(mask.size)
    logging.info("MASK SANITY: flood pixels=%d of %d (%.2f%%), allowed %.2f%%-%.0f%%",
                 np.count_nonzero(mask), mask.size, coverage * 100,
                 settings.MASK_MIN_COVERAGE_FRACTION * 100, settings.MASK_MAX_COVERAGE_FRACTION * 100)
    summary["mask_coverage"] = coverage
    if not (settings.MASK_MIN_COVERAGE_FRACTION <= coverage <= settings.MASK_MAX_COVERAGE_FRACTION):
        logging.error("MASK SANITY FAILED: flood mask covers %.2f%% of the image; HSV sampling picked "
                      "non-flood colours. Re-sample flood pixels (--sample-points or interactive).", coverage * 100)
        sys.exit(1)

    if not args.no_gui:
        show_image_safe("Flood Mask", mask, wait=True)

    # 2. Predict flood spread
    logging.info(f"Predicting flood spread for horizon={args.horizon} hours...")
    forecasts = predict_spread(mask, weather, args.horizon)
    if not forecasts:
        logging.error("No future flood forecast frames were generated.")
        sys.exit(1)
    pred_mask = forecasts[max(forecasts)]

    # Combine masks for obstacle avoidance
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

    # Safe points from cluster edges (with multi-point boundary delivery for large contours)
    try:
        # Drops must stay dry for the whole forecast horizon, and must be judged
        # against the same mask the connectivity filter and D* Lite grids use.
        safe_points = find_safe_drop_points(cluster_edges, list(range(len(points))), points,
                                            obstacle_mask=combined_obstacle_mask, raw_contours=contours)
    except Exception as e:
        logging.error(f"find_safe_drop_points failed: {e}")
        safe_points = []

    if not safe_points:
        logging.warning("No safe drop points computed.")
        sys.exit(0)
    logging.info("Safe delivery points: %d of %d detected zones", len(safe_points), len(contours))
    generated_drop_count = len(safe_points)
    zones_with_point = sum(
        1 for c in contours
        if any(cv2.pointPolygonTest(c, (float(x), float(y)), True) >= -25 for x, y in safe_points))
    logging.info("STAGE COUNTS: raw contours=%d -> zones (area >= %d)=%d -> zones with a drop point=%d "
                 "(%d drop points)", len(raw_contours), args.min_area, len(contours), zones_with_point,
                 generated_drop_count)

    # Sample base candidates on a ~40x40 spatial grid of dry pixels.
    stride = max(1, int(max(image.shape[:2]) / 40))
    grid_y, grid_x = np.mgrid[stride // 2:image.shape[0]:stride, stride // 2:image.shape[1]:stride]
    grid_points = list(zip(grid_x.ravel().tolist(), grid_y.ravel().tolist()))
    dry = [(x, y) for x, y in grid_points if combined_obstacle_mask[y, x] == 0]
    # Every base keeps HOME_CLEARANCE_PX from flood edges (current and forecast).
    clearance = cv2.distanceTransform((combined_obstacle_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    candidates = [(x, y) for x, y in dry if clearance[y, x] >= settings.HOME_CLEARANCE_PX]
    logging.info("Base candidates with >= %.0f px flood clearance: %d of %d dry grid points",
                 settings.HOME_CLEARANCE_PX, len(candidates), len(dry))
    summary["home_candidates"] = {"grid_points": len(grid_points), "dry": len(dry),
                                  "with_clearance": len(candidates)}
    if not candidates:
        summary["home_outcome"] = "NO_CANDIDATES_SAMPLED"
        logging.error("BASE SELECTION: NO_CANDIDATES_SAMPLED - no dry grid point with %.0f px flood clearance "
                      "(grid_points=%d dry=%d).", settings.HOME_CLEARANCE_PX, len(grid_points), len(dry))
        log_deviation("No safe base candidate; mission infeasible.")
        write_summary(summary, session_dir)
        sys.exit(2)

    # Reach and cost use routed distances around flood zones on the D* Lite grid.
    headwind = wind_kmh_to_mps(weather.get("wind_speed_10m", 0.0))
    routing_grid = build_routing_grid(combined_obstacle_mask, 5)
    routed_distance = routed_distance_function(routing_grid, safe_points, settings.METERS_PER_PIXEL)
    selection = select_bases(
        candidates, safe_points, routed_distance, combined_obstacle_mask,
        headwind_mps=headwind, wind_from_deg=weather.get("wind_direction_10m"),
        per_drop_kg=settings.PAYLOAD_PER_DROP_KG, payload_capacity_kg=settings.PAYLOAD_CAPACITY_KG,
        reserve=settings.BATTERY_RESERVE_FRACTION, meters_per_pixel=settings.METERS_PER_PIXEL,
        max_bases=settings.MAX_BASES, min_separation_px=settings.MIN_BASE_SEPARATION_PX)
    bases = selection["bases"]
    if not bases:
        summary["home_outcome"] = "NO_BASE_REACHES_ANY_DROP"
        logging.error("BASE SELECTION: no flood-clear base site can serve any drop point within battery range.")
        log_deviation("Mission infeasible: no base can reach any drop point.")
        write_summary(summary, session_dir)
        sys.exit(2)
    summary["home_outcome"] = "SINGLE_BASE" if len(bases) == 1 else f"MULTI_BASE_{len(bases)}"
    logging.info("BASE SELECTION: %s - %d base(s) at %s; best single site reached %d of %d coverable drop "
                 "points.", summary["home_outcome"], len(bases), bases, selection["best_single_reach"],
                 selection["coverable"])

    # Zone-to-base assignment table (each drop point to exactly one base, or reported unreachable).
    logging.info("ASSIGNMENT TABLE (drop -> base, routed distance from base):")
    for index, point in enumerate(safe_points):
        owner = selection["assignment"].get(index)
        if owner is None:
            logging.warning("  drop %2d %s -> UNREACHABLE BY ANY BASE (%s)", index, tuple(map(int, point)),
                            selection["unreachable"][index])
        else:
            logging.info("  drop %2d %s -> BASE %d at %s, %.0f m", index, tuple(map(int, point)), owner + 1,
                         bases[owner], routed_distance(bases[owner], point))
    separations = [float(np.hypot(a[0] - b[0], a[1] - b[1]))
                   for i, a in enumerate(bases) for b in bases[i + 1:]]
    if separations:
        logging.info("BASE SEPARATION: min %.0f px (required >= %.0f px)", min(separations),
                     settings.MIN_BASE_SEPARATION_PX)

    def plan_base(number, home, drops):
        """Existing single-drone pipeline for one base: connectivity, TSP, time-aware
        D* Lite, battery/payload sortie splitting with reloads, waypoint export."""
        tag = f"BASE {number}"
        result = {"base": number, "home": tuple(map(int, home)), "assigned": len(drops),
                  "clearance_px": float(clearance[int(home[1]), int(home[0])])}
        logging.info("%s at %s (flood clearance %.1f px): planning %d assigned drop points.",
                     tag, result["home"], result["clearance_px"], len(drops))
        drops, excluded = filter_drops_by_home_connectivity(drops, home, combined_obstacle_mask,
                                                            downsample_factor=5)
        result["connectivity_excluded"] = len(excluded)
        logging.info("%s drop connectivity filter: reachable=%d excluded=%d", tag, len(drops), len(excluded))
        if not drops:
            logging.error("%s: no reachable drop points after connectivity filtering.", tag)
            return None
        path_order, _ = nearest_neighbor_tsp(drops, home=home)
        ordered_drops = [drops[i] for i in path_order]

        def plan(routed):
            return plan_mission_stops(home, ordered_drops, forecasts, mask,
                meters_per_pixel=settings.METERS_PER_PIXEL, headwind_mps=headwind,
                initial_battery=1.0, routed_leg_distances_m=routed)

        def routed_distances(candidate_route):
            _, _, leg_distances = compute_full_path(candidate_route.points, combined_obstacle_mask,
                forecast_masks=forecasts, meters_per_pixel=settings.METERS_PER_PIXEL)
            return map_routed_drop_distances(candidate_route, ordered_drops, leg_distances, home)

        max_iterations = min(6, len(ordered_drops) + 2)
        route, refine_status, iterations = refine_route(plan, routed_distances, max_iterations)
        result["refinement"] = {"status": refine_status, "iterations": iterations, "cap": max_iterations}
        if refine_status == "converged":
            logging.info("%s ROUTE REFINEMENT: CONVERGED after %d iteration(s).", tag, iterations)
        else:
            logging.warning("%s ROUTE REFINEMENT: %s after %d iteration(s) - NOT converged; battery is "
                            "re-verified on the routed geometry below.", tag,
                            "CYCLE DETECTED" if refine_status == "cycle" else "ITERATION CAP", iterations)
        if not route.drop_indices:
            logging.error("%s: no reachable drop points after resource planning.", tag)
            return None
        if route.unreachable:
            logging.warning("%s unreachable drop indices in TSP order: %s", tag, route.unreachable)

        leg_report = []
        full_path, stop_indices, _ = compute_full_path(route.points, combined_obstacle_mask,
            forecast_masks=forecasts, meters_per_pixel=settings.METERS_PER_PIXEL, leg_report=leg_report)
        drop_indices = sorted(stop_indices[i] for i in route.drop_indices if i < len(stop_indices))
        reload_indices = sorted(stop_indices[i] for i in route.reload_indices if i < len(stop_indices))

        statuses = [leg["status"] for leg in leg_report]
        fallbacks = [leg for leg in leg_report if leg["status"] == "fallback"]
        arrival_crossings = sum(leg["route_flooded_px"] for leg in leg_report)
        current_crossings = polyline_flood_pixels(full_path, mask)
        logging.info("%s D* LITE ROUTING: legs=%d dstar=%d same_cell=%d fallback=%d; route pixels inside the "
                     "arrival-time flood mask=%d, inside the detected (t=0) flood mask=%d", tag,
                     len(leg_report), statuses.count("dstar"), statuses.count("same_cell"), len(fallbacks),
                     arrival_crossings, current_crossings)
        for leg in fallbacks:
            logging.warning("%s FALLBACK LEG %d %s -> %s: %s (crosses %d flooded px)", tag,
                            leg["leg"], leg["start"], leg["goal"], leg["reason"], leg["flooded_px"])
        sorties = sortie_battery_report(full_path, drop_indices, reload_indices,
                                        meters_per_pixel=settings.METERS_PER_PIXEL, headwind_mps=headwind)
        limit = 1.0 - settings.BATTERY_RESERVE_FRACTION
        for k, sortie in enumerate(sorties, 1):
            logging.info("%s ROUTED SORTIE %d: drops=%d distance=%.0fm battery=%.1f%% (limit %.0f%%)%s", tag, k,
                         sortie["drops"], sortie["distance_m"], sortie["battery_fraction"] * 100, limit * 100,
                         "  ** EXCEEDS LIMIT **" if sortie["battery_fraction"] > limit + 1e-9 else "")

        waypoint_file = session_dir / f"base{number}.waypoints"
        try:
            generate_mission_file(full_path, drop_indices, home,
                                  (image.shape[1] // 2, image.shape[0] // 2), (args.lat, args.lon),
                                  settings.METERS_PER_PIXEL, filename=str(waypoint_file),
                                  reload_indices=reload_indices)
            problems = validate_waypoints_file(str(waypoint_file))
        except Exception as e:
            problems = [f"generation failed: {e}"]
        if problems:
            logging.error("%s WAYPOINT VALIDATION FAILED (%d problems): %s", tag, len(problems), problems[:5])
        else:
            logging.info("%s WAYPOINT VALIDATION: valid QGC WPL 110 (%s)", tag, waypoint_file)
        result.update({
            "full_path": full_path, "drop_indices": drop_indices, "reload_indices": reload_indices,
            "stop_indices": stop_indices, "route_points": list(route.points),
            "visited": len(drop_indices), "unreachable": len(route.unreachable),
            "legs": {"total": len(leg_report), "dstar": statuses.count("dstar"),
                     "same_cell": statuses.count("same_cell"), "fallback": len(fallbacks)},
            "route_flooded_px_arrival": arrival_crossings, "route_flooded_px_current": current_crossings,
            "planner_sortie_battery": route.sortie_battery,
            "routed_sortie_battery": [s["battery_fraction"] for s in sorties],
            "waypoint_file": str(waypoint_file), "waypoint_problems": problems})
        return result

    def plan_assigned(b):
        assigned = [safe_points[i] for i, owner in sorted(selection["assignment"].items()) if owner == b]
        if not assigned:
            return None
        try:
            return plan_base(b + 1, bases[b], assigned)
        except Exception as e:
            logging.error("BASE %d planning failed: %s", b + 1, e)
            return None

    plans_by_base = {b: plan_assigned(b) for b in range(len(bases))}

    # Post-routing check: independently planned bases must not share airspace.
    # Airspace deconfliction repair pipeline:
    # 1. Reassignment: reassign the crossing sortie to the other base if within reach.
    # 2. Tier 1: locally reroute conflicting legs with D* Lite using crossing paths as soft obstacles.
    # 3. Tier 2: nudge conflicting base position within safe clearance and separation.
    # 4. Hard error (sys.exit(2)): block mission export if any overlap remains.
    coord_index = {tuple(map(int, p)): i for i, p in enumerate(safe_points)}
    base_reach = single_drop_reach(bases, safe_points, routed_distance, headwind_mps=headwind,
                                   per_drop_kg=settings.PAYLOAD_PER_DROP_KG,
                                   reserve=settings.BATTERY_RESERVE_FRACTION)

    repairs, remaining_crossings = execute_overlap_repairs(
        plans_by_base, bases, selection, safe_points, base_reach, coord_index,
        combined_obstacle_mask, forecasts, clearance, routed_distance, headwind,
        plan_base, plan_assigned, session_dir, image.shape[:2], (args.lat, args.lon),
        max_rounds=5
    )

    plans = [plans_by_base[b] for b in sorted(plans_by_base) if plans_by_base[b]]
    if not plans:
        logging.error("mission infeasible - no base produced a route")
        write_summary(summary, session_dir)
        sys.exit(2)

    overlaps = []
    for i, first in enumerate(plans):
        for second in plans[i + 1:]:
            hits = polyline_intersections(first["full_path"], second["full_path"])
            if hits:
                overlaps.append({"bases": (first["base"], second["base"]), "points": len(hits)})
                logging.error("PATH OVERLAP: BASE %d and BASE %d routes intersect at %d point(s), e.g. %s",
                              first["base"], second["base"], len(hits), hits[:3])

    if overlaps:
        summary["path_overlaps"] = overlaps
        summary["overlap_repairs"] = repairs
        logging.error("MISSION INFEASIBLE: Unresolved path overlap between bases (%d conflict(s)). "
                      "Airspace deconfliction failed all repair tiers.", len(overlaps))
        log_deviation("Mission infeasible: unresolved path overlap between bases.")
        write_summary(summary, session_dir)
        sys.exit(2)

    if len(plans) > 1 and not overlaps:
        logging.info("PATH OVERLAP CHECK: no intersections between any two bases' routes (%d bases).", len(plans))

    # Territories derived from the final assignment: every grid cell belongs to the base whose
    # service points (the base and its visited drops) are routed-closest. Non-overlapping by
    # construction; each visited drop must fall inside its own base's territory.
    territory_labels = None
    territory_violations = []
    if len(plans) > 1:
        fields = np.stack([grid_distance_field(routing_grid, [locate_cell(q, routing_grid)[0] for q in
                                                              [p["home"]] + [p["full_path"][d] for d in p["drop_indices"]]])
                           for p in plans])
        labels = np.where(np.isfinite(fields).any(axis=0), np.argmin(fields, axis=0), -1).astype(np.int32)
        territory_labels = cv2.resize(labels, (routing_grid.width * routing_grid.factor,
                                               routing_grid.height * routing_grid.factor),
                                      interpolation=cv2.INTER_NEAREST)
        territory_labels = cv2.copyMakeBorder(
            territory_labels, 0, mask.shape[0] - territory_labels.shape[0], 0,
            mask.shape[1] - territory_labels.shape[1], cv2.BORDER_REPLICATE)
        for k, plan in enumerate(plans):
            for d in plan["drop_indices"]:
                cx, cy = locate_cell(plan["full_path"][d], routing_grid)[0]
                if labels[cy, cx] != k:
                    territory_violations.append({"base": plan["base"], "drop": tuple(map(int, plan["full_path"][d]))})
        if territory_violations:
            logging.warning("TERRITORY CHECK: %d visited drop(s) lie outside their base's territory: %s",
                            len(territory_violations), territory_violations[:5])
        else:
            logging.info("TERRITORY CHECK: every visited drop lies inside its own base's territory "
                         "(routed-distance partition of base service points, non-overlapping by construction).")

    visited = sum(p["visited"] for p in plans)
    unreachable_any = len(selection["unreachable"])
    logging.info("MISSION COVERAGE: drop points=%d assigned=%d unreachable_by_any_base=%d visited=%d "
                 "(bases=%d)", len(safe_points), len(selection["assignment"]), unreachable_any, visited, len(plans))
    summary.update({
        "bases": [{k: v for k, v in p.items() if k not in ("full_path", "drop_indices", "reload_indices")}
                  for p in plans],
        "base_separation_min_px": min(separations) if separations else None,
        "base_separation_required_px": settings.MIN_BASE_SEPARATION_PX,
        "path_overlaps": overlaps, "overlap_repairs": repairs,
        "territory_violations": territory_violations,
        "assignment": {str(i): b + 1 for i, b in selection["assignment"].items()},
        "unreachable_by_any_base": {str(i): reason for i, reason in selection["unreachable"].items()},
        "home": list(plans[0]["home"]), "home_clearance_px": min(p["clearance_px"] for p in plans),
        "drops_visited": visited, "reloads": sum(len(p["reload_indices"]) for p in plans),
        "unreachable": sum(p["unreachable"] for p in plans),
        "connectivity_excluded": sum(p["connectivity_excluded"] for p in plans),
        "legs": {key: sum(p["legs"][key] for p in plans) for key in ("total", "dstar", "same_cell", "fallback")},
        "route_flooded_px_arrival": sum(p["route_flooded_px_arrival"] for p in plans),
        "route_flooded_px_current": sum(p["route_flooded_px_current"] for p in plans),
        "planner_sortie_battery": [x for p in plans for x in p["planner_sortie_battery"]],
        "routed_sortie_battery": [x for p in plans for x in p["routed_sortie_battery"]],
        "waypoint_problems": [f"base{p['base']}: {msg}" for p in plans for msg in p["waypoint_problems"]],
        "refinement": {"status": "converged" if all(p["refinement"]["status"] == "converged" for p in plans)
                       else "not_converged", "per_base": [p["refinement"] for p in plans]},
        "stage_counts": {"raw_contours": len(raw_contours), "zones": len(contours),
                         "zones_with_drop_point": zones_with_point, "drop_points": generated_drop_count,
                         "unreachable_by_any_base": unreachable_any,
                         "assigned": len(selection["assignment"]), "visited": visited},
    })

    # Combined map: all bases and sorties
    try:
        rendered = display_path_on_map(
            image.copy(), contours, pred_mask, plans[0]["full_path"], plans[0]["drop_indices"],
            plans[0]["home"], reload_indices=plans[0]["reload_indices"],
            save_path=str(session_dir / "mission_route_map.png"), show=(not args.no_gui),
            bases=[{"home": p["home"], "full_path": p["full_path"], "drop_indices": p["drop_indices"],
                    "reload_indices": p["reload_indices"]} for p in plans],
            territories=territory_labels)
        map_ok, map_details = basemap_sanity(rendered, image)
        summary["map"] = {"ok": map_ok, **map_details}
        if map_ok:
            logging.info("MAP SANITY: basemap intact (dominant colour %.1f%%, correlation with source %.3f)",
                         map_details["dominant_fraction"] * 100, map_details["correlation"])
        else:
            logging.error("MAP SANITY FAILED: %s", map_details)
    except Exception as e:
        summary["map"] = {"ok": False, "error": str(e)}
        logging.warning(f"display_path_on_map failed: {e}")

    write_summary(summary, session_dir)
    logging.info("Done. Session output: %s", session_dir)


if __name__ == "__main__":
    main()
