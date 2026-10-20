"""Single-UAV (legacy) planning mode.

Extracted from the ``main()`` function's single-mode code path.
"""

import logging
import os
import sys

import numpy as np

from src.routing.pathfinding import nearest_neighbor_tsp, compute_full_path
from src.mission.safe_dropzone import find_safe_drop_points
from src.mission.mission_output import generate_mission_file, display_path_on_map
from config import settings

from src.pipeline.common import remove_mission_file


def _main_attr(name, default):
    main_mod = sys.modules.get("main")
    return getattr(main_mod, name, default) if main_mod is not None else default


def run_single_base(image, mask, forecasts, contours, cluster_edges, points,
                    geo_center, scale_factor, combined_obstacle_mask, pred_mask,
                    output_file="data/output/enriched_drone_mission.waypoints",
                    show=True, coordinate_fn=None):
    """Legacy single-UAV planning: one waypoints file to ``output_file``."""

    # Safe drop points
    try:
        safe_points = find_safe_drop_points(cluster_edges, list(range(len(points))), points,
                                            obstacle_mask=combined_obstacle_mask, raw_contours=contours)
    except Exception as e:
        logging.error(f"find_safe_drop_points failed: {e}")
        safe_points = []

    if not safe_points:
        logging.warning("No safe drop points computed.")
        sys.exit(0)

    safe_points_arr = np.array(safe_points)
    center = np.array([image.shape[1] // 2, image.shape[0] // 2])
    distances = np.linalg.norm(safe_points_arr - center, axis=1)
    home_index = int(np.argmin(distances))
    home = tuple(safe_points[home_index])
    logging.info(f"Selected HOME at safe point {home}")

    try:
        path_order, ordered_points = nearest_neighbor_tsp(safe_points, home=home)
    except Exception as e:
        logging.error(f"nearest_neighbor_tsp failed: {e}")
        sys.exit(1)

    logging.info(f"TSP Path order: {path_order}")
    try:
        res = compute_full_path(ordered_points, combined_obstacle_mask)
        full_path, drop_indices = res[0], res[1]
    except Exception as e:
        logging.error(f"compute_full_path failed: {e}")
        sys.exit(1)

    gen_fn = _main_attr("generate_mission_file", generate_mission_file)
    disp_fn = _main_attr("display_path_on_map", display_path_on_map)
    rm_fn = _main_attr("remove_mission_file", remove_mission_file)

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        image_center_px = ((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)
        meters_per_pixel = settings.METERS_PER_PIXEL / scale_factor
        gen_fn(
            full_path, drop_indices, home, image_center_px, geo_center,
            meters_per_pixel, filename=output_file, coordinate_fn=coordinate_fn
        )
        logging.info(f"Mission file '{output_file}' generated successfully.")
    except Exception as e:
        logging.error(f"generate_mission_file failed: {e}")
        rm_fn(output_file)

    try:
        disp_fn(image.copy(), contours, pred_mask, full_path, drop_indices, home=home,
                save_path="data/output/mission_route_map.png", show=show)
    except Exception as e:
        logging.warning(f"display_path_on_map failed: {e}")

    logging.info("Done.")
