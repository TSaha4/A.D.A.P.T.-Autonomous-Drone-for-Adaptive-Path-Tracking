"""Shared pipeline helpers.

Every function that tests or other modules import from ``main`` lives here
so that ``main.py`` can delegate without duplicating code.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Drop-point indexing
# ---------------------------------------------------------------------------

def drop_index_by_point(points):
    """{(x, y): [drop indices]}. Two adjacent regions can choose the same dry pixel, so a point can stand for
    several drops; the repair step must move all of them together."""
    index = {}
    for i, p in enumerate(points):
        index.setdefault(tuple(map(int, p)), []).append(i)
    return index


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def path_length_px(path):
    return float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])))


# ---------------------------------------------------------------------------
# GUI helpers
# ---------------------------------------------------------------------------

def show_image_safe(title: str, img: np.ndarray, wait: bool = True) -> None:
    """Safe cv2.imshow wrapper."""
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


# ---------------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------------

def to_point_list(cluster_centers: Any) -> List[Tuple[int, int]]:
    """Convert cluster_centers to list of (x,y) int tuples."""
    if cluster_centers is None:
        return []
    arr = np.asarray(cluster_centers)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("cluster_centers must be convertible to Nx2 array")
    pts = [(int(round(float(x))), int(round(float(y)))) for x, y in arr[:, :2]]
    return pts


# ---------------------------------------------------------------------------
# Routed-distance mapping
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mission file management
# ---------------------------------------------------------------------------

def remove_mission_file(path: str) -> bool:
    """Delete a mission file if present; returns False if removal fails."""
    if not os.path.lexists(path):
        return True
    try:
        os.remove(path)
        return True
    except OSError as e:
        logging.error(f"Could not remove mission file '{path}': {e}")
        return False


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
    try:
        with open(session_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
    except OSError as e:
        logging.warning(f"Could not write run summary: {e}")


# ---------------------------------------------------------------------------
# Route refinement loop
# ---------------------------------------------------------------------------

def refine_route(plan, routed_distances, max_iterations):
    """Alternate resource planning and D* Lite routing until the stop list settles."""
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
