import numpy as np
import cv2
from src.routing.dstarlite import DStarLite
from src.mission.constraints import estimate_straight_leg
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Time-aware single-leg planning (Task 2)
# --------------------------------------------------------------------------- #
def downsample_mask(mask, downsample_factor):
    """Resize an obstacle mask to the D* Lite planning grid."""
    h, w = mask.shape
    new_w = w // downsample_factor
    new_h = h // downsample_factor
    if new_w == 0 or new_h == 0:
        new_w, new_h = w, h
        downsample_factor = 1
    small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return small, downsample_factor


def mask_to_obstacles(small_mask):
    y_idx, x_idx = np.where(small_mask > 0)
    return set(zip(x_idx, y_idx))


def plan_single_leg(start_px, goal_px, obstacle_mask, downsample_factor=5):
    """
    D* Lite route between two pixels against ONE static obstacle snapshot.

    Returns (path_px, ok). ``path_px`` begins exactly at ``start_px`` and ends
    exactly at ``goal_px``; intermediate points lie on the downsampled grid
    upscaled back to pixel space.  If no path exists after snapping, ok=False.
    """
    start_px = (int(start_px[0]), int(start_px[1]))
    goal_px = (int(goal_px[0]), int(goal_px[1]))

    if start_px == goal_px:
        return [start_px], True

    small, factor = downsample_mask(obstacle_mask, downsample_factor)
    new_w, new_h = small.shape[1], small.shape[0]
    obstacles = mask_to_obstacles(small)

    s_grid = (start_px[0] // factor, start_px[1] // factor)
    g_grid = (goal_px[0] // factor, goal_px[1] // factor)
    s_grid = (min(max(0, s_grid[0]), new_w - 1), min(max(0, s_grid[1]), new_h - 1))
    g_grid = (min(max(0, g_grid[0]), new_w - 1), min(max(0, g_grid[1]), new_h - 1))

    if s_grid in obstacles:
        s_grid, _ = snap_to_nearest_free_cell(s_grid, obstacles, new_w, new_h)
    if g_grid in obstacles:
        g_grid, _ = snap_to_nearest_free_cell(g_grid, obstacles, new_w, new_h)

    if s_grid == g_grid:
        return [start_px, goal_px], True

    dstar = DStarLite((new_h, new_w), s_grid, g_grid)
    path = dstar.plan_path(obstacles)
    if not path:
        return [], False

    out = [start_px]
    for pt in path[1:-1]:
        out.append((pt[0] * factor, pt[1] * factor))
    out.append(goal_px)
    return out, True


def path_length_m(path_px, meters_per_pixel):
    total = 0.0
    for a, b in zip(path_px, path_px[1:]):
        total += np.hypot(b[0] - a[0], b[1] - a[1])
    return float(total * meters_per_pixel)


def plan_timed_leg(start_px, goal_px, timeline, departure_min, spec,
                   weather, meters_per_pixel, downsample_factor=5):
    """
    Plan one flight leg against the predicted obstacle state at the *estimated
    arrival time* of that leg, not the mission-start snapshot (Task 2).

    Steps
    -----
    1. Estimate arrival = departure + straight-line distance / wind-adjusted
       ground speed -> pick the predicted flood frame valid at that time.
    2. D* Lite against that frame.
    3. Recompute the arrival from the *actual* (obstacle-avoiding) path length;
       if it moved to a different forecast slice, replan once against it.

    Returns a dict with ``path`` (px), ``ok``, ``arrival_min``,
    ``obstacle_time_min`` (which forecast slice was used), ``duration_s``,
    ``distance_m`` and ``headwind_mps``.
    """
    est = estimate_straight_leg(start_px, goal_px, meters_per_pixel, spec, weather)
    arrival_min = departure_min + est["duration_s"] / 60.0
    obstacle_time = timeline.nearest_time_min(arrival_min)
    mask = timeline.obstacle_mask_at(arrival_min, mode="nearest")

    path, ok = plan_single_leg(start_px, goal_px, mask, downsample_factor)
    if not ok:
        return {
            "path": [], "ok": False, "arrival_min": arrival_min,
            "obstacle_time_min": obstacle_time, "duration_s": float("inf"),
            "distance_m": float("inf"), "headwind_mps": est["headwind_mps"],
        }

    dist_m = path_length_m(path, meters_per_pixel)
    dur_s = spec.leg_duration_s(dist_m, est["headwind_mps"])
    actual_arrival = departure_min + dur_s / 60.0

    # If the detour pushed the ETA onto a different forecast slice, replan once.
    refined_time = timeline.nearest_time_min(actual_arrival)
    if refined_time != obstacle_time:
        mask2 = timeline.obstacle_mask_at(actual_arrival, mode="nearest")
        path2, ok2 = plan_single_leg(start_px, goal_px, mask2, downsample_factor)
        if ok2:
            path = path2
            obstacle_time = refined_time
            dist_m = path_length_m(path, meters_per_pixel)
            dur_s = spec.leg_duration_s(dist_m, est["headwind_mps"])
            actual_arrival = departure_min + dur_s / 60.0

    return {
        "path": path, "ok": True, "arrival_min": actual_arrival,
        "obstacle_time_min": obstacle_time, "duration_s": dur_s,
        "distance_m": dist_m, "headwind_mps": est["headwind_mps"],
    }


def nearest_neighbor_tsp(points, home=None):
    # Simple Nearest Neighbor TSP with HOME start/return.
    if not points:
        return [], []

    pts = np.array(points)
    n = len(pts)
    visited = [False] * n
    order = []

    if home is not None:
        current = np.array(home)
    else:
        current = pts[0]

    for _ in range(n):
        dists = np.linalg.norm(pts - current, axis=1)
        dists = [d if not visited[i] else np.inf for i, d in enumerate(dists)]
        idx = int(np.argmin(dists))
        order.append(idx)
        visited[idx] = True
        current = pts[idx]

    ordered_points = [points[i] for i in order]
    if home is not None:
        ordered_points.insert(0, home)
        ordered_points.append(home)

    return order, ordered_points

def snap_to_nearest_free_cell(point, obstacles, max_w, max_h, max_search_radius=10):
    """
    Finds the nearest non-obstacle cell using BFS.
    """
    if point not in obstacles:
        return point, 0
        
    queue = [(point, 0)]
    visited = {point}
    
    while queue:
        current, dist = queue.pop(0)
        
        if dist > max_search_radius:
            break
            
        if current not in obstacles:
            return current, dist
            
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < max_w and 0 <= ny < max_h:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
                    
    return point, max_search_radius

def compute_full_path(ordered_points, obstacle_mask, downsample_factor=5):
    """
    Computes obstacle-aware paths between consecutive TSP-ordered points using D* Lite.
    Uses a downsampled mask for computational efficiency.
    Returns:
        full_path: List of (x,y) points including intermediate waypoints.
        drop_indices: List of indices in full_path where drops should occur.
    """
    if len(ordered_points) < 2:
        return ordered_points, list(range(len(ordered_points)))
        
    # Downsample mask
    h, w = obstacle_mask.shape
    new_w = w // downsample_factor
    new_h = h // downsample_factor
    if new_w == 0 or new_h == 0:
        new_w, new_h = w, h
        downsample_factor = 1
        
    small_mask = cv2.resize(obstacle_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Extract obstacles as a set of (x,y)
    y_idx, x_idx = np.where(small_mask > 0)
    obstacles = set(zip(x_idx, y_idx))
    
    full_path = []
    drop_indices = []
    
    for i in range(len(ordered_points) - 1):
        start = ordered_points[i]
        goal = ordered_points[i+1]
        
        # Add current point to full path and mark as a drop index
        drop_indices.append(len(full_path))
        full_path.append(start)
        
        # Convert to small grid coords
        s_grid = (start[0] // downsample_factor, start[1] // downsample_factor)
        g_grid = (goal[0] // downsample_factor, goal[1] // downsample_factor)
        
        # Ensure within bounds
        s_grid = (min(max(0, s_grid[0]), new_w-1), min(max(0, s_grid[1]), new_h-1))
        g_grid = (min(max(0, g_grid[0]), new_w-1), min(max(0, g_grid[1]), new_h-1))
        
        # Check if start/goal are obstacles and snap if necessary
        s_obs = s_grid in obstacles
        g_obs = g_grid in obstacles
        
        if s_obs:
            logging.info(f"Start node {s_grid} is an obstacle. Attempting to snap to free cell...")
            s_grid, s_dist = snap_to_nearest_free_cell(s_grid, obstacles, new_w, new_h)
            logging.info(f"Snapped start node to {s_grid} (distance: {s_dist} cells).")
            
        if g_obs:
            logging.info(f"Goal node {g_grid} is an obstacle. Attempting to snap to free cell...")
            g_grid, g_dist = snap_to_nearest_free_cell(g_grid, obstacles, new_w, new_h)
            logging.info(f"Snapped goal node to {g_grid} (distance: {g_dist} cells).")
        
        # If start and goal are same in small grid, just use direct line
        if s_grid == g_grid:
            continue
            
        logging.info(f"D* Lite planning from {s_grid} to {g_grid} on {new_w}x{new_h} grid...")
        dstar = DStarLite((new_h, new_w), s_grid, g_grid)
        path = dstar.plan_path(obstacles)
        
        if not path:
            logging.error(f"No path found between {start} and {goal} even after snapping. Using unvalidated direct-line fallback that may cross obstacles.")
        else:
            # Upsample and add to full path (skip first and last to avoid duplicates)
            for pt in path[1:-1]:
                full_path.append((pt[0] * downsample_factor, pt[1] * downsample_factor))
                
    # Add final point (usually home)
    drop_indices.append(len(full_path))
    full_path.append(ordered_points[-1])
    
    return full_path, drop_indices

