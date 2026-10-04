from collections import deque, namedtuple

import numpy as np
import cv2
from src.routing.dstarlite import DStarLite
import logging

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

def snap_to_nearest_free_cell(point, obstacles, max_w, max_h, max_search_radius=15, target_component=None, labels=None,
                              accept=None):
    """
    Finds the nearest non-obstacle cell using BFS.
    If target_component and labels are provided, only cells belonging to target_component are accepted.
    ``accept`` is an optional extra predicate a candidate cell must satisfy.
    When no acceptable cell lies within ``max_search_radius`` the nearest free
    cell (any component) is returned instead; callers that need a specific
    component must check the returned cell's label.
    """
    if point not in obstacles and (accept is None or accept(point)):
        if target_component is None or labels is None:
            return point, 0
        if 0 <= point[0] < max_w and 0 <= point[1] < max_h and labels[point[1], point[0]] == target_component:
            return point, 0

    queue = deque([(point, 0)])
    visited = {point}
    fallback = None
    fallback_dist = max_search_radius

    while queue:
        current, dist = queue.popleft()

        if dist > max_search_radius:
            break

        if current not in obstacles and (accept is None or accept(current)):
            if target_component is None or labels is None:
                return current, dist
            if 0 <= current[0] < max_w and 0 <= current[1] < max_h and labels[current[1], current[0]] == target_component:
                return current, dist
            elif fallback is None:
                fallback = current
                fallback_dist = dist

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < max_w and 0 <= ny < max_h:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    if fallback is not None:
        return fallback, fallback_dist
    return point, max_search_radius


RoutingGrid = namedtuple("RoutingGrid", "occupancy obstacles labels factor width height mask")


def build_routing_grid(obstacle_mask, downsample_factor=5):
    """Downsampled 8-connected occupancy grid shared by D* Lite and drop filtering.

    A cell is blocked when any pixel of its block, after a one-pixel safety
    dilation, is flooded. Sampling a single pixel per block would let flood
    strips narrower than the block slip between samples and be routed through.
    """
    mask = np.asarray(obstacle_mask)
    h, w = mask.shape[:2]
    factor = int(downsample_factor)
    width, height = w // factor, h // factor
    if width == 0 or height == 0:
        factor, width, height = 1, w, h
    occupied = (mask > 0).astype(np.uint8)
    if factor > 1:
        occupied = cv2.dilate(occupied, np.ones((3, 3), np.uint8))
        blocks = occupied[:height * factor, :width * factor]
        occupancy = blocks.reshape(height, factor, width, factor).max(axis=(1, 3))
    else:
        occupancy = occupied
    _, labels = cv2.connectedComponents((occupancy == 0).astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(occupancy)
    obstacles = set(zip(xs.tolist(), ys.tolist()))
    return RoutingGrid(occupancy, obstacles, labels, factor, width, height, mask)


def cell_of(point, grid):
    """Grid cell containing a full-resolution pixel, clamped to the grid."""
    x, y = int(point[0]) // grid.factor, int(point[1]) // grid.factor
    return min(max(0, x), grid.width - 1), min(max(0, y), grid.height - 1)


def cell_center(cell, grid):
    """Full-resolution pixel at the centre of a grid cell."""
    half = grid.factor // 2
    return int(cell[0]) * grid.factor + half, int(cell[1]) * grid.factor + half


def locate_cell(point, grid, target_component=None, max_search_radius=4):
    """Resolve a full-resolution pixel to a free grid cell.

    Returns ``(cell, component, snap_distance)``; ``component`` is 0 when the
    point cannot be placed on a usable free cell. A point whose own cell is
    free keeps that cell and its component, even if it differs from
    ``target_component``: being on the far side of an obstacle means being
    unreachable, not being moved to the near side. Only a point whose cell is
    blocked (a dry pixel inside a conservatively blocked cell) is snapped,
    within ``max_search_radius`` cells, to a free ``target_component`` cell
    whose straight connector from the real pixel crosses no flooded pixel.

    Both the pre-routing connectivity filter and D* Lite resolve endpoints
    with this function, so they agree on which points are reachable.
    """
    cell = cell_of(point, grid)
    if cell not in grid.obstacles:
        return cell, int(grid.labels[cell[1], cell[0]]), 0

    def connector_clear(candidate):
        return polyline_flood_pixels([point, cell_center(candidate, grid)], grid.mask) == 0

    snapped, distance = snap_to_nearest_free_cell(
        cell, grid.obstacles, grid.width, grid.height, max_search_radius,
        target_component=target_component, labels=grid.labels, accept=connector_clear)
    if snapped in grid.obstacles or not connector_clear(snapped):
        return snapped, 0, distance
    component = int(grid.labels[snapped[1], snapped[0]])
    if target_component is not None and component != target_component:
        return snapped, 0, distance
    return snapped, component, distance


def grid_distance_field(grid, source_cell):
    """Shortest 8-connected path length (in cells) from ``source_cell`` to every free cell.

    ``source_cell`` may also be a list of cells (distance to the nearest one).
    Obstacle and unreachable cells are ``inf``. Used to score HOME candidates
    by routed distance around flood zones rather than straight-line distance.
    """
    import heapq
    dist = np.full((grid.height, grid.width), np.inf)
    free = grid.occupancy == 0
    sources = [source_cell] if np.ndim(source_cell) == 1 else list(source_cell)
    heap = []
    for sx, sy in sources:
        if free[sy, sx] and dist[sy, sx] > 0:
            dist[sy, sx] = 0.0
            heap.append((0.0, sx, sy))
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5)]
    while heap:
        d, x, y = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dx, dy, w in steps:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height and free[ny, nx] and d + w < dist[ny, nx]:
                dist[ny, nx] = d + w
                heapq.heappush(heap, (d + w, nx, ny))
    return dist


def routed_distance_function(grid, targets, meters_per_pixel):
    """``distance(a, b)`` in metres along the routing grid, for legs touching ``targets``.

    One distance field is computed per target point; a leg between two
    non-target points falls back to straight-line distance.
    """
    fields = {}
    cells = {}

    def cell(point):
        key = tuple(map(int, point))
        if key not in cells:
            cells[key] = locate_cell(key, grid)[0]
        return cells[key]

    for target in targets:
        key = tuple(map(int, target))
        if key not in fields:
            fields[key] = grid_distance_field(grid, cell(key))
    scale = grid.factor * meters_per_pixel

    def distance(a, b):
        a, b = tuple(map(int, a)), tuple(map(int, b))
        if b in fields:
            cx, cy = cell(a)
            return float(fields[b][cy, cx]) * scale
        if a in fields:
            cx, cy = cell(b)
            return float(fields[a][cy, cx]) * scale
        return float(np.linalg.norm(np.subtract(a, b))) * meters_per_pixel
    return distance


def polyline_flood_pixels(points, mask):
    """Count flooded mask pixels touched by the 1-px rasterised polyline."""
    if len(points) < 2:
        return 0
    canvas = np.zeros(np.asarray(mask).shape[:2], dtype=np.uint8)
    pts = np.asarray([[int(round(p[0])), int(round(p[1]))] for p in points], dtype=np.int32)
    cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, 255, 1)
    return int(np.count_nonzero((canvas > 0) & (np.asarray(mask) > 0)))


def compute_full_path(ordered_points, obstacle_mask, downsample_factor=5, *, forecast_masks=None,
                      meters_per_pixel=2.0, speed_mps=None, leg_report=None):
    """
    Computes obstacle-aware paths between consecutive TSP-ordered points using D* Lite.
    Uses a downsampled mask for computational efficiency.

    Every leg is either a verified D* Lite route or an explicitly logged
    straight-line fallback; pass a list as ``leg_report`` to receive one dict
    per leg describing which (``status`` is ``dstar``, ``same_cell`` or
    ``fallback``) plus how many flooded pixels a fallback crosses.

    Returns:
        full_path: List of (x,y) points including intermediate waypoints.
        stop_indices: Index in full_path of each entry of ordered_points.
        leg_distances_m: Routed length of each leg in metres.
    """
    from config import settings
    speed_mps = settings.DRONE_SPEED_MPS if speed_mps is None else speed_mps
    if len(ordered_points) < 2:
        return ordered_points, list(range(len(ordered_points))), []

    full_path = []
    stop_indices = []
    elapsed_hours = 0.0
    anchor = ordered_points[0]
    for i in range(len(ordered_points) - 1):
        start = ordered_points[i]
        goal = ordered_points[i+1]

        # Record semantic stop location before adding routed intermediates.
        stop_indices.append(len(full_path))
        full_path.append(start)

        leg_distance_m = float(np.linalg.norm(np.asarray(goal, dtype=float) - np.asarray(start, dtype=float))) * meters_per_pixel
        estimated_arrival = elapsed_hours + leg_distance_m / max(speed_mps, 0.1) / 3600.0
        leg_mask = obstacle_mask
        if forecast_masks:
            from src.weather.flood_spread import obstacle_mask_at
            leg_mask = obstacle_mask_at(forecast_masks, estimated_arrival, obstacle_mask)
        grid = build_routing_grid(leg_mask, downsample_factor)

        # Every route point must lie in HOME's free-space component; snapping
        # either endpoint into another component would make D* Lite fail.
        _, home_comp, _ = locate_cell(anchor, grid)
        target_comp = home_comp or None
        s_grid, s_comp, s_dist = locate_cell(start, grid, target_comp)
        g_grid, g_comp, g_dist = locate_cell(goal, grid, target_comp)
        if s_dist or g_dist:
            logging.info("Leg %d grid endpoints: start %s->%s (snap %d cells, comp %d), "
                         "goal %s->%s (snap %d cells, comp %d), HOME comp %d.",
                         i, cell_of(start, grid), s_grid, s_dist, s_comp,
                         cell_of(goal, grid), g_grid, g_dist, g_comp, home_comp)

        entry = {"leg": i, "start": tuple(map(int, start)), "goal": tuple(map(int, goal)),
                 "arrival_h": estimated_arrival, "status": "dstar", "reason": "",
                 "flooded_px": 0, "route_flooded_px": 0}
        leg_points = []
        if s_grid == g_grid:
            entry["status"] = "same_cell"
        else:
            reason = ""
            path = []
            if s_comp == 0 or g_comp == 0 or s_comp != g_comp:
                reason = (f"endpoints not in one free component (start comp {s_comp}, "
                          f"goal comp {g_comp}, HOME comp {home_comp})")
            else:
                logging.info(f"D* Lite planning from {s_grid} to {g_grid} on {grid.width}x{grid.height} grid...")
                path = DStarLite((grid.height, grid.width), s_grid, g_grid).plan_path(grid.obstacles)
                if not path:
                    reason = "D* Lite found no path"
                elif path[0] != s_grid or path[-1] != g_grid or any(p in grid.obstacles for p in path):
                    reason = "D* Lite returned an incomplete or blocked path"
                    path = []
            if path:
                # Snapped endpoints keep their cell centre so the connector to
                # the real pixel is explicit rather than an implied jump.
                cells = path if (s_dist or g_dist) else path[1:-1]
                if not s_dist and cells and cells[0] == s_grid:
                    cells = cells[1:]
                if not g_dist and cells and cells[-1] == g_grid:
                    cells = cells[:-1]
                leg_points = [cell_center(c, grid) for c in cells]
            else:
                entry["status"] = "fallback"
                entry["reason"] = reason
                entry["flooded_px"] = polyline_flood_pixels([start, goal], leg_mask)
                logging.warning(
                    "FALLBACK STRAIGHT-LINE SEGMENT (not a D* Lite route) on leg %d %s -> %s at "
                    "arrival +%.2fh: %s. Segment %s %d flooded pixel(s) of the arrival-time mask.",
                    i, tuple(map(int, start)), tuple(map(int, goal)), estimated_arrival, reason,
                    "CROSSES" if entry["flooded_px"] else "is clear of", entry["flooded_px"])
        full_path.extend(leg_points)
        # Route length, rather than the straight line, sets subsequent arrival time.
        routed = [start] + leg_points + [goal]
        entry["route_flooded_px"] = polyline_flood_pixels(routed, leg_mask)
        routed_px = sum(float(np.linalg.norm(np.subtract(b, a))) for a, b in zip(routed, routed[1:]))
        elapsed_hours += routed_px * meters_per_pixel / max(speed_mps, 0.1) / 3600.0
        if leg_report is not None:
            leg_report.append(entry)

    # Add final point (usually home)
    stop_indices.append(len(full_path))
    full_path.append(ordered_points[-1])
    leg_distances_m = []
    for leg_idx, route_idx in enumerate(stop_indices):
        end_idx = stop_indices[leg_idx + 1] if leg_idx + 1 < len(stop_indices) else len(full_path) - 1
        segment = full_path[route_idx:end_idx + 1]
        if len(segment) < 2:
            leg_distances_m.append(0.0)
        else:
            leg_distances_m.append(sum(float(np.linalg.norm(np.subtract(b, a))) for a, b in zip(segment, segment[1:])) * meters_per_pixel)
    return full_path, stop_indices, leg_distances_m


def polyline_intersections(path_a, path_b, return_segments=False):
    """Points where a segment of ``path_a`` crosses or touches a segment of ``path_b``.

    Used to verify that independently planned bases do not share airspace.
    With ``return_segments`` the result is a list of ``(i, j)`` pairs: segment
    ``i`` (from ``path_a[i]`` to ``path_a[i + 1]``) meets segment ``j`` of ``path_b``.
    """
    def without_repeats(path):
        # Reload stops repeat the base point; zero-length segments are not geometry.
        pts = np.asarray(path, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            return pts, np.arange(0)
        keep = np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]
        return pts[keep], np.nonzero(keep)[0]

    a, a_index = without_repeats(path_a)
    b, b_index = without_repeats(path_b)
    if len(a) < 2 or len(b) < 2:
        return []
    p, r = a[:-1, None, :], (a[1:] - a[:-1])[:, None, :]
    q, s = b[None, :-1, :], (b[1:] - b[:-1])[None, :, :]

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    denom = cross(r, s)
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cross(qp, s) / denom
        u = cross(qp, r) / denom
    hit = (denom != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    # Collinear overlapping segments
    collinear = (denom == 0) & (cross(qp, r) == 0)
    if collinear.any():
        rr = np.maximum((r * r).sum(-1), 1e-12)
        t0 = (qp * r).sum(-1) / rr
        t1 = t0 + (s * r).sum(-1) / rr
        overlap = collinear & (np.maximum(t0, t1) >= 0) & (np.minimum(t0, t1) <= 1)
        hit |= overlap
    i, j = np.nonzero(hit)
    if return_segments:
        # Map back to segment indices of the original (repeat-containing) paths.
        return [(int(a_index[k + 1]) - 1, int(b_index[m + 1]) - 1) for k, m in zip(i, j)]
    return [tuple(float(v) for v in (p[k, 0] + np.clip(np.nan_to_num(t[k, m]), 0, 1) * r[k, 0]).round(1))
            for k, m in zip(i, j)]
