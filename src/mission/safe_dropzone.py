import numpy as np
import cv2
import logging

def point_line_distance(point, line_start, line_end):
    # Find the shortest distance from a point to a line segment.
    line_vec = line_end - line_start
    pnt_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(pnt_vec)

    line_unitvec = line_vec / line_len
    proj_length = np.dot(pnt_vec, line_unitvec)

    if proj_length < 0:
        return np.linalg.norm(point - line_start)
    elif proj_length > line_len:
        return np.linalg.norm(point - line_end)
    else:
        proj_point = line_start + proj_length * line_unitvec
        return np.linalg.norm(point - proj_point)


def find_safe_drop_points(clustered_contours, path_order, path_points, obstacle_mask=None,
                          raw_contours=None,
                          large_contour_area_threshold=None,
                          large_contour_area_step=None,
                          max_drops_per_contour=None,
                          min_drop_separation_px=None):
    """Choose dry delivery points near each flood contour and the planned path.

    Contour vertices themselves belong to the flooded region, so they cannot
    be treated as safe delivery points. With an obstacle mask, candidates are
    searched in a narrow band outside each contour and validated as dry.

    For large contours whose area exceeds ``large_contour_area_threshold``,
    multiple safe drop points are generated along the contour boundary,
    spaced around the perimeter and filtered by ``min_drop_separation_px``.
    """
    from config import settings
    large_contour_area_threshold = (settings.LARGE_CONTOUR_AREA_THRESHOLD
                                    if large_contour_area_threshold is None
                                    else large_contour_area_threshold)
    large_contour_area_step = (settings.LARGE_CONTOUR_AREA_STEP
                               if large_contour_area_step is None
                               else large_contour_area_step)
    max_drops_per_contour = (settings.MAX_DROPS_PER_CONTOUR
                             if max_drops_per_contour is None
                             else max_drops_per_contour)
    min_drop_separation_px = (settings.MIN_DROP_SEPARATION_PX
                              if min_drop_separation_px is None
                              else min_drop_separation_px)

    safe_points = []
    path_points = [np.array(p, dtype=float) for p in path_points]

    for cluster_idx in path_order:
        # Prefer the original raw contour for area and perimeter calculations if available
        if raw_contours is not None and cluster_idx < len(raw_contours):
            cnt_ref = raw_contours[cluster_idx]
        else:
            cnt_ref = clustered_contours[cluster_idx]

        contour_points = np.squeeze(cnt_ref)

        if contour_points.ndim == 1:
            contour_points = np.expand_dims(contour_points, axis=0)

        area = float(cv2.contourArea(cnt_ref))

        # Check if contour qualifies as large
        target_points = 1
        if area >= large_contour_area_threshold:
            extra = int((area - large_contour_area_threshold) // max(1.0, large_contour_area_step))
            target_points = min(max_drops_per_contour, 1 + extra + 1)

        # Generate dry candidates just outside the contour.
        candidates = []
        perimeter_candidates = None
        if obstacle_mask is not None:
            mask = np.asarray(obstacle_mask)
            contour_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            polygon = np.rint(contour_points).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(contour_mask, [polygon], 255)
            max_radius = max(2, min(20, int(np.ceil(max(mask.shape) * 0.03))))
            for radius in range(1, max_radius + 1):
                kernel_size = 2 * radius + 1
                expanded = cv2.dilate(contour_mask, np.ones((kernel_size, kernel_size), np.uint8))
                band = (expanded > 0) & (mask == 0)
                ys, xs = np.where(band)
                if len(xs):
                    candidates.extend(np.column_stack((xs, ys)))
                if candidates:
                    break
            if target_points > 1:
                # Multi-point zones search the whole band: the first dry ring
                # can be a single spot (e.g. where forecast growth is least),
                # which would collapse every perimeter anchor onto one point.
                kernel_size = 2 * max_radius + 1
                expanded = cv2.dilate(contour_mask, np.ones((kernel_size, kernel_size), np.uint8))
                ys, xs = np.where((expanded > 0) & (mask == 0))
                perimeter_candidates = np.column_stack((xs, ys))
        else:
            candidates = contour_points

        if len(candidates) == 0:
            continue

        if target_points > 1 and len(contour_points) >= 3:
            # Multi-point perimeter sampling for large flood contour
            diffs = np.diff(contour_points, axis=0, append=contour_points[:1])
            seg_lens = np.linalg.norm(diffs, axis=1)
            cum_lens = np.cumsum(seg_lens)
            total_len = cum_lens[-1] if len(cum_lens) and cum_lens[-1] > 0 else 1.0

            cand_arr = np.asarray(candidates if perimeter_candidates is None else perimeter_candidates,
                                  dtype=float)
            chosen_contour_points = []
            for k in range(target_points):
                target_s = (k / target_points) * total_len
                idx = int(np.searchsorted(cum_lens, target_s))
                idx = min(idx, len(contour_points) - 1)
                anchor = contour_points[idx]
                dists = np.linalg.norm(cand_arr - anchor, axis=1)
                sorted_cand_indices = np.argsort(dists)

                for cand_idx in sorted_cand_indices:
                    pt_np = cand_arr[cand_idx]
                    x, y = int(round(pt_np[0])), int(round(pt_np[1]))
                    if obstacle_mask is not None:
                        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]) or mask[y, x] > 0:
                            continue
                    # Check separation from existing points in this contour
                    pt_tuple = (x, y)
                    if all(np.linalg.norm(np.array(pt_tuple) - np.array(prev)) >= min_drop_separation_px
                           for prev in chosen_contour_points):
                        chosen_contour_points.append(pt_tuple)
                        break

            if chosen_contour_points:
                safe_points.extend(chosen_contour_points)
                logging.info(
                    "Large flood contour %d (area=%.1f px^2 >= %.1f) allocated %d drop points: %s",
                    cluster_idx, area, large_contour_area_threshold, len(chosen_contour_points),
                    chosen_contour_points)
                continue

        # Standard single-point selection (closest to planned path / center)
        min_dist = float('inf')
        best_pt = None
        for pt in candidates:
            pt_np = np.array(pt, dtype=float)
            x, y = np.rint(pt_np).astype(int)
            if obstacle_mask is not None:
                if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]) or mask[y, x] > 0:
                    continue
            if len(path_points) < 2:
                dist = min((np.linalg.norm(pt_np - p) for p in path_points), default=0.0)
                if dist < min_dist:
                    min_dist, best_pt = dist, pt_np
                continue
            for i in range(len(path_points) - 1):
                dist = point_line_distance(pt_np, path_points[i], path_points[i + 1])
                if dist < min_dist:
                    min_dist = dist
                    best_pt = pt_np

        if best_pt is not None:
            safe_points.append(tuple(np.rint(best_pt).astype(int)))

    return safe_points



def filter_drops_by_home_connectivity(drop_points, home, obstacle_mask, downsample_factor=5):
    """Keep drops in the same 8-connected free-space component as HOME.

    Uses ``build_routing_grid`` and ``locate_cell`` from the router, i.e. the
    exact grid construction and component-aware endpoint snapping D* Lite
    applies, so a point that passes this filter is one D* Lite can reach.
    Pass the most conservative (largest) obstacle mask the mission will see:
    forecast frames are cumulative, so connectivity here implies connectivity
    in every arrival-time grid.
    """
    if not drop_points:
        return [], []
    if obstacle_mask is None or np.asarray(obstacle_mask).ndim != 2:
        raise ValueError("A 2D obstacle mask is required for connectivity filtering")
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be >= 1")

    from src.routing.pathfinding import build_routing_grid, locate_cell

    grid = build_routing_grid(obstacle_mask, downsample_factor)
    home_cell, home_component, _ = locate_cell(home, grid)
    if home_component == 0:
        logging.warning("HOME %s has no free connected-component cell after D* Lite grid snapping.", home)
        return [], list(range(len(drop_points)))

    reachable, excluded = [], []
    for index, point in enumerate(drop_points):
        grid_cell, component, _ = locate_cell(point, grid, home_component)
        if component == home_component:
            reachable.append(point)
        else:
            excluded.append(index)
            logging.warning(
                "FORCED MISSION DEVIATION: Excluding drop %d at %s before TSP: "
                "downsampled free-space component %d (cell=%s) is unreachable "
                "from HOME %s component %d (cell=%s).",
                index, point, component, grid_cell, home, home_component, home_cell)
    return reachable, excluded
