import numpy as np
import cv2
import logging
import math

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
                          moderate_contour_area_threshold=None,
                          large_contour_area_threshold=None,
                          huge_contour_area_threshold=None,
                          large_contour_area_step=None,
                          max_drops_per_contour=None,
                          min_drop_separation_px=None,
                          **kwargs):
    """Choose dry delivery points near each flood contour and the planned path.

    Contour vertices themselves belong to the flooded region, so they cannot
    be treated as safe delivery points. With an obstacle mask, candidates are
    searched in a narrow band outside each contour and validated as dry.

    Contour visits scale gracefully with size:
      - Normal contours (< moderate threshold): 1 drop point
      - Moderate contours (>= moderate threshold): 2 drop points
      - Big contours (>= large threshold): 3 drop points
      - Huge contours (>= huge threshold): up to 4 drop points (capped by max_drops)

    Multiple drop points on the same contour are distributed around the perimeter
    and spaced widely apart according to the contour's perimeter and bounding diameter.
    """
    from config import settings
    moderate_thresh = (settings.MODERATE_CONTOUR_AREA_THRESHOLD
                       if moderate_contour_area_threshold is None
                       else float(moderate_contour_area_threshold))
    large_thresh = (settings.LARGE_CONTOUR_AREA_THRESHOLD
                    if large_contour_area_threshold is None
                    else float(large_contour_area_threshold))
    huge_thresh = (settings.HUGE_CONTOUR_AREA_THRESHOLD
                   if huge_contour_area_threshold is None
                   else float(huge_contour_area_threshold))
    max_drops = (settings.MAX_DROPS_PER_CONTOUR
                 if max_drops_per_contour is None
                 else int(max_drops_per_contour))
    min_drop_sep = (settings.MIN_DROP_SEPARATION_PX
                    if min_drop_separation_px is None
                    else float(min_drop_separation_px))

    safe_points = []
    path_points = [np.array(p, dtype=float) for p in path_points]

    dist_dry = None
    dry_labels = None
    label_sizes = None
    if obstacle_mask is not None:
        mask_arr = np.asarray(obstacle_mask)
        dist_dry = cv2.distanceTransform((mask_arr == 0).astype(np.uint8), cv2.DIST_L2, 5)
        _, dry_labels = cv2.connectedComponents((mask_arr == 0).astype(np.uint8), connectivity=8)
        label_sizes = np.bincount(dry_labels.ravel())

    for cluster_idx in path_order:
        # Use the clustered contour geometry directly — cluster_idx is an index
        # into the filtered contours (via cluster_contours), NOT into raw_contours
        # whose ordering comes from cv2.findContours on the full mask and doesn't
        # align with the filtered/clustered indices.
        cnt_ref = clustered_contours[cluster_idx]

        contour_points = np.squeeze(cnt_ref)

        if contour_points.ndim == 1:
            contour_points = np.expand_dims(contour_points, axis=0)

        area = float(cv2.contourArea(cnt_ref))

        # Size-based visit formulation:
        # Normal (< moderate): 1 drop point
        # Moderate: 2 drop points
        # Big: 3 drop points
        # Huge: 4 drop points (capped by max_drops)
        if area >= huge_thresh:
            target_points = min(max_drops, 4)
            cat_label = "Huge"
        elif area >= large_thresh:
            target_points = min(max_drops, 3)
            cat_label = "Big"
        elif area >= moderate_thresh:
            target_points = min(max_drops, 2)
            cat_label = "Moderate"
        else:
            target_points = 1
            cat_label = "Standard"

        # Generate dry candidates just outside the contour with safe clearance.
        candidates = []
        perimeter_candidates = None
        if obstacle_mask is not None:
            mask = np.asarray(obstacle_mask)
            contour_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            polygon = np.rint(contour_points).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(contour_mask, [polygon], 255)
            max_radius = max(5, min(35, int(np.ceil(max(mask.shape) * 0.04))))
            kernel_size = 2 * max_radius + 1
            expanded = cv2.dilate(contour_mask, np.ones((kernel_size, kernel_size), np.uint8))
            cand_mask = (expanded > 0) & (mask == 0)

            for min_c in [5.0, 3.5, 2.0, 0.5]:
                ys, xs = np.where(cand_mask & (dist_dry >= min_c))
                if len(xs):
                    pts_cand = np.column_stack((xs, ys))
                    if dry_labels is not None and label_sizes is not None:
                        c_labels = dry_labels[ys, xs]
                        c_sizes = label_sizes[c_labels]
                        max_s = np.max(c_sizes)
                        if max_s > 0:
                            pts_cand = pts_cand[c_sizes >= max_s * 0.5]
                    candidates = pts_cand
                    break
            if len(candidates) == 0:
                ys, xs = np.where(cand_mask)
                candidates = np.column_stack((xs, ys)) if len(xs) else contour_points
            perimeter_candidates = candidates
        else:
            candidates = contour_points

        if len(candidates) == 0 and (perimeter_candidates is None or len(perimeter_candidates) == 0):
            continue

        if target_points > 1 and len(contour_points) >= 3:
            # Multi-point perimeter sampling: distribute drops around the contour perimeter
            diffs = np.diff(contour_points, axis=0, append=contour_points[:1])
            seg_lens = np.linalg.norm(diffs, axis=1)
            cum_lens = np.cumsum(seg_lens)
            total_len = cum_lens[-1] if len(cum_lens) and cum_lens[-1] > 0 else 1.0

            # Contour bounding box diameter to prevent points from collapsing
            bx, by, bw, bh = cv2.boundingRect(cnt_ref)
            contour_diag = math.hypot(bw, bh)

            # Target separation ensures drops are placed far apart across the contour
            target_sep = max(min_drop_sep, min(total_len / (1.5 * target_points), contour_diag * 0.45))

            cand_pool = perimeter_candidates if perimeter_candidates is not None and len(perimeter_candidates) else candidates
            cand_arr = np.asarray(cand_pool, dtype=float)
            chosen_contour_points = []

            for k in range(target_points):
                # Anchor stations spaced around the perimeter (e.g., 0 and 0.5 for 2 drops; 0, 1/3, 2/3 for 3 drops)
                target_s = (k / target_points) * total_len
                idx = int(np.searchsorted(cum_lens, target_s))
                idx = min(idx, len(contour_points) - 1)
                anchor = contour_points[idx]
                dists = np.linalg.norm(cand_arr - anchor, axis=1)
                sorted_cand_indices = np.argsort(dists)

                found_pt = None
                # Progressively try separation from target_sep down to min_drop_sep
                sep_tiers = [target_sep, target_sep * 0.8, target_sep * 0.6, min_drop_sep]
                for sep_req in sep_tiers:
                    for cand_idx in sorted_cand_indices:
                        pt_np = cand_arr[cand_idx]
                        x, y = int(round(pt_np[0])), int(round(pt_np[1]))
                        if obstacle_mask is not None:
                            if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]) or mask[y, x] > 0:
                                continue
                        pt_tuple = (x, y)
                        # Check separation from other drop points on this contour
                        if all(math.hypot(x - prev[0], y - prev[1]) >= sep_req for prev in chosen_contour_points):
                            found_pt = pt_tuple
                            break
                    if found_pt is not None:
                        break

                if found_pt is not None:
                    chosen_contour_points.append(found_pt)

            if chosen_contour_points:
                safe_points.extend(chosen_contour_points)
                logging.info(
                    "%s flood contour %d (area=%.1f px^2) allocated %d spaced drop points (target_sep=%.1f px): %s",
                    cat_label, cluster_idx, area, len(chosen_contour_points), target_sep,
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



def filter_drops_by_home_connectivity(drop_points, home, obstacle_mask, downsample_factor=3):
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


# Alias for compatibility with paper scripts
find_dry_drop_points = find_safe_drop_points
