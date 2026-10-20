"""Multi-base path overlap repair strategies.

Enforces zero-overlap airspace deconfliction between independently planned bases.
When paths cross:
1. Reassignment: reassign the crossing sortie to the other base if within reach.
2. Tier 1: locally reroute conflicting legs using D* Lite with crossing path cells as soft obstacles.
3. Tier 2: nudge base location within safe separation and clearance constraints.
4. Hard error: block mission export (exit code 2) if overlaps cannot be resolved.

Ported from the MAIN branch. Repairs change plans only; main.run_multi_base writes the
base<N>.waypoints files once, after the final overlap check, so no file can describe a
plan that was later replaced.
"""
import logging
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from config import settings
from src.mission.mission_output import split_sorties
from src.mission.constraints import sortie_battery_report
from src.mission.multi_base import single_drop_reach
from src.routing.dstarlite import DStarLite
from src.routing.pathfinding import (
    build_routing_grid,
    cell_center,
    cell_of,
    locate_cell,
    polyline_flood_pixels,
    polyline_intersections,
)


def find_leg_index(stop_indices: List[int], seg_index: int) -> int:
    """Identify which leg index in stop_indices contains the given segment index."""
    for i in range(len(stop_indices) - 1):
        if stop_indices[i] <= seg_index < stop_indices[i + 1]:
            return i
    return max(0, len(stop_indices) - 2)


def sortie_drops(plan: Dict[str, Any], segment: int, coord_index: Dict[Tuple[int, int], int]) -> Set[int]:
    """Find the set of drop indices belonging to the sortie that contains segment.

    ``coord_index`` maps a point to its drop index, or to a list of indices when several drops share the point
    (adjacent regions can choose the same dry pixel); all of them are returned.
    """
    for sortie in split_sorties(plan["full_path"], plan["drop_indices"], plan["reload_indices"]):
        if sortie["start"] <= segment < sortie["rtb_end"]:
            found = set()
            for d in sortie["drops"]:
                ids = coord_index.get(tuple(map(int, plan["full_path"][d])), ())
                found.update(ids if isinstance(ids, (list, tuple, set)) else (ids,))
            return found
    return set()


def try_reroute_leg(
    plan: Dict[str, Any],
    other_plan: Dict[str, Any],
    segs: List[Tuple[int, int]],
    combined_obstacle_mask: np.ndarray,
    forecasts: Any,
    headwind: float,
    *,
    meters_per_pixel: float,
    other_bases_plans: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Tuple[Dict[str, Any], int, int]]:
    """Tier 1: Locally reroute a crossing leg with soft obstacles.

    Returns (updated_plan, leg_idx, remaining_intersections) on success, or None.
    """
    stop_indices = plan.get("stop_indices")
    route_points = plan.get("route_points")
    if not stop_indices or not route_points or len(stop_indices) < 2:
        return None

    # Group crossing segments by leg in plan
    crossing_legs: Dict[int, List[int]] = {}
    for seg_a, seg_b in segs:
        leg_idx = find_leg_index(stop_indices, seg_a)
        crossing_legs.setdefault(leg_idx, []).append(seg_b)

    for leg_idx, crossing_segs_b in crossing_legs.items():
        if leg_idx + 1 >= len(stop_indices) or leg_idx + 1 >= len(route_points):
            continue

        s_idx = stop_indices[leg_idx]
        g_idx = stop_indices[leg_idx + 1]
        start_pt = route_points[leg_idx]
        goal_pt = route_points[leg_idx + 1]

        # Gather points along other_plan's conflicting legs to avoid
        other_stop_indices = other_plan.get("stop_indices")
        avoid_indices = set()
        if other_stop_indices and len(other_stop_indices) >= 2:
            for sb in crossing_segs_b:
                o_leg = find_leg_index(other_stop_indices, sb)
                o_start = other_stop_indices[o_leg]
                o_end = min(len(other_plan["full_path"]) - 1, other_stop_indices[o_leg + 1])
                for idx in range(o_start, o_end):
                    avoid_indices.add(idx)
        else:
            for sb in crossing_segs_b:
                idx_min = max(0, sb - 15)
                idx_max = min(len(other_plan["full_path"]) - 1, sb + 16)
                for idx in range(idx_min, idx_max):
                    avoid_indices.add(idx)

        pts_to_avoid = []
        for p_idx in sorted(avoid_indices):
            p1 = other_plan["full_path"][p_idx]
            p2 = other_plan["full_path"][p_idx + 1]
            dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            num_s = max(2, int(np.ceil(dist / 2.0)))
            for t in np.linspace(0, 1, num_s):
                pts_to_avoid.append((p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])))

        # Elapsed mission time for arrival mask
        elapsed_dist_m = sum(
            float(np.hypot(b[0] - a[0], b[1] - a[1]))
            for a, b in zip(plan["full_path"][:s_idx], plan["full_path"][1:s_idx + 1])
        ) * meters_per_pixel
        elapsed_hours = elapsed_dist_m / max(settings.DRONE_SPEED_MPS, 0.1) / 3600.0
        leg_dist_m = float(np.hypot(goal_pt[0] - start_pt[0], goal_pt[1] - start_pt[1])) * meters_per_pixel
        estimated_arrival = elapsed_hours + leg_dist_m / max(settings.DRONE_SPEED_MPS, 0.1) / 3600.0

        leg_mask = combined_obstacle_mask
        if forecasts:
            from src.weather.flood_spread import obstacle_mask_at
            leg_mask = obstacle_mask_at(forecasts, estimated_arrival, combined_obstacle_mask)

        grid = build_routing_grid(leg_mask, 5)
        anchor = route_points[0]
        _, home_comp, _ = locate_cell(anchor, grid)
        s_grid, s_comp, s_dist = locate_cell(start_pt, grid, home_comp or None)
        g_grid, g_comp, g_dist = locate_cell(goal_pt, grid, home_comp or None)

        if s_grid == g_grid or s_comp == 0 or g_comp == 0 or s_comp != g_comp:
            continue

        conflict_cells = {cell_of(pt, grid) for pt in pts_to_avoid}
        buffered = set()
        for cx, cy in conflict_cells:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < grid.width and 0 <= ny < grid.height:
                        buffered.add((nx, ny))

        buffered.discard(s_grid)
        buffered.discard(g_grid)
        conflict_cells.discard(s_grid)
        conflict_cells.discard(g_grid)

        # Plan with buffer first, then exact conflict cells
        path = DStarLite((grid.height, grid.width), s_grid, g_grid).plan_path(grid.obstacles | buffered)
        if not path or path[0] != s_grid or path[-1] != g_grid or any(p in grid.obstacles for p in path):
            path = DStarLite((grid.height, grid.width), s_grid, g_grid).plan_path(grid.obstacles | conflict_cells)

        if not path or path[0] != s_grid or path[-1] != g_grid or any(p in grid.obstacles for p in path):
            continue

        cells = path if (s_dist or g_dist) else path[1:-1]
        if not s_dist and cells and cells[0] == s_grid:
            cells = cells[1:]
        if not g_dist and cells and cells[-1] == g_grid:
            cells = cells[:-1]

        new_leg_points = [cell_center(c, grid) for c in cells]
        routed = [start_pt] + new_leg_points + [goal_pt]

        if polyline_flood_pixels(routed, leg_mask) > 0:
            continue

        new_full_path = plan["full_path"][:s_idx + 1] + new_leg_points + plan["full_path"][g_idx:]
        delta = len(new_full_path) - len(plan["full_path"])
        new_stop_indices = [idx if i <= leg_idx else idx + delta for i, idx in enumerate(plan["stop_indices"])]
        new_drop_indices = [idx if idx <= s_idx else idx + delta for idx in plan["drop_indices"]]
        new_reload_indices = [idx if idx <= s_idx else idx + delta for idx in plan["reload_indices"]]

        # Battery verification
        new_sorties = sortie_battery_report(
            new_full_path, new_drop_indices, new_reload_indices,
            meters_per_pixel=meters_per_pixel, headwind_mps=headwind
        )
        limit = 1.0 - settings.BATTERY_RESERVE_FRACTION
        if any(s["battery_fraction"] > limit + 1e-9 for s in new_sorties):
            continue

        # Check intersection reduction
        old_hits = len(polyline_intersections(plan["full_path"], other_plan["full_path"]))
        new_hits = len(polyline_intersections(new_full_path, other_plan["full_path"]))
        if new_hits >= old_hits:
            continue

        # Ensure no new intersection introduced with any other base
        if other_bases_plans:
            has_other_crossing = False
            for other_bp in other_bases_plans:
                if other_bp and polyline_intersections(new_full_path, other_bp["full_path"]):
                    has_other_crossing = True
                    break
            if has_other_crossing:
                continue

        # Valid reroute found (the waypoint file is written after deconfliction)
        updated_plan = dict(plan)
        updated_plan["full_path"] = new_full_path
        updated_plan["stop_indices"] = new_stop_indices
        updated_plan["drop_indices"] = new_drop_indices
        updated_plan["reload_indices"] = new_reload_indices
        updated_plan["routed_sortie_battery"] = [s["battery_fraction"] for s in new_sorties]

        return updated_plan, leg_idx, new_hits

    return None


def try_nudge_base(
    b: int,
    plans_by_base: Dict[int, Any],
    selection: Dict[str, Any],
    safe_points: List[Tuple[int, int]],
    clearance: np.ndarray,
    routed_distance: Any,
    headwind: float,
    plan_base_fn: Any,
    image_shape: Tuple[int, int],
    min_separation_px: Optional[float] = None,
) -> Optional[Tuple[Tuple[int, int], Dict[str, Any]]]:
    """Tier 2: Nudge a base position within local neighborhood."""
    curr_home = tuple(map(int, plans_by_base[b]["home"]))
    assigned = [safe_points[i] for i, owner in sorted(selection["assignment"].items()) if owner == b]
    other_homes = [plans_by_base[k]["home"] for k in sorted(plans_by_base) if k != b and plans_by_base[k]]
    sep_req = settings.MIN_BASE_SEPARATION_PX if min_separation_px is None else float(min_separation_px)

    candidate_offsets = []
    for r in range(5, 55, 5):
        for dx in range(-r, r + 1, 5):
            for dy in range(-r, r + 1, 5):
                if dx * dx + dy * dy <= r * r and (dx != 0 or dy != 0):
                    candidate_offsets.append((dx, dy, dx * dx + dy * dy))
    candidate_offsets.sort(key=lambda item: item[2])

    for dx, dy, _ in candidate_offsets:
        nx, ny = curr_home[0] + dx, curr_home[1] + dy
        if not (0 <= nx < image_shape[1] and 0 <= ny < image_shape[0]):
            continue
        if clearance[ny, nx] < settings.HOME_CLEARANCE_PX:
            continue
        if any(np.hypot(nx - ox, ny - oy) < sep_req for ox, oy in other_homes):
            continue

        reach = single_drop_reach(
            [(nx, ny)], assigned, routed_distance,
            headwind_mps=headwind,
            per_drop_kg=settings.PAYLOAD_PER_DROP_KG,
            reserve=settings.BATTERY_RESERVE_FRACTION
        )
        if len(reach[0]) < len(assigned):
            continue

        new_plan = plan_base_fn(b + 1, (nx, ny), assigned)
        if not new_plan:
            continue

        # Check intersections with all other bases
        has_crossings = False
        for k in sorted(plans_by_base):
            if k != b and plans_by_base[k]:
                if polyline_intersections(new_plan["full_path"], plans_by_base[k]["full_path"]):
                    has_crossings = True
                    break

        if not has_crossings:
            return (nx, ny), new_plan

    return None


def execute_overlap_repairs(
    plans_by_base: Dict[int, Any],
    bases: List[Tuple[int, int]],
    selection: Dict[str, Any],
    safe_points: List[Tuple[int, int]],
    base_reach: List[Set[int]],
    coord_index: Dict[Tuple[int, int], int],
    combined_obstacle_mask: np.ndarray,
    forecasts: Any,
    clearance: np.ndarray,
    routed_distance: Any,
    headwind: float,
    plan_base_fn: Any,
    plan_assigned_fn: Any,
    image_shape: Tuple[int, int],
    *,
    meters_per_pixel: float,
    max_rounds: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, Any]]]:
    """Execute complete two-tier repair workflow when reassignment fails."""
    def find_crossings():
        live = [b for b in sorted(plans_by_base) if plans_by_base[b]]
        found = []
        for i, first in enumerate(live):
            for second in live[i + 1:]:
                segs = polyline_intersections(
                    plans_by_base[first]["full_path"],
                    plans_by_base[second]["full_path"],
                    return_segments=True
                )
                if segs:
                    found.append((first, second, segs))
        return found

    def refresh_reach(b):
        base_reach[b] = single_drop_reach(
            [bases[b]], safe_points, routed_distance, headwind_mps=headwind,
            per_drop_kg=settings.PAYLOAD_PER_DROP_KG, reserve=settings.BATTERY_RESERVE_FRACTION)[0]

    repairs = []
    for round_idx in range(max_rounds):
        crossings = find_crossings()
        if not crossings:
            break

        first, second, segs = crossings[0]
        moves = []
        for seg_a, seg_b in segs:
            drops_a = sortie_drops(plans_by_base[first], seg_a, coord_index)
            drops_b = sortie_drops(plans_by_base[second], seg_b, coord_index)
            if drops_a and drops_a <= base_reach[second]:
                moves.append((len(drops_a), sorted(drops_a), first, second))
            if drops_b and drops_b <= base_reach[first]:
                moves.append((len(drops_b), sorted(drops_b), second, first))

        if moves:
            _, moved, source, target = min(moves)
            for i in moved:
                selection["assignment"][i] = target
            repairs.append({"tier": 0, "type": "reassign", "drops": moved, "from_base": source + 1, "to_base": target + 1})
            logging.info(
                "PATH OVERLAP REPAIR (Reassignment): BASE %d and BASE %d routes crossed; reassigned drop(s) %s from "
                "BASE %d to BASE %d and re-planned both.",
                first + 1, second + 1, moved, source + 1, target + 1
            )
            plans_by_base[source] = plan_assigned_fn(source)
            plans_by_base[target] = plan_assigned_fn(target)
            continue

        logging.info(
            "PATH OVERLAP REPAIR: Reassignment cannot resolve BASE %d / BASE %d crossing. Attempting Tier 1 (local rerouting)...",
            first + 1, second + 1
        )

        # Tier 1: Try locally rerouting the crossing leg of first around second
        other_bp_first = [plans_by_base[k] for k in sorted(plans_by_base) if k not in (first, second) and plans_by_base[k]]
        reroute_result = try_reroute_leg(
            plans_by_base[first], plans_by_base[second], segs,
            combined_obstacle_mask, forecasts, headwind,
            meters_per_pixel=meters_per_pixel, other_bases_plans=other_bp_first
        )
        if reroute_result:
            updated_plan, leg_idx, remaining_hits = reroute_result
            plans_by_base[first] = updated_plan
            repairs.append({"tier": 1, "type": "reroute", "base": first + 1, "leg": leg_idx, "remaining_crossings": remaining_hits})
            logging.info(
                "PATH OVERLAP REPAIR (Tier 1): Rerouted leg %d of BASE %d around BASE %d conflict (remaining hits=%d).",
                leg_idx, first + 1, second + 1, remaining_hits
            )
            continue

        # Tier 1 fallback: Try locally rerouting the crossing leg of second around first
        segs_reversed = [(sb, sa) for sa, sb in segs]
        other_bp_second = [plans_by_base[k] for k in sorted(plans_by_base) if k not in (first, second) and plans_by_base[k]]
        reroute_result_b = try_reroute_leg(
            plans_by_base[second], plans_by_base[first], segs_reversed,
            combined_obstacle_mask, forecasts, headwind,
            meters_per_pixel=meters_per_pixel, other_bases_plans=other_bp_second
        )
        if reroute_result_b:
            updated_plan, leg_idx, remaining_hits = reroute_result_b
            plans_by_base[second] = updated_plan
            repairs.append({"tier": 1, "type": "reroute", "base": second + 1, "leg": leg_idx, "remaining_crossings": remaining_hits})
            logging.info(
                "PATH OVERLAP REPAIR (Tier 1): Rerouted leg %d of BASE %d around BASE %d conflict (remaining hits=%d).",
                leg_idx, second + 1, first + 1, remaining_hits
            )
            continue

        logging.info(
            "PATH OVERLAP REPAIR: Tier 1 rerouting failed for BASE %d / BASE %d. Attempting Tier 2 (base nudging)...",
            first + 1, second + 1
        )

        # Tier 2: Try nudging base position for first
        nudge_result_a = try_nudge_base(
            first, plans_by_base, selection, safe_points, clearance,
            routed_distance, headwind, plan_base_fn, image_shape
        )
        if nudge_result_a:
            new_site, new_plan = nudge_result_a
            old_site = bases[first]
            bases[first] = new_site
            refresh_reach(first)
            plans_by_base[first] = new_plan
            repairs.append({"tier": 2, "type": "nudge", "base": first + 1, "from": old_site, "to": new_site})
            logging.info(
                "PATH OVERLAP REPAIR (Tier 2): Nudged BASE %d from %s to %s; crossings resolved.",
                first + 1, old_site, new_site
            )
            continue

        # Tier 2 fallback: Try nudging base position for second
        nudge_result_b = try_nudge_base(
            second, plans_by_base, selection, safe_points, clearance,
            routed_distance, headwind, plan_base_fn, image_shape
        )
        if nudge_result_b:
            new_site, new_plan = nudge_result_b
            old_site = bases[second]
            bases[second] = new_site
            refresh_reach(second)
            plans_by_base[second] = new_plan
            repairs.append({"tier": 2, "type": "nudge", "base": second + 1, "from": old_site, "to": new_site})
            logging.info(
                "PATH OVERLAP REPAIR (Tier 2): Nudged BASE %d from %s to %s; crossings resolved.",
                second + 1, old_site, new_site
            )
            continue

        logging.error(
            "PATH OVERLAP REPAIR: Reassignment, Tier 1 rerouting, and Tier 2 base nudging all failed "
            "to resolve crossing between BASE %d and BASE %d.",
            first + 1, second + 1
        )
        break

    return repairs, find_crossings()
