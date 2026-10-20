"""Multi-base mission planning engine.

Extracted from ``main.py`` to keep the top-level entry point slim while
preserving every log message, error path, and feature.
"""

import json
import logging
import shutil
import sys

import cv2
import numpy as np

from src.routing.pathfinding import (
    compute_full_path, polyline_flood_pixels, build_routing_grid,
    routed_distance_function, polyline_intersections, grid_distance_field,
    locate_cell, nearest_neighbor_tsp,
)
from src.mission.safe_dropzone import (
    find_safe_drop_points, filter_drops_by_home_connectivity,
)
from src.mission.mission_output import (
    generate_mission_file, display_multi_base_map, validate_waypoints_file,
    basemap_sanity,
)
from src.mission.constraints import (
    plan_mission_stops, log_deviation, wind_kmh_to_mps, sortie_battery_report,
)
from src.mission.multi_base import select_bases, single_drop_reach
from src.mission.overlap_repair import execute_overlap_repairs
from config import settings

from src.pipeline.common import (
    drop_index_by_point, map_routed_drop_distances, create_session_dir,
    write_summary, refine_route,
)


def _main_attr(name, default):
    main_mod = sys.modules.get("main")
    return getattr(main_mod, name, default) if main_mod is not None else default


def run_multi_base(image, mask, forecasts, contours, cluster_edges, points, weather, geo_center,
                   scale_factor, max_bases, session_root="data/output", show=True, coordinate_fn=None,
                   raw_contours=None):
    """Multi-base mission planning, one QGC WPL 110 file per base."""
    meters_per_pixel = settings.METERS_PER_PIXEL / scale_factor
    image_center_px = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)
    pred_mask = forecasts[max(forecasts)]
    combined_obstacle_mask = cv2.bitwise_or(mask, pred_mask)

    session_dir = create_session_dir(session_root)
    logging.info("SESSION OUTPUT: %s", session_dir)
    summary = {
        "mode": "multi", "max_bases": int(max_bases), "geo_anchor": list(geo_center),
        "scale_factor": scale_factor, "meters_per_pixel": meters_per_pixel,
        "meters_per_pixel_display": meters_per_pixel, "image_center_px": list(image_center_px),
        "session_dir": str(session_dir), "weather": dict(weather)
    }

    # Safe points from cluster edges
    try:
        safe_points = find_safe_drop_points(cluster_edges, list(range(len(points))), points,
                                            obstacle_mask=combined_obstacle_mask, raw_contours=raw_contours or contours)
    except Exception as e:
        logging.error(f"find_safe_drop_points failed: {e}")
        safe_points = []

    if not safe_points:
        logging.warning("No safe drop points computed.")
        sys.exit(0)
    safe_points = [tuple(map(int, p)) for p in safe_points]
    summary["drop_points"] = [list(p) for p in safe_points]
    logging.info("Safe delivery points: %d of %d detected zones", len(safe_points), len(contours))
    generated_drop_count = len(safe_points)
    zones_with_point = sum(
        1 for c in contours
        if any(cv2.pointPolygonTest(c, (float(x), float(y)), True) >= -25 for x, y in safe_points))
    raw_count = len(raw_contours) if raw_contours is not None else len(contours)
    logging.info("STAGE COUNTS: raw contours=%d -> zones (area >= %d)=%d -> zones with a drop point=%d "
                 "(%d drop points)", raw_count, 200, len(contours), zones_with_point,
                 generated_drop_count)

    # Sample base candidates on a ~60x60 spatial grid of dry pixels.
    stride = max(1, int(max(image.shape[:2]) / 60))
    grid_y, grid_x = np.mgrid[stride // 2:image.shape[0]:stride, stride // 2:image.shape[1]:stride]
    grid_points = list(zip(grid_x.ravel().tolist(), grid_y.ravel().tolist()))
    dry = [(x, y) for x, y in grid_points if combined_obstacle_mask[y, x] == 0]
    clearance = cv2.distanceTransform((combined_obstacle_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    candidates = [(x, y) for x, y in dry if clearance[y, x] >= settings.HOME_CLEARANCE_PX]
    logging.info("Base candidates with >= %.0f px flood clearance: %d of %d dry grid points",
                 settings.HOME_CLEARANCE_PX, len(candidates), len(dry))
    summary["home_candidates"] = {"grid_points": len(grid_points), "dry": len(dry),
                                  "with_clearance": len(candidates)}
    if not candidates:
        summary["home_outcome"] = "NO_CANDIDATES_SAMPLED"
        summary["outcome"] = "NO_CANDIDATES_SAMPLED"
        summary["selection"] = None
        logging.error("BASE SELECTION: NO_CANDIDATES_SAMPLED - no dry grid point with %.0f px flood clearance "
                      "(grid_points=%d dry=%d).", settings.HOME_CLEARANCE_PX, len(grid_points), len(dry))
        log_deviation("No safe base candidate; mission infeasible.")
        write_summary(summary, session_dir)
        sys.exit(2)

    # Reach and cost use routed distances around flood zones on the D* Lite grid.
    headwind = wind_kmh_to_mps(weather.get("wind_speed_10m", 0.0))
    routing_grid = build_routing_grid(combined_obstacle_mask, downsample_factor=3)
    routed_distance = routed_distance_function(routing_grid, safe_points, meters_per_pixel)
    selection = select_bases(
        candidates, safe_points, routed_distance, combined_obstacle_mask,
        headwind_mps=headwind, wind_from_deg=weather.get("wind_direction_10m"),
        per_drop_kg=settings.PAYLOAD_PER_DROP_KG, payload_capacity_kg=settings.PAYLOAD_CAPACITY_KG,
        reserve=settings.BATTERY_RESERVE_FRACTION, meters_per_pixel=meters_per_pixel,
        max_bases=max_bases, min_separation_px=settings.MIN_BASE_SEPARATION_PX)
    bases = selection["bases"]
    if not bases:
        summary["home_outcome"] = "NO_BASE_REACHES_ANY_DROP"
        summary["outcome"] = "NO_BASE_REACHES_ANY_DROP"
        summary["selection"] = selection
        logging.error("BASE SELECTION: no flood-clear base site can serve any drop point within battery range.")
        log_deviation("Mission infeasible: no base can reach any drop point.")
        write_summary(summary, session_dir)
        sys.exit(2)
    summary["home_outcome"] = "SINGLE_BASE" if len(bases) == 1 else f"MULTI_BASE_{len(bases)}"
    logging.info("BASE SELECTION: %s - %d base(s) at %s; best single site reached %d of %d coverable drop "
                 "points.", summary["home_outcome"], len(bases), bases, selection["best_single_reach"],
                 selection["coverable"])

    # Zone-to-base assignment table
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
        tag = f"BASE {number}"
        result = {"base": number, "home": tuple(map(int, home)), "assigned": len(drops),
                  "clearance_px": float(clearance[int(home[1]), int(home[0])])}
        logging.info("%s at %s (flood clearance %.1f px): planning %d assigned drop points.",
                     tag, result["home"], result["clearance_px"], len(drops))
        drops, excluded = filter_drops_by_home_connectivity(drops, home, combined_obstacle_mask,
                                                            downsample_factor=3)
        result["connectivity_excluded"] = len(excluded)
        logging.info("%s drop connectivity filter: reachable=%d excluded=%d", tag, len(drops), len(excluded))
        if not drops:
            logging.error("%s: no reachable drop points after connectivity filtering.", tag)
            return None
        path_order, _ = nearest_neighbor_tsp(drops, home=home)
        ordered_drops = [drops[i] for i in path_order]

        def plan(routed):
            plan_stops = _main_attr("plan_mission_stops", plan_mission_stops)
            return plan_stops(home, ordered_drops, forecasts, mask,
                meters_per_pixel=meters_per_pixel, headwind_mps=headwind,
                initial_battery=1.0, routed_leg_distances_m=routed)

        def routed_distances(candidate_route):
            _, _, leg_distances = compute_full_path(candidate_route.points, combined_obstacle_mask,
                forecast_masks=forecasts, meters_per_pixel=meters_per_pixel)
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
            forecast_masks=forecasts, meters_per_pixel=meters_per_pixel, leg_report=leg_report)
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
                                        meters_per_pixel=meters_per_pixel, headwind_mps=headwind)
        limit = 1.0 - settings.BATTERY_RESERVE_FRACTION
        for k, sortie in enumerate(sorties, 1):
            logging.info("%s ROUTED SORTIE %d: drops=%d distance=%.0fm battery=%.1f%% (limit %.0f%%)%s", tag, k,
                         sortie["drops"], sortie["distance_m"], sortie["battery_fraction"] * 100, limit * 100,
                         "  ** EXCEEDS LIMIT **" if sortie["battery_fraction"] > limit + 1e-9 else "")

        result.update({
            "full_path": full_path, "drop_indices": drop_indices, "reload_indices": reload_indices,
            "stop_indices": stop_indices, "route_points": list(route.points),
            "visited": len(drop_indices), "unreachable": len(route.unreachable),
            "legs": {"total": len(leg_report), "dstar": statuses.count("dstar"),
                     "same_cell": statuses.count("same_cell"), "fallback": len(fallbacks)},
            "route_flooded_px_arrival": arrival_crossings, "route_flooded_px_current": current_crossings,
            "planner_sortie_battery": route.sortie_battery,
            "routed_sortie_battery": [s["battery_fraction"] for s in sorties]})
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

    # Airspace deconfliction repair pipeline
    coord_index = drop_index_by_point(safe_points)
    base_reach = single_drop_reach(bases, safe_points, routed_distance, headwind_mps=headwind,
                                   per_drop_kg=settings.PAYLOAD_PER_DROP_KG,
                                   reserve=settings.BATTERY_RESERVE_FRACTION)

    repairs, remaining_crossings = execute_overlap_repairs(
        plans_by_base, bases, selection, safe_points, base_reach, coord_index,
        combined_obstacle_mask, forecasts, clearance, routed_distance, headwind,
        plan_base, plan_assigned, image.shape[:2], meters_per_pixel=meters_per_pixel,
        max_rounds=5
    )

    # Drops assigned to a base that produced no route are neither visited nor otherwise reported: list them
    not_planned = {i: f"assigned to BASE {b + 1}, which produced no route (see the BASE {b + 1} log lines)"
                   for i, b in sorted(selection["assignment"].items()) if not plans_by_base.get(b)}
    summary["not_planned"] = {str(i): reason for i, reason in not_planned.items()}
    for i, reason in not_planned.items():
        logging.warning("  drop %2d %s NOT PLANNED: %s", i, safe_points[i], reason)

    plans = [plans_by_base[b] for b in sorted(plans_by_base) if plans_by_base[b]]
    if not plans:
        logging.error("mission infeasible - no base produced a route")
        summary["outcome"] = "NO_CANDIDATES_SAMPLED"
        summary["selection"] = selection
        write_summary(summary, session_dir)
        sys.exit(2)

    overlaps = []
    intersect_fn = _main_attr("polyline_intersections", polyline_intersections)
    for i, first in enumerate(plans):
        for second in plans[i + 1:]:
            hits = intersect_fn(first["full_path"], second["full_path"])
            if hits:
                overlaps.append({"bases": (first["base"], second["base"]), "points": len(hits)})
                logging.error("PATH OVERLAP: BASE %d and BASE %d routes intersect at %d point(s), e.g. %s",
                              first["base"], second["base"], len(hits), hits[:3])

    if overlaps:
        summary["path_overlaps"] = overlaps
        summary["overlap_repairs"] = repairs
        summary["outcome"] = "UNRESOLVED_PATH_OVERLAP"
        summary["selection"] = selection
        logging.error("MISSION INFEASIBLE: Unresolved path overlap between bases (%d conflict(s)). "
                      "Airspace deconfliction failed all repair tiers.", len(overlaps))
        log_deviation("Mission infeasible: unresolved path overlap between bases.")
        write_summary(summary, session_dir)
        sys.exit(2)

    if len(plans) > 1 and not overlaps:
        logging.info("PATH OVERLAP CHECK: no intersections between any two bases' routes (%d bases).", len(plans))

    # Export finalized waypoints and validator sidecars for each base
    for plan in plans:
        waypoint_file = session_dir / f"base{plan['base']}.waypoints"
        sidecar_file = session_dir / f"base{plan['base']}.sidecar.json"
        route_file = session_dir / f"base{plan['base']}.route.json"
        try:
            generate_mission_file(
                plan["full_path"], plan["drop_indices"], plan["home"],
                image_center_px, geo_center,
                meters_per_pixel, filename=str(waypoint_file),
                reload_indices=plan["reload_indices"],
                coordinate_fn=coordinate_fn
            )
            problems = validate_waypoints_file(str(waypoint_file))
        except Exception as e:
            problems = [f"generation failed: {e}"]
        plan["waypoint_file"] = str(waypoint_file)
        plan["waypoint_problems"] = problems
        if problems:
            logging.error("BASE %d WAYPOINT VALIDATION FAILED (%d problems): %s", plan["base"], len(problems), problems[:5])
        else:
            logging.info("BASE %d WAYPOINT VALIDATION: valid QGC WPL 110 (%s)", plan["base"], waypoint_file)

        sidecar = {
            "full_path": [list(map(float, p)) for p in plan["full_path"]],
            "geo_center": list(geo_center),
            "meters_per_pixel": meters_per_pixel,
            "image_center_px": list(image_center_px),
            "drop_indices": list(plan["drop_indices"]),
            "reload_indices": list(plan["reload_indices"]),
            "home": list(map(float, plan["home"])),
        }
        for sf in (sidecar_file, route_file):
            try:
                with open(sf, "w") as f:
                    json.dump(sidecar, f, indent=2)
            except Exception as e:
                logging.warning("Could not write sidecar %s: %s", sf, e)

    # For compatibility with single-mission checkers, link or copy base1 to data/output/enriched_drone_mission.waypoints
    if session_root == "data/output" and plans:
        try:
            b1_wp = session_dir / "base1.waypoints"
            if b1_wp.exists():
                shutil.copyfile(str(b1_wp), "data/output/enriched_drone_mission.waypoints")
        except Exception:
            pass

    # Base territories derived from final assignment
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
        "not_planned": summary.get("not_planned", {}),
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
        "stage_counts": {"raw_contours": raw_count, "zones": len(contours),
                         "zones_with_drop_point": zones_with_point, "drop_points": generated_drop_count,
                         "unreachable_by_any_base": unreachable_any,
                         "assigned": len(selection["assignment"]), "visited": visited},
    })
    # Add required outcome and selection keys for teammate's test format
    if not plans:
        outcome = "NO_CANDIDATES_SAMPLED"
    elif any(p.get("overlap_issues") for p in plans):
        outcome = "UNRESOLVED_PATH_OVERLAP"
    elif len(plans) == 1:
        outcome = "SINGLE_BASE"
    else:
        outcome = f"MULTI_BASE_{len(plans)}"
    summary["outcome"] = outcome
    summary["selection"] = selection
    if "home_outcome" not in summary:
        summary["home_outcome"] = outcome

    # Combined map: all bases and sorties
    try:
        rendered = display_multi_base_map(
            image.copy(), contours, pred_mask,
            [{"number": p["base"], "home": p["home"], "full_path": p["full_path"], "drop_indices": p["drop_indices"],
              "reload_indices": p["reload_indices"]} for p in plans],
            territories=territory_labels, save_path=str(session_dir / "mission_route_map.png"), show=show)
        map_ok, map_details = basemap_sanity(rendered, image)
        summary["map"] = {"ok": map_ok, **map_details}
        if map_ok:
            logging.info("MAP SANITY: basemap intact (dominant colour %.1f%%, correlation with source %.3f)",
                         map_details["dominant_fraction"] * 100, map_details["correlation"])
        else:
            logging.error("MAP SANITY FAILED: %s", map_details)
    except Exception as e:
        summary["map"] = {"ok": False, "error": str(e)}
        logging.warning(f"display_multi_base_map failed: {e}")

    write_summary(summary, session_dir)
    logging.info("Done. Session output: %s", session_dir)
    return summary
