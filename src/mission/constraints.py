"""Mission-level resource checks and safe-home candidate scoring."""
import logging
import numpy as np
from dataclasses import dataclass, field
from config import settings

def leg_battery_fraction(distance_m, payload_kg=0.0, headwind_mps=0.0):
    """Battery fraction consumed, using loaded endurance as the budget unit."""
    speed = max(settings.DRONE_SPEED_MPS, 0.1)
    flight_time_min = max(0.0, distance_m) / speed / 60.0
    flight_time_min *= (1.0 + max(0.0, payload_kg) * settings.PAYLOAD_TIME_FACTOR_PER_KG
                        + max(0.0, headwind_mps) * settings.WIND_PENALTY_PER_MPS)
    return flight_time_min / settings.DRONE_LOADED_ENDURANCE_MIN


def wind_kmh_to_mps(wind_speed_kmh):
    """Convert Open-Meteo's km/h wind speed to the battery model's m/s input."""
    return max(0.0, float(wind_speed_kmh)) / 3.6


def headwind_component_mps(wind_speed_mps, wind_from_deg, start, end):
    """Positive headwind component for a pixel-space leg, using met wind-from bearing."""
    delta = np.asarray(end, float) - np.asarray(start, float)
    if np.linalg.norm(delta) == 0:
        return 0.0
    # Image y increases south; convert course to compass heading clockwise from north.
    heading_deg = (np.degrees(np.arctan2(delta[0], -delta[1])) + 360.0) % 360.0
    component = float(wind_speed_mps) * np.cos(np.radians(float(wind_from_deg) - heading_deg))
    return max(0.0, component)


def score_home_candidates(candidates, drops, obstacle_mask, meters_per_pixel=2.0,
                          speed_mps=None, payload_kg=0.0, headwind_mps=0.0,
                          per_drop_payload_kg=None, payload_capacity_kg=None,
                          wind_from_deg=None,
                          return_details=False, distance_fn=None):
    """Score sampled homes by estimated time and payload-aware battery use.

    ``distance_fn(a, b)`` gives leg length in metres (e.g. routed around flood
    zones); by default straight-line pixel distance times ``meters_per_pixel``.
    """
    def leg_m(a, b):
        if distance_fn is not None:
            return distance_fn(a, b)
        return float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float))) * meters_per_pixel

    speed_mps = settings.DRONE_SPEED_MPS if speed_mps is None else speed_mps
    per_drop_payload_kg = settings.PAYLOAD_PER_DROP_KG if per_drop_payload_kg is None else per_drop_payload_kg
    payload_capacity_kg = settings.PAYLOAD_CAPACITY_KG if payload_capacity_kg is None else payload_capacity_kg
    h, w = obstacle_mask.shape[:2]
    scored, rejected, sortie_metrics = [], [], []
    evaluated = 0
    dry_candidates = 0
    for candidate in candidates:
        x, y = map(int, candidate)
        if not (0 <= x < w and 0 <= y < h):
            continue
        if obstacle_mask[y, x] > 0:
            continue
        dry_candidates += 1
        evaluated += 1
        # Greedily build the same kind of closed sorties as the mission planner:
        # close a sortie before payload capacity or the battery reserve is exceeded.
        # Match the actual TSP planner: nearest-neighbor order from this home.
        pending = list(drops)
        ordered = []
        cursor_for_order = np.asarray(candidate, float)
        while pending:
            nearest_index = min(range(len(pending)), key=lambda i:
                leg_m(cursor_for_order, pending[i]))
            target = pending.pop(nearest_index)
            ordered.append(target)
            cursor_for_order = np.asarray(target, float)
        pending = ordered
        sortie_battery = []
        unreachable_battery = []
        distance_m = 0.0
        reachable_drops = []
        unreachable_drops = []
        max_drops_capacity = max(1, int(payload_capacity_kg // max(per_drop_payload_kg, 1e-9)))
        reserve_threshold = 1.0 - settings.BATTERY_RESERVE_FRACTION

        while pending:
            # Check battery cumulative total BEFORE committing a leg to the current sortie.
            # Greedily expand the sortie drops prefix as long as total battery (including return to base) <= reserve_threshold.
            best_sortie_drops = []
            best_sortie_battery = 0.0
            best_sortie_dist = 0.0
            best_leg_cumulatives = []

            for num_drops in range(1, min(len(pending), max_drops_capacity) + 1):
                proposed_drops = pending[:num_drops]
                curr_cursor = np.asarray(candidate, float)
                curr_carried = payload_kg + num_drops * per_drop_payload_kg
                total_used = 0.0
                sortie_dist = 0.0
                leg_cumulatives = []

                for d in proposed_drops:
                    d_pt = np.asarray(d, float)
                    leg_d = leg_m(curr_cursor, d_pt)
                    leg_w = (headwind_component_mps(headwind_mps, wind_from_deg, curr_cursor, d_pt)
                             if wind_from_deg is not None else headwind_mps)
                    leg_drain = leg_battery_fraction(leg_d, curr_carried, leg_w)
                    total_used += leg_drain
                    sortie_dist += leg_d
                    leg_cumulatives.append(total_used)
                    curr_carried = max(payload_kg, curr_carried - per_drop_payload_kg)
                    curr_cursor = d_pt

                ret_d = leg_m(curr_cursor, candidate)
                ret_w = (headwind_component_mps(headwind_mps, wind_from_deg, curr_cursor, candidate)
                         if wind_from_deg is not None else headwind_mps)
                ret_drain = leg_battery_fraction(ret_d, payload_kg, ret_w)
                total_used += ret_drain
                sortie_dist += ret_d

                if total_used <= reserve_threshold:
                    best_sortie_drops = proposed_drops
                    best_sortie_battery = total_used
                    best_sortie_dist = sortie_dist
                    best_leg_cumulatives = leg_cumulatives
                else:
                    # Leg would push cumulative use past reserve threshold (80%).
                    # Close sortie before this drop so it starts the next sortie.
                    break

            if not best_sortie_drops:
                # Even 1 single drop cannot be served within battery reserve from this home candidate.
                target_drop = pending[0]
                d_pt = np.asarray(target_drop, float)
                leg_d = leg_m(candidate, d_pt)
                w1 = (headwind_component_mps(headwind_mps, wind_from_deg, candidate, d_pt)
                      if wind_from_deg is not None else headwind_mps)
                w2 = (headwind_component_mps(headwind_mps, wind_from_deg, d_pt, candidate)
                      if wind_from_deg is not None else headwind_mps)
                out_drain = leg_battery_fraction(leg_d, payload_kg + per_drop_payload_kg, w1)
                ret_drain = leg_battery_fraction(leg_d, payload_kg, w2)
                single_battery = out_drain + ret_drain
                unreachable_battery.append(single_battery)
                distance_m += 2.0 * leg_d
                unreachable_drops.append(tuple(map(int, target_drop)))
                if evaluated == 1:
                    logging.info(
                        "Unreachable drop %s (single-drop sortie, not flown): out_leg=%.2f%%, rtb=%.2f%%, total=%.2f%% exceeds reserve_limit=%.1f%%",
                        tuple(target_drop), out_drain * 100.0, ret_drain * 100.0,
                        single_battery * 100.0, reserve_threshold * 100.0)
                pending.pop(0)
            else:
                sortie_battery.append(best_sortie_battery)
                distance_m += best_sortie_dist
                reachable_drops.extend([tuple(map(int, d)) for d in best_sortie_drops])
                if evaluated == 1:
                    logging.info(
                        "Sortie %d committed: %d drops, cumulative leg totals=%s, total_with_rtb=%.2f%% (reserve_limit=%.1f%%)",
                        len(sortie_battery), len(best_sortie_drops),
                        [round(x * 100.0, 2) for x in best_leg_cumulatives],
                        best_sortie_battery * 100.0, reserve_threshold * 100.0)
                pending = pending[len(best_sortie_drops):]

        max_sortie_battery = max(sortie_battery, default=0.0)
        total_mission_drain = sum(sortie_battery)
        sortie_metrics.append({"candidate": tuple(candidate),
            "sortie_count": len(sortie_battery),
            "sortie_drain_fractions": sortie_battery,
            "unreachable_drop_drain_fractions": unreachable_battery,
            "total_mission_drain_fraction": total_mission_drain,
            "recharge_cycles": max(0, len(sortie_battery) - 1),
            "reachable_drops": reachable_drops,
            "unreachable_drops": unreachable_drops,
            "reachable_count": len(reachable_drops)})
        hours = distance_m / max(speed_mps, .1) / 3600.0
        if unreachable_drops:
            # Rejected: at least one drop needs more than the usable battery
            # even alone. Report the worst single-drop requirement.
            worst = max(unreachable_battery + sortie_battery)
            rejected.append((hours + total_mission_drain + sum(unreachable_battery),
                             tuple(candidate), hours, worst))
        else:
            scored.append((hours + total_mission_drain, tuple(candidate), hours, max_sortie_battery))
        if evaluated == 1:
            logging.info("Battery audit candidate=%s per_sortie_battery_pct=%s unreachable_single_drop_pct=%s total_mission_cumulative_drain_pct=%.3f recharge_cycles=%d reserve_threshold_per_sortie_pct=%.1f",
                tuple(candidate), [round(x * 100.0, 3) for x in sortie_battery],
                [round(x * 100.0, 3) for x in unreachable_battery],
                total_mission_drain * 100.0, max(0, len(sortie_battery) - 1),
                (1.0 - settings.BATTERY_RESERVE_FRACTION) * 100.0)
    details = {"sampled": len(candidates), "dry_candidates": dry_candidates, "evaluated": evaluated,
               "rejected": sorted(rejected), "scored": sorted(scored),
               "sortie_metrics": sortie_metrics}
    return details if return_details else sorted(scored)

def plan_payload_batches(drop_count, capacity=None, per_drop_kg=None):
    capacity = settings.PAYLOAD_CAPACITY_KG if capacity is None else capacity
    per_drop_kg = settings.PAYLOAD_PER_DROP_KG if per_drop_kg is None else per_drop_kg
    if drop_count < 0 or capacity <= 0 or per_drop_kg <= 0:
        raise ValueError("drop count and payload parameters must be positive")
    drops_per_load = max(1, int(capacity // per_drop_kg))
    return [list(range(i, min(i + drops_per_load, drop_count))) for i in range(0, drop_count, drops_per_load)]


def log_deviation(message):
    logging.warning("FORCED MISSION DEVIATION: %s", message)


@dataclass
class PlannedRoute:
    points: list
    drop_indices: list
    reload_indices: list
    unreachable: list
    remaining_battery: list
    deviations: list
    sortie_battery: list = field(default_factory=list)


def _routed_distance(routed, home, start, end, drop_id, leg_number, straight_m):
    """Leg length: routed when known for this exact leg, else straight line.

    Dict keys ``("pair", from_xy, to_xy)`` give the routed length of that exact
    leg. Legacy keys (``drop_id`` / ``("home", drop_id)``, distance into a drop
    from its previous route point) are honoured only when no pair keys exist.
    A list gives one routed distance per committed outbound leg, in order.
    """
    if isinstance(routed, dict):
        pair = ("pair", start, end)
        if pair in routed:
            return float(routed[pair])
        has_pairs = any(isinstance(k, tuple) and k and k[0] == "pair" for k in routed)
        if drop_id is not None and not has_pairs:
            if start == home and ("home", drop_id) in routed:
                return float(routed[("home", drop_id)])
            if drop_id in routed:
                return float(routed[drop_id])
        return straight_m
    if routed is not None and drop_id is not None and leg_number < len(routed):
        return float(routed[leg_number])
    return straight_m


def plan_mission_stops(home, drop_points, forecasts, current_mask, *, capacity=None,
                       per_drop_kg=None, meters_per_pixel=2.0, speed_mps=None,
                       headwind_mps=0.0, initial_battery=None, reserve=None,
                       routed_leg_distances_m=None):
    """Build an executable stop sequence of closed sorties, checking flooding and battery.

    Each sortie departs HOME carrying one package per planned drop, delivers
    them in TSP order (getting lighter after each drop) and returns to HOME.
    A drop joins the current sortie only if the whole sortie, including the
    return leg from that drop, stays within the battery reserve and payload
    capacity; otherwise the sortie is closed with a return/reload and the drop
    starts the next sortie. A drop that cannot be served by a single-drop
    sortie from a full battery is marked unreachable. Flooded drops are moved
    to the front once; if still flooded at arrival they are marked unreachable.

    ``headwind_mps`` is applied to every leg (worst case, direction unknown).
    """
    from src.weather.flood_spread import obstacle_mask_at, forecast_at
    capacity = settings.PAYLOAD_CAPACITY_KG if capacity is None else capacity
    per_drop_kg = settings.PAYLOAD_PER_DROP_KG if per_drop_kg is None else per_drop_kg
    speed_mps = settings.DRONE_SPEED_MPS if speed_mps is None else speed_mps
    initial_battery = settings.INITIAL_BATTERY_FRACTION if initial_battery is None else initial_battery
    reserve = settings.BATTERY_RESERVE_FRACTION if reserve is None else reserve
    if not drop_points:
        msg = "mission infeasible — no reachable drop points"
        logging.error(msg)
        return PlannedRoute([tuple(map(int, home))], [], [], [], [float(initial_battery)], [msg])
    home = tuple(map(int, home))
    targets = [tuple(map(int, p)) for p in drop_points]
    speed = max(speed_mps, .1)
    routed = routed_leg_distances_m

    # Predict arrival at each target using its scheduled position and promote a
    # target if its first estimate is flooded at the original visit time.
    ranked = []
    cursor = home
    elapsed_s = 0.0
    first_forecast_h = min((t for t in forecasts if t > 0), default=float("inf"))
    for idx, target in enumerate(targets):
        d = float(np.linalg.norm(np.asarray(target, float) - np.asarray(cursor, float))) * meters_per_pixel
        arrival_h = elapsed_s / 3600.0 + d / speed / 3600.0
        # Use current conditions for arrivals before the first forecast frame;
        # never promote based on flooding that only appears later.
        query_h = arrival_h if arrival_h >= first_forecast_h else 0.0
        mask = obstacle_mask_at(forecasts, query_h, current_mask)
        x, y = target
        ranked.append((idx, bool(0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0)))
        elapsed_s += d / speed
        cursor = target
    # Move forecast-flooded visits ahead; the subsequent actual arrival test is authoritative.
    order = [i for i, flooded in ranked if flooded] + [i for i, flooded in ranked if not flooded]

    state = {"remaining": float(initial_battery), "clock_s": 0.0, "committed_legs": 0}
    points = [home]
    drops, reloads, unreachable, deviations = [], [], [], []
    battery_history = [state["remaining"]]
    sortie_battery = []

    def simulate(sortie):
        """Fly ``sortie`` (drop ids) from HOME and back, starting at the current mission time."""
        carried = per_drop_kg * len(sortie)
        cursor, clock, total, legs = home, state["clock_s"], 0.0, []
        for k, drop_id in enumerate(sortie):
            target = targets[drop_id]
            straight = float(np.linalg.norm(np.subtract(target, cursor))) * meters_per_pixel
            distance = _routed_distance(routed, home, cursor, target, drop_id,
                                        state["committed_legs"] + k, straight)
            drain = leg_battery_fraction(distance, carried, headwind_mps)
            clock += distance / speed
            total += drain
            legs.append({"drop": drop_id, "distance_m": distance, "drain": drain,
                         "payload_kg": carried, "arrival_h": clock / 3600.0})
            carried = max(0.0, carried - per_drop_kg)
            cursor = target
        straight = float(np.linalg.norm(np.subtract(home, cursor))) * meters_per_pixel
        rtb_m = _routed_distance(routed, home, cursor, home, None, 0, straight)
        rtb = leg_battery_fraction(rtb_m, carried, headwind_mps)
        return {"legs": legs, "rtb_m": rtb_m, "rtb_drain": rtb, "total": total + rtb,
                "end_clock_s": clock + rtb_m / speed}

    def flooded_at_arrival(drop_id, arrival_h):
        arrival_mask, forecast_info = forecast_at(forecasts, arrival_h, current_mask)
        tx, ty = targets[drop_id]
        inside = 0 <= ty < arrival_mask.shape[0] and 0 <= tx < arrival_mask.shape[1]
        lo, hi = forecast_info["lower_hours"], forecast_info["upper_hours"]
        lo_value = int(forecasts[lo][ty, tx]) if inside and lo in forecasts else None
        hi_value = int(forecasts[hi][ty, tx]) if inside and hi in forecasts else None
        current_value = int(current_mask[ty, tx]) if inside and current_mask is not None else None
        sampled_value = int(arrival_mask[ty, tx]) if inside else None
        is_flooded = (not inside) or bool(arrival_mask[ty, tx] > 0)
        logging.info("Drop forecast check drop=%d point=(%d,%d) arrival_h=%.4f frame=%s [%.4f,%.4f] alpha=%s current_pixel=%s bracket_pixels=[%s,%s] raw_pixel=%s flooded=%s",
            drop_id, tx, ty, arrival_h, forecast_info["source"],
            lo if lo is not None else -1, hi if hi is not None else -1,
            f"{forecast_info['alpha']:.3f}" if forecast_info["alpha"] is not None else "n/a",
            current_value, lo_value, hi_value, sampled_value, is_flooded)
        return is_flooded

    def reload(reason):
        if points[-1] != home:
            points.append(home)
        reloads.append(len(points) - 1)
        deviations.append(reason); log_deviation(reason)
        state["remaining"] = 1.0
        battery_history.append(1.0)
        points.append(home)  # explicit re-departure waypoint in the route

    def commit(sim):
        level = state["remaining"]
        for leg in sim["legs"]:
            level -= leg["drain"]
            points.append(targets[leg["drop"]])
            drops.append(len(points) - 1)
            battery_history.append(level)
            logging.info("Plan leg to drop %d at (%d, %d): dist=%.1fm, payload=%.2fkg, leg_drain=%.2f%%, "
                         "remaining=%.2f%% (reserve=%.1f%%)", leg["drop"], *targets[leg["drop"]],
                         leg["distance_m"], leg["payload_kg"], leg["drain"] * 100.0, level * 100.0,
                         reserve * 100.0)
        level -= sim["rtb_drain"]
        battery_history.append(level)
        used = state["remaining"] - level
        sortie_battery.append(used)
        logging.info("Sortie %d committed: drops=%s, outbound=%.2f%%, rtb=%.1fm/%.2f%%, "
                     "total_with_rtb=%.2f%% (start=%.1f%%, end=%.1f%%, reserve=%.1f%%)",
                     len(sortie_battery), [leg["drop"] for leg in sim["legs"]],
                     (used - sim["rtb_drain"]) * 100.0, sim["rtb_m"], sim["rtb_drain"] * 100.0,
                     used * 100.0, state["remaining"] * 100.0, level * 100.0, reserve * 100.0)
        state["remaining"] = level
        state["committed_legs"] += len(sim["legs"])
        state["clock_s"] = sim["end_clock_s"]
        points.append(home)

    sortie, sortie_sim = [], None
    for idx in order:
        if ranked[idx][1]:
            msg = f"Drop {idx} reprioritized earlier because forecast showed flooding at initial arrival estimate."
            deviations.append(msg); log_deviation(msg)
        if sortie and per_drop_kg * (len(sortie) + 1) > capacity + 1e-9:
            commit(sortie_sim); sortie, sortie_sim = [], None
            reload(f"Payload capacity reached before drop {idx}; inserted return-to-home reload and re-departure.")
        # Pre-commit check: the sortie including this drop AND the return leg
        # from it must fit inside the usable battery.
        sim = simulate(sortie + [idx])
        if sortie and state["remaining"] - sim["total"] < reserve:
            alone = simulate([idx])
            if 1.0 - alone["total"] < reserve:
                # Not servable even alone from a full battery: do not close the
                # current sortie (a reload would buy nothing).
                unreachable.append(idx)
                msg = (f"Drop {idx} unreachable: single-drop sortie needs {alone['total']:.0%} battery "
                       f"including return, violating the {reserve:.0%} reserve.")
                deviations.append(msg); log_deviation(msg)
                continue
            commit(sortie_sim); sortie, sortie_sim = [], None
            reload(f"Battery reserve would be breached before drop {idx}; inserted return-to-home and reload.")
            sim = simulate([idx])
        if state["remaining"] - sim["total"] < reserve:
            if state["remaining"] < 1.0 - 1e-9 and 1.0 - sim["total"] >= reserve:
                reload(f"Battery reserve would be breached before drop {idx}; recharged at HOME before departure.")
            else:
                unreachable.append(idx)
                msg = (f"Drop {idx} unreachable: single-drop sortie needs {sim['total']:.0%} battery "
                       f"including return, violating the {reserve:.0%} reserve.")
                deviations.append(msg); log_deviation(msg)
                continue
        arrival_h = sim["legs"][-1]["arrival_h"]
        if flooded_at_arrival(idx, arrival_h):
            unreachable.append(idx)
            msg = f"Drop {idx} unreachable: point flooded at routed arrival +{arrival_h:.2f}h."
            deviations.append(msg); log_deviation(msg)
            continue
        sortie, sortie_sim = sortie + [idx], sim
        if per_drop_kg * len(sortie) > capacity + 1e-9:
            raise AssertionError("Payload capacity exceeded before reload insertion")
    if sortie:
        commit(sortie_sim)
    if points[-1] != home:
        points.append(home)
    if not drops:
        msg = "mission infeasible — no reachable drop points"
        logging.error(msg)
        deviations.append(msg)
    return PlannedRoute(points, drops, reloads, unreachable, battery_history, deviations, sortie_battery)


def sortie_battery_report(full_path, drop_indices, reload_indices, *, meters_per_pixel=2.0,
                          per_drop_kg=None, headwind_mps=0.0):
    """Per-sortie battery use measured along the final routed ``full_path``.

    Sorties end at each reload index and at the final point. Payload starts at
    one package per drop in the sortie and falls by one package at each drop.
    This is the authoritative post-routing check: it uses the exported D* Lite
    geometry rather than the planner's distance estimates.
    """
    per_drop_kg = settings.PAYLOAD_PER_DROP_KG if per_drop_kg is None else per_drop_kg
    drop_set = set(drop_indices)
    bounds = sorted({int(i) for i in reload_indices}) + [len(full_path) - 1]
    report, start = [], 0
    for end in bounds:
        n_drops = sum(1 for i in range(start, end + 1) if i in drop_set)
        if end > start and n_drops:
            carried, drain, distance = per_drop_kg * n_drops, 0.0, 0.0
            for i in range(start, end):
                leg_m = float(np.linalg.norm(np.subtract(full_path[i + 1], full_path[i]))) * meters_per_pixel
                drain += leg_battery_fraction(leg_m, carried, headwind_mps)
                distance += leg_m
                if i + 1 in drop_set:
                    carried = max(0.0, carried - per_drop_kg)
            report.append({"start_index": start, "end_index": end, "drops": n_drops,
                           "distance_m": distance, "battery_fraction": drain})
        start = end + 1
    return report
