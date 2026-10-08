"""Base placement: one base when it can reach every drop point, else the fewest bases (capped).

Reach uses the existing battery model on routed (around-flood) distances: a
drop is reachable from a base when a single-drop sortie (out loaded, back
empty, worst-case headwind on both legs) fits inside the usable battery, the
same test ``plan_mission_stops`` applies before committing a drop. Bases run
independently afterwards; there is no live multi-agent coordination.
"""
import logging

import numpy as np

from src.mission.constraints import leg_battery_fraction, score_home_candidates


def single_drop_reach(candidates, drops, distance_fn, *, headwind_mps, per_drop_kg, reserve):
    """For each candidate, the set of drop indices a single-drop sortie can serve."""
    limit = 1.0 - reserve
    reach = []
    for candidate in candidates:
        served = set()
        for index, drop in enumerate(drops):
            distance = distance_fn(candidate, drop)
            if not np.isfinite(distance):
                continue
            if (leg_battery_fraction(distance, per_drop_kg, headwind_mps)
                    + leg_battery_fraction(distance, 0.0, headwind_mps)) <= limit:
                served.add(index)
        reach.append(served)
    return reach


def _separated(point, others, min_separation_px):
    return all(np.hypot(point[0] - o[0], point[1] - o[1]) >= min_separation_px for o in others)


def _assign(chosen, reach, candidates, drops, distance_fn):
    """Each coverable drop -> the chosen base with the lowest routed distance that can reach it."""
    assignment = {}
    for index, drop in enumerate(drops):
        options = [(distance_fn(candidates[c], drop), b) for b, c in enumerate(chosen) if index in reach[c]]
        if options:
            assignment[index] = min(options)[1]
    return assignment


def select_bases(candidates, drops, distance_fn, obstacle_mask, *, headwind_mps, wind_from_deg,
                 per_drop_kg, payload_capacity_kg, reserve, meters_per_pixel, max_bases,
                 min_separation_px, refine_rounds=3):
    """Choose base positions among ``candidates`` and assign every drop to at most one base.

    Returns a dict with ``mode`` ("single", "multi" or "none"), ``bases``
    (positions), ``assignment`` ({drop index: base index}), ``unreachable``
    ({drop index: reason}) and ``reach_counts`` diagnostics.
    """
    reach = single_drop_reach(candidates, drops, distance_fn, headwind_mps=headwind_mps,
                              per_drop_kg=per_drop_kg, reserve=reserve)
    coverable = set().union(*reach) if reach else set()
    unreachable = {i: "unreachable by any base: no flood-clear base site can serve it within battery"
                   for i in range(len(drops)) if i not in coverable}
    if not coverable:
        return {"mode": "none", "bases": [], "assignment": {}, "unreachable": unreachable,
                "best_single_reach": 0, "coverable": 0}

    def score(positions, drop_indices):
        """Existing routed home scorer: minimum total time + battery cost over all sorties."""
        result = score_home_candidates(positions, [drops[i] for i in drop_indices], obstacle_mask,
                                       meters_per_pixel, headwind_mps=headwind_mps,
                                       wind_from_deg=wind_from_deg, payload_kg=0.0,
                                       per_drop_payload_kg=per_drop_kg,
                                       payload_capacity_kg=payload_capacity_kg,
                                       return_details=True, distance_fn=distance_fn)
        return result["scored"]

    best_single = max(len(r) for r in reach)
    if best_single == len(coverable):
        full = [c for c, r in zip(candidates, reach) if len(r) == len(coverable)]
        scored = score(full, sorted(coverable))
        base = scored[0][1] if scored else full[0]
        logging.info("BASE PLANNING: single base suffices - %d candidate sites reach all %d coverable drop "
                     "points; selected minimum routed-cost site %s.", len(full), len(coverable), base)
        return {"mode": "single", "bases": [tuple(base)], "assignment": {i: 0 for i in coverable},
                "unreachable": unreachable, "best_single_reach": best_single, "coverable": len(coverable)}

    logging.info("BASE PLANNING: best single site reaches %d of %d coverable drop points; adding bases "
                 "(cap %d, min separation %.0f px).", best_single, len(coverable), max_bases, min_separation_px)
    chosen, assignment = [], {}
    for _ in range(max_bases):
        # Greedy fewest-bases cover: add the separated site covering the most uncovered drops
        # (ties broken by lower routed distance to them) until all are covered or the cap is hit.
        uncovered = set(coverable) - set(assignment)
        before = (list(chosen), dict(assignment))
        while uncovered and len(chosen) < max_bases:
            best = None
            for c, candidate in enumerate(candidates):
                if c in chosen or not _separated(candidate, [candidates[k] for k in chosen], min_separation_px):
                    continue
                gain = reach[c] & uncovered
                if not gain:
                    continue
                cost = sum(distance_fn(candidate, drops[i]) for i in gain)
                key = (len(gain), -cost)
                if best is None or key > best[0]:
                    best = (key, c)
            if best is None:
                break
            chosen.append(best[1])
            uncovered -= reach[best[1]]

        # Per-cluster placement: re-place each base at the minimum routed-cost valid site that
        # still reaches all its assigned drops and keeps separation from the other bases.
        assignment = _assign(chosen, reach, candidates, drops, distance_fn)
        for _ in range(refine_rounds):
            changed = False
            for b in range(len(chosen)):
                mine = sorted(i for i, owner in assignment.items() if owner == b)
                if not mine:
                    continue
                others = [candidates[k] for j, k in enumerate(chosen) if j != b]
                valid = [c for c in range(len(candidates))
                         if set(mine) <= reach[c] and _separated(candidates[c], others, min_separation_px)]
                if not valid:
                    continue
                scored = score([candidates[c] for c in valid], mine)
                if scored:
                    best_site = valid[[candidates[c] for c in valid].index(tuple(scored[0][1]))]
                    if best_site != chosen[b]:
                        chosen[b] = best_site
                        changed = True
            new_assignment = _assign(chosen, reach, candidates, drops, distance_fn)
            changed = changed or new_assignment != assignment
            assignment = new_assignment
            if not changed:
                break
        # Drop bases left with no assigned drops (fewest bases), re-indexing the assignment;
        # freed capacity is reused by the next greedy pass for any drop still uncovered.
        used = sorted(set(assignment.values()))
        remap = {old: new for new, old in enumerate(used)}
        chosen = [chosen[old] for old in used]
        assignment = {i: remap[b] for i, b in assignment.items()}
        if not (set(coverable) - set(assignment)) or (chosen, assignment) == before:
            break
    for i in coverable:
        if i not in assignment:
            if len(chosen) >= max_bases:
                unreachable[i] = (f"unreachable by any base: the {max_bases}-base cap was reached before a "
                                  "base covering it could be placed")
            else:
                unreachable[i] = (f"unreachable by any base: every site that can reach it is closer than "
                                  f"{min_separation_px:.0f} px (MIN_BASE_SEPARATION_PX) to an existing base")
    mode = "multi" if len(chosen) > 1 else "single"
    return {"mode": mode, "bases": [tuple(candidates[c]) for c in chosen], "assignment": assignment,
            "unreachable": unreachable, "best_single_reach": best_single, "coverable": len(coverable)}
