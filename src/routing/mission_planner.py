"""
Mission planner: turns flood-blocked drop zones into an obstacle-avoiding,
time-aware, battery/payload-constrained set of sorties from an optimised home.

Responsibilities
----------------
* Task 2 -- D* Lite legs are planned against the *predicted flood contour at the
  estimated arrival time of each leg* (via :class:`FloodTimeline`), not against
  a single mission-start snapshot.
* Task 3 -- battery drains per leg as a function of distance + current payload +
  headwind (see :mod:`src.mission.constraints`); payload depletes on each drop;
  a greedy return-to-base *reload leg* is inserted when the next drop would not
  fit the remaining load, and a return-to-base *battery* reroute is triggered
  when the projected remaining route would exceed the safety reserve.
* Task 4 -- home/base selection minimises a weighted (time + battery) cost over
  a grid of safe, non-flooded candidate pixels; alternatives are logged.
* Task 5 -- graceful degradation instead of crashes or silent wrong routes:
  zones flooded out before their ETA are skipped/reprioritised, un-reachable
  zones are dropped with a loud log, and an infeasible mission is reported as
  such.

The route is emitted as one or more *sorties* (each starts/ends at home and is
a standalone, battery-feasible flight), because reloading payload or swapping a
battery requires landing at base.
"""
import logging
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.mission.constraints import (
    DroneSpec,
    estimate_straight_leg,
)
from src.routing.pathfinding import (
    nearest_neighbor_tsp,
    plan_timed_leg,
)
from src.weather.flood_predictor import FloodTimeline

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
@dataclass
class DropZone:
    zone_id: str
    pixel: Tuple[int, int]
    weight_kg: float = 0.5
    critical: bool = False          # degraded plans try harder to keep these

    def __hash__(self):  # pragma: no cover - trivial
        return hash(self.zone_id)


@dataclass
class Sortie:
    number: int
    home_px: Tuple[int, int]
    path: List[Tuple[int, int]] = dc_field(default_factory=list)      # px, starts & ends at home
    drop_indices: List[int] = dc_field(default_factory=list)          # indexes into path
    drop_zone_ids: List[str] = dc_field(default_factory=list)
    energy_wh: float = 0.0
    duration_min: float = 0.0
    payload_kg: float = 0.0           # total delivered on this sortie
    start_clock_min: float = 0.0
    end_clock_min: float = 0.0
    closed_reason: str = "completed"
    notes: List[str] = dc_field(default_factory=list)


@dataclass
class MissionPlan:
    home_px: Tuple[int, int]
    zones: List[DropZone]
    sorties: List[Sortie] = dc_field(default_factory=list)
    skipped: List[Tuple[DropZone, str]] = dc_field(default_factory=list)
    notes: List[str] = dc_field(default_factory=list)
    total_energy_wh: float = 0.0      # sum across sorties (batteries used)
    total_duration_min: float = 0.0   # wall-clock incl. reloads at base
    served_ids: List[str] = dc_field(default_factory=list)

    @property
    def status(self) -> str:
        if self.served_ids:
            return "partial" if self.skipped else "ok"
        return "infeasible"

    def served_kg(self) -> float:
        return sum(z.weight_kg for z in self.zones if z.zone_id in self.served_ids)


# --------------------------------------------------------------------------- #
class MissionPlanner:
    """Config + orchestration.  One instance is reused to score home candidates."""

    def __init__(
        self,
        spec: Optional[DroneSpec] = None,
        weather: Optional[Dict[str, float]] = None,
        timeline: Optional[FloodTimeline] = None,
        static_obstacle_mask: Optional[np.ndarray] = None,
        meters_per_pixel: float = 2.0,
        downsample_factor: int = 5,
        drop_service_time_s: float = 45.0,   # loiter/descend/servo/ascend per drop
        reload_time_min: float = 5.0,        # payload reload / battery swap at base
        time_weight: float = 1.0,            # Task 4 cost: minutes
        energy_weight: float = 1.0,          # Task 4 cost: Wh
        skip_zone_penalty: float = 1e6,      # dominant: prefer homes serving all zones
    ):
        self.spec = spec or DroneSpec()
        self.weather = weather or {
            "precipitation": 0.0,
            "wind_speed_10m": 0.0,
            "wind_direction_10m": 0.0,
        }
        if timeline is None and static_obstacle_mask is not None:
            # Constant (static) obstacle world -> a single-frame timeline.
            timeline = FloodTimeline([0.0], [static_obstacle_mask], weather=self.weather,
                                     source="static")
        self.timeline = timeline
        if self.timeline is None:
            raise ValueError("MissionPlanner needs a flood timeline or a static mask.")
        self.meters_per_pixel = meters_per_pixel
        self.downsample_factor = downsample_factor
        self.drop_service_time_s = drop_service_time_s
        self.reload_time_min = reload_time_min
        self.time_weight = time_weight
        self.energy_weight = energy_weight
        self.skip_zone_penalty = skip_zone_penalty

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    def _service_min(self) -> float:
        return self.drop_service_time_s / 60.0

    def _home_is_safe(self, home_px: Tuple[int, int]) -> bool:
        """Home must be dry now AND at the end of the forecast horizon."""
        t0 = self.timeline.obstacle_mask_at(0.0)
        tend = self.timeline.obstacle_mask_at(self.timeline.times_min[-1])
        x, y = int(home_px[0]), int(home_px[1])
        if y < 0 or x < 0 or y >= t0.shape[0] or x >= t0.shape[1]:
            return False
        return t0[y, x] == 0 and tend[y, x] == 0

    def _blocked_after(self, zone: DropZone) -> float:
        if self.timeline is None:
            return float("inf")
        return self.timeline.blocked_after_min(zone.pixel)

    # ------------------------------------------------------------------ #
    # Ordering
    # ------------------------------------------------------------------ #
    def _nn_order(self, home_px, zones: List[DropZone]) -> List[int]:
        pts = [z.pixel for z in zones]
        order, _ = nearest_neighbor_tsp(pts, home=home_px)
        return list(order)

    def _order_for_flood(self, home_px, zones: List[DropZone]) -> Tuple[List[int], List[Tuple[int, str]]]:
        """Task 5b: zones threatened by the predicted spread move earlier in the
        TSP order; zones already underwater at t=0 (or too heavy for the drone)
        are dropped here with a reason."""
        order = self._nn_order(home_px, zones)
        skipped: List[Tuple[int, str]] = []
        active = []
        for i in order:
            z = zones[i]
            if z.weight_kg > self.spec.payload_capacity_kg:
                skipped.append((i, "package heavier than drone capacity"))
                continue
            if self._blocked_after(z) == 0.0:
                skipped.append((i, "already flooded at mission start"))
                continue
            active.append(i)

        def eta_along(seq: List[int]) -> List[float]:
            etas, t = [], 0.0
            pos = home_px
            for i in seq:
                est = estimate_straight_leg(pos, zones[i].pixel, self.meters_per_pixel,
                                            self.spec, self.weather)
                t += est["duration_s"] / 60.0 + self._service_min()
                etas.append(t)
                pos = zones[i].pixel
            return etas

        changed = True
        guard = 0
        while changed and guard < 2 * len(active) + 1:
            changed = False
            guard += 1
            etas = eta_along(active)
            for idx, i in enumerate(active):
                block = self._blocked_after(zones[i])
                if block == float("inf") or etas[idx] <= block:
                    continue
                if idx == 0:
                    # Already earliest possible -> cannot be helped here; the
                    # sortie simulation will skip it with a clear reason.
                    continue
                # Move this threatened zone as early as it needs to be.
                best = None
                for j in range(idx):
                    trial = active[:j] + [i] + active[j:idx] + active[idx + 1:]
                    if eta_along(trial)[j] <= block:
                        best = j
                        break
                if best is None:
                    best = 0
                active = active[:best] + [i] + active[best:idx] + active[idx + 1:]
                changed = True
                break

        return active, skipped

    # ------------------------------------------------------------------ #
    # Sortie simulation (Tasks 3 & 5)
    # ------------------------------------------------------------------ #
    def plan_for_home(self, home_px, zones: List[DropZone]) -> MissionPlan:
        plan = MissionPlan(home_px=home_px, zones=list(zones))
        if not self._home_is_safe(home_px):
            plan.notes.append(
                f"Home candidate {home_px} is flooded now or by the end of the "
                "forecast horizon -- infeasible."
            )
            return plan

        ordered, skipped_now = self._order_for_flood(home_px, zones)
        for i, why in skipped_now:
            plan.skipped.append((zones[i], why))

        clock = 0.0
        sortie_no = 0
        cap = self.spec.payload_capacity_kg
        usable = self.spec.usable_energy_wh
        pending = list(ordered)          # zone indices still to serve

        while pending:
            # ---- capacity batch: greedy load until the hold is full ------
            batch: List[int] = []
            tot = 0.0
            while pending:
                w = zones[pending[0]].weight_kg
                if w > cap:
                    plan.skipped.append(
                        (zones[pending.pop(0)], "package heavier than drone capacity"))
                    continue
                if tot + w > cap + 1e-9:
                    break                 # next package needs a reload -> next sortie
                tot += w
                batch.append(pending.pop(0))
            if not batch:
                continue

            # ---------------- open a new sortie ------------------------- #
            sortie_no += 1
            sortie = Sortie(number=sortie_no, home_px=home_px)
            sortie.path = [home_px]
            sortie.start_clock_min = clock
            remaining_load = tot          # these packages are on board at take-off
            energy = 0.0
            pos = home_px
            served_here = False
            sortie_notes: List[str] = []

            k = 0
            while k < len(batch):
                zone = zones[batch[k]]
                w = zone.weight_kg

                # ---- battery feasibility for the projected route --------
                in_est = estimate_straight_leg(pos, zone.pixel, self.meters_per_pixel,
                                               self.spec, self.weather, payload_kg=remaining_load)
                aft = remaining_load - w
                ret_est = estimate_straight_leg(zone.pixel, home_px, self.meters_per_pixel,
                                                self.spec, self.weather, payload_kg=aft)
                svc_energy = self.spec.hover_energy_wh(self.drop_service_time_s, remaining_load)
                projected = energy + in_est["energy_wh"] + svc_energy + ret_est["energy_wh"]

                if projected > usable:
                    if served_here:
                        sortie.closed_reason = (
                            f"battery reserve: continuing to {zone.zone_id} would exceed "
                            f"{self.spec.usable_battery_fraction * 100:.0f}% of capacity")
                        break
                    plan.skipped.append(
                        (zone, f"beyond round-trip battery range from home "
                               f"(projected {projected:.1f} Wh > {usable:.1f} Wh reserve)"))
                    batch.pop(k)
                    continue

                # ---- flood blocking at projected arrival -----------------
                block = self._blocked_after(zone)
                arr_est = clock + in_est["duration_s"] / 60.0
                if block != float("inf") and arr_est > block:
                    plan.skipped.append(
                        (zone, f"predicted flood reaches it at t={block:.0f} min before "
                               f"estimated arrival t={arr_est:.0f} min"))
                    batch.pop(k)
                    continue

                # ---- time-aware D* Lite inbound leg ----------------------
                leg = plan_timed_leg(pos, zone.pixel, self.timeline, clock, self.spec,
                                     self.weather, self.meters_per_pixel,
                                     self.downsample_factor)
                if not leg["ok"]:
                    plan.skipped.append(
                        (zone, "no obstacle-avoiding path found to the drop point"))
                    batch.pop(k)
                    continue

                # Re-check battery against the *actual* (longer) detour path.
                in_energy = self.spec.leg_energy_wh(leg["distance_m"], remaining_load,
                                                    leg["headwind_mps"])
                projected2 = energy + in_energy + svc_energy + ret_est["energy_wh"]
                if projected2 > usable:
                    if served_here:
                        sortie.closed_reason = (
                            f"battery reserve after obstacle detour before {zone.zone_id}")
                        break
                    plan.skipped.append(
                        (zone, f"beyond round-trip battery range from home "
                               f"(projected {projected2:.1f} Wh > {usable:.1f} Wh reserve)"))
                    batch.pop(k)
                    continue

                if block != float("inf") and leg["arrival_min"] > block:
                    plan.skipped.append(
                        (zone, f"predicted flood reaches it at t={block:.0f} min before "
                               f"actual arrival t={leg['arrival_min']:.0f} min"))
                    batch.pop(k)
                    continue

                # ---------------- commit this drop ------------------------ #
                sortie.path.extend(leg["path"][1:])
                sortie.drop_indices.append(len(sortie.path) - 1)
                sortie.drop_zone_ids.append(zone.zone_id)
                sortie_notes.append(
                    f"leg to {zone.zone_id} planned against flood contour @ "
                    f"t={leg['obstacle_time_min']:.0f} min (ETA {leg['arrival_min']:.0f} min)")
                energy += in_energy + svc_energy
                clock = leg["arrival_min"] + self._service_min()
                remaining_load = aft
                pos = zone.pixel
                served_here = True
                sortie.payload_kg += w
                plan.served_ids.append(zone.zone_id)
                k += 1

            # ---------------- close / return home ------------------------- #
            if served_here:
                ret = plan_timed_leg(pos, home_px, self.timeline, clock, self.spec,
                                     self.weather, self.meters_per_pixel,
                                     self.downsample_factor)
                if ret["ok"]:
                    ret_energy = self.spec.leg_energy_wh(ret["distance_m"], remaining_load,
                                                         ret["headwind_mps"])
                    sortie.path.extend(ret["path"][1:])
                    energy += ret_energy
                    clock = ret["arrival_min"]
                else:
                    plan.notes.append(
                        f"Sortie {sortie_no}: no path found back to home -- "
                        "route would be unsafe; flagging mission as degraded.")
                sortie.energy_wh = energy
                sortie.end_clock_min = clock
                sortie.duration_min = clock - sortie.start_clock_min
                plan.sorties.append(sortie)
                sortie.notes.extend(sortie_notes)
                plan.notes.extend(sortie_notes)
                if sortie_no > 1:
                    plan.notes.append(
                        f"Sortie {sortie_no} begins after a base turnaround "
                        f"(payload reload / battery swap) at t={sortie.start_clock_min:.1f} min.")
                if sortie.closed_reason != "completed":
                    plan.notes.append(
                        f"Constraint deviation: sortie {sortie_no} ended early "
                        f"({sortie.closed_reason}) -- returning to base early.")
                if k < len(batch):          # undelivered packages return to base
                    pending = batch[k:] + pending
            elif k < len(batch):
                pending = batch[k:] + pending

            # Only a real turnaround (return to base) needs the reload dwell.
            if served_here and pending:
                clock += self.reload_time_min

        plan.total_energy_wh = sum(s.energy_wh for s in plan.sorties)
        plan.total_duration_min = (plan.sorties[-1].end_clock_min
                                   if plan.sorties else 0.0)
        if not plan.sorties and not plan.skipped:
            plan.notes.append("Mission infeasible: no drop zone could be served.")
        return plan

    # ------------------------------------------------------------------ #
    # Task 4 -- home optimisation
    # ------------------------------------------------------------------ #
    def _candidate_homes(self, zones: List[DropZone], max_candidates: int = 24) -> List[Tuple[int, int]]:
        t0 = self.timeline.obstacle_mask_at(0.0)
        tend = self.timeline.obstacle_mask_at(self.timeline.times_min[-1])
        H, W = t0.shape
        if H == 0 or W == 0:
            return []
        step = max(2, int(round(min(H, W) / 12.0)))
        zone_px = {tuple(map(int, z.pixel)) for z in zones}
        cands = []
        for y in range(step // 2, H, step):
            for x in range(step // 2, W, step):
                if t0[y, x] or tend[y, x]:
                    continue
                p = (x, y)
                if p in zone_px:
                    continue
                cands.append(p)
        if len(cands) > max_candidates:
            rng = np.random.default_rng(0)  # deterministic sampling
            idx = rng.choice(len(cands), size=max_candidates, replace=False)
            cands = [cands[int(i)] for i in idx]
        return cands

    def cost(self, plan: MissionPlan) -> float:
        skip_penalty = self.skip_zone_penalty * len(plan.skipped)
        return (self.time_weight * plan.total_duration_min
                + self.energy_weight * plan.total_energy_wh
                + skip_penalty)

    def optimize_home(
        self, zones: List[DropZone], max_candidates: int = 24
    ) -> Tuple[Optional[Tuple[int, int]], MissionPlan, List[dict]]:
        """Select the home that minimises weighted (time + battery) mission cost
        across the full TSP + time-aware D* Lite route.  Logs alternatives."""
        best_home, best_plan, best_cost = None, None, float("inf")
        results: List[dict] = []
        for cand in self._candidate_homes(zones, max_candidates=max_candidates):
            plan = self.plan_for_home(cand, zones)
            c = self.cost(plan)
            results.append({
                "home": cand, "cost": c, "time_min": plan.total_duration_min,
                "energy_wh": plan.total_energy_wh, "sorties": len(plan.sorties),
                "served": len(plan.served_ids), "skipped": len(plan.skipped),
            })
            if c < best_cost:
                best_cost, best_home, best_plan = c, cand, plan

        results.sort(key=lambda r: r["cost"])
        if best_home is None:
            logger.error("No feasible home candidate found -- mission infeasible.")
            return None, MissionPlan(home_px=(-1, -1), zones=list(zones),
                                     notes=["no feasible home candidate"]), results

        # Visible, defensible choice: log chosen vs 2-3 alternatives.
        shown = results[:min(4, len(results))]
        for r in shown:
            tag = "CHOSEN" if r["home"] == best_home else "alt"
            logger.info(
                "HOME candidate %s %s: est time=%6.1f min, battery=%6.1f Wh "
                "(%d sortie(s), %d served, %d skipped), cost=%9.1f",
                tag, r["home"], r["time_min"], r["energy_wh"], r["sorties"],
                r["served"], r["skipped"], r["cost"])
        return best_home, best_plan, results
