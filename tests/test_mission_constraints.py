import unittest
import logging
import numpy as np

from src.mission.constraints import DroneSpec, estimate_straight_leg, headwind_component_mps
from src.routing.mission_planner import MissionPlanner, DropZone
from src.weather.flood_predictor import FloodTimeline

CALM = {"precipitation": 0.0, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0}


def clear_timeline(size, times=(0.0, 60.0), weather=None):
    """A flood timeline whose masks are empty (no obstacles)."""
    masks = [np.zeros((size, size), dtype=np.uint8) for _ in times]
    return FloodTimeline(list(times), masks, weather=weather or CALM)


def flood_at_timeline(size, times, flooded_pixels_by_index, weather=None):
    """Timeline where the listed pixels are flooded only in the given frames."""
    masks = [np.zeros((size, size), dtype=np.uint8) for _ in times]
    for idx, pixels in flooded_pixels_by_index.items():
        for (x, y) in pixels:
            masks[idx][y, x] = 255
    return FloodTimeline(list(times), masks, weather=weather or CALM)


def make_planner(size, spec=None, timeline=None, mpp=10.0, factor=2, **kw):
    tl = timeline if timeline is not None else clear_timeline(size)
    return MissionPlanner(spec=spec, weather=CALM, timeline=tl,
                          meters_per_pixel=mpp, downsample_factor=factor, **kw)


class TestBatteryModel(unittest.TestCase):
    """Task 3: documented linear battery model (DJI Mavic-class anchors)."""

    def test_sourced_anchors(self):
        spec = DroneSpec()
        self.assertEqual(spec.battery_wh, 77.0)        # DJI Mavic 3 5000mAh/15.4V
        self.assertEqual(spec.hover_endurance_min, 46.0)
        self.assertEqual(spec.usable_energy_wh, 77.0 * 0.8)  # 80% reserve

    def test_energy_increases_with_distance(self):
        spec = DroneSpec()
        e1 = spec.leg_energy_wh(1000.0, 0.0, 0.0)
        e2 = spec.leg_energy_wh(2000.0, 0.0, 0.0)
        self.assertAlmostEqual(e2, 2.0 * e1, places=6)

    def test_energy_increases_with_payload(self):
        spec = DroneSpec()
        e0 = spec.leg_energy_wh(1000.0, 0.0, 0.0)
        e1 = spec.leg_energy_wh(1000.0, 1.0, 0.0)
        self.assertGreater(e1, e0)
        # linear payload term: +15% energy per kg carried
        self.assertAlmostEqual(e1, e0 * (1.0 + spec.payload_energy_per_kg), places=6)

    def test_headwind_slows_and_costs_more(self):
        spec = DroneSpec()
        calm = spec.leg_energy_wh(1000.0, 0.0, 0.0)
        into_wind = spec.leg_energy_wh(1000.0, 0.0, 5.0)   # 5 m/s headwind
        self.assertGreater(into_wind, calm)
        self.assertLess(spec.ground_speed_mps(5.0), spec.ground_speed_mps(0.0))
        # tailwind speeds the drone up
        self.assertGreater(spec.ground_speed_mps(-5.0), spec.ground_speed_mps(0.0))

    def test_headwind_component_sign(self):
        # wind from the North (0 deg): travelling North is a headwind.
        # (0,-1) = north in image coords.
        self.assertGreater(headwind_component_mps(0.0, -1.0,
                                                  {"wind_speed_10m": 6.0, "wind_direction_10m": 0.0}), 0.0)
        # travelling South is a tailwind (negative headwind).
        self.assertLess(headwind_component_mps(0.0, 1.0,
                                               {"wind_speed_10m": 6.0, "wind_direction_10m": 0.0}), 0.0)


class TestPayloadAndBatteryRouting(unittest.TestCase):
    """Task 3: payload reload legs + battery safety margin on a small map."""

    def test_payload_reload_splits_sorties(self):
        # Capacity 1.0 kg; three 0.5 kg zones -> need two sorties.
        zones = [DropZone("a", (60, 10), 0.5), DropZone("b", (10, 60), 0.5),
                 DropZone("c", (60, 60), 0.5)]
        pl = make_planner(80, spec=DroneSpec(payload_capacity_kg=1.0), mpp=4.0)
        plan = pl.plan_for_home((10, 10), zones)
        self.assertEqual(plan.status, "ok")
        self.assertEqual(len(plan.served_ids), 3)
        self.assertEqual(len(plan.sorties), 2)
        # each sortie carries at most capacity and delivers a positive amount
        self.assertEqual(sorted(round(s.payload_kg, 6) for s in plan.sorties), [0.5, 1.0])
        for s in plan.sorties:
            self.assertLessEqual(s.payload_kg, 1.0 + 1e-9)
            # every sortie starts AND ends at home (reload happens at base)
            self.assertEqual(s.path[0], (10, 10))
            self.assertEqual(s.path[-1], (10, 10))
            self.assertEqual(len(s.drop_zone_ids), len(s.drop_indices))
        # a payload-reload turnaround was logged
        self.assertTrue(any("turnaround" in n for n in plan.notes) or len(plan.sorties) > 1)

    def test_payload_depletes_after_each_drop(self):
        # Single sortie of two 0.5 kg drops on a 2.0 kg capacity drone: the
        # return-home leg (final) is flown empty, so both packages fit and are
        # delivered in one sortie.
        zones = [DropZone("a", (60, 10), 0.5), DropZone("b", (10, 60), 0.5)]
        pl = make_planner(80, spec=DroneSpec(payload_capacity_kg=2.0), mpp=4.0)
        plan = pl.plan_for_home((10, 10), zones)
        self.assertEqual(len(plan.sorties), 1)
        self.assertEqual(round(plan.sorties[0].payload_kg, 6), 1.0)

    def test_battery_reserve_reroutes_and_recovers(self):
        # Home centred between two equidistant zones on opposite sides, each
        # ~7 km away.  Serving the second after the first pushes the flight time
        # past the 80% energy reserve, but each is fine on its own fresh sortie.
        spec = DroneSpec(payload_capacity_kg=2.0)
        home = (100, 100)
        zones = [DropZone("a", (30, 100), 0.5), DropZone("c", (170, 100), 0.5)]
        pl = make_planner(200, spec=spec, mpp=100.0, factor=1)
        plan = pl.plan_for_home(home, zones)
        self.assertEqual(plan.status, "ok")
        self.assertEqual(len(plan.served_ids), 2)
        self.assertGreaterEqual(len(plan.sorties), 2)
        reasons = [s.closed_reason for s in plan.sorties]
        self.assertTrue(any("battery" in r for r in reasons))
        self.assertTrue(any("Constraint deviation" in n for n in plan.notes))
        # Every sortie returns to base, so the battery can be swapped.
        for s in plan.sorties:
            self.assertEqual(s.path[-1], home)

    def test_battery_beyond_round_trip_skips_gracefully(self):
        # A far zone whose round trip alone exceeds the 80% reserve is dropped
        # with a clear reason (near zone still served) - no crash.
        spec = DroneSpec(payload_capacity_kg=2.0)
        zones = [DropZone("a", (60, 20), 0.5),
                 DropZone("c", (20, 260), 0.5)]     # ~13 km one-way
        pl = make_planner(300, spec=spec, mpp=100.0, factor=1)
        plan = pl.plan_for_home((20, 20), zones)
        self.assertEqual(plan.status, "partial")
        self.assertIn("a", plan.served_ids)
        skipped_reasons = [r for z, r in plan.skipped]
        self.assertTrue(any("battery" in r for r in skipped_reasons))


class TestFloodThreatenedZones(unittest.TestCase):
    """Task 5b: predicted spread blocking a drop point."""

    def setUp(self):
        # Home sits between two zones on a line.  O (East) is *closer* to home
        # than T (West), so the plain nearest-neighbour TSP visits O first and
        # would reach T too late -- unless the flood-threat reprioritisation
        # moves T to the front.
        self.home = (150, 150)
        self.size = 300
        self.zones = [DropZone("T", (0, 150), 0.5),    # 150 px West (6 km @ 40 m/px)
                      DropZone("O", (250, 150), 0.5)]  # 100 px East (4 km)

    def _threatened_timeline(self, flood_min):
        # T becomes flooded at flood_min (and stays flooded). O never floods.
        times = sorted(set([0.0, float(flood_min), 30.0]))
        idx = times.index(float(flood_min))
        pixels = {k: [(0, 150)] for k in range(idx, len(times))}
        return flood_at_timeline(self.size, times, pixels)

    def test_threatened_zone_reprioritised_earlier(self):
        # Natural TSP = [O, T]; T would be flooded (t=18) by the time the drone
        # reached it second (~20 min), so it must move first and be served.
        tl = self._threatened_timeline(18.0)
        pl = make_planner(self.size, spec=DroneSpec(payload_capacity_kg=2.0),
                          timeline=tl, mpp=40.0, factor=5)
        plan = pl.plan_for_home(self.home, self.zones)
        self.assertEqual(plan.status, "ok")
        self.assertIn("T", plan.served_ids)
        self.assertIn("O", plan.served_ids)
        first_sortie = plan.sorties[0]
        self.assertEqual(first_sortie.drop_zone_ids[0], "T")

    def test_zone_underwater_before_any_eta_is_skipped_with_log(self):
        # Flood reaches T almost immediately (t=3, before even the 6 km direct
        # leg) -> cannot be served even first; skipped with a clear reason.
        tl = self._threatened_timeline(3.0)
        pl = make_planner(self.size, spec=DroneSpec(payload_capacity_kg=2.0),
                          timeline=tl, mpp=40.0, factor=5)
        plan = pl.plan_for_home(self.home, self.zones)
        self.assertEqual(plan.status, "partial")
        reasons = [r for z, r in plan.skipped if z.zone_id == "T"]
        self.assertTrue(reasons)
        self.assertIn("predicted flood", reasons[0])


class TestHomeOptimisation(unittest.TestCase):
    """Task 4: base location is optimised over safe candidates, then logged."""

    def test_optimizer_picks_cheapest_safe_home_and_logs(self):
        zones = [DropZone("a", (120, 30), 0.5), DropZone("b", (140, 40), 0.5),
                 DropZone("c", (120, 50), 0.5)]
        pl = make_planner(160, spec=DroneSpec(payload_capacity_kg=2.0), mpp=5.0)

        with self.assertLogs("src.routing.mission_planner", level=logging.INFO) as cm:
            home, plan, results = pl.optimize_home(zones, max_candidates=12)

        self.assertIsNotNone(home)
        self.assertEqual(plan.status, "ok")
        # chosen home is a dry pixel now and at horizon end
        t0 = plan.sorties and pl.timeline.obstacle_mask_at(0.0)
        self.assertIsNotNone(t0)
        self.assertEqual(t0[int(home[1]), int(home[0])], 0)
        # results sorted by cost, chosen has the minimum cost
        self.assertEqual(results[0]["home"], home)
        self.assertEqual(min(r["cost"] for r in results), results[0]["cost"])
        # the choice was logged alongside alternatives, not black-box
        log_text = "\n".join(cm.output)
        self.assertIn("CHOSEN", log_text)

    def test_optimizer_beats_centre_heuristic(self):
        # Drop zones cluster in the top-right; a centre-based home is clearly
        # worse than the optimised pick.
        zones = [DropZone("a", (120, 30), 0.5), DropZone("b", (140, 40), 0.5),
                 DropZone("c", (120, 50), 0.5)]
        pl = make_planner(160, spec=DroneSpec(payload_capacity_kg=2.0), mpp=5.0)
        centre = (80, 80)
        centre_plan = pl.plan_for_home(centre, zones)
        home, plan, _ = pl.optimize_home(zones, max_candidates=12)
        self.assertLessEqual(pl.cost(plan), pl.cost(centre_plan))


class TestGracefulDegradation(unittest.TestCase):
    """Task 5a: infeasibility degrades gracefully (no crash / silent wrong route)."""

    def test_mission_infeasible_when_nothing_reachable(self):
        # Both zones are beyond the round-trip battery range -> the planner
        # reports an infeasible mission instead of crashing / silent wrong route.
        spec = DroneSpec(payload_capacity_kg=2.0)
        zones = [DropZone("x", (20, 140), 0.5), DropZone("y", (140, 20), 0.5)]
        pl = make_planner(150, spec=spec, mpp=100.0, factor=1)
        plan = pl.plan_for_home((20, 20), zones)
        self.assertEqual(plan.status, "infeasible")
        self.assertEqual(plan.sorties, [])
        self.assertTrue(plan.skipped)      # zones were dropped, with reasons

    def test_flooded_home_candidate_is_rejected(self):
        zones = [DropZone("a", (60, 20), 0.5)]
        tl = clear_timeline(80)
        m = tl.masks[0].copy()
        m[10, 10] = 255                     # candidate home flooded at t=0
        tl2 = FloodTimeline([0.0, 60.0], [m, np.zeros_like(m)], weather=CALM)
        pl = make_planner(80, timeline=tl2)
        plan = pl.plan_for_home((10, 10), zones)
        self.assertEqual(plan.status, "infeasible")
        self.assertTrue(any("flooded" in n for n in plan.notes))

    def test_all_paths_return_home(self):
        # Structural property: whatever the plan, every sortie ends at home.
        spec = DroneSpec(battery_wh=5.0)
        zones = [DropZone("a", (60, 20), 0.5), DropZone("b", (20, 60), 0.5),
                 DropZone("c", (60, 60), 0.5)]
        pl = make_planner(80, spec=spec, mpp=20.0, factor=1)
        plan = pl.plan_for_home((10, 10), zones)
        for s in plan.sorties:
            self.assertEqual(s.path[0], (10, 10))
            self.assertEqual(s.path[-1], (10, 10))


class TestCostModel(unittest.TestCase):
    def test_estimate_straight_leg(self):
        spec = DroneSpec()
        est = estimate_straight_leg((0, 0), (300, 400), 2.0, spec, CALM)
        self.assertAlmostEqual(est["distance_m"], 1000.0, places=3)  # 500 px * 2 m
        self.assertAlmostEqual(est["duration_s"], 1000.0 / spec.cruise_airspeed_mps, places=3)
        self.assertGreater(est["energy_wh"], 0.0)


if __name__ == "__main__":
    unittest.main()
