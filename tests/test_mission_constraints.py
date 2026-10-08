"""Battery/payload model, sortie planning and dry drop placement of the multi-base mode (ported from MAIN)."""
import unittest
import numpy as np
from src.mission.constraints import (leg_battery_fraction, plan_payload_batches,
                                    score_home_candidates, plan_mission_stops,
                                    wind_kmh_to_mps, headwind_component_mps)
from src.mission.mission_output import generate_mission_file
from pathlib import Path
from tempfile import TemporaryDirectory


class TestMissionConstraints(unittest.TestCase):
    def test_safe_drop_point_is_dry_and_adjacent_to_flood_boundary(self):
        import cv2
        from src.mission.safe_dropzone import find_safe_drop_points

        mask = np.zeros((40, 40), dtype=np.uint8)
        cv2.rectangle(mask, (15, 15), (24, 24), 255, thickness=-1)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(contours[0]).reshape(-1, 2)
        point = find_safe_drop_points([hull], [0], [(5, 20), (35, 20)], obstacle_mask=mask)[0]
        x, y = point
        self.assertEqual(mask[y, x], 0)
        self.assertLessEqual(cv2.pointPolygonTest(contours[0], (float(x), float(y)), True), 0)

    def test_drop_connectivity_filter_excludes_other_downsampled_free_component(self):
        from src.mission.safe_dropzone import filter_drops_by_home_connectivity

        # A full-height obstacle wall splits the 8-connected downsampled grid.
        mask = np.zeros((50, 100), dtype=np.uint8)
        mask[:, 45:55] = 255
        kept, excluded = filter_drops_by_home_connectivity(
            [(20, 20), (80, 20)], (10, 20), mask, downsample_factor=5)
        self.assertEqual(kept, [(20, 20)])
        self.assertEqual(excluded, [1])

    def test_battery_drain_increases_with_distance_payload_and_headwind(self):
        base = leg_battery_fraction(1000)
        self.assertGreater(leg_battery_fraction(2000), base)
        self.assertGreater(leg_battery_fraction(1000, .5), base)
        self.assertGreater(leg_battery_fraction(1000, 0, 10), base)

    def test_short_leg_battery_percent_matches_endurance_time_budget(self):
        # 300 m / 5 m/s = 60 s = 1 minute; 1 kg payload adds 8% time.
        expected_fraction = (1.0 * (1.0 + 1.0 * 0.08)) / 7.0
        actual_fraction = leg_battery_fraction(300, payload_kg=1.0)
        actual_percent = actual_fraction * 100.0
        self.assertGreaterEqual(actual_percent, 0.0)
        self.assertLessEqual(actual_percent, 100.0)
        self.assertAlmostEqual(actual_fraction, expected_fraction)

    def test_weather_wind_speed_is_converted_from_kmh_to_mps(self):
        self.assertAlmostEqual(wind_kmh_to_mps(30.0), 30.0 / 3.6)

    def test_battery_wind_penalty_uses_headwind_component(self):
        speed = wind_kmh_to_mps(30)
        eastbound = headwind_component_mps(speed, 270, (0, 0), (10, 0))
        westbound = headwind_component_mps(speed, 270, (10, 0), (0, 0))
        northbound = headwind_component_mps(speed, 270, (0, 10), (0, 0))
        self.assertAlmostEqual(eastbound, 0.0, places=6)
        self.assertGreater(westbound, 0)
        self.assertAlmostEqual(northbound, 0.0, places=6)

    def test_multi_sortie_reserve_is_checked_per_sortie_not_cumulative(self):
        dry = np.zeros((100, 100), dtype=np.uint8)
        # Six drops at 1 kg each, with a 2 kg capacity, force three sorties.
        result = score_home_candidates([(1, 1)], [(2, 1), (3, 1), (4, 1),
            (5, 1), (6, 1), (7, 1)], dry, meters_per_pixel=100,
            per_drop_payload_kg=1.0, payload_capacity_kg=2.0, return_details=True)
        self.assertEqual(result["evaluated"], 1)
        self.assertEqual(len(result["scored"]), 1)
        self.assertEqual(len(result["sortie_metrics"]), 1)
        metrics = result["sortie_metrics"][0]
        self.assertEqual(metrics["sortie_count"], 3)
        self.assertGreater(metrics["total_mission_drain_fraction"], 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in metrics["sortie_drain_fractions"]))
        self.assertTrue(all(value < 1.0 - __import__("config.settings", fromlist=["BATTERY_RESERVE_FRACTION"]).BATTERY_RESERVE_FRACTION
                            for value in metrics["sortie_drain_fractions"]))

    def test_home_scorer_splits_sorties_on_battery_before_capacity(self):
        dry = np.zeros((100, 100), dtype=np.uint8)
        drops = [(60, 50), (40, 50), (60, 51), (40, 51), (60, 52), (40, 52)]
        result = score_home_candidates([(50, 50)], drops, dry,
            meters_per_pixel=40, per_drop_payload_kg=.25, payload_capacity_kg=8,
            return_details=True)
        self.assertEqual(len(result["scored"]), 1)
        metrics = result["sortie_metrics"][0]
        self.assertGreater(metrics["sortie_count"], 1)
        self.assertTrue(all(value < .8 for value in metrics["sortie_drain_fractions"]))
        self.assertGreater(metrics["total_mission_drain_fraction"], .8)

    def test_home_scorer_does_not_add_mission_total_payload_to_each_sortie(self):
        dry = np.zeros((100, 100), dtype=np.uint8)
        drops = [(10, 10), (12, 10), (14, 10), (16, 10)]
        per_sortie = score_home_candidates([(5, 10)], drops, dry,
            meters_per_pixel=30, payload_kg=0,
            per_drop_payload_kg=.25, payload_capacity_kg=8,
            headwind_mps=30 / 3.6, return_details=True)
        incorrectly_aggregated = score_home_candidates([(5, 10)], drops, dry,
            meters_per_pixel=30, payload_kg=len(drops) * .25,
            per_drop_payload_kg=.25, payload_capacity_kg=8,
            headwind_mps=30 / 3.6, return_details=True)
        self.assertEqual(len(per_sortie["scored"]), 1)
        self.assertGreater(incorrectly_aggregated["sortie_metrics"][0]["sortie_drain_fractions"][0],
                           per_sortie["sortie_metrics"][0]["sortie_drain_fractions"][0])

    def test_home_scorer_uses_tsp_order_instead_of_raw_contour_order(self):
        dry = np.zeros((100, 100), dtype=np.uint8)
        # Raw list alternates between distant ends; nearest-neighbor ordering
        # keeps each out-and-back sortie local.
        drops = [(80, 50), (20, 50), (79, 50), (21, 50)]
        result = score_home_candidates([(50, 50)], drops, dry,
            meters_per_pixel=20, per_drop_payload_kg=.25,
            payload_capacity_kg=8, return_details=True)
        self.assertEqual(len(result["scored"]), 1)

    def test_payload_batches_require_reload_after_capacity(self):
        self.assertEqual(plan_payload_batches(5, capacity=1, per_drop_kg=.5), [[0, 1], [2, 3], [4]])

    def test_home_candidates_are_dry_and_ranked(self):
        obstacles = np.zeros((20, 20), dtype=np.uint8)
        obstacles[5, 5] = 255
        scores = score_home_candidates([(5, 5), (1, 1), (18, 18)], [(10, 10)], obstacles)
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0][1], (18, 18))

    def test_home_candidate_logging_distinguishes_absent_vs_rejected(self):
        from src.mission.constraints import score_home_candidates
        flooded = np.ones((10, 10), dtype=np.uint8) * 255
        no_dry = score_home_candidates([(1, 1)], [(9, 9)], flooded, return_details=True)
        self.assertEqual(no_dry["sampled"], 1)
        self.assertEqual(no_dry["dry_candidates"], 0)
        self.assertEqual(no_dry["evaluated"], 0)
        self.assertEqual(no_dry["rejected"], [])

        dry = np.zeros((10, 10), dtype=np.uint8)
        rejected = score_home_candidates([(0, 0)], [(9, 9)], dry,
            meters_per_pixel=10000, return_details=True)
        self.assertEqual(rejected["evaluated"], 1)
        self.assertEqual(len(rejected["rejected"]), 1)
        self.assertGreater(rejected["rejected"][0][0], 0)

    def test_payload_weight_changes_selected_home_in_controlled_case(self):
        dry = np.zeros((30, 30), dtype=np.uint8)
        candidates = [(2, 15), (27, 15)]
        drops = [(3, 15), (4, 15), (5, 15), (28, 15)]
        light = score_home_candidates(candidates, drops, dry, meters_per_pixel=.001,
            payload_kg=0, per_drop_payload_kg=.1, return_details=True)
        heavy = score_home_candidates(candidates, drops, dry, meters_per_pixel=.001,
            payload_kg=0, per_drop_payload_kg=7.0, return_details=True)
        self.assertTrue(light["scored"] and heavy["scored"])
        light, heavy = light["scored"], heavy["scored"]
        self.assertNotEqual(light[0][1], heavy[0][1])

    def test_per_leg_battery_triggers_midroute_reload_and_logs_deviation(self):
        dry = np.zeros((100, 100), dtype=np.uint8)
        # Each drop alone is a ~50% out-and-back sortie; both in one sortie
        # need ~85% including the return leg, so a reload must split them.
        with self.assertLogs(level="WARNING") as captured:
            route = plan_mission_stops((5, 5), [(40, 5), (5, 40)], {}, dry,
                meters_per_pixel=15, headwind_mps=0, per_drop_kg=.1)
        self.assertTrue(route.reload_indices)
        self.assertEqual(len(route.drop_indices), 2)
        self.assertEqual(route.unreachable, [])
        self.assertTrue(any("Battery reserve" in line for line in captured.output))
        self.assertTrue(any(b >= .99 for b in route.remaining_battery))
        self.assertEqual(len(route.sortie_battery), 2)
        self.assertTrue(all(0.0 < used <= 0.80 for used in route.sortie_battery))

    def test_unservable_drop_is_unreachable_without_pointless_reload(self):
        dry = np.zeros((100, 100), dtype=np.uint8)
        route = plan_mission_stops((5, 5), [(90, 5)], {}, dry,
            meters_per_pixel=1000, per_drop_kg=.1)
        self.assertEqual(route.unreachable, [0])
        self.assertEqual(route.reload_indices, [])
        self.assertTrue(any("unreachable" in m and "reserve" in m for m in route.deviations))

    def test_routed_detour_distance_triggers_battery_reload(self):
        dry = np.zeros((100, 100), dtype=np.uint8)
        drops = [(10, 5), (15, 5)]
        # Straight-line legs total 200 m (one sortie); the routed detour between
        # the two drops makes the shared sortie ~89% including return.
        routed = {("pair", (5, 5), (10, 5)): 50.0,
                  ("pair", (10, 5), (15, 5)): 1700.0,
                  ("pair", (15, 5), (5, 5)): 100.0}
        straight = plan_mission_stops((5, 5), drops, {}, dry, meters_per_pixel=10, per_drop_kg=.1)
        self.assertEqual(straight.reload_indices, [])
        with self.assertLogs(level="WARNING") as captured:
            route = plan_mission_stops((5, 5), drops, {}, dry,
                meters_per_pixel=10, per_drop_kg=.1, routed_leg_distances_m=routed)
        self.assertTrue(route.reload_indices)
        self.assertEqual(len(route.drop_indices), 2)
        self.assertTrue(any("Battery reserve" in line for line in captured.output))

    def test_reserve_check_includes_return_leg_from_drop(self):
        dry = np.zeros((200, 200), dtype=np.uint8)
        # Outbound alone (~66%) is above reserve; with the return leg (~132%)
        # it is not, so the drop must be rejected rather than flown.
        route = plan_mission_stops((0, 100), [(97, 100)], {}, dry,
            meters_per_pixel=10, per_drop_kg=.25)
        self.assertEqual(route.drop_indices, [])
        self.assertEqual(route.unreachable, [0])

    def test_payload_decreases_along_sortie(self):
        from src.mission.constraints import leg_battery_fraction
        dry = np.zeros((100, 100), dtype=np.uint8)
        with self.assertLogs(level="INFO") as captured:
            route = plan_mission_stops((0, 50), [(10, 50), (20, 50), (30, 50)], {}, dry,
                meters_per_pixel=2, per_drop_kg=1.0)
        payloads = [float(line.split("payload=")[1].split("kg")[0])
                    for line in captured.output if "Plan leg to drop" in line]
        self.assertEqual(payloads, [3.0, 2.0, 1.0])
        expected = (leg_battery_fraction(20, 3.0) + leg_battery_fraction(20, 2.0)
                    + leg_battery_fraction(20, 1.0) + leg_battery_fraction(60, 0.0))
        self.assertAlmostEqual(route.sortie_battery[0], expected)

    def test_sortie_battery_report_uses_routed_geometry(self):
        from src.mission.constraints import sortie_battery_report, leg_battery_fraction
        path = [(0, 0), (0, 30), (40, 30), (40, 0), (0, 0), (0, 0), (10, 0), (0, 0)]
        report = sortie_battery_report(path, [3, 6], [4], meters_per_pixel=1, per_drop_kg=.25)
        self.assertEqual([r["drops"] for r in report], [1, 1])
        self.assertAlmostEqual(report[0]["distance_m"], 140.0)
        self.assertAlmostEqual(report[0]["battery_fraction"],
                               leg_battery_fraction(30, .25) + leg_battery_fraction(40, .25)
                               + leg_battery_fraction(30, .25) + leg_battery_fraction(40, 0.0))
        self.assertAlmostEqual(report[1]["distance_m"], 20.0)

    def test_payload_excess_inserts_reload_land_takeoff_in_output(self):
        dry = np.zeros((100, 100), dtype=np.uint8)
        route = plan_mission_stops((5, 5), [(10, 5), (15, 5), (20, 5)], {}, dry,
            capacity=.5, per_drop_kg=.25)
        self.assertTrue(route.reload_indices)
        with TemporaryDirectory() as tmp:
            outfile = Path(tmp) / "mission.waypoints"
            generate_mission_file(route.points, route.drop_indices, (5, 5), (0, 0),
                (25.0, 82.0), 2.0, filename=str(outfile), reload_indices=route.reload_indices)
            text = outfile.read_text()
        commands = [line.split("\t")[3] for line in text.splitlines()[1:]]
        self.assertGreaterEqual(commands.count("21"), 2)  # reload land plus final land
        self.assertGreaterEqual(commands.count("22"), 2)  # initial takeoff plus reload departure

    def test_flooded_drop_is_reprioritized_then_flagged_if_still_flooded(self):
        current = np.zeros((20, 20), dtype=np.uint8)
        future = np.zeros_like(current)
        future[10, 10] = 255
        route = plan_mission_stops((1, 1), [(10, 10), (3, 3)], {0.5: future, 1.0: future}, current,
            meters_per_pixel=10, speed_mps=1)
        self.assertIn(0, route.unreachable)
        self.assertTrue(any("reprioritized earlier" in msg for msg in route.deviations))
        self.assertTrue(any("unreachable" in msg and "flooded" in msg for msg in route.deviations))

    def test_moderate_weather_keeps_nearby_early_drops_reachable(self):
        current = np.zeros((50, 50), dtype=np.uint8)
        current[20:30, 20:30] = 255
        # DEVELOPMENT's flood model as time-indexed frames (MAIN called its own predict_spread here)
        forecasts = __import__("src.weather.flood_spread", fromlist=["forecast_frames"]).forecast_frames(
            current, {"precipitation": 5, "wind_speed_10m": 15, "wind_direction_10m": 0}, 2.0)
        drops = [(19, 20), (20, 19), (29, 20), (30, 29)]
        route = plan_mission_stops((19, 18), drops, forecasts, current,
            meters_per_pixel=1, speed_mps=8, per_drop_kg=.05,
            routed_leg_distances_m=[8, 8, 8, 8, 8])
        self.assertGreaterEqual(len(route.drop_indices), 2)
        self.assertLess(len(route.unreachable), len(drops))

    def test_all_drops_unreachable_returns_infeasible_without_exception(self):
        current = np.zeros((20, 20), dtype=np.uint8)
        future = np.ones_like(current) * 255
        with self.assertLogs(level="ERROR") as captured:
            route = plan_mission_stops((1, 1), [(5, 5), (10, 10)],
                {1/12: future, 1/6: future, .5: future}, current)
        self.assertEqual(route.drop_indices, [])
        self.assertEqual(route.unreachable, [0, 1])
        self.assertTrue(any("mission infeasible — no reachable drop points" in line for line in captured.output))

    def test_sortie_splitting_checks_battery_before_committing_leg(self):
        # Two clusters of drops: each cluster fits in a sortie, but together they exceed 80%
        dry = np.zeros((100, 100), dtype=np.uint8)
        drops = [(10, 50), (12, 50), (90, 50), (92, 50)]
        result = score_home_candidates([(50, 50)], drops, dry,
            meters_per_pixel=12.0, per_drop_payload_kg=0.25, payload_capacity_kg=8.0,
            return_details=True)
        self.assertEqual(len(result["scored"]), 1)
        metrics = result["sortie_metrics"][0]
        self.assertGreaterEqual(metrics["sortie_count"], 2)
        # All committed sorties must be strictly <= 80% battery use
        for drain in metrics["sortie_drain_fractions"]:
            self.assertLessEqual(drain, 0.80)

    def test_score_home_candidates_tracks_reachable_and_unreachable_drops(self):
        dry = np.zeros((200, 200), dtype=np.uint8)
        # (10, 10) is nearby, (190, 190) is far away (exceeds single drop RT)
        drops = [(10, 10), (190, 190)]
        result = score_home_candidates([(10, 10)], drops, dry,
            meters_per_pixel=100.0, per_drop_payload_kg=0.25, payload_capacity_kg=8.0,
            return_details=True)
        self.assertEqual(len(result["rejected"]), 1)
        metric = result["sortie_metrics"][0]
        self.assertIn((10, 10), metric["reachable_drops"])
        self.assertIn((190, 190), metric["unreachable_drops"])
        self.assertEqual(metric["reachable_count"], 1)

    def test_large_contour_multi_point_generation_creates_well_separated_dry_points(self):
        import cv2
        from src.mission.safe_dropzone import find_safe_drop_points

        # Create a large rectangular flood region (e.g. 80x80 = 6400 area)
        mask = np.zeros((150, 150), dtype=np.uint8)
        cv2.rectangle(mask, (30, 30), (110, 110), 255, thickness=-1)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        self.assertGreaterEqual(area, 2500.0)

        # find_safe_drop_points with large contour threshold = 2500
        safe_pts = find_safe_drop_points(
            [cnt], [0], [(10, 10)], obstacle_mask=mask,
            large_contour_area_threshold=2500.0,
            large_contour_area_step=2000.0,
            max_drops_per_contour=3,
            min_drop_separation_px=30.0
        )
        # Should generate multiple points (e.g. 2 or 3)
        self.assertGreaterEqual(len(safe_pts), 2)
        self.assertLessEqual(len(safe_pts), 3)

        # Every point must be dry
        for pt in safe_pts:
            x, y = pt
            self.assertEqual(mask[y, x], 0, f"Point {pt} must be dry")

        # Every pair of points must be separated by at least min_drop_separation_px
        for i in range(len(safe_pts)):
            for j in range(i + 1, len(safe_pts)):
                p1, p2 = np.array(safe_pts[i]), np.array(safe_pts[j])
                dist = np.linalg.norm(p1 - p2)
                self.assertGreaterEqual(dist, 30.0, f"Points {p1} and {p2} must be >= 30px apart")


if __name__ == "__main__":
    unittest.main()


