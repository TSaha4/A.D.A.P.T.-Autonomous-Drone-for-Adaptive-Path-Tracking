"""Multi-UAV / multi-base mode (main.py --mode multi), integrated from the MAIN branch.

End-to-end cases run the unmodified main.main() on synthetic north-up maps whose flood zones are pure
red squares. Only I/O is stubbed: the interactive HSV click step (thresholds set directly), GUI windows,
and the weather request (calm weather, so the flood mask is not dilated). Every exported mission is
checked with the independent validator (tests/wpl_validator.py) against its route sidecar.

Battery geometry used below (settings defaults, calm air): a single-drop sortie fits the 80 % usable
battery when the routed base-drop distance is at most ~830 m. With METERS_PER_PIXEL = 4 and no
resizing, one base therefore reaches drops within ~205 px.
"""
import contextlib
import glob
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)

import main as main_module
from config import settings
from src.mission import mission_output
from src.mission.constraints import leg_battery_fraction, plan_mission_stops
from src.mission.multi_base import select_bases, single_drop_reach
from src.mission.overlap_repair import execute_overlap_repairs
from src.routing.pathfinding import (compute_routed_path, polyline_intersections, snap_to_nearest_free_cell)
from src.vision.image_processing import ImageProcessor
from src.weather.flood_spread import (forecast_at, forecast_frames, interpolate_forecast_mask, obstacle_mask_at,
                                      predict_spread)
from tests import wpl_validator

CALM = {"precipitation": 0.0, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0, "status": "success"}
GEO = (25.0, 82.0)
MPP = 4.0
SIZE = 750
# Flood zones (x0, y0, x1, y1). Two clusters ~550 px apart: no single base reaches both.
WEST = [(70, 330, 105, 365), (120, 400, 150, 430), (60, 440, 90, 470)]
EAST = [(620, 330, 655, 365), (660, 400, 690, 430)]
SOUTH = [(350, 650, 385, 685), (410, 640, 440, 670)]
NORTH_WEST = [(70, 60, 105, 95), (130, 80, 160, 110)]
NORTH_EAST = [(630, 60, 665, 95), (580, 100, 610, 130)]


def make_map(path, squares, size=SIZE):
    img = np.full((size, size, 3), 255, np.uint8)
    for x0, y0, x1, y1 in squares:
        img[y0:y1 + 1, x0:x1 + 1] = (0, 0, 255)
    cv2.imwrite(path, img)


def fake_select(self, image):
    self.image = image.copy()
    self.hsv_lower = np.array([0, 100, 100], np.uint8)
    self.hsv_upper = np.array([179, 255, 255], np.uint8)
    self.is_red_wrap = True
    return self.hsv_lower, self.hsv_upper


def run_main(argv_tail, cwd, mpp=MPP, weather=CALM, extra=()):
    """Runs main.main() in cwd; returns the exit status (0 when main returns normally)."""
    argv = ["main.py"] + list(argv_tail) + ["--lat", str(GEO[0]), "--lon", str(GEO[1])]
    with contextlib.ExitStack() as stack:
        stack.enter_context(contextlib.chdir(cwd))
        stack.enter_context(mock.patch.object(sys, "argv", argv))
        stack.enter_context(mock.patch.object(settings, "METERS_PER_PIXEL", mpp))
        stack.enter_context(mock.patch.object(ImageProcessor, "select_sample_points", fake_select))
        stack.enter_context(mock.patch.object(main_module, "get_weather_data", return_value=dict(weather)))
        stack.enter_context(mock.patch.object(main_module, "show_image_safe"))
        stack.enter_context(mock.patch.object(main_module, "display_path_on_map"))
        stack.enter_context(mock.patch("src.mission.mission_output.plt.show"))
        for target in extra:
            stack.enter_context(target)
        try:
            main_module.main()
            return 0
        except SystemExit as e:
            return e.code


class MultiRun:
    """One multi-mode run on a synthetic map, inside its own temporary directory."""

    def __init__(self, squares, max_bases=None, mpp=MPP, size=SIZE, extra=()):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        image = os.path.join(self.dir, "flood.png")
        make_map(image, squares, size)
        tail = [image, "--mode", "multi", "--session-root", os.path.join(self.dir, "sessions")]
        if max_bases is not None:
            tail += ["--max-bases", str(max_bases)]
        self.exit = run_main(tail, self.dir, mpp=mpp, extra=extra)
        sessions = sorted(glob.glob(os.path.join(self.dir, "sessions", "session_*")))
        self.session = sessions[-1] if sessions else None
        self.summary = None
        if self.session and os.path.exists(os.path.join(self.session, "run_summary.json")):
            with open(os.path.join(self.session, "run_summary.json")) as f:
                self.summary = json.load(f)

    def missions(self):
        return sorted(glob.glob(os.path.join(self.session, "base*.waypoints")))

    def sidecar(self, mission):
        with open(mission[:-len(".waypoints")] + ".route.json") as f:
            return json.load(f)

    def close(self):
        self.tmp.cleanup()


class MultiRunCase(unittest.TestCase):
    def run_multi(self, *args, **kwargs):
        run = MultiRun(*args, **kwargs)
        self.addCleanup(run.close)
        return run

    def assert_valid_missions(self, run):
        """Every mission passes the independent validator against its own route, drops are not duplicated
        across bases, routes do not cross, and the summary's visited count matches the files."""
        missions = run.missions()
        self.assertEqual(len(missions), len(run.summary["bases"]))
        drop_positions, paths = [], []
        for mission in missions:
            sidecar = run.sidecar(mission)
            findings, items = wpl_validator.validate(mission, sidecar, servo_channel=9, servo_pwm=2000)
            self.assertEqual([f for f in findings if f.level == "ERROR"], [], mission)
            drop_positions += [(r["lat"], r["lon"]) for r in items if r["cmd"] == wpl_validator.DO_SET_SERVO]
            paths.append([tuple(p) for p in sidecar["full_path"]])
            # Export uses DEVELOPMENT's georeference: centre (W-1)/2, metres per display pixel.
            self.assertEqual(sidecar["image_center_px"], [(SIZE - 1) / 2, (SIZE - 1) / 2])
            self.assertAlmostEqual(sidecar["meters_per_pixel"], run.summary["meters_per_pixel_display"])
        self.assertEqual(len(drop_positions), len(set(drop_positions)), "a drop point is served twice")
        self.assertEqual(len(drop_positions), run.summary["drops_visited"])
        for i, a in enumerate(paths):
            for b in paths[i + 1:]:
                self.assertEqual(polyline_intersections(a, b), [])
        limit = 1.0 - settings.BATTERY_RESERVE_FRACTION
        self.assertTrue(all(f <= limit + 1e-9 for f in run.summary["routed_sortie_battery"]))
        return paths


class TestSingleUav(MultiRunCase):
    def test_max_bases_one_gives_one_base_one_route_one_mission(self):
        run = self.run_multi(WEST + EAST, max_bases=1)
        self.assertEqual(run.exit, 0)
        self.assertEqual(run.summary["outcome"], "SINGLE_BASE")
        self.assertEqual(len(run.summary["bases"]), 1)
        self.assertEqual(len(run.missions()), 1)
        self.assertTrue(all(owner == 1 for owner in run.summary["assignment"].values()))
        # The one base cannot reach the far cluster: those drops are reported, not silently dropped.
        self.assertEqual(len(run.summary["assignment"]) + len(run.summary["unreachable_by_any_base"]),
                         len(run.summary["drop_points"]))
        self.assertTrue(run.summary["unreachable_by_any_base"])
        self.assertTrue(all("cap was reached" in r for r in run.summary["unreachable_by_any_base"].values()))
        self.assert_valid_missions(run)

    def test_all_drops_reachable_from_one_site_needs_one_base_even_with_cap_four(self):
        run = self.run_multi(WEST)
        self.assertEqual(run.exit, 0)
        self.assertEqual(run.summary["max_bases"], settings.MAX_BASES)
        self.assertEqual(run.summary["selection"]["mode"], "single")
        self.assertEqual(len(run.missions()), 1)
        self.assertEqual(run.summary["drops_visited"], len(run.summary["drop_points"]))
        self.assertEqual(run.summary["unreachable_by_any_base"], {})
        self.assert_valid_missions(run)


class TestTwoUavs(MultiRunCase):
    def test_two_far_clusters_get_two_bases_exclusive_assignment_and_valid_missions(self):
        run = self.run_multi(WEST + EAST, max_bases=4)
        self.assertEqual(run.exit, 0)
        self.assertEqual(run.summary["outcome"], "MULTI_BASE_2")
        self.assertEqual(run.summary["unreachable_by_any_base"], {})
        self.assertEqual(len(run.summary["assignment"]), len(run.summary["drop_points"]))
        self.assertEqual(run.summary["drops_visited"], len(run.summary["drop_points"]))
        self.assertGreaterEqual(run.summary["base_separation_min_px"], settings.MIN_BASE_SEPARATION_PX)
        paths = self.assert_valid_missions(run)
        # Each cluster is served entirely by one base, and the two bases serve different clusters.
        drops = [tuple(p) for p in run.summary["drop_points"]]
        owner = {drops[int(i)]: b for i, b in run.summary["assignment"].items()}
        west = {b for p, b in owner.items() if p[0] < SIZE / 2}
        east = {b for p, b in owner.items() if p[0] >= SIZE / 2}
        self.assertEqual(len(west), 1)
        self.assertEqual(len(east), 1)
        self.assertNotEqual(west, east)
        self.assertEqual(run.summary["territory_violations"], [])
        self.assertTrue(os.path.exists(os.path.join(run.session, "mission_route_map.png")))
        self.assertNotIn("error", run.summary["map"])  # rendered (basemap_sanity targets real maps, not white images)
        self.assertEqual(len(paths), 2)


class TestMultipleUavs(MultiRunCase):
    def test_three_separated_clusters_get_three_bases(self):
        run = self.run_multi(WEST + EAST + SOUTH, max_bases=4)
        self.assertEqual(run.exit, 0)
        self.assertEqual(run.summary["outcome"], "MULTI_BASE_3")
        self.assertEqual(run.summary["drops_visited"], len(run.summary["drop_points"]))
        self.assertEqual(sorted(b["base"] for b in run.summary["bases"]), [1, 2, 3])
        self.assert_valid_missions(run)

    def test_base_cap_limits_the_number_of_uavs(self):
        squares = NORTH_WEST + NORTH_EAST + WEST + EAST + SOUTH
        capped = self.run_multi(squares, max_bases=2)
        self.assertEqual(capped.exit, 0)
        self.assertLessEqual(len(capped.summary["bases"]), 2)
        self.assertEqual(len(capped.summary["assignment"]) + len(capped.summary["unreachable_by_any_base"]),
                         len(capped.summary["drop_points"]))
        self.assertTrue(capped.summary["unreachable_by_any_base"])
        self.assert_valid_missions(capped)
        free = self.run_multi(squares, max_bases=4)
        self.assertEqual(free.exit, 0)
        self.assertGreater(len(free.summary["bases"]), 2)
        self.assertGreater(free.summary["drops_visited"], capped.summary["drops_visited"])
        self.assert_valid_missions(free)


class TestBatteryConstraints(unittest.TestCase):
    def straight(self, mpp):
        return lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1])) * mpp

    def test_drop_beyond_single_sortie_range_is_never_assigned(self):
        mask = np.zeros((400, 1200), np.uint8)
        drops = [(110, 200), (400, 200)]  # 2nd drop: 290 px = 1160 m from the only site
        result = select_bases([(100, 200)], drops, self.straight(4.0), mask, headwind_mps=0.0, wind_from_deg=None,
                              per_drop_kg=0.25, payload_capacity_kg=8.0, reserve=0.2, meters_per_pixel=4.0,
                              max_bases=4, min_separation_px=150)
        self.assertEqual(result["assignment"], {0: 0})
        self.assertIn(1, result["unreachable"])
        d = 290 * 4.0
        self.assertGreater(leg_battery_fraction(d, 0.25) + leg_battery_fraction(d), 0.8)

    def test_headwind_shrinks_reach(self):
        drops = [(100 + 200, 200)]
        calm = single_drop_reach([(100, 200)], drops, self.straight(4.0), headwind_mps=0.0, per_drop_kg=0.25,
                                 reserve=0.2)
        windy = single_drop_reach([(100, 200)], drops, self.straight(4.0), headwind_mps=10.0, per_drop_kg=0.25,
                                  reserve=0.2)
        self.assertEqual(calm, [{0}])
        self.assertEqual(windy, [set()])

    def test_planner_inserts_reload_instead_of_breaching_reserve(self):
        current = np.zeros((300, 300), np.uint8)
        home = (150, 150)
        drops = [(150, 10), (290, 150), (150, 290), (10, 150)]  # 140 px = 560 m each way at 4 m/px
        route = plan_mission_stops(home, drops, {0.0: current}, current, meters_per_pixel=4.0)
        self.assertEqual(len(route.drop_indices), 4)
        self.assertTrue(route.reload_indices)
        self.assertTrue(all(s <= 0.8 + 1e-9 for s in route.sortie_battery))


class TestUnreachableWaypoints(MultiRunCase):
    def test_drop_sealed_off_by_flood_is_reported_and_everything_else_is_flown(self):
        # Zone Z sits in the image corner, sealed off by an L-shaped flood band 12 px away: its dry drop
        # point has no flood-free connection to any base site (and no site in the pocket has clearance).
        pocket = 713
        l_band = [(600, 600, 749, pocket - 1), (600, pocket, pocket - 1, 749)]
        zone = [(725, 725, 745, 745)]
        run = self.run_multi(WEST + l_band + zone)
        self.assertEqual(run.exit, 0)
        drops = [tuple(p) for p in run.summary["drop_points"]]
        unreachable = {int(i) for i in run.summary["unreachable_by_any_base"]}
        self.assertTrue(unreachable)
        for i, (x, y) in enumerate(drops):
            in_pocket = x >= pocket and y >= pocket
            self.assertEqual(i in unreachable, in_pocket, (i, (x, y)))
        self.assertTrue(all("unreachable by any base" in r for r in run.summary["unreachable_by_any_base"].values()))
        self.assertEqual(run.summary["drops_visited"], len(drops) - len(unreachable))
        self.assertEqual(run.summary["legs"]["fallback"], 0)
        self.assertEqual(run.summary["route_flooded_px_current"], 0)
        self.assert_valid_missions(run)


class TestOverlapRepair(unittest.TestCase):
    @staticmethod
    def straight_plan(number, home, drops):
        path = [tuple(home)] + [tuple(d) for d in drops] + [tuple(home)]
        return {"base": number, "home": tuple(home), "full_path": path, "stop_indices": list(range(len(path))),
                "route_points": list(path), "drop_indices": list(range(1, len(path) - 1)), "reload_indices": []}

    def test_tier0_reassignment_removes_crossing_and_keeps_every_drop(self):
        bases = [(10, 50), (90, 50)]
        safe_points = [(90, 10), (10, 10)]
        selection = {"assignment": {0: 0, 1: 1}}

        def plan_assigned(b):
            assigned = [safe_points[i] for i, owner in sorted(selection["assignment"].items()) if owner == b]
            return self.straight_plan(b + 1, bases[b], assigned) if assigned else None

        plans = {0: plan_assigned(0), 1: plan_assigned(1)}
        self.assertTrue(polyline_intersections(plans[0]["full_path"], plans[1]["full_path"]))
        dist = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))
        reach = single_drop_reach(bases, safe_points, dist, headwind_mps=0.0, per_drop_kg=0.25, reserve=0.2)
        repairs, remaining = execute_overlap_repairs(
            plans, bases, selection, safe_points, reach, {p: i for i, p in enumerate(safe_points)},
            np.zeros((100, 100), np.uint8), None, np.full((100, 100), 30.0, np.float32), dist, 0.0,
            lambda n, h, d: self.straight_plan(n, h, d), plan_assigned, (100, 100), meters_per_pixel=1.0)
        self.assertEqual(remaining, [])
        self.assertEqual([r["tier"] for r in repairs], [0])
        self.assertEqual(sorted(selection["assignment"]), [0, 1])  # both drops still assigned
        live = [p for p in plans.values() if p]
        self.assertEqual(sum(len(p["drop_indices"]) for p in live), 2)

    def test_nudged_base_reach_is_refreshed(self):
        clearance = np.full((150, 150), 30.0, dtype=np.float32)
        plan1 = {"base": 1, "home": (20, 20), "full_path": [(x, x) for x in range(20, 65, 5)],
                 "drop_indices": [8], "reload_indices": []}  # no stop_indices: Tier 1 cannot reroute
        plan2 = {"base": 2, "home": (20, 60), "full_path": [(x, 80 - x) for x in range(20, 65, 5)],
                 "drop_indices": [8], "reload_indices": []}
        plans = {0: plan1, 1: plan2}
        bases = [(20, 20), (20, 60)]
        safe_points = [(60, 60), (60, 20)]
        dist = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1])) * 2.0

        def plan_base(num, home, assigned):
            if tuple(home) == (20, 20):
                return plan1
            return {"base": num, "home": tuple(home), "full_path": [tuple(home), (home[0], assigned[0][1]),
                                                                    tuple(assigned[0])],
                    "drop_indices": [2], "reload_indices": []}

        reach = [set(), set()]  # stale on purpose: nothing reachable, so Tier 0 cannot move drops
        with mock.patch.object(settings, "MIN_BASE_SEPARATION_PX", 30.0):
            repairs, remaining = execute_overlap_repairs(
                plans, bases, {"assignment": {0: 0, 1: 1}}, safe_points, reach, {}, np.zeros((150, 150), np.uint8),
                None, clearance, dist, 0.0, plan_base, lambda b: plans[b], (150, 150), meters_per_pixel=2.0)
        self.assertEqual(remaining, [])
        self.assertEqual(repairs[0]["tier"], 2)
        expected = single_drop_reach([bases[0]], safe_points, dist, headwind_mps=0.0,
                                     per_drop_kg=settings.PAYLOAD_PER_DROP_KG,
                                     reserve=settings.BATTERY_RESERVE_FRACTION)[0]
        self.assertNotEqual(bases[0], (20, 20))
        self.assertEqual(reach[0], expected)
        self.assertTrue(expected)

    def test_repairs_write_no_files(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            p1 = [(x, 50) for x in range(10, 105, 5)]
            p2 = [(50, y) for y in range(10, 105, 5)]
            plans = {0: {"base": 1, "home": p1[0], "full_path": p1, "stop_indices": [0, len(p1) - 1],
                         "route_points": [p1[0], p1[-1]], "drop_indices": [len(p1) - 1], "reload_indices": []},
                     1: {"base": 2, "home": p2[0], "full_path": p2, "stop_indices": [0, len(p2) - 1],
                         "route_points": [p2[0], p2[-1]], "drop_indices": [len(p2) - 1], "reload_indices": []}}
            repairs, remaining = execute_overlap_repairs(
                plans, [p1[0], p2[0]], {"assignment": {}}, [], [set(), set()], {}, np.zeros((120, 120), np.uint8),
                None, np.zeros((120, 120), np.float32), lambda a, b: 1.0, 0.0, lambda *a: None, lambda b: None,
                (120, 120), meters_per_pixel=2.0)
            self.assertEqual(remaining, [])
            self.assertEqual(repairs[0]["tier"], 1)
            self.assertEqual(os.listdir(tmp), [])


class TestOverlapGate(MultiRunCase):
    def test_unresolved_crossing_exits_2_and_exports_nothing(self):
        fake_hit = mock.patch.object(main_module, "polyline_intersections", return_value=[(1.0, 1.0)])
        run = self.run_multi(WEST + EAST, max_bases=4, extra=[fake_hit])
        self.assertEqual(run.exit, 2)
        self.assertEqual(run.summary["outcome"], "UNRESOLVED_PATH_OVERLAP")
        self.assertEqual(run.missions(), [])


class TestEmptyAndSmallMissions(MultiRunCase):
    def test_no_flood_ends_before_any_session_is_created(self):
        run = self.run_multi([])
        self.assertEqual(run.exit, 0)
        self.assertIsNone(run.session)

    def test_single_flood_zone_gives_one_base_one_drop(self):
        run = self.run_multi([(350, 350, 385, 385)])
        self.assertEqual(run.exit, 0)
        self.assertEqual(len(run.summary["drop_points"]), 1)
        self.assertEqual(run.summary["drops_visited"], 1)
        self.assertEqual(run.summary["reloads"], 0)
        self.assert_valid_missions(run)

    def test_no_base_site_with_clearance_exits_2_without_missions(self):
        # Flood stripes every 20 px: no dry pixel is 15 px from flood.
        stripes = [(0, y, SIZE - 1, y + 9) for y in range(0, SIZE, 20)]
        run = self.run_multi(stripes)
        self.assertEqual(run.exit, 2)
        self.assertEqual(run.summary["outcome"], "NO_CANDIDATES_SAMPLED")
        self.assertEqual(run.missions(), [])


class TestScaleConvention(MultiRunCase):
    def test_resized_input_plans_and_exports_in_metres_per_display_pixel(self):
        # 1500 px input shown at 750 px: each display pixel covers 2 original pixels.
        run = self.run_multi([(2 * x0, 2 * y0, 2 * x1, 2 * y1) for x0, y0, x1, y1 in WEST], max_bases=1,
                             mpp=MPP / 2, size=2 * SIZE)
        self.assertEqual(run.exit, 0)
        self.assertAlmostEqual(run.summary["scale_factor"], 0.5)
        self.assertAlmostEqual(run.summary["meters_per_pixel_display"], MPP)
        self.assert_valid_missions(run)


class TestCommandLine(unittest.TestCase):
    def test_multi_only_options_are_rejected_in_single_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("sys.stderr"):
                self.assertEqual(run_main(["x.png", "--max-bases", "2"], tmp), 2)
                self.assertEqual(run_main(["x.png", "--session-root", "s"], tmp), 2)
                self.assertEqual(run_main(["x.png", "--mode", "multi", "--max-bases", "0"], tmp), 2)

    def test_single_mode_is_unchanged_and_writes_no_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = os.path.join(tmp, "flood.png")
            make_map(image, WEST + EAST)
            self.assertEqual(run_main([image], tmp), 0)
            out = os.path.join(tmp, "data", "output", "enriched_drone_mission.waypoints")
            findings, items = wpl_validator.validate(out)
            self.assertEqual([f for f in findings if f.level == "ERROR"], [])
            self.assertEqual(sum(r["cmd"] == wpl_validator.DO_SET_SERVO for r in items), len(WEST + EAST))
            self.assertFalse(glob.glob(os.path.join(tmp, "data", "output", "session_*")))


class TestPreservedInterfaces(unittest.TestCase):
    def test_reload_free_export_is_byte_identical(self):
        path = [(5, 5), (40, 5), (40, 30), (5, 5)]
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a.wp"), os.path.join(tmp, "b.wp")
            mission_output.generate_mission_file(path, [0, 1, 2, 3], (5, 5), (24.5, 24.5), GEO, 2.0, filename=a)
            mission_output.generate_mission_file(path, [0, 1, 2, 3], (5, 5), (24.5, 24.5), GEO, 2.0, filename=b,
                                                 reload_indices=None)
            with open(a, "rb") as fa, open(b, "rb") as fb:
                self.assertEqual(fa.read(), fb.read())

    def test_reload_index_must_be_an_intermediate_return_to_home(self):
        path = [(5, 5), (40, 5), (5, 5), (5, 5), (60, 5), (5, 5)]
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "m.wp")
            for bad in ([1], [0], [5], [4]):
                with self.assertRaises(ValueError):
                    mission_output.generate_mission_file(path, [1, 4], (5, 5), (24.5, 24.5), GEO, 2.0, filename=out,
                                                         reload_indices=bad)
                self.assertFalse(os.path.exists(out))
            mission_output.generate_mission_file(path, [1, 4], (5, 5), (24.5, 24.5), GEO, 2.0, filename=out,
                                                 reload_indices=[2])
            sidecar = {"full_path": path, "drop_indices": [1, 4], "reload_indices": [2], "home": [5, 5],
                       "image_center_px": [24.5, 24.5], "geo_center": list(GEO), "meters_per_pixel": 2.0}
            findings, _ = wpl_validator.validate(out, sidecar)
            self.assertEqual([f for f in findings if f.level == "ERROR"], [])
            # Without declaring reloads, the independent validator still rejects an intermediate LAND.
            findings, _ = wpl_validator.validate(out)
            self.assertTrue(any(f.check == "return" and f.level == "ERROR" for f in findings))

    def test_legacy_drop_selection_without_mask_is_unchanged(self):
        from src.mission.safe_dropzone import find_safe_drop_points
        hull = np.array([[10, 10], [30, 10], [30, 30], [10, 30]])
        points = find_safe_drop_points([hull], [0], [(0, 0), (50, 0)])
        self.assertEqual(points, [(10.0, 10.0)])  # hull vertex closest to the path, as before
        with self.assertRaises(TypeError):
            find_safe_drop_points([hull], [0], [(0, 0), (50, 0)], raw_contours=[hull])

    def test_snap_default_radius_is_still_ten(self):
        obstacles = {(x, y) for x in range(30) for y in range(30)} - {(11, 0)}
        self.assertEqual(snap_to_nearest_free_cell((0, 0), obstacles, 30, 30), ((0, 0), 10))
        self.assertEqual(snap_to_nearest_free_cell((0, 0), obstacles, 30, 30, 11), ((11, 0), 11))


class TestForecastFrames(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((100, 100), np.uint8)
        self.mask[40:60, 40:60] = 255
        self.weather = {"precipitation": 10.0, "wind_speed_10m": 30.0, "wind_direction_10m": 270.0}

    def test_hourly_frames_are_developments_model(self):
        frames = forecast_frames(self.mask, self.weather, 3.0)
        self.assertEqual(sorted(frames), [0.0, 1.0, 2.0, 3.0])
        self.assertTrue(np.array_equal(frames[0.0], self.mask))
        for k in (1, 2, 3):
            self.assertTrue(np.array_equal(frames[float(k)], predict_spread(self.mask, self.weather, k)))

    def test_last_frame_is_the_single_mode_prediction(self):
        for horizon, times in ((2.5, [0.0, 1.0, 2.0]), (0.5, [0.0, 0.5]), (2.0, [0.0, 1.0, 2.0])):
            frames = forecast_frames(self.mask, self.weather, horizon)
            self.assertEqual(sorted(frames), times)
            self.assertTrue(np.array_equal(frames[max(frames)], predict_spread(self.mask, self.weather, horizon)))

    def test_non_positive_horizon_keeps_the_one_iteration_as_present(self):
        frames = forecast_frames(self.mask, self.weather, 0)
        self.assertEqual(list(frames), [0.0])
        self.assertTrue(np.array_equal(frames[0.0], predict_spread(self.mask, self.weather, 0)))

    def test_frames_are_cumulative_and_queries_advance_with_time(self):
        frames = forecast_frames(self.mask, self.weather, 2.0)
        areas = [np.count_nonzero(obstacle_mask_at(frames, t)) for t in np.linspace(0, 2.5, 11)]
        self.assertTrue(all(b >= a for a, b in zip(areas, areas[1:])))
        self.assertEqual(areas[0], np.count_nonzero(self.mask))
        self.assertEqual(areas[-1], np.count_nonzero(frames[2.0]))


class TestPortedInterfaces(unittest.TestCase):
    """MAIN tests whose files share names with DEVELOPMENT tests (test_flood_spread, test_dstarlite,
    test_mission_output); kept here so DEVELOPMENT's files stay untouched."""

    def test_arrival_time_selects_future_obstacle_frame(self):
        a = np.zeros((2, 2), dtype=np.uint8)
        b = np.ones((2, 2), dtype=np.uint8) * 255
        self.assertFalse(obstacle_mask_at({0.5: a, 1.0: b}, 0.6).any())
        self.assertTrue(obstacle_mask_at({0.5: a, 1.0: b}, 0.9).any())
        self.assertTrue(obstacle_mask_at({0.5: a, 1.0: b}, 5.0).any())

    def test_routed_path_home_only_uses_three_value_contract(self):
        result = compute_routed_path([(4, 4)], np.zeros((10, 10), dtype=np.uint8))
        self.assertEqual(result, ([(4, 4)], [0], []))

    def test_interpolated_mask_is_blended_then_thresholded(self):
        current = np.zeros((5, 5), dtype=np.uint8)
        low = current.copy(); low[2, 2] = 42
        higher = current.copy(); higher[2, 2] = 85
        grown = current.copy(); grown[2, 2] = 255
        frames = {0.0: current, 1/12: low, 1/6: higher, .5: grown}
        sampled, details = forecast_at(frames, .12, current)
        self.assertEqual(details["source"], "interpolated")
        self.assertAlmostEqual(details["lower_hours"], 1/12)
        self.assertAlmostEqual(details["upper_hours"], 1/6)
        self.assertEqual(int(sampled[2, 2]), 0)

    def test_obstacle_mask_interpolation_uses_alpha_before_binary_threshold(self):
        lower = np.zeros((2, 2), dtype=np.uint8)
        upper = np.zeros_like(lower)
        upper[0, 0] = 255
        frames = {0.0: lower, 1.0: upper}
        early, _ = interpolate_forecast_mask(frames, 0.4)
        late, _ = interpolate_forecast_mask(frames, 0.8)
        self.assertEqual(int(early[0, 0]), 102)
        self.assertEqual(int(late[0, 0]), 204)
        self.assertEqual(int(obstacle_mask_at(frames, 0.4)[0, 0]), 0)
        self.assertEqual(int(obstacle_mask_at(frames, 0.8)[0, 0]), 255)

    def test_sorties_have_distinct_colors_and_base_reload_labels(self):
        image = np.zeros((40, 110, 3), dtype=np.uint8)
        path = [(5, 20), (20, 20), (30, 20), (45, 20), (100, 20)]
        with mock.patch("src.mission.mission_output.plt.show"), \
                mock.patch("src.mission.mission_output.cv2.putText", wraps=cv2.putText) as put_text:
            rendered = mission_output.display_multi_base_map(
                image, [], None, [{"home": (5, 5), "full_path": path, "drop_indices": [1, 3], "reload_indices": [2]}],
                show=False)
        self.assertTupleEqual(tuple(rendered[20, 10]), (0, 0, 213))  # sortie 1: red #d50000
        self.assertTupleEqual(tuple(rendered[20, 65]), (126, 35, 26))  # sortie 2: navy #1a237e, inside an RTB dash
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("BASE", labels)
        self.assertIn("RELOAD", labels)

    def test_map_labels_use_real_base_numbers(self):
        image = np.zeros((60, 200, 3), dtype=np.uint8)
        bases = [{"number": 1, "home": (10, 30), "full_path": [(10, 30), (40, 30), (10, 30)], "drop_indices": [1],
                  "reload_indices": []},
                 {"number": 3, "home": (150, 30), "full_path": [(150, 30), (180, 30), (150, 30)], "drop_indices": [1],
                  "reload_indices": []}]
        with mock.patch("src.mission.mission_output.plt.show"), \
                mock.patch("src.mission.mission_output.cv2.putText", wraps=cv2.putText) as put_text:
            mission_output.display_multi_base_map(image, [], None, bases, show=False)
        labels = {call.args[1] for call in put_text.call_args_list}
        self.assertIn("BASE 3", labels)
        self.assertNotIn("BASE 2", labels)

    def test_routed_distance_mapping_follows_drop_semantics_across_reload(self):
        class Route:
            points = [(0, 0), (10, 0), (0, 0), (0, 0), (20, 0), (0, 0)]
            drop_indices = [1, 4]

        mapped = main_module.map_routed_drop_distances(Route(), [(10, 0), (20, 0)], [10, 11, 0, 30, 31, 0], (0, 0))
        self.assertEqual(mapped[0], 10.0)
        self.assertEqual(mapped[1], 30.0)
        self.assertEqual(mapped[("home", 1)], 30.0)

    def test_concurrent_session_folders_never_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders = {main_module.create_session_dir(tmp) for _ in range(5)}
            self.assertEqual(len(folders), 5)
            self.assertTrue(all(f.is_dir() for f in folders))


if __name__ == "__main__":
    unittest.main()
