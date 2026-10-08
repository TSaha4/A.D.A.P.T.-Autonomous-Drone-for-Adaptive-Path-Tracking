"""Regression tests for the defects found by the October 2026 adversarial QA pass (see docs/QA_REPORT.md).

Each test reproduces one defect: an uncaught exception or a silent loss of information for an input that the
program can receive (command line, environment, weather API, weather file, file system), or a bookkeeping error
in the multi-base mode. All end-to-end cases run the unmodified main.main(); only I/O is stubbed.
"""
import contextlib
import json
import math
import os
import subprocess
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
from src.mission.multi_base import single_drop_reach
from src.mission.overlap_repair import execute_overlap_repairs
from src.vision.image_processing import ImageProcessor
from src.weather import weather_api
from src.weather.flood_spread import forecast_frames, predict_spread

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALM = {"precipitation": 0.0, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0, "status": "success"}
OUT = os.path.join("data", "output", "enriched_drone_mission.waypoints")


def make_map(path, squares, size=(750, 750)):
    img = np.full((size[1], size[0], 3), 255, np.uint8)
    for x0, y0, x1, y1 in squares:
        img[y0:y1 + 1, x0:x1 + 1] = (0, 0, 255)
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(path)  # also works for non-ASCII paths on Windows


def fake_select(self, image):
    self.image = image.copy()
    self.hsv_lower = np.array([0, 100, 100], np.uint8)
    self.hsv_upper = np.array([179, 255, 255], np.uint8)
    self.is_red_wrap = True
    return self.hsv_lower, self.hsv_upper


def run_main(args, cwd, weather=CALM, mpp=4.0):
    """Returns the exit status; any exception other than SystemExit propagates (and fails the test)."""
    argv = ["main.py"] + list(args) + ["--lat", "25.0", "--lon", "82.0"]
    with contextlib.chdir(cwd), \
            mock.patch.object(sys, "argv", argv), \
            mock.patch.object(settings, "METERS_PER_PIXEL", mpp), \
            mock.patch.object(ImageProcessor, "select_sample_points", fake_select), \
            mock.patch.object(main_module, "get_weather_data", return_value=weather), \
            mock.patch.object(main_module, "show_image_safe"), \
            mock.patch.object(main_module, "display_path_on_map"), \
            mock.patch("src.mission.mission_output.plt.show"), \
            mock.patch("sys.stderr"):
        try:
            main_module.main()
            return 0
        except SystemExit as e:
            return e.code


class Case(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.image = os.path.join(self.tmp, "flood.png")
        make_map(self.image, [(100, 100, 150, 150), (500, 500, 540, 540)])

    def tearDown(self):
        self._tmp.cleanup()


class TestCommandLineAndEnvironment(Case):
    def test_non_finite_horizon_is_rejected_before_any_work(self):
        """QA-01: --horizon nan/inf crashed deep in the spread rule (ValueError/OverflowError)."""
        for mode in ("single", "multi"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(mode=mode, horizon=value):
                    self.assertEqual(run_main([self.image, "--mode", mode, "--horizon", value], self.tmp), 2)

    def test_session_root_that_is_a_file_is_rejected_before_any_work(self):
        """QA-06: an existing file given as --session-root failed with FileExistsError after the whole run."""
        blocker = os.path.join(self.tmp, "blocker")
        open(blocker, "w").close()
        with mock.patch.object(ImageProcessor, "select_sample_points", side_effect=AssertionError("too late")):
            code = run_main([self.image, "--mode", "multi", "--session-root", blocker], self.tmp)
        self.assertEqual(code, 2)

    def test_invalid_numeric_environment_variable_gives_a_clear_message(self):
        """QA-05: METERS_PER_PIXEL=abc (and the like) failed at import with a bare float()/int() traceback."""
        for var, value in (("METERS_PER_PIXEL", "abc"), ("MAX_BASES", "2.5"), ("HOME_CLEARANCE_PX", "x")):
            with self.subTest(var=var):
                r = subprocess.run([sys.executable, "main.py", "--help"], cwd=ROOT, capture_output=True, text=True,
                                   env=dict(os.environ, **{var: value}))
                self.assertNotEqual(r.returncode, 0)
                self.assertNotIn("Traceback", r.stderr)
                self.assertIn(var, r.stderr)

    def test_image_under_a_non_ascii_path_is_read(self):
        """QA-07: cv2.imread cannot open non-ASCII paths on Windows; the run reported 'Error loading image file'."""
        folder = os.path.join(self.tmp, "बाढ़ मानचित्र")
        os.mkdir(folder)
        image = os.path.join(folder, "map.png")
        make_map(image, [(100, 100, 150, 150)])
        self.assertEqual(run_main([image], self.tmp), 0)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, OUT)))


class TestWeatherInputs(Case):
    BAD = [{"precipitation": None, "wind_speed_10m": 5.0, "wind_direction_10m": 90.0, "status": "success"},
           {"precipitation": 1.0, "wind_speed_10m": None, "wind_direction_10m": 90.0, "status": "success"},
           {"precipitation": 1.0, "wind_speed_10m": 5.0, "wind_direction_10m": None, "status": "success"},
           {"precipitation": float("nan"), "wind_speed_10m": float("inf"), "wind_direction_10m": 90.0},
           {"precipitation": "3", "wind_speed_10m": True, "wind_direction_10m": [1]},
           {"status": "success"}]

    def test_incomplete_or_non_numeric_weather_does_not_crash(self):
        """QA-02: a null/NaN/missing weather field raised TypeError/ValueError/KeyError in main."""
        for mode in ("single", "multi"):
            for weather in self.BAD:
                with self.subTest(mode=mode, weather=weather):
                    self.assertEqual(run_main([self.image, "--mode", mode], self.tmp, weather=weather), 0)

    def test_api_nulls_become_calm_values_with_a_warning(self):
        response = mock.Mock()
        response.json.return_value = {"current": {"precipitation": None, "wind_speed_10m": 12.5,
                                                  "wind_direction_10m": float("nan")}}
        response.raise_for_status.return_value = None
        weather_api.get_weather_data.cache_clear()
        try:
            with mock.patch("src.weather.weather_api.requests.get", return_value=response), \
                    self.assertLogs(level="WARNING") as logs:
                w = weather_api.get_weather_data(10.0, 20.0)
        finally:
            weather_api.get_weather_data.cache_clear()
        self.assertEqual((w["precipitation"], w["wind_speed_10m"], w["wind_direction_10m"]), (0.0, 12.5, 0.0))
        self.assertEqual(w["status"], "partial")
        self.assertTrue(any("precipitation" in line for line in logs.output))

    def test_malformed_dummy_weather_file_does_not_crash(self):
        """QA-03: a dummy weather file without all fields (or not an object) raised KeyError/TypeError."""
        os.makedirs(os.path.join(self.tmp, "data", "input"))
        for content in ("{}", '{"precipitation": 5}', "[]", '"text"', '{"precipitation": null, '
                        '"wind_speed_10m": 1, "wind_direction_10m": 0}'):
            with self.subTest(content=content):
                with open(os.path.join(self.tmp, "data", "input", "dummy_weather.json"), "w") as f:
                    f.write(content)
                self.assertEqual(run_main([self.image, "--dummy-weather"], self.tmp), 0)

    def test_extreme_precipitation_does_not_exhaust_memory(self):
        """QA-04: precipitation 1e6 mm built a ~400 000 px dilation element (160 GB allocation, cv2 OOM)."""
        mask = np.zeros((60, 80), np.uint8)
        mask[20:30, 20:30] = 255
        out = predict_spread(mask, {"precipitation": 1e6, "wind_speed_10m": 0, "wind_direction_10m": 0}, 2.0)
        self.assertTrue(np.all(out == 255))  # growth beyond the image diagonal floods every pixel
        # clamping the element must not change any result inside the image
        moderate = {"precipitation": 400.0, "wind_speed_10m": 0, "wind_direction_10m": 0}  # 80 px, below the 100 px diagonal
        ref = mask.copy()
        for _ in range(2):
            ref = cv2.dilate(ref, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (161, 161)))
        self.assertTrue(np.array_equal(predict_spread(mask, moderate, 2.0), ref))
        w = {"precipitation": 1e6, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0, "status": "success"}
        self.assertEqual(run_main([self.image], self.tmp, weather=w), 0)


class TestMultiBaseBookkeeping(unittest.TestCase):
    @staticmethod
    def plan(number, home, drops):
        path = [tuple(home)] + [tuple(d) for d in drops] + [tuple(home)]
        return {"base": number, "home": tuple(home), "full_path": path, "stop_indices": list(range(len(path))),
                "route_points": list(path), "drop_indices": list(range(1, len(path) - 1)), "reload_indices": []}

    def test_coincident_drop_points_are_reassigned_together(self):
        """QA-09: two regions can share a drop pixel; Tier 0 moved only one of them (coordinate->index map kept
        one index), so the point stayed with both bases and a repairable plan ended as infeasible."""
        bases = [(10, 50), (90, 50)]
        safe = [(90, 10), (90, 10), (10, 10)]
        selection = {"assignment": {0: 0, 1: 0, 2: 1}}

        def assigned(b):
            drops = [safe[i] for i, owner in sorted(selection["assignment"].items()) if owner == b]
            return self.plan(b + 1, bases[b], drops) if drops else None

        plans = {0: assigned(0), 1: assigned(1)}
        dist = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))  # noqa: E731
        reach = single_drop_reach(bases, safe, dist, headwind_mps=0.0, per_drop_kg=0.25, reserve=0.2)
        coord_index = main_module.drop_index_by_point(safe)
        repairs, remaining = execute_overlap_repairs(
            plans, bases, selection, safe, reach, coord_index, np.zeros((100, 100), np.uint8), None,
            np.full((100, 100), 30.0, np.float32), dist, 0.0, self.plan, assigned, (100, 100), meters_per_pixel=1.0)
        self.assertEqual(remaining, [])
        self.assertEqual(selection["assignment"][0], selection["assignment"][1])
        owners = [b for b, p in plans.items() if p and (90, 10) in p["full_path"][1:-1]]
        self.assertEqual(len(owners), 1)

    def test_drops_of_a_base_without_a_plan_are_reported(self):
        """QA-08: when a selected base produced no route (e.g. its drop fails the sortie battery check on the
        exact geometry), its assigned drops disappeared from run_summary.json."""
        with tempfile.TemporaryDirectory() as tmp:
            image = os.path.join(tmp, "flood.png")
            make_map(image, [(70, 330, 105, 365), (620, 330, 655, 365)])
            real = main_module.plan_mission_stops

            def failing_for_second_base(home, *a, **kw):
                route = real(home, *a, **kw)
                if home[0] > 375:  # the eastern base: pretend every drop failed the battery check
                    route.drop_indices, route.unreachable = [], list(range(len(a[0])))
                return route

            with mock.patch.object(main_module, "plan_mission_stops", failing_for_second_base):
                code = run_main([image, "--mode", "multi", "--session-root", os.path.join(tmp, "s")], tmp)
            self.assertEqual(code, 0)
            session = [os.path.join(tmp, "s", d) for d in os.listdir(os.path.join(tmp, "s"))][0]
            with open(os.path.join(session, "run_summary.json")) as f:
                summary = json.load(f)
        drops = len(summary["drop_points"])
        reported = (summary["drops_visited"] + len(summary["unreachable_by_any_base"]) + summary["unreachable"]
                    + summary["connectivity_excluded"] + len(summary["not_planned"]))
        self.assertEqual(reported, drops)
        self.assertTrue(summary["not_planned"])
        self.assertTrue(all("no route" in reason for reason in summary["not_planned"].values()))


class TestForecastFrameCost(unittest.TestCase):
    def test_frames_need_one_spread_iteration_per_hour(self):
        """QA-10: forecast_frames recomputed every frame from scratch (O(N^2) iterations; 7.5 s for 168 h)."""
        mask = np.zeros((60, 60), np.uint8)
        mask[25:35, 25:35] = 255
        w = {"precipitation": 6.0, "wind_speed_10m": 15.0, "wind_direction_10m": 90.0}
        calls = []

        def counting(m, weather, hours):
            calls.append(max(1, int(hours)) if math.isfinite(hours) else None)
            return predict_spread(m, weather, hours)

        frames = forecast_frames(mask, w, 30.0, predictor=counting)
        self.assertLessEqual(sum(calls), 30)
        for k in (1, 7, 30):
            self.assertTrue(np.array_equal(frames[float(k)], predict_spread(mask, w, k)))


if __name__ == "__main__":
    unittest.main()
