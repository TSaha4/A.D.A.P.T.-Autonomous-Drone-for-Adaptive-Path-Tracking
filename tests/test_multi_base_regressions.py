"""Multi-base mode regression tests, ported from MAIN's tests/test_audit_regressions.py.

Adapted to the integrated API: MAIN's multi-base renderer is display_multi_base_map, its router is
compute_routed_path, forecasts are DEVELOPMENT's flood model exposed as frames (forecast_frames), the
servo parameters are DEVELOPMENT's generate_mission_file defaults (9 / 2000), and Varanasi replaces the
Bhopal map (not part of this branch). MAIN's automatic HSV sampling was not ported, so its two tests
are not included.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)

ROOT = Path(__file__).resolve().parents[1]


def blob_mask(seed, shape=(120, 160), blobs=7):
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=np.uint8)
    for _ in range(blobs):
        cx, cy = int(rng.integers(0, shape[1])), int(rng.integers(0, shape[0]))
        axes = (int(rng.integers(5, 30)), int(rng.integers(3, 18)))
        cv2.ellipse(mask, (cx, cy), axes, float(rng.integers(0, 180)), 0, 360, 255, -1)
    return mask


class TestMapRendering(unittest.TestCase):
    def render(self, image, path, drops, home, reloads, tmp):
        from src.mission.mission_output import display_multi_base_map
        save = str(Path(tmp) / "map.png")
        with patch("src.mission.mission_output.plt.show"):
            vis = display_multi_base_map(image.copy(), [], None, [{"home": home, "full_path": path,
                                                                   "drop_indices": drops, "reload_indices": reloads}],
                                         save_path=save, show=False)
        return vis, save

    def test_rendered_map_keeps_real_basemap(self):
        """Permanent guard: the map once silently rendered as a solid-colour pattern."""
        from src.mission.mission_output import basemap_sanity
        image = cv2.imread(str(ROOT / "data" / "input" / "varanasi.png"))
        self.assertIsNotNone(image)
        path = [(300, 250), (320, 250), (340, 260), (360, 270), (340, 260), (300, 250)]
        with TemporaryDirectory() as tmp:
            vis, save = self.render(image, path, [3], (300, 250), [], tmp)
            figure = cv2.imread(save)
            annotated = cv2.imread(str(Path(tmp) / "mission_map_annotated.png"))
        ok, details = basemap_sanity(vis, image)
        self.assertTrue(ok, details)
        self.assertEqual(annotated.shape, image.shape)
        self.assertGreater(details["unchanged_fraction"], 0.9)
        # The saved figure must not be dominated by one colour either.
        _, counts = np.unique(figure.reshape(-1, 3), axis=0, return_counts=True)
        self.assertLess(counts.max() / counts.sum(), 0.5)

    def test_basemap_sanity_rejects_corrupted_renders(self):
        from src.mission.mission_output import basemap_sanity
        image = cv2.imread(str(ROOT / "data" / "input" / "varanasi.png"))
        solid = np.full_like(image, (0, 0, 255))
        self.assertFalse(basemap_sanity(solid, image)[0])
        stripes = np.zeros_like(image)
        stripes[::2] = (255, 0, 255)
        self.assertFalse(basemap_sanity(stripes, image)[0])
        self.assertFalse(basemap_sanity(image[:100], image)[0])

    def test_rtb_leg_made_of_grid_steps_is_visibly_dashed(self):
        image = np.zeros((60, 220, 3), dtype=np.uint8)
        # Outbound to the drop at x=10, then a 200 px RTB leg of 5 px D* Lite steps.
        path = [(5, 30), (10, 30)] + [(x, 30) for x in range(15, 211, 5)]
        with TemporaryDirectory() as tmp:
            vis, _ = self.render(image, path, [1], (210, 30), [], tmp)
        row = vis[30, 30:200]
        coloured = row.max(axis=1) > 128  # ignore anti-aliased fringes
        transitions = int(np.count_nonzero(coloured[1:] != coloured[:-1]))
        self.assertGreaterEqual(transitions, 10, "RTB leg must alternate dash/gap along its length")
        self.assertGreater(np.count_nonzero(~coloured), 30)

    def test_legend_has_entries_only_for_drawn_sorties(self):
        from matplotlib.axes import Axes
        image = np.zeros((40, 120, 3), dtype=np.uint8)
        # Second sortie (indices 4..5) carries no drop and must not get a legend entry.
        path = [(5, 20), (30, 20), (5, 20), (5, 20), (5, 20), (60, 20), (5, 20)]
        captured = {}
        original = Axes.legend

        def spy(ax, *args, **kwargs):
            captured["labels"] = [h.get_label() for h in kwargs["handles"]]
            return original(ax, *args, **kwargs)

        with TemporaryDirectory() as tmp, patch.object(Axes, "legend", spy):
            self.render(image, path, [1, 5], (5, 20), [2, 4], tmp)
        sortie_labels = [label for label in captured["labels"] if label.startswith("Sortie")]
        self.assertEqual(sortie_labels, ["Sortie 1 Outbound", "Sortie 1 RTB",
                                         "Sortie 2 Outbound", "Sortie 2 RTB"])
        self.assertIn("Reload stop", captured["labels"])

    def test_base_and_reload_labels_do_not_overprint(self):
        image = np.zeros((80, 160, 3), dtype=np.uint8)
        path = [(20, 40), (80, 40), (20, 40), (20, 40), (120, 40), (20, 40)]
        with TemporaryDirectory() as tmp, \
                patch("src.mission.mission_output.cv2.putText", wraps=cv2.putText) as put_text:
            self.render(image, path, [1, 4], (20, 40), [2], tmp)
        positions = {}
        for call in put_text.call_args_list:
            positions.setdefault(call.args[1], set()).add(call.args[2])
        self.assertIn("BASE", positions)
        self.assertIn("RELOAD x1", positions)
        self.assertNotIn("RELOAD", positions)
        self.assertTrue(positions["BASE"].isdisjoint(positions["RELOAD x1"]))


class TestMapPalette(unittest.TestCase):
    @staticmethod
    def lab(colour):
        rgb = np.array([[[int(colour[i:i + 2], 16) for i in (1, 3, 5)]]], np.uint8)
        l, a, b = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)[0, 0]
        return np.array([l * 100 / 255, a - 128, b - 128])

    def test_sortie_palette_is_distinct_and_has_no_green(self):
        import itertools
        from src.mission.mission_output import SORTIE_PALETTE_HEX
        fixed = {"drop blue": "#0000ff", "base black": "#000000", "reload cyan": "#00ffff",
                 "label yellow": "#ffff00", "contour green": "#00c800", "forecast brown": "#8c4b1e"}
        self.assertEqual(len(set(SORTIE_PALETTE_HEX)), len(SORTIE_PALETTE_HEX))
        for a, b in itertools.combinations(SORTIE_PALETTE_HEX, 2):
            self.assertGreaterEqual(np.linalg.norm(self.lab(a) - self.lab(b)), 30, (a, b))
        for colour in SORTIE_PALETTE_HEX:
            hue = cv2.cvtColor(np.array([[[int(colour[i:i + 2], 16) for i in (5, 3, 1)]]], np.uint8),
                               cv2.COLOR_BGR2HSV)[0, 0, 0]
            self.assertFalse(20 <= hue <= 100, f"{colour} is green/teal")   # OpenCV hue = degrees / 2
            for name, other in fixed.items():
                self.assertGreaterEqual(np.linalg.norm(self.lab(colour) - self.lab(other)), 35, (colour, name))
        self.assertGreaterEqual(len(SORTIE_PALETTE_HEX), 12)


class TestRoutingConsistency(unittest.TestCase):
    def test_free_point_beyond_wall_is_not_snapped_across(self):
        from src.routing.pathfinding import build_routing_grid, locate_cell
        mask = np.zeros((50, 100), dtype=np.uint8)
        mask[:, 45:55] = 255
        grid = build_routing_grid(mask, 5)
        _, home_comp, _ = locate_cell((10, 20), grid)
        cell, comp, dist = locate_cell((80, 20), grid, home_comp)
        self.assertNotEqual(comp, home_comp)
        self.assertEqual(dist, 0)
        self.assertGreaterEqual(cell[0], 11)

    def test_blocked_cell_snaps_only_with_clear_connector(self):
        from src.routing.pathfinding import build_routing_grid, locate_cell, cell_center, polyline_flood_pixels
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        grid = build_routing_grid(mask, 5)
        point = (41, 30)  # dry pixel right next to the flood, inside a blocked cell
        self.assertIn((8, 6), grid.obstacles)
        cell, comp, dist = locate_cell(point, grid, int(grid.labels[0, 0]))
        self.assertGreater(comp, 0)
        self.assertGreater(dist, 0)
        self.assertEqual(polyline_flood_pixels([point, cell_center(cell, grid)], mask), 0)

    def test_connectivity_filter_agrees_with_dstar_lite(self):
        """Every kept drop routes with D* Lite; every excluded drop cannot."""
        from src.mission.safe_dropzone import filter_drops_by_home_connectivity
        from src.routing.pathfinding import compute_routed_path
        rng = np.random.default_rng(7)
        checked = excluded_checked = 0
        for seed in range(25):
            mask = blob_mask(seed)
            dry = np.argwhere(mask == 0)
            home = tuple(int(v) for v in dry[rng.integers(len(dry))][::-1])
            drops = [tuple(int(v) for v in dry[i][::-1]) for i in rng.integers(len(dry), size=6)]
            kept, excluded = filter_drops_by_home_connectivity(drops, home, mask)
            for drop in kept:
                report = []
                compute_routed_path([home, drop], mask, leg_report=report)
                self.assertNotEqual(report[0]["status"], "fallback", (seed, home, drop, report))
                self.assertEqual(report[0]["route_flooded_px"], 0, (seed, home, drop))
                checked += 1
            for index in excluded:
                report = []
                with self.assertLogs(level="WARNING"):
                    compute_routed_path([home, drops[index]], mask, leg_report=report)
                self.assertEqual(report[0]["status"], "fallback", (seed, home, drops[index]))
                excluded_checked += 1
        self.assertGreater(checked, 50)
        self.assertGreater(excluded_checked, 0)

    def test_unreachable_leg_is_explicit_fallback_with_crossing_count(self):
        from src.routing.pathfinding import compute_routed_path
        mask = np.zeros((60, 60), dtype=np.uint8)
        cv2.rectangle(mask, (30, 10), (55, 50), 255, 4)  # closed ring around the goal
        report = []
        with self.assertLogs(level="WARNING") as captured:
            path, stops, _ = compute_routed_path([(5, 30), (42, 30)], mask, leg_report=report)
        self.assertEqual(report[0]["status"], "fallback")
        self.assertGreater(report[0]["flooded_px"], 0)
        self.assertTrue(any("FALLBACK STRAIGHT-LINE SEGMENT (not a D* Lite route)" in line
                            and "CROSSES" in line for line in captured.output))
        self.assertEqual(path, [(5, 30), (42, 30)])

    def test_dstar_route_is_grid_connected_and_avoids_flood(self):
        from src.routing.pathfinding import compute_routed_path
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (40, 0), (55, 80), 255, -1)
        report = []
        path, stops, _ = compute_routed_path([(10, 50), (90, 50)], mask, leg_report=report)
        self.assertEqual(report[0]["status"], "dstar")
        self.assertEqual(report[0]["route_flooded_px"], 0)
        steps = [np.linalg.norm(np.subtract(b, a)) for a, b in zip(path, path[1:])]
        self.assertLessEqual(max(steps), 5 * np.sqrt(2) + 3)  # grid steps + endpoint connectors
        self.assertGreater(max(p[1] for p in path), 80)  # detours around the wall's open end


class TestDropPlacement(unittest.TestCase):
    def test_large_zone_gets_separated_points_when_first_dry_ring_is_one_notch(self):
        from src.mission.safe_dropzone import find_safe_drop_points
        zone = np.zeros((700, 700), dtype=np.uint8)  # map-sized: band search radius is 3% of size
        cv2.rectangle(zone, (250, 250), (450, 450), 255, -1)
        contour = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        # Horizon mask: the zone grown by 6 px everywhere except a notch on the left side,
        # so the nearest dry ring touches the zone at one spot only.
        horizon = cv2.dilate(zone, np.ones((13, 13), np.uint8))
        horizon[345:355, 240:250] = 0
        points = find_safe_drop_points([contour], [0], [(5, 5)], obstacle_mask=horizon,
                                       raw_contours=[contour], max_drops_per_contour=4,
                                       min_drop_separation_px=40.0)
        self.assertGreaterEqual(len(points), 3, points)
        for x, y in points:
            self.assertEqual(horizon[y, x], 0)
        for i, a in enumerate(points):
            for b in points[i + 1:]:
                self.assertGreaterEqual(np.hypot(a[0] - b[0], a[1] - b[1]), 40.0)


class TestForecastInterpolation(unittest.TestCase):
    def test_obstacle_mask_front_advances_progressively_with_alpha(self):
        from src.weather.flood_spread import obstacle_mask_at
        lower = np.zeros((80, 80), dtype=np.uint8)
        lower[35:45, 35:45] = 255
        upper = cv2.dilate(lower, np.ones((17, 17), np.uint8))
        frames = {0.0: lower, 1.0: upper}
        new = int(np.count_nonzero(upper)) - int(np.count_nonzero(lower))
        fractions = [(np.count_nonzero(obstacle_mask_at(frames, a)) - np.count_nonzero(lower)) / new
                     for a in np.linspace(0, 1, 11)]
        self.assertEqual(fractions[0], 0.0)
        self.assertEqual(fractions[-1], 1.0)
        self.assertTrue(all(b >= a for a, b in zip(fractions, fractions[1:])))
        self.assertGreaterEqual(len({round(f, 3) for f in fractions}), 8, fractions)
        # The front itself moves linearly: along the centre row it advances
        # 8 px (lower edge x=44 -> upper edge x=52) proportionally to alpha.
        for alpha in np.linspace(0, 1, 11):
            row = obstacle_mask_at(frames, alpha)[40]
            edge = int(np.nonzero(row)[0].max())
            self.assertLessEqual(abs(edge - (44 + 8 * alpha)), 1.0, (alpha, edge))

    def test_forecast_frames_are_binary_and_cumulative(self):
        from src.weather.flood_spread import forecast_frames
        mask = blob_mask(3)
        for weather in ({"precipitation": 10, "wind_speed_10m": 30, "wind_direction_10m": 270},
                        {"precipitation": 3, "wind_speed_10m": 12, "wind_direction_10m": 45}):
            frames = forecast_frames(mask, weather, 2.0)
            times = sorted(frames)
            for t in times:
                self.assertTrue(set(np.unique(frames[t])).issubset({0, 255}))
            for a, b in zip(times, times[1:]):
                self.assertFalse(np.any((frames[a] > 0) & (frames[b] == 0)))

    def test_moderate_weather_does_not_flood_quickly(self):
        from src.weather.flood_spread import forecast_frames, obstacle_mask_at
        mask = blob_mask(5)
        frames = forecast_frames(mask, {"precipitation": 3, "wind_speed_10m": 12,
                                        "wind_direction_10m": 270}, 2.0)
        for minutes in (1, 5, 10):
            self.assertTrue(np.array_equal(obstacle_mask_at(frames, minutes / 60), frames[0.0]))
        grown = np.count_nonzero(frames[2.0]) / np.count_nonzero(mask)
        self.assertLess(grown, 1.6)


class TestWaypointExport(unittest.TestCase):
    def test_generated_mission_is_valid_qgc_wpl_110(self):
        from src.mission.mission_output import generate_mission_file, validate_waypoints_file
        path = [(5, 5), (40, 5), (40, 30), (5, 5), (5, 5), (60, 60), (5, 5)]
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "m.waypoints"
            generate_mission_file(path, [2, 5], (5, 5), (50, 50), (25.14, 83.11), 2.0,
                                  filename=str(out), reload_indices=[3])
            self.assertEqual(validate_waypoints_file(str(out)), [])
            rows = [line.split("\t") for line in out.read_text().splitlines()[1:]]
        self.assertEqual(rows[0][1:4], ["1", "3", "16"])  # DEVELOPMENT writes the HOME row in frame 3 (MAIN: 0)
        servo = [r for r in rows if r[3] == "183"]
        self.assertEqual(len(servo), 2)
        self.assertTrue(all(float(r[4]) == 9 and float(r[5]) == 2000 for r in servo))
        for r in rows:
            self.assertLess(abs(float(r[8]) - 25.14), 0.01)
            self.assertLess(abs(float(r[9]) - 83.11), 0.01)

    def test_validator_rejects_malformed_items(self):
        from src.mission.mission_output import validate_waypoints_file
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "bad.waypoints"
            out.write_text("QGC WPL 110\n"
                           "0\t1\t0\t16\t0\t0\t0\t0\t25.1\t83.1\t0\t1\n"
                           "1\t0\t3\t183\t0\t0\t2000\t0\t25.1\t83.1\t10\t1\n"
                           "3\t0\t3\t16\t0\t0\t0\t25.1\t83.1\t100\t1\n")
            problems = validate_waypoints_file(str(out))
        self.assertTrue(any("DO_SET_SERVO" in p for p in problems))
        self.assertTrue(any("fields" in p for p in problems))


class TestVisionSampling(unittest.TestCase):
    def test_original_hsv_algorithm_reproduces_reference_varanasi_mask(self):
        """The interactive (click) path must keep the original HSV + morphology result."""
        from src.vision.image_processing import ImageProcessor
        image = cv2.imread(str(ROOT / "data" / "input" / "varanasi.png"))
        image = cv2.resize(image, (750, 748), interpolation=cv2.INTER_AREA)
        proc = ImageProcessor()
        proc.image = image.copy()
        proc.sample_points = [(400, 340), (462, 264), (546, 314)]
        proc.compute_dynamic_hsv()
        mask = proc.mask_flood_areas(image)
        self.assertEqual(proc.hsv_lower.tolist(), [0, 21, 142])
        self.assertEqual(int(np.count_nonzero(mask)), 48828)
        self.assertEqual(len(proc.find_filtered_contours(mask, min_area=200)), 18)


class TestRouteRefinement(unittest.TestCase):
    class Route:
        def __init__(self, points):
            self.points = points

    def test_statuses_distinguish_convergence_cycle_and_cap(self):
        from main import refine_route
        R = self.Route
        sequence = iter([R([1]), R([2]), R([2])])
        route, status, n = refine_route(lambda d: next(sequence), lambda r: {}, 5)
        self.assertEqual((status, n, route.points), ("converged", 2, [2]))

        sequence = iter([R([1]), R([2]), R([1])])
        _, status, n = refine_route(lambda d: next(sequence), lambda r: {}, 5)
        self.assertEqual((status, n), ("cycle", 2))

        counter = iter(range(100))
        _, status, n = refine_route(lambda d: R([next(counter)]), lambda r: {}, 3)
        self.assertEqual((status, n), ("iteration_cap", 3))

    def test_routed_distances_accumulate_across_iterations(self):
        from main import refine_route
        seen = []
        sequence = iter([self.Route(["a"]), self.Route(["b"]), self.Route(["b"])])

        def plan(distances):
            seen.append(distances)
            return next(sequence)

        refine_route(plan, lambda r: {r.points[0]: 1.0}, 5)
        self.assertEqual(seen, [None, {"a": 1.0}, {"a": 1.0, "b": 1.0}])


if __name__ == "__main__":
    unittest.main()
