import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.mission.multi_base import select_bases
from src.routing.pathfinding import polyline_intersections


def straight(meters_per_pixel):
    return lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1])) * meters_per_pixel


class TestMultiBase(unittest.TestCase):
    def select(self, candidates, drops, **kw):
        mask = np.zeros((400, 1200), dtype=np.uint8)
        args = dict(headwind_mps=0.0, wind_from_deg=None, per_drop_kg=0.25, payload_capacity_kg=8.0,
                    reserve=0.2, meters_per_pixel=2.0, max_bases=4, min_separation_px=150)
        args.update(kw)
        return select_bases(candidates, drops, straight(args["meters_per_pixel"]), mask, **args)

    def test_single_base_when_one_site_reaches_everything(self):
        result = self.select([(100, 100), (150, 100)], [(120, 100), (140, 110), (160, 90)])
        self.assertEqual(result["mode"], "single")
        self.assertEqual(len(result["bases"]), 1)
        self.assertEqual(set(result["assignment"].values()), {0})

    def test_two_far_clusters_get_two_separated_bases_and_exclusive_assignment(self):
        # Single-drop reach at 2 m/px is ~410 px; clusters ~1000 px apart need two bases.
        candidates = [(x, 200) for x in range(50, 1200, 50)]
        drops = [(80, 180), (120, 220), (100, 260), (1080, 180), (1120, 220)]
        result = self.select(candidates, drops)
        self.assertEqual(result["mode"], "multi")
        self.assertEqual(len(result["bases"]), 2)
        (a, b) = result["bases"]
        self.assertGreaterEqual(np.hypot(a[0] - b[0], a[1] - b[1]), 150)
        self.assertEqual(sorted(result["assignment"]), list(range(5)))   # each drop exactly once
        west = {result["assignment"][i] for i in (0, 1, 2)}
        east = {result["assignment"][i] for i in (3, 4)}
        self.assertEqual(len(west), 1)
        self.assertEqual(len(east), 1)
        self.assertNotEqual(west, east)
        self.assertEqual(result["unreachable"], {})

    def test_separation_is_enforced_even_if_it_costs_coverage(self):
        candidates = [(100, 200), (180, 200)]     # only 80 px apart
        drops = [(100, 200), (100, 600), (180, 200)]
        result = self.select(candidates, drops, meters_per_pixel=20.0, min_separation_px=150)
        bases = result["bases"]
        for i, p in enumerate(bases):
            for q in bases[i + 1:]:
                self.assertGreaterEqual(np.hypot(p[0] - q[0], p[1] - q[1]), 150)

    def test_unreachable_drops_are_reported_with_reason(self):
        result = self.select([(100, 100)], [(110, 100), (1100, 390)])
        self.assertIn(1, result["unreachable"])
        self.assertIn("unreachable by any base", result["unreachable"][1])
        self.assertNotIn(1, result["assignment"])

    def test_base_cap_is_respected_and_leftovers_reported(self):
        candidates = [(x, 200) for x in range(50, 1200, 50)]
        drops = [(80, 200), (500, 200), (900, 200), (1150, 200)]
        result = self.select(candidates, drops, meters_per_pixel=40.0, max_bases=2, min_separation_px=100)
        self.assertLessEqual(len(result["bases"]), 2)
        self.assertEqual(len(result["assignment"]) + len(result["unreachable"]), len(drops))


class TestPathOverlap(unittest.TestCase):
    def test_crossing_and_overlap_detected_repeated_base_points_ignored(self):
        self.assertEqual(polyline_intersections([(0, 0), (10, 10)], [(0, 10), (10, 0)]), [(5.0, 5.0)])
        self.assertEqual(polyline_intersections([(0, 0), (10, 0)], [(0, 5), (10, 5)]), [])
        self.assertTrue(polyline_intersections([(0, 0), (10, 0)], [(5, 0), (15, 0)]))
        self.assertEqual(polyline_intersections([(0, 0), (0, 0), (10, 0)], [(5, 5), (5, 5), (5, 8)]), [])


class TestDistanceFields(unittest.TestCase):
    def test_multi_source_field_is_min_of_single_fields(self):
        from src.routing.pathfinding import build_routing_grid, grid_distance_field
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:, 45:55] = 255
        grid = build_routing_grid(mask, 5)
        a, b = grid_distance_field(grid, (2, 2)), grid_distance_field(grid, (17, 17))
        both = grid_distance_field(grid, [(2, 2), (17, 17)])
        self.assertTrue(np.array_equal(both, np.minimum(a, b)))
        self.assertTrue(np.isinf(a[2, 17]))          # wall separates the halves


class TestSessionFolders(unittest.TestCase):
    def test_each_run_gets_a_new_session_folder(self):
        from main import create_session_dir
        with TemporaryDirectory() as tmp:
            first = create_session_dir(tmp)
            second = create_session_dir(tmp)
            self.assertNotEqual(first, second)
            self.assertTrue(first.name.startswith("session_") and first.is_dir() and second.is_dir())
            self.assertEqual(Path(first).parent, Path(tmp))


if __name__ == "__main__":
    unittest.main()
