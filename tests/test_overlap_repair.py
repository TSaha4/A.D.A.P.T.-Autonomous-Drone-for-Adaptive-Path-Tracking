import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config import settings
from src.mission.overlap_repair import (
    execute_overlap_repairs,
    find_leg_index,
    try_nudge_base,
    try_reroute_leg,
)
from src.routing.pathfinding import polyline_intersections


class TestOverlapRepair(unittest.TestCase):
    def test_find_leg_index(self):
        stop_indices = [0, 10, 25, 40]
        self.assertEqual(find_leg_index(stop_indices, 0), 0)
        self.assertEqual(find_leg_index(stop_indices, 5), 0)
        self.assertEqual(find_leg_index(stop_indices, 10), 1)
        self.assertEqual(find_leg_index(stop_indices, 24), 1)
        self.assertEqual(find_leg_index(stop_indices, 25), 2)
        self.assertEqual(find_leg_index(stop_indices, 39), 2)

    def test_tier1_reroute_resolves_crossing(self):
        # Create Base 1 flying horizontal across the map, Base 2 flying vertical
        # Base 1: (10, 50) -> (100, 50)
        # Base 2: (50, 10) -> (50, 100)
        # They cross at (50, 50).
        mask = np.zeros((120, 120), dtype=np.uint8)
        p1 = [(x, 50) for x in range(10, 105, 5)]
        p2 = [(50, y) for y in range(10, 105, 5)]

        plan1 = {
            "base": 1,
            "home": (10, 50),
            "full_path": p1,
            "stop_indices": [0, len(p1) - 1],
            "route_points": [(10, 50), (100, 50)],
            "drop_indices": [len(p1) - 1],
            "reload_indices": [],
        }
        plan2 = {
            "base": 2,
            "home": (50, 10),
            "full_path": p2,
            "stop_indices": [0, len(p2) - 1],
            "route_points": [(50, 10), (50, 100)],
            "drop_indices": [len(p2) - 1],
            "reload_indices": [],
        }

        segs = polyline_intersections(p1, p2, return_segments=True)
        self.assertTrue(len(segs) > 0)

        with TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            res = try_reroute_leg(
                plan1, plan2, segs, mask, None, 0.0, session_dir,
                (120, 120), (25.0, 82.0)
            )
            self.assertIsNotNone(res)
            updated_plan1, leg_idx, hits = res
            self.assertEqual(hits, 0)
            # Verify new intersections is 0
            new_hits = polyline_intersections(updated_plan1["full_path"], p2)
            self.assertEqual(len(new_hits), 0)

    def test_tier2_nudge_resolves_crossing(self):
        # When rerouting cannot detour (e.g. wall bounds), nudging base resolves it
        mask = np.zeros((150, 150), dtype=np.uint8)
        clearance = np.full((150, 150), 30.0, dtype=np.float32)

        # Base 1 at (20, 20) with drop at (60, 60)
        # Base 2 at (20, 60) with drop at (60, 20)
        # Crossing at (40, 40)
        p1 = [(x, x) for x in range(20, 65, 5)]
        p2 = [(x, 80 - x) for x in range(20, 65, 5)]

        plan1 = {
            "base": 1,
            "home": (20, 20),
            "full_path": p1,
            "stop_indices": [0, len(p1) - 1],
            "route_points": [(20, 20), (60, 60)],
            "drop_indices": [len(p1) - 1],
            "reload_indices": [],
        }
        plan2 = {
            "base": 2,
            "home": (20, 60),
            "full_path": p2,
            "stop_indices": [0, len(p2) - 1],
            "route_points": [(20, 60), (60, 20)],
            "drop_indices": [len(p2) - 1],
            "reload_indices": [],
        }
        plans_by_base = {0: plan1, 1: plan2}
        selection = {"assignment": {0: 0, 1: 1}}
        safe_points = [(60, 60), (60, 20)]

        # Mock distance function
        def dist_fn(a, b):
            return float(np.hypot(a[0] - b[0], a[1] - b[1])) * 2.0

        def mock_plan_base(num, home, assigned):
            # If base nudged away, route goes around without crossing
            if home[0] != 20 or home[1] != 20:
                # new path from home to assigned
                pts = [(home[0], home[1]), (home[0], assigned[0][1]), (assigned[0][0], assigned[0][1])]
                return {
                    "base": num,
                    "home": home,
                    "full_path": pts,
                    "stop_indices": [0, 2],
                    "route_points": [home, assigned[0]],
                    "drop_indices": [2],
                    "reload_indices": [],
                }
            return plan1

        res = try_nudge_base(
            0, plans_by_base, selection, safe_points, clearance,
            dist_fn, 0.0, mock_plan_base, (150, 150), min_separation_px=30.0
        )
        self.assertIsNotNone(res)
        new_site, new_plan = res
        self.assertNotEqual(new_site, (20, 20))
        self.assertEqual(len(polyline_intersections(new_plan["full_path"], plan2["full_path"])), 0)

    def test_bhopal_session_172629_case_resolves_to_zero_overlap(self):
        # Test exact waypoints from Bhopal session 172629
        w1_path = Path("data/output/session_20261004_172629/base1.waypoints")
        w2_path = Path("data/output/session_20261004_172629/base2.waypoints")
        if not w1_path.exists() or not w2_path.exists():
            self.skipTest("Session 172629 waypoints not present")

        geo_lat, geo_lon = 23.37, 78.03
        cx, cy = 616 // 2, 501 // 2
        m_per_px = 2.0

        def read_wpl_px(path):
            pts = []
            with open(path) as f:
                for line in f:
                    if not line.strip() or line.startswith("QGC"):
                        continue
                    p = line.strip().split("\t")
                    if int(p[3]) in (16, 22):
                        lat, lon = float(p[8]), float(p[9])
                        dy_m = (lat - geo_lat) * 111320.0
                        dx_m = (lon - geo_lon) * (111320.0 * np.cos(np.radians(geo_lat)))
                        pts.append((int(round(cx + dx_m / m_per_px)), int(round(cy - dy_m / m_per_px))))
            return pts

        p1 = read_wpl_px(w1_path)
        p2 = read_wpl_px(w2_path)
        initial_hits = polyline_intersections(p1, p2, return_segments=True)
        self.assertEqual(len(initial_hits), 15)

        # Plan 1 with crossing leg
        # Find stops in p1
        home1 = (352, 277)
        drops1 = [(384, 322), (369, 359), (426, 361), (454, 325), (478, 312),
                  (525, 359), (301, 336), (285, 280), (186, 205), (198, 185), (216, 176)]
        stops1_idx = [i for i, pt in enumerate(p1) if pt == home1 or pt in drops1]
        route_points1 = [p1[i] for i in stops1_idx]

        plan1 = {
            "base": 1,
            "home": home1,
            "full_path": p1,
            "stop_indices": stops1_idx,
            "route_points": route_points1,
            "drop_indices": [i for i, pt in enumerate(p1) if pt in drops1],
            "reload_indices": [i for i, pt in enumerate(p1) if pt == home1 and 0 < i < len(p1) - 1],
        }
        plan2 = {
            "base": 2,
            "home": (172, 442),
            "full_path": p2,
            "stop_indices": [0, len(p2) - 1],
            "route_points": [(172, 442), p2[-1]],
            "drop_indices": [len(p2) - 1],
            "reload_indices": [],
        }

        with TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            res = try_reroute_leg(
                plan1, plan2, initial_hits, np.zeros((501, 616), dtype=np.uint8),
                None, 8.33, session_dir, (501, 616), (23.37, 78.03)
            )
            self.assertIsNotNone(res)
            updated_p1, leg_idx, hits = res
            self.assertEqual(hits, 0)
            remaining = polyline_intersections(updated_p1["full_path"], p2)
            self.assertEqual(len(remaining), 0)

    def test_hard_error_enforced_when_all_tiers_fail(self):
        # Two parallel walls forming a 1-cell corridor where paths must cross and cannot detour
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        # Cut a 1-pixel plus-shaped corridor
        mask[25, 5:45] = 0
        mask[5:45, 25] = 0

        p1 = [(x, 25) for x in range(5, 45, 5)]
        p2 = [(25, y) for y in range(5, 45, 5)]

        plan1 = {
            "base": 1, "home": (5, 25), "full_path": p1,
            "stop_indices": [0, len(p1) - 1],
            "route_points": [(5, 25), (40, 25)],
            "drop_indices": [len(p1) - 1], "reload_indices": [],
        }
        plan2 = {
            "base": 2, "home": (25, 5), "full_path": p2,
            "stop_indices": [0, len(p2) - 1],
            "route_points": [(25, 5), (25, 40)],
            "drop_indices": [len(p2) - 1], "reload_indices": [],
        }
        plans_by_base = {0: plan1, 1: plan2}
        bases = [(5, 25), (25, 5)]
        selection = {"assignment": {0: 0, 1: 1}}
        safe_points = [(40, 25), (25, 40)]
        clearance = np.zeros((50, 50), dtype=np.float32)

        def mock_plan(num, home, assigned):
            return None

        with TemporaryDirectory() as tmp:
            repairs, remaining = execute_overlap_repairs(
                plans_by_base, bases, selection, safe_points, [set(), set()],
                {}, mask, None, clearance, lambda a, b: 100.0, 0.0,
                mock_plan, mock_plan, Path(tmp), (50, 50), (25.0, 82.0),
                max_rounds=2
            )
            # Unresolved crossing remains
            self.assertGreater(len(remaining), 0)


if __name__ == "__main__":
    unittest.main()
