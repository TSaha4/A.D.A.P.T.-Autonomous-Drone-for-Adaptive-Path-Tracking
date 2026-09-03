import unittest
import numpy as np
from src.routing.dstarlite import DStarLite
from src.routing.pathfinding import plan_single_leg, plan_timed_leg
from src.mission.constraints import DroneSpec
from src.weather.flood_predictor import FloodTimeline

class TestDStarLite(unittest.TestCase):
    def test_shortest_path_empty_grid(self):
        # 10x10 grid, start at 0,0, goal at 9,9
        dstar = DStarLite((10, 10), (0, 0), (9, 9))
        path = dstar.plan_path()
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (9, 9))
        self.assertEqual(len(path), 10) # 0 to 9 is 10 steps via diagonal

    def test_replanning_with_obstacle(self):
        dstar = DStarLite((10, 10), (0, 0), (9, 9))
        path1 = dstar.plan_path()
        
        # Add an obstacle that blocks the direct diagonal path
        obstacles = set()
        for i in range(0, 5):
            obstacles.add((i, 4))
            
        path2 = dstar.plan_path(obstacles)
        self.assertIsNotNone(path2)
        
        # Path should avoid obstacles
        for p in path2:
            self.assertNotIn(p, obstacles)
            
        self.assertNotEqual(path1, path2)

    def test_snap_to_nearest_free_cell(self):
        from src.routing.pathfinding import snap_to_nearest_free_cell
        obstacles = {(5, 5), (5, 6), (6, 5), (6, 6)}
        
        # Test a point that is an obstacle
        snapped_point, dist = snap_to_nearest_free_cell((5, 5), obstacles, 10, 10)
        self.assertNotIn(snapped_point, obstacles)
        self.assertGreater(dist, 0)
        
        # Test a point that is not an obstacle
        snapped_point2, dist2 = snap_to_nearest_free_cell((1, 1), obstacles, 10, 10)
        self.assertEqual(snapped_point2, (1, 1))
        self.assertEqual(dist2, 0)


class TestTimeAwareLeg(unittest.TestCase):
    """Task 2: D* Lite plans against the obstacle state *at the leg ETA*, not
    against the mission-start snapshot."""

    def setUp(self):
        self.spec = DroneSpec()
        self.weather = {"precipitation": 0.0, "wind_speed_10m": 0.0,
                        "wind_direction_10m": 0.0}

    def _timeline_with_masks(self, masks, times=(0.0, 30.0, 60.0)):
        return FloodTimeline(list(times), masks, weather=self.weather)

    def test_uses_forecast_valid_at_eta_not_mission_start(self):
        # A wall exists at x~30 in frame t=0 but has gone (well: the world the
        # drone will actually fly through at its ETA) by t=30.
        m0 = np.zeros((20, 60), dtype=np.uint8)
        m0[4:7, 28:33] = 255            # frame @ 0 min: wall across the corridor
        m30 = np.zeros_like(m0)         # frame @ 30 min: no wall
        m60 = np.zeros_like(m0)
        tl = self._timeline_with_masks([m0, m30, m60])

        # Straight-line ETA lands between 30 and 60 min -> must use the t=30 slice.
        leg = plan_timed_leg((2, 5), (55, 5), tl, departure_min=15.0, spec=self.spec,
                             weather=self.weather, meters_per_pixel=250.0,
                             downsample_factor=1)
        self.assertTrue(leg["ok"])
        self.assertEqual(leg["obstacle_time_min"], 30.0)
        # Because the t=30 world is clear, the drone flies straight through the
        # cell that was an obstacle at t=0.  A static (t=0) plan could not.
        self.assertIn((30, 5), leg["path"])
        self.assertAlmostEqual(leg["arrival_min"], 15.0 + 53 * 250.0 / self.spec.cruise_airspeed_mps / 60.0,
                               delta=1.0)

    def test_uses_t0_snapshot_for_early_eta(self):
        m0 = np.zeros((20, 60), dtype=np.uint8)
        m0[4:7, 28:33] = 255
        m30 = np.zeros_like(m0)
        m60 = np.zeros_like(m0)
        tl = self._timeline_with_masks([m0, m30, m60])

        # Very short leg -> ETA ~= 0 min -> plan against the t=0 frame, avoiding the wall.
        leg = plan_timed_leg((2, 5), (55, 5), tl, departure_min=0.0, spec=self.spec,
                             weather=self.weather, meters_per_pixel=10.0,
                             downsample_factor=1)
        self.assertTrue(leg["ok"])
        self.assertEqual(leg["obstacle_time_min"], 0.0)
        self.assertNotIn((30, 5), leg["path"])
        self.assertEqual(leg["path"][-1], (55, 5))

    def test_arrival_between_frames_rounds_to_nearest(self):
        # Nothing to avoid; we only check the selected forecast slice for an ETA
        # that falls between frames.
        m = [np.zeros((20, 60), dtype=np.uint8) for _ in range(3)]
        tl = self._timeline_with_masks(m)
        leg = plan_timed_leg((2, 5), (55, 5), tl, departure_min=14.0, spec=self.spec,
                             weather=self.weather, meters_per_pixel=250.0,
                             downsample_factor=1)
        # arrival ~ 14 + 18.4 = 32.4 min, closer to 30 than to 60.
        self.assertEqual(leg["obstacle_time_min"], 30.0)
        self.assertTrue(leg["ok"])

    def test_plan_single_leg_against_one_snapshot(self):
        mask = np.zeros((20, 60), dtype=np.uint8)
        mask[4:7, 28:33] = 255
        path, ok = plan_single_leg((2, 5), (55, 5), mask, downsample_factor=1)
        self.assertTrue(ok)
        self.assertNotIn((30, 5), path)
        self.assertEqual(path[0], (2, 5))
        self.assertEqual(path[-1], (55, 5))


if __name__ == '__main__':
    unittest.main()
