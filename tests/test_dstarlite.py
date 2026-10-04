import unittest
from src.routing.dstarlite import DStarLite

class TestDStarLite(unittest.TestCase):
    def test_arrival_time_selects_future_obstacle_frame(self):
        import numpy as np
        from src.weather.flood_spread import obstacle_mask_at
        a = np.zeros((2, 2), dtype=np.uint8)
        b = np.ones((2, 2), dtype=np.uint8) * 255
        self.assertFalse(obstacle_mask_at({0.5: a, 1.0: b}, 0.6).any())
        self.assertTrue(obstacle_mask_at({0.5: a, 1.0: b}, 0.9).any())
        self.assertTrue(obstacle_mask_at({0.5: a, 1.0: b}, 5.0).any())

    def test_compute_full_path_home_only_uses_three_value_contract(self):
        import numpy as np
        from src.routing.pathfinding import compute_full_path
        result = compute_full_path([(4, 4)], np.zeros((10, 10), dtype=np.uint8))
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], [(4, 4)])
        self.assertEqual(result[1], [0])
        self.assertEqual(result[2], [])

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

if __name__ == '__main__':
    unittest.main()
