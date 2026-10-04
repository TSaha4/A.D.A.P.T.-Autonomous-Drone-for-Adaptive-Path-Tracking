import unittest
import numpy as np
import cv2
from src.weather.flood_spread import predict_spread

class TestFloodSpread(unittest.TestCase):
    def setUp(self):
        # Create a 100x100 blank mask with a single 10x10 flood block in the center
        self.mask = np.zeros((100, 100), dtype=np.uint8)
        self.mask[45:55, 45:55] = 255

    def test_no_spread_with_zero_weather(self):
        weather = {"precipitation": 0.0, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0}
        pred = predict_spread(self.mask, weather, horizon_hours=1.0)
        self.assertEqual(set(pred), {0.0, 1/12, 1/6, 0.5, 1.0})
        self.assertTrue(np.array_equal(self.mask, pred[1.0]))

    def test_isotropic_spread_with_precipitation(self):
        # High precip -> should expand in all directions
        weather = {"precipitation": 10.0, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0}
        pred = predict_spread(self.mask, weather, horizon_hours=1.0)
        
        # Check that center is still flooded
        pred = pred[1.0]
        self.assertEqual(pred[50, 50], 255)
        # Check that it spread outside the original 10x10 box
        self.assertTrue(np.sum(pred) > np.sum(self.mask))
        
        # Should spread symmetrically (more or less) by 2 pixels
        self.assertEqual(pred[43, 50], 255) # Top
        self.assertEqual(pred[56, 50], 255) # Bottom
        self.assertEqual(pred[50, 43], 255) # Left
        self.assertEqual(pred[50, 56], 255) # Right

    def test_directional_spread_with_wind(self):
        # Strong wind from North (0 deg) -> blowing South
        weather = {"precipitation": 0.0, "wind_speed_10m": 50.0, "wind_direction_10m": 0.0}
        pred = predict_spread(self.mask, weather, horizon_hours=1.0)[1.0]
        
        # South is positive Y in image coords
        # Top edge of original was Y=45. With Southward blow, it should not expand up.
        self.assertEqual(pred[40, 50], 0)
        
        # Bottom edge was Y=55. It should expand down.
        self.assertEqual(pred[58, 50], 255)

    def test_temporal_sequence_has_increasing_horizons(self):
        weather = {"precipitation": 10, "wind_speed_10m": 0, "wind_direction_10m": 0}
        frames = predict_spread(self.mask, weather, 2.0)
        self.assertEqual(list(frames), [0.0, 1/12, 1/6, 0.5, 1.0, 2.0])
        self.assertLessEqual(np.sum(frames[1/12] > 0), np.sum(frames[1.0] > 0))
        self.assertLessEqual(np.sum(frames[1.0] > 0), np.sum(frames[2.0] > 0))

    def test_sub_half_hour_forecast_scales_to_elapsed_time(self):
        from src.weather.flood_spread import obstacle_mask_at
        weather = {"precipitation": 5.0, "wind_speed_10m": 15.0, "wind_direction_10m": 0.0}
        frames = predict_spread(self.mask, weather, 2.0)
        early = obstacle_mask_at(frames, 10 / 60)
        half_hour = frames[0.5]
        self.assertLessEqual(np.sum(early > 0), np.sum(half_hour > 0))
        self.assertLessEqual(np.sum(early > 0), np.sum(frames[0.5] > 0))

    def test_interpolated_mask_is_blended_then_thresholded(self):
        from src.weather.flood_spread import forecast_at
        current = np.zeros((5, 5), dtype=np.uint8)
        low = current.copy(); low[2, 2] = 42
        higher = current.copy(); higher[2, 2] = 85
        grown = current.copy(); grown[2, 2] = 255
        frames = {0.0: current, 1/12: low, 1/6: higher, .5: grown}
        sampled, details = forecast_at(frames, .12, current)
        self.assertEqual(details["source"], "interpolated")
        self.assertAlmostEqual(details["lower_hours"], 1/12)
        self.assertAlmostEqual(details["upper_hours"], 1/6)
        self.assertEqual((int(low[2, 2]), int(higher[2, 2])), (42, 85))
        self.assertEqual(int(sampled[2, 2]), 0)

    def test_obstacle_mask_interpolation_uses_alpha_before_binary_threshold(self):
        from src.weather.flood_spread import (obstacle_mask_at,
                                              interpolate_forecast_mask)
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

    def test_forecast_frames_binary_and_intermediate_time_is_blended(self):
        from src.weather.flood_spread import (predict_spread, forecast_at,
                                              interpolate_forecast_mask)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, 45:55] = 255
        frames = predict_spread(mask, {
            "precipitation": 10.0,
            "wind_speed_10m": 30.0,
            "wind_direction_10m": 270.0,
        }, 2.0)
        self.assertTrue(all(set(np.unique(frame)).issubset({0, 255}) for frame in frames.values()))
        blended, details = interpolate_forecast_mask(frames, 0.3, mask)
        sampled, _ = forecast_at(frames, 0.3, mask)
        self.assertAlmostEqual(details["lower_hours"], 1/6)
        self.assertAlmostEqual(details["upper_hours"], 0.5)
        self.assertAlmostEqual(details["alpha"], 0.4)
        self.assertEqual(int(frames[1/6][44, 50]), 0)
        self.assertEqual(int(frames[0.5][44, 50]), 255)
        self.assertEqual(int(blended[44, 50]), 102)
        self.assertEqual(int(sampled[44, 50]), 0)
        self.assertEqual(int(sampled[50, 50]), 255)

if __name__ == '__main__':
    unittest.main()
