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
        self.assertTrue(np.array_equal(self.mask, pred))

    def test_isotropic_spread_with_precipitation(self):
        # High precip -> should expand in all directions
        weather = {"precipitation": 10.0, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0}
        pred = predict_spread(self.mask, weather, horizon_hours=1.0)
        
        # Check that center is still flooded
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
        pred = predict_spread(self.mask, weather, horizon_hours=1.0)
        
        # South is positive Y in image coords
        # Top edge of original was Y=45. With Southward blow, it should not expand up.
        self.assertEqual(pred[40, 50], 0)
        
        # Bottom edge was Y=55. It should expand down.
        self.assertEqual(pred[58, 50], 255)

if __name__ == '__main__':
    unittest.main()
