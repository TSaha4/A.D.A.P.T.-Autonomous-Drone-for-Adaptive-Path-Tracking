import math
import unittest
from src.mission.coordinates import pixel_to_latlon


class TestCoordinates(unittest.TestCase):
    def test_center_pixel(self):
        # Center pixel should map exactly to geo_center
        cx, cy = 500, 500
        geo_lat, geo_lon = 25.0, 82.0
        meters_per_pixel = 2.0

        lat, lon = pixel_to_latlon(cx, cy, (cx, cy), (geo_lat, geo_lon), meters_per_pixel)

        self.assertTrue(math.isclose(lat, geo_lat, abs_tol=1e-7))
        self.assertTrue(math.isclose(lon, geo_lon, abs_tol=1e-7))

    def test_direction_signs(self):
        cx, cy = 500, 500
        geo_lat, geo_lon = 25.0, 82.0
        meters_per_pixel = 2.0

        # Move North: py < cy (y-axis inverted in images)
        lat_north, lon_north = pixel_to_latlon(500, 400, (cx, cy), (geo_lat, geo_lon), meters_per_pixel)
        self.assertGreater(lat_north, geo_lat)
        self.assertTrue(math.isclose(lon_north, geo_lon, abs_tol=1e-7))

        # Move South: py > cy
        lat_south, lon_south = pixel_to_latlon(500, 600, (cx, cy), (geo_lat, geo_lon), meters_per_pixel)
        self.assertLess(lat_south, geo_lat)
        self.assertTrue(math.isclose(lon_south, geo_lon, abs_tol=1e-7))

        # Move East: px > cx
        lat_east, lon_east = pixel_to_latlon(600, 500, (cx, cy), (geo_lat, geo_lon), meters_per_pixel)
        self.assertTrue(math.isclose(lat_east, geo_lat, abs_tol=1e-7))
        self.assertGreater(lon_east, geo_lon)

        # Move West: px < cx
        lat_west, lon_west = pixel_to_latlon(400, 500, (cx, cy), (geo_lat, geo_lon), meters_per_pixel)
        self.assertTrue(math.isclose(lat_west, geo_lat, abs_tol=1e-7))
        self.assertLess(lon_west, geo_lon)


if __name__ == "__main__":
    unittest.main()
