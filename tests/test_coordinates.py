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

    def test_metric_scale(self):
        # 100 px north at 2 m/px is 200 m = 200 / 111320 degrees of latitude.
        lat, _ = pixel_to_latlon(500, 400, (500, 500), (25.0, 82.0), 2.0)
        self.assertAlmostEqual((lat - 25.0) * 111320.0, 200.0, places=6)

    def test_region_presets_anchor_each_test_image_in_its_own_region(self):
        from config import settings
        # Rough district boxes for the three bundled maps.
        boxes = {"varanasi": (24.9, 25.6, 82.6, 83.5),
                 "kanpur": (25.9, 27.0, 79.6, 80.7),
                 "bhopal": (22.8, 24.2, 77.2, 78.6)}
        for name, (lat0, lat1, lon0, lon1) in boxes.items():
            preset = settings.REGION_PRESETS[name]
            self.assertTrue(lat0 <= preset["lat"] <= lat1 and lon0 <= preset["lon"] <= lon1, name)
        # No two presets share a location (no leftover copy-pasted city).
        anchors = {(p["lat"], p["lon"]) for p in settings.REGION_PRESETS.values()}
        self.assertEqual(len(anchors), len(settings.REGION_PRESETS))


if __name__ == "__main__":
    unittest.main()
