"""Synthetic geospatial ground truth for pixel_to_latlon.

1000 x 1000 px image, reference (25.000000, 82.000000) at the geometric centre (499.5, 499.5),
10 m/px. Expected positions come from tests/geodesy_reference.py (WGS84 Vincenty), never from the
implementation. Two properties are checked:
  - conformance: the implementation equals the documented flat-earth model (1 deg = 111,320 m),
    re-derived independently, to floating-point precision;
  - physical accuracy: the distance to the WGS84 ground truth is within the model's derived error bound.
"""
import itertools
import math
import random
import unittest

from src.mission.coordinates import pixel_to_latlon
from tests import geodesy_reference as ref

W = H = 1000
CENTER = ((W - 1) / 2, (H - 1) / 2)
GEO = (25.0, 82.0)
MPP = 10.0
CONTROL_POINTS = {
    "centre": CENTER, "top-centre": (CENTER[0], 0), "bottom-centre": (CENTER[0], H - 1),
    "left-centre": (0, CENTER[1]), "right-centre": (W - 1, CENTER[1]), "top-left": (0, 0),
    "top-right": (W - 1, 0), "bottom-left": (0, H - 1), "bottom-right": (W - 1, H - 1),
}


class TestSyntheticGroundTruth(unittest.TestCase):
    def test_conforms_to_documented_model(self):
        for name, (px, py) in CONTROL_POINTS.items():
            lat, lon = pixel_to_latlon(px, py, CENTER, GEO, MPP)
            exp = ref.documented_model(px, py, CENTER, GEO, MPP)
            self.assertAlmostEqual(lat, exp[0], delta=1e-12, msg=name)
            self.assertAlmostEqual(lon, exp[1], delta=1e-12, msg=name)

    def test_within_physical_error_bound(self):
        for name, (px, py) in CONTROL_POINTS.items():
            lat, lon = pixel_to_latlon(px, py, CENTER, GEO, MPP)
            t_lat, t_lon = ref.physical_ground_truth(px, py, CENTER, GEO, MPP)
            err, _ = ref.vincenty_inverse(lat, lon, t_lat, t_lon)
            east, north = ref.pixel_offset_meters(px, py, CENTER, MPP)
            self.assertLessEqual(err, ref.model_error_bound_m(east, north, GEO[0]), name)

    def test_centre_pixel_is_reference(self):
        self.assertEqual(pixel_to_latlon(*CENTER, CENTER, GEO, MPP), GEO)

    def test_axis_directions(self):
        lat0, lon0 = GEO
        up = pixel_to_latlon(CENTER[0], CENTER[1] - 1, CENTER, GEO, MPP)
        right = pixel_to_latlon(CENTER[0] + 1, CENTER[1], CENTER, GEO, MPP)
        self.assertGreater(up[0], lat0)       # image up = north
        self.assertEqual(up[1], lon0)
        self.assertGreater(right[1], lon0)    # image right = east
        self.assertEqual(right[0], lat0)
        # One pixel = 10 m in the documented model
        self.assertAlmostEqual((up[0] - lat0) * ref.MODEL_METERS_PER_DEGREE, MPP, delta=1e-9)
        self.assertAlmostEqual((right[1] - lon0) * ref.MODEL_METERS_PER_DEGREE * math.cos(math.radians(lat0)),
                               MPP, delta=1e-9)

    def test_all_hemispheres_bearing(self):
        # Every quadrant: the geodesic bearing from the reference matches the pixel direction
        for geo in [(25.0, 82.0), (40.7, -74.0), (-33.9, 151.2), (-22.9, -43.2), (0.0, 30.0), (51.5, 0.0)]:
            for (px, py), az in [((999, 0), 45.0), ((999, 999), 135.0), ((0, 999), 225.0), ((0, 0), 315.0)]:
                lat, lon = pixel_to_latlon(px, py, CENTER, geo, MPP)
                _, bearing = ref.vincenty_inverse(geo[0], geo[1], lat, lon)
                self.assertAlmostEqual(bearing, az, delta=0.5, msg=f"{geo} {(px, py)}")

    def test_round_trip_through_test_only_inverse(self):
        for geo in [(25.0, 82.0), (-33.9, 151.2), (60.0, -120.0), (0.0, 0.0)]:
            for px, py in itertools.product([0, 250, 499.5, 777, 999], repeat=2):
                lat, lon = pixel_to_latlon(px, py, CENTER, geo, MPP)
                rx, ry = ref.documented_model_inverse(lat, lon, CENTER, geo, MPP)
                self.assertAlmostEqual(rx, px, delta=1e-6)
                self.assertAlmostEqual(ry, py, delta=1e-6)

    def test_valid_inputs_unchanged_by_safety_validation(self):
        # Regression for G1: for every in-range input the result is the documented model, exactly as before
        rng = random.Random(7)
        for _ in range(2000):
            geo = (rng.uniform(-80, 80), rng.uniform(-179, 179))
            px, py, mpp = rng.uniform(-200, 1200), rng.uniform(-200, 1200), rng.uniform(0.1, 50)
            got = pixel_to_latlon(px, py, CENTER, geo, mpp)
            exp = ref.documented_model(px, py, CENTER, geo, mpp)
            self.assertAlmostEqual(got[0], exp[0], delta=1e-12)
            self.assertAlmostEqual(got[1], exp[1], delta=1e-12)


if __name__ == '__main__':
    unittest.main()
