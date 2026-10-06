"""G1: pixel_to_latlon must reject inputs it cannot convert safely, wrap longitude across the
antimeridian, and generate_mission_file must never leave a partial or non-finite mission."""
import math
import os
import tempfile
import unittest

from src.mission.coordinates import pixel_to_latlon
from src.mission.mission_output import generate_mission_file
from tests import geodesy_reference as ref
from tests import wpl_validator

C = (499.5, 499.5)


class TestInputValidation(unittest.TestCase):
    def test_rejects_zero_and_negative_scale(self):
        for mpp in (0.0, -0.0, -10.0, -1e-9):
            with self.assertRaises(ValueError, msg=f"mpp={mpp}"):
                pixel_to_latlon(999, 0, C, (25.0, 82.0), mpp)

    def test_rejects_non_finite_scale(self):
        for mpp in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError, msg=f"mpp={mpp}"):
                pixel_to_latlon(999, 0, C, (25.0, 82.0), mpp)

    def test_rejects_invalid_latitude(self):
        for lat in (90.0001, 100.0, -90.0001, -91.0, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError, msg=f"lat={lat}"):
                pixel_to_latlon(999, 0, C, (lat, 82.0), 10.0)

    def test_rejects_polar_reference(self):
        # cos(lat) = 0: the longitude scale of the method is undefined at the poles
        for lat in (90.0, -90.0):
            with self.assertRaises(ValueError, msg=f"lat={lat}"):
                pixel_to_latlon(999, 0, C, (lat, 0.0), 10.0)

    def test_rejects_invalid_longitude(self):
        for lon in (180.0001, 500.0, -180.0001, -500.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=f"lon={lon}"):
                pixel_to_latlon(999, 0, C, (25.0, lon), 10.0)

    def test_accepts_boundary_longitudes(self):
        for lon in (180.0, -180.0):
            lat, out = pixel_to_latlon(499.5, 499.5, C, (25.0, lon), 10.0)
            self.assertEqual(lat, 25.0)
            self.assertEqual(out, lon)

    def test_rejects_non_finite_pixel_and_centre(self):
        nan, inf = float("nan"), float("inf")
        for px, py, centre in ((nan, 0, C), (0, inf, C), (0, 0, (nan, 499.5)), (0, 0, (499.5, -inf))):
            with self.assertRaises(ValueError):
                pixel_to_latlon(px, py, centre, (25.0, 82.0), 10.0)

    def test_rejects_offset_beyond_north_pole(self):
        with self.assertRaises(ValueError):
            pixel_to_latlon(499.5, -500.5, C, (89.995, 0.0), 1.0)    # 1000 m north of 89.995 N

    def test_rejects_offset_beyond_south_pole(self):
        with self.assertRaises(ValueError):
            pixel_to_latlon(499.5, 1499.5, C, (-89.995, 0.0), 1.0)   # 1000 m south of 89.995 S

    def test_offset_reaching_but_not_crossing_pole_is_valid(self):
        north_m = (90.0 - 89.99) * ref.MODEL_METERS_PER_DEGREE  # exactly reaches 90 N in the model
        lat, _ = pixel_to_latlon(499.5, 499.5 - north_m, C, (89.99, 0.0), 1.0)
        self.assertLessEqual(lat, 90.0)
        self.assertAlmostEqual(lat, 90.0, delta=1e-9)


class TestLongitudeWrap(unittest.TestCase):
    def check_wrap(self, px, ref_lon, expected_azimuth):
        lat, lon = pixel_to_latlon(px, 499.5, C, (65.0, ref_lon), 1.0)
        self.assertTrue(-180.0 <= lon < 180.0, lon)
        expected = ref.documented_model(px, 499.5, C, (65.0, ref_lon), 1.0)
        self.assertAlmostEqual(lon, ref.wrap_lon(expected[1]), delta=1e-12)
        self.assertEqual(lat, expected[0])
        d, az = ref.vincenty_inverse(65.0, ref_lon, lat, lon)
        self.assertAlmostEqual(d, 1000.0, delta=ref.model_error_bound_m(1000.0, 0.0, 65.0))
        self.assertAlmostEqual(az, expected_azimuth, delta=0.1)

    def test_wraps_east_across_antimeridian(self):
        self.check_wrap(1499.5, 179.995, 90.0)       # 1000 m east

    def test_wraps_west_across_antimeridian(self):
        self.check_wrap(-500.5, -179.995, 270.0)     # 1000 m west

    def test_large_offset_wraps_only_while_position_is_unique(self):
        # 50 km east at 89 N spans ~25.7 deg of longitude: crosses 180 E and wraps to a unique position.
        lat, lon = pixel_to_latlon(499.5 + 5000, 499.5, C, (89.0, 179.0), 10.0)
        self.assertTrue(-180.0 <= lon < 180.0, lon)
        self.assertAlmostEqual(lon, ref.wrap_lon(ref.documented_model(499.5 + 5000, 499.5, C, (89.0, 179.0), 10.0)[1]),
                               delta=1e-9)
        # 500 km east at 89 N would span ~257 deg: more than half the parallel, so wrapping would alias it
        # onto a westward position. (This case previously asserted a silently wrapped result.)
        with self.assertRaises(ValueError):
            pixel_to_latlon(499.5 + 50000, 499.5, C, (89.0, 179.0), 10.0)


class TestMissionFileSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def out(self, name):
        return os.path.join(self.tmp.name, f"{name}.waypoints")

    def assert_rejected_without_file(self, name, path, home=(500, 500), center=(500, 500), geo=(25.0, 82.0), mpp=10.0):
        out = self.out(name)
        with self.assertRaises(ValueError):
            generate_mission_file(path, list(range(len(path))), home, center, geo, mpp, filename=out)
        self.assertFalse(os.path.exists(out), f"{name}: a partial mission file was written")

    def test_invalid_scale_writes_nothing(self):
        self.assert_rejected_without_file("bad_scale", [(500, 500), (500, 500), (500, 400), (500, 500)], mpp=-2.0)

    def test_pole_crossing_mid_route_writes_nothing(self):
        # Only the 3rd route point (10 km north at 100 m/px) is invalid: nothing may be written
        self.assert_rejected_without_file("pole", [(500, 500), (500, 500), (500, 400), (500, 500)],
                                          geo=(89.999, 0.0), mpp=100.0)

    def test_non_finite_route_point_writes_nothing(self):
        self.assert_rejected_without_file("nan_point", [(500, 500), (500, 500), (float("nan"), 400), (500, 500)])

    def test_invalid_reference_writes_nothing(self):
        self.assert_rejected_without_file("bad_ref", [(500, 500), (520, 480), (500, 500)], geo=(95.0, 82.0))

    def test_valid_mission_has_only_finite_in_range_coordinates(self):
        out = self.out("antimeridian")
        path = [(500, 500), (500, 500), (900, 450), (950, 520), (500, 500)]  # crosses 180 E
        generate_mission_file(path, [0, 1, 2, 3, 4], (500, 500), (500, 500), (-17.0, 179.995), 10.0, filename=out)
        findings, items = wpl_validator.validate(out)
        self.assertEqual([f for f in findings if f.level == "ERROR"], [])
        self.assertTrue(any(r["lon"] < 0 for r in items), "route should have wrapped to negative longitude")
        for r in items:
            self.assertTrue(math.isfinite(r["lat"]) and math.isfinite(r["lon"]))


if __name__ == '__main__':
    unittest.main()
