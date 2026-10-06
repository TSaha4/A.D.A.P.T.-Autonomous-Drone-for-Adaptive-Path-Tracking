"""Validates the independent test reference (tests/geodesy_reference.py) against published values,
so the geospatial ground-truth tests do not rest on an unverified oracle."""
import math
import unittest
from tests import geodesy_reference as ref


def dms(d, m, s):
    sign = -1 if d < 0 else 1
    return sign * (abs(d) + m / 60.0 + s / 3600.0)


class TestGeodesyReference(unittest.TestCase):
    # Geoscience Australia worked example (Vincenty, GRS80; GRS80 and WGS84 differ by
    # ~1e-11 in flattening, a sub-millimetre effect over this 55 km line).
    FLINDERS = (dms(-37, 57, 3.72030), dms(144, 25, 29.52440))
    BUNINYONG = (dms(-37, 39, 10.15610), dms(143, 55, 35.38390))
    DISTANCE = 54972.271
    AZIMUTH = dms(306, 52, 5.37)

    def test_inverse_published_example(self):
        s, az = ref.vincenty_inverse(*self.FLINDERS, *self.BUNINYONG)
        self.assertAlmostEqual(s, self.DISTANCE, delta=0.001)
        self.assertAlmostEqual(az, self.AZIMUTH, delta=0.01 / 3600)

    def test_direct_published_example(self):
        lat, lon = ref.vincenty_direct(*self.FLINDERS, self.AZIMUTH, self.DISTANCE)
        self.assertAlmostEqual(lat, self.BUNINYONG[0], delta=1e-7)
        self.assertAlmostEqual(lon, self.BUNINYONG[1], delta=1e-7)

    def test_one_degree_along_equator(self):
        s, az = ref.vincenty_inverse(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(s, ref.WGS84_A * math.pi / 180.0, delta=1e-6)  # 111319.4908 m
        self.assertAlmostEqual(az, 90.0, delta=1e-9)

    def test_meridian_quadrant(self):
        # WGS84 equator-to-pole meridian arc length: 10,001,965.729 m
        s, _ = ref.vincenty_inverse(0.0, 0.0, 89.999999999, 0.0)
        self.assertAlmostEqual(s, 10001965.729, delta=0.01)

    def test_direct_inverse_consistency(self):
        for lat, lon, az, d in [(25.0, 82.0, 37.0, 7071.0), (-33.9, 151.2, 200.0, 5000.0),
                                (60.0, -179.99, 90.0, 3000.0), (0.0, 0.0, 315.0, 1500.0)]:
            lat2, lon2 = ref.vincenty_direct(lat, lon, az, d)
            s, az1 = ref.vincenty_inverse(lat, lon, lat2, lon2)
            self.assertAlmostEqual(s, d, delta=1e-6)
            self.assertAlmostEqual(az1, az, delta=1e-8)

    def test_documented_model_inverse_round_trip(self):
        center, geo = (499.5, 499.5), (25.0, 82.0)
        for px, py in [(0, 0), (999, 999), (499.5, 499.5), (123.25, 876.75)]:
            lat, lon = ref.documented_model(px, py, center, geo, 10.0)
            rx, ry = ref.documented_model_inverse(lat, lon, center, geo, 10.0)
            self.assertAlmostEqual(rx, px, delta=1e-7)
            self.assertAlmostEqual(ry, py, delta=1e-7)


if __name__ == '__main__':
    unittest.main()
