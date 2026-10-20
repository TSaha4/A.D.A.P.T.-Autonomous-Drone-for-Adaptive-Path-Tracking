"""
Adversarial tests for pixel_to_latlon. Expected values come from tests/geodesy_reference.py, which
does not import production code. Every rejection test states why rejecting is the correct outcome.
"""
import math
import random
import unittest

from src.mission.coordinates import pixel_to_latlon
from tests import geodesy_reference as ref

W, H = 1000, 800
C = ((W - 1) / 2, (H - 1) / 2)
REFERENCES = {
    "+lat +lon": (25.3176, 82.9739), "+lat -lon": (40.7128, -74.0060), "-lat +lon": (-33.8688, 151.2093),
    "-lat -lon": (-22.9068, -43.1729), "equator": (0.0, 36.8), "prime meridian": (51.4779, 0.0),
    "equator + prime meridian": (0.0, 0.0), "near +90": (89.9, 45.0), "near -90": (-89.9, -45.0),
    "near +180": (65.0, 179.999), "near -180": (-65.0, -179.999),
}
PIXELS = {
    "centre": C, "top-left": (0, 0), "top-right": (W - 1, 0), "bottom-left": (0, H - 1),
    "bottom-right": (W - 1, H - 1), "top-mid": (C[0], 0), "bottom-mid": (C[0], H - 1), "left-mid": (0, C[1]),
    "right-mid": (W - 1, C[1]), "just inside top-left": (1, 1), "just inside bottom-right": (W - 2, H - 2),
    "outside left/top": (-37, -11), "outside right/bottom": (W + 250, H + 90), "fractional": (123.375, 456.625),
}


def expected(px, py, geo, mpp):
    lat, lon = ref.documented_model(px, py, C, geo, mpp)
    return lat, ref.wrap_lon(lon)


class TestValidDomain(unittest.TestCase):
    def test_every_reference_pixel_and_scale_matches_independent_model(self):
        # Representability is decided independently: the model point exists iff its latitude stays within
        # [-90, 90] and its longitude offset is less than half a parallel. Otherwise a ValueError is required.
        for (gname, geo), (pname, (px, py)), mpp in [(g, p, m) for g in REFERENCES.items()
                                                     for p in PIXELS.items() for m in (1e-6, 2.0, 50.0)]:
            with self.subTest(ref=gname, pixel=pname, mpp=mpp):
                raw_lat, raw_lon = ref.documented_model(px, py, C, geo, mpp)
                if not (-90.0 <= raw_lat <= 90.0 and abs(raw_lon - geo[1]) < 180.0):
                    with self.assertRaises(ValueError):
                        pixel_to_latlon(px, py, C, geo, mpp)
                    continue
                lat, lon = pixel_to_latlon(px, py, C, geo, mpp)
                e_lat, e_lon = expected(px, py, geo, mpp)
                self.assertTrue(-90 <= lat <= 90 and -180 <= lon <= 180)
                self.assertAlmostEqual(lat, e_lat, delta=1e-12)
                # compare longitudes on the circle (+180 and -180 are the same meridian)
                self.assertAlmostEqual(abs(ref.wrap_lon(lon - e_lon)), 0.0, delta=1e-12)

    def test_large_scale_valid_at_low_latitude(self):
        # 2 km/px puts the corners ~1000 km away: still a valid (if crude) position in the model
        lat, lon = pixel_to_latlon(0, 0, C, (10.0, 20.0), 2000.0)
        self.assertAlmostEqual(lat, expected(0, 0, (10.0, 20.0), 2000.0)[0], delta=1e-9)

    def test_boundary_references_accepted(self):
        for geo in [(89.999999, 0.0), (-89.999999, 0.0), (0.0, 180.0), (0.0, -180.0)]:
            self.assertEqual(pixel_to_latlon(*C, C, geo, 1.0), geo)


class TestRejections(unittest.TestCase):
    def assert_rejects(self, why, *args):
        with self.assertRaises(ValueError, msg=why):
            pixel_to_latlon(*args)

    def test_scale(self):
        # A ground distance per pixel must be a positive finite length: 0 collapses every pixel onto the
        # reference, a negative value mirrors the map through it, NaN/inf have no position at all.
        for mpp in (0.0, -0.0, -2.0, float("nan"), float("inf"), float("-inf")):
            self.assert_rejects(f"mpp={mpp}", 0, 0, C, (25.0, 82.0), mpp)

    def test_reference(self):
        # Latitude outside [-90, 90] / longitude outside [-180, 180] are not positions on Earth; at exactly
        # +/-90 the model divides by cos(90 deg) = 0, so no east-west scale exists.
        for geo in [(90.0, 0.0), (-90.0, 0.0), (90.5, 0.0), (-91.0, 0.0), (0.0, 180.5), (0.0, -181.0),
                    (float("nan"), 0.0), (0.0, float("nan")), (float("inf"), 0.0), (0.0, float("-inf"))]:
            self.assert_rejects(f"geo={geo}", *C, C, geo, 1.0)

    def test_overflowing_offset_never_returns_nan(self):
        # Finite inputs whose product overflows to inf have no representable position; returning NaN
        # (the pre-fix behaviour) would put a non-number into the mission file.
        self.assert_rejects("E overflows", 1e306, C[1], C, (25.0, 82.0), 1e3)
        self.assert_rejects("N overflows", C[0], 1e306, C, (25.0, 82.0), 1e3)

    def test_offset_of_half_a_parallel_or_more_is_rejected(self):
        # The model maps an eastward offset E to dlon = E / (111320 cos lat0). Once |dlon| >= 180 deg two
        # different offsets land on the same longitude after wrapping (aliasing), so the output no longer
        # identifies a unique position. Pre-fix: 100 m and 100 m + one revolution returned identical results.
        lat0 = 89.9
        revolution_m = 360 * ref.MODEL_METERS_PER_DEGREE * math.cos(math.radians(lat0))
        pixel_to_latlon(C[0] + 100, C[1], C, (lat0, 10.0), 1.0)                       # valid
        self.assert_rejects("one revolution east", C[0] + 100 + revolution_m, C[1], C, (lat0, 10.0), 1.0)
        self.assert_rejects("half a parallel west", C[0] - revolution_m / 2, C[1], C, (lat0, 10.0), 1.0)
        self.assert_rejects("1 m east at 89.9999999", C[0] + 1, C[1], C, (89.9999999, 0.0), 1.0)

    def test_half_parallel_limit_is_exclusive(self):
        lat0 = 60.0
        half_m = 180 * ref.MODEL_METERS_PER_DEGREE * math.cos(math.radians(lat0))
        pixel_to_latlon(C[0] + half_m * (1 - 1e-9), C[1], C, (lat0, 0.0), 1.0)     # just under: valid
        self.assert_rejects("exactly half a parallel", C[0] + half_m * (1 + 1e-9), C[1], C, (lat0, 0.0), 1.0)


class TestLongitudeWrapping(unittest.TestCase):
    def test_east_across_plus_180(self):
        for d_m in (1.0, 50.0, 5000.0):
            lat, lon = pixel_to_latlon(C[0] + d_m, C[1], C, (0.0, 179.999), 1.0)
            raw = 179.999 + d_m / ref.MODEL_METERS_PER_DEGREE
            exp = raw - 360.0 if raw > 180.0 else raw          # independent: only wrap if past +180
            self.assertAlmostEqual(lon, exp, delta=1e-11, msg=d_m)
            self.assertTrue(-180.0 <= lon <= 180.0)

    def test_west_across_minus_180(self):
        for d_m in (1.0, 50.0, 5000.0):
            lat, lon = pixel_to_latlon(C[0] - d_m, C[1], C, (0.0, -179.999), 1.0)
            raw = -179.999 - d_m / ref.MODEL_METERS_PER_DEGREE
            exp = raw + 360.0 if raw < -180.0 else raw
            self.assertAlmostEqual(lon, exp, delta=1e-11, msg=d_m)
            self.assertTrue(-180.0 <= lon <= 180.0)

    def test_non_wrapping_values_untouched(self):
        # Wrapping must never alter an in-range result (bit-for-bit)
        rng = random.Random(11)
        for _ in range(5000):
            geo = (rng.uniform(-85, 85), rng.uniform(-179.9, 179.9))
            px, py = rng.uniform(-500, 1500), rng.uniform(-500, 1300)
            lat, lon = pixel_to_latlon(px, py, C, geo, 1.0)
            raw = geo[1] + (px - C[0]) * 1.0 / (111320.0 * math.cos(math.radians(geo[0])))
            if -180.0 <= raw <= 180.0:
                self.assertEqual(lon, raw)

    def test_plus_and_minus_180_are_the_same_meridian(self):
        a = pixel_to_latlon(C[0] + 10, C[1], C, (30.0, 180.0), 1.0)
        b = pixel_to_latlon(C[0] + 10, C[1], C, (30.0, -180.0), 1.0)
        self.assertEqual(a[0], b[0])
        self.assertAlmostEqual(abs(ref.wrap_lon(a[1] - b[1])), 0.0, delta=1e-12)
        d, _ = ref.vincenty_inverse(a[0], a[1], b[0], b[1])
        self.assertLess(d, 1e-6)


class TestRoundTrip(unittest.TestCase):
    def test_randomised_round_trip_through_independent_inverse(self):
        # pixel -> production -> tests-only inverse (no production import) -> pixel, across sizes, scales,
        # mid-latitudes, near-polar and near-antimeridian references. Only representable cases are compared.
        rng = random.Random(42)
        worst, compared = 0.0, 0
        for _ in range(20000):
            w, h = rng.choice([(750, 529), (999, 799), (4000, 3000)])
            centre = ((w - 1) / 2, (h - 1) / 2)
            geo = rng.choice([(rng.uniform(-60, 60), rng.uniform(-170, 170)),
                              (rng.choice([-1, 1]) * rng.uniform(85, 89.9), rng.uniform(-180, 180)),
                              (rng.uniform(-60, 60), rng.choice([-1, 1]) * rng.uniform(179.5, 180))])
            mpp = rng.choice([0.05, 2.592, 93.0, 633.0])
            px, py = rng.uniform(-0.1 * w, 1.1 * w), rng.uniform(-0.1 * h, 1.1 * h)
            try:
                lat, lon = pixel_to_latlon(px, py, centre, geo, mpp)
            except ValueError:
                continue
            rx, ry = ref.documented_model_inverse(lat, lon, centre, geo, mpp)
            worst = max(worst, abs(rx - px), abs(ry - py))
            compared += 1
        self.assertGreater(compared, 15000)
        self.assertLess(worst, 1e-6)


class TestPoles(unittest.TestCase):
    def test_below_at_and_beyond_pole(self):
        for sign in (1, -1):
            lat0 = sign * 89.99
            to_pole_m = (90.0 - abs(lat0)) * ref.MODEL_METERS_PER_DEGREE
            py_toward = lambda m: C[1] - sign * m          # north = up for +, south = down for -
            lat, _ = pixel_to_latlon(C[0], py_toward(to_pole_m - 1.0), C, (lat0, 0.0), 1.0)
            self.assertLess(abs(lat), 90.0)
            lat, _ = pixel_to_latlon(C[0], py_toward(to_pole_m), C, (lat0, 0.0), 1.0)   # exactly reaches
            self.assertAlmostEqual(abs(lat), 90.0, delta=1e-9)
            self.assertLessEqual(abs(lat), 90.0)
            # Past the pole the model would emit |lat| > 90, which is not a latitude; the flat model cannot
            # continue "over" the pole onto the opposite meridian, so it must refuse.
            with self.assertRaises(ValueError):
                pixel_to_latlon(C[0], py_toward(to_pole_m + 1.0), C, (lat0, 0.0), 1.0)


if __name__ == '__main__':
    unittest.main()
