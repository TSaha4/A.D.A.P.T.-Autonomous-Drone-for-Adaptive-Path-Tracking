import unittest
import numpy as np
import cv2
from src.weather.flood_spread import predict_spread
from src.weather.flood_predictor import (
    FloodTimeline,
    predict_flood_sequence,
    weather_feature_vector,
    ca_sequence,
)

RAIN = {"precipitation": 10.0, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0}
CALM = {"precipitation": 0.0, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0}


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
        self.assertEqual(pred[43, 50], 255)  # Top
        self.assertEqual(pred[56, 50], 255)  # Bottom
        self.assertEqual(pred[50, 43], 255)  # Left
        self.assertEqual(pred[50, 56], 255)  # Right

    def test_directional_spread_with_wind(self):
        # Strong wind from North (0 deg) -> blowing South
        weather = {"precipitation": 0.0, "wind_speed_10m": 50.0, "wind_direction_10m": 0.0}
        pred = predict_spread(self.mask, weather, horizon_hours=1.0)

        # South is positive Y in image coords
        # Top edge of original was Y=45. With Southward blow, it should not expand up.
        self.assertEqual(pred[40, 50], 0)

        # Bottom edge was Y=55. It should expand down.
        self.assertEqual(pred[58, 50], 255)


class TestFloodTimeline(unittest.TestCase):
    """Task 1: the prediction must be time-indexed, not a single static contour."""

    def setUp(self):
        self.base = np.zeros((100, 100), dtype=np.uint8)
        self.base[40:50, 40:50] = 255  # 10x10 blob (CA expands ~2 px/hour here)

    # --------------------------------------------------------------- #
    def test_sequence_returns_time_indexed_frames(self):
        tl = predict_flood_sequence(self.base, RAIN, timestamps_min=[60, 30], backend="ca")
        # Frame 0 is the current observation, then one frame per requested time.
        self.assertIsInstance(tl, FloodTimeline)
        np.testing.assert_allclose(tl.times_min, [0.0, 30.0, 60.0])
        self.assertEqual(len(tl), 3)
        self.assertEqual(tl.n_predicted_frames(), 2)
        # Weather conditioning is carried for downstream use / logging.
        self.assertEqual(tl.weather["precipitation"], 10.0)

    def test_every_timestep_is_a_distinct_prediction(self):
        # 30min != 1h != 2h: the routing layer must be able to distinguish
        # "flood at the time the drone arrives", not just "the forecast".
        tl = predict_flood_sequence(self.base, RAIN, [30, 60, 120], backend="ca")
        t0, t30, t60, t120 = tl.masks
        self.assertTrue(np.array_equal(t0, self.base))
        self.assertGreater(np.count_nonzero(t30), np.count_nonzero(t0))
        self.assertGreater(np.count_nonzero(t60), np.count_nonzero(t30))
        self.assertGreater(np.count_nonzero(t120), np.count_nonzero(t60))

    def test_monotonic_spread(self):
        tl = predict_flood_sequence(self.base, RAIN, [30, 60, 120], backend="ca")
        # Later contours are supersets of earlier ones (flood only grows here).
        for a, b in zip(tl.masks, tl.masks[1:]):
            self.assertEqual(np.count_nonzero(b & (a == 0)), np.count_nonzero(b) - np.count_nonzero(a))
            self.assertTrue(np.array_equal(b | a, b))

    # --------------------------------------------------------------- #
    def test_obstacle_mask_query_rounds_to_nearest_frame(self):
        tl = predict_flood_sequence(self.base, RAIN, [30, 60], backend="ca")
        before = tl.obstacle_mask_at(0.0)
        self.assertTrue(np.array_equal(before, self.base))
        # t=5 min is much closer to frame 0 than frame 30.
        self.assertTrue(np.array_equal(tl.obstacle_mask_at(5.0), tl.masks[0]))
        # t=30 lands exactly on the 30 min frame.
        self.assertTrue(np.array_equal(tl.obstacle_mask_at(30.0), tl.masks[1]))
        # Beyond the horizon we clamp to the last frame.
        self.assertTrue(np.array_equal(tl.obstacle_mask_at(1e9), tl.masks[-1]))
        # 45 is exactly between 30 and 60 -> round up (safer / later).
        self.assertTrue(np.array_equal(tl.obstacle_mask_at(45.0), tl.masks[2]))

    def test_obstacle_mask_query_interpolates_between_frames(self):
        tl = predict_flood_sequence(self.base, RAIN, [60], backend="ca")
        # Interpolation at the boundary times reproduces the bracketing frames.
        self.assertTrue(np.array_equal(tl.obstacle_mask_at(0.0, mode="interpolate"), tl.masks[0]))
        self.assertTrue(np.array_equal(tl.obstacle_mask_at(60.0, mode="interpolate"), tl.masks[1]))
        # An interpolated frame never removes earlier flood (binary threshold blend).
        mid = tl.obstacle_mask_at(30.0, mode="interpolate")
        self.assertTrue(np.array_equal(mid | tl.masks[0], mid))

    def test_blocked_after_min_reports_first_flood_time(self):
        tl = predict_flood_sequence(self.base, RAIN, [30, 60, 120], backend="ca")
        # Already flooded at t=0.
        self.assertEqual(tl.blocked_after_min((45, 45)), 0.0)
        # Edge of the 2h spread envelope -> blocked at 120 min.
        self.assertEqual(tl.blocked_after_min((45, 53)), 120.0)
        # Far away and never reached.
        self.assertEqual(tl.blocked_after_min((5, 5)), float("inf"))

    # --------------------------------------------------------------- #
    def test_weather_conditions_the_forecast(self):
        tl_wet = predict_flood_sequence(self.base, RAIN, [60], backend="ca")
        tl_dry = predict_flood_sequence(self.base, CALM, [60], backend="ca")
        self.assertGreater(np.count_nonzero(tl_wet.masks[-1]), np.count_nonzero(tl_dry.masks[-1]))
        # No rain, no wind -> no spread -> frame equals the current contour.
        self.assertTrue(np.array_equal(tl_dry.masks[-1], self.base))

    def test_auto_backend_falls_back_when_dl_unavailable(self):
        tl = predict_flood_sequence(self.base, RAIN, [30], backend="auto")
        self.assertIn(tl.source, ("ca-baseline", "convlstm"))
        self.assertEqual(len(tl), 2)

    def test_ca_sequence_is_deterministic(self):
        f1, _ = ca_sequence(self.base, RAIN, [60])
        f2, _ = ca_sequence(self.base, RAIN, [60])
        self.assertTrue(np.array_equal(f1[0], f2[0]))
        self.assertTrue(np.array_equal(f1[1], f2[1]))

    def test_timeline_validation(self):
        with self.assertRaises(ValueError):
            FloodTimeline([0.0, 0.0], [self.base, self.base])  # non-increasing
        with self.assertRaises(ValueError):
            FloodTimeline([0.0], [self.base, self.base])  # length mismatch

    def test_weather_feature_vector_is_normalised(self):
        v = weather_feature_vector(RAIN)
        self.assertEqual(v.shape, (3,))
        self.assertTrue(np.all(np.isfinite(v)))
        # wind_direction wraps into [0, 1)
        self.assertGreaterEqual(v[2], 0.0)
        self.assertLess(v[2], 1.0)


if __name__ == "__main__":
    unittest.main()
