"""
Phase 3: image-resize / pixel-scale consistency of the full main() pipeline.

A synthetic north-up image with a single red square is run through the unmodified main().
Only I/O is stubbed: the interactive HSV click step (thresholds set directly), GUI windows,
the weather request (calm weather, so the flood mask is not dilated), and the mission
filename (redirected to a temp dir). The physical location of every waypoint is then
compared with an independent WGS84 ground truth computed from ORIGINAL-image geometry.
"""
import contextlib
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import main as main_module
from config import settings
from src.mission import mission_output
from src.vision.image_processing import ImageProcessor
from tests import geodesy_reference as ref

GEO_CENTER = (25.0, 82.0)
MPP = 10.0
CALM = {"precipitation": 0.0, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0, "status": "success"}


def make_image(path, width, height, square):
    img = np.full((height, width, 3), 255, np.uint8)
    x0, y0, x1, y1 = square
    img[y0:y1 + 1, x0:x1 + 1] = (0, 0, 255)  # pure red (BGR), inclusive bounds
    cv2.imwrite(path, img)


def fake_select(self, image):
    self.image = image.copy()
    self.hsv_lower = np.array([0, 100, 100], np.uint8)
    self.hsv_upper = np.array([179, 255, 255], np.uint8)
    self.is_red_wrap = True
    return self.hsv_lower, self.hsv_upper


def run_pipeline(image_path, display_width, tmpdir, lat=GEO_CENTER[0], lon=GEO_CENTER[1], redirect=True,
                 generate=None, mpp=MPP, select=fake_select):
    """Runs main.main() inside tmpdir (so main's relative data/output path never touches the repository)
    and returns what it passed to generate_mission_file. With redirect=False main writes its own output
    file, data/output/enriched_drone_mission.waypoints, under tmpdir. `generate` replaces
    generate_mission_file entirely."""
    captured = {}

    def capture(full_path, drop_indices, home, image_center_px, geo_center, meters_per_pixel, filename=None, **kw):
        captured.update(full_path=full_path, drop_indices=drop_indices, home=home,
                        image_center_px=image_center_px, geo_center=geo_center,
                        meters_per_pixel=meters_per_pixel)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out = os.path.join(tmpdir, f"{stem}_w{display_width}.waypoints")
        captured["file"] = out
        return mission_output.generate_mission_file(full_path, drop_indices, home, image_center_px,
                                                    geo_center, meters_per_pixel, filename=out, **kw)

    def record_only(*args, **kwargs):
        captured.update(home=args[2], image_center_px=args[3], meters_per_pixel=args[5])
        return mission_output.generate_mission_file(*args, **kwargs)

    argv = ["main.py", os.path.abspath(image_path), "--display-width", str(display_width),
            "--lat", str(lat), "--lon", str(lon)]
    with contextlib.chdir(tmpdir), \
            mock.patch.object(sys, "argv", argv), \
            mock.patch.object(settings, "METERS_PER_PIXEL", mpp), \
            mock.patch.object(ImageProcessor, "select_sample_points", select), \
            mock.patch.object(main_module, "get_weather_data", return_value=dict(CALM)), \
            mock.patch.object(main_module, "show_image_safe"), \
            mock.patch.object(main_module, "display_path_on_map"), \
            mock.patch.object(main_module, "generate_mission_file",
                              side_effect=generate or (capture if redirect else record_only)):
        main_module.main()
    return captured


def read_nav_rows(path):
    with open(path) as f:
        rows = [line.rstrip("\n").split("\t") for line in f.readlines()[1:]]
    return [(float(r[8]), float(r[9])) for r in rows]


class TestResizeConsistency(unittest.TestCase):
    # 1200 x 900 original image; square spans original pixels x 200..499, y 150..449
    W0, H0 = 1200, 900
    SQUARE = (200, 150, 499, 449)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.image = os.path.join(cls.tmp.name, "synthetic.png")
        make_image(cls.image, cls.W0, cls.H0, cls.SQUARE)
        x0, y0, x1, y1 = cls.SQUARE
        center0 = ((cls.W0 - 1) / 2, (cls.H0 - 1) / 2)  # geometric centre, original pixel-centre coords
        cls.corners = [ref.physical_ground_truth(x, y, center0, GEO_CENTER, MPP)
                       for x, y in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]]
        cls.results = {}
        for width in (1600, 1200, 900, 600, 300):
            cls.results[width] = run_pipeline(cls.image, width, cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def scale(self, width):
        return min(1.0, width / self.W0)

    def nearest_corner(self, lat, lon):
        dists = [ref.vincenty_inverse(lat, lon, c[0], c[1])[0] for c in self.corners]
        i = int(np.argmin(dists))
        return i, dists[i]

    def test_a_no_resize_uses_configured_scale(self):
        # Original width 1200 <= display width -> image untouched, scale factor 1
        for width in (1600, 1200):
            r = self.results[width]
            self.assertEqual(r["meters_per_pixel"], MPP)

    def test_a_no_resize_home_is_square_corner(self):
        lat, lon = read_nav_rows(self.results[1200]["file"])[0]
        _, err = self.nearest_corner(lat, lon)
        # Only sources of error: model approximation (bound) + main.py's centre convention (<= 0.5 px)
        x0, y0, _, _ = self.SQUARE
        e, n = (x0 - (self.W0 - 1) / 2) * MPP, ((self.H0 - 1) / 2 - y0) * MPP
        self.assertLessEqual(err, ref.model_error_bound_m(e, n, GEO_CENTER[0]) + 0.5 * MPP * math.sqrt(2))

    def test_b_resize_preserves_physical_location(self):
        for width in (900, 600, 300):
            s = self.scale(width)
            lat, lon = read_nav_rows(self.results[width]["file"])[0]
            _, err = self.nearest_corner(lat, lon)
            # A hull vertex can move by ~1 resized pixel through INTER_AREA resampling + thresholding,
            # plus <= 0.5 px centre convention, plus model approximation.
            x0, y0, _, _ = self.SQUARE
            e, n = (x0 - (self.W0 - 1) / 2) * MPP, ((self.H0 - 1) / 2 - y0) * MPP
            tol = 1.5 * math.sqrt(2) * MPP / s + ref.model_error_bound_m(e, n, GEO_CENTER[0])
            self.assertLessEqual(err, tol, f"display width {width}: HOME {err:.1f} m from true corner (tol {tol:.1f} m)")

    def test_c_display_width_does_not_move_waypoints(self):
        homes = {w: read_nav_rows(r["file"])[0] for w, r in self.results.items()}
        corner_ids = {w: self.nearest_corner(*h)[0] for w, h in homes.items()}
        self.assertEqual(len(set(corner_ids.values())), 1, f"HOME snapped to different corners: {corner_ids}")
        ref_lat, ref_lon = homes[1200]
        for width, (lat, lon) in homes.items():
            d, _ = ref.vincenty_inverse(ref_lat, ref_lon, lat, lon)
            tol = 2 * 1.5 * math.sqrt(2) * MPP / self.scale(width)
            self.assertLessEqual(d, tol, f"display width {width} moved HOME by {d:.1f} m (tol {tol:.1f} m)")

    def test_every_waypoint_maps_its_pixel_through_the_documented_model(self):
        # No hidden transformation between pathfinding output and mission rows
        for width, r in self.results.items():
            rows = read_nav_rows(r["file"])
            expected_home = ref.documented_model(r["home"][0], r["home"][1], r["image_center_px"],
                                                 r["geo_center"], r["meters_per_pixel"])
            for lat, lon in rows:
                self.assertAlmostEqual(lat, expected_home[0], delta=1e-12)
                self.assertAlmostEqual(lon, expected_home[1], delta=1e-12)


if __name__ == '__main__':
    unittest.main()
