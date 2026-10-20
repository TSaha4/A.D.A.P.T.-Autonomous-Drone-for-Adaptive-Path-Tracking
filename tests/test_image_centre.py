"""G2: the geographic reference (--lat/--lon) is the ground position of the image's geometric centre,
which in OpenCV pixel-centre coordinates is ((W-1)/2, (H-1)/2) for every width/height parity."""
import os
import tempfile
import unittest

from tests import geodesy_reference as ref
from tests.test_resize_consistency import GEO_CENTER, MPP, make_image, read_nav_rows, run_pipeline

SQUARE = (200, 150, 499, 449)


class TestImageCentre(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.runs = {}
        # (width, height): even/even, odd/odd, even width + odd height, odd width + even height.
        # All are narrower than the 1600 px display width, so no resizing takes place.
        for size in [(1000, 800), (999, 799), (1000, 799), (999, 800)]:
            path = os.path.join(cls.tmp.name, f"img_{size[0]}x{size[1]}.png")
            make_image(path, *size, SQUARE)
            cls.runs[size] = run_pipeline(path, 1600, cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_reference_pixel_is_geometric_centre(self):
        for (w, h), r in self.runs.items():
            self.assertEqual(tuple(r["image_center_px"]), ((w - 1) / 2, (h - 1) / 2), f"{w}x{h}")

    def test_odd_dimension_centre_is_a_pixel_centre(self):
        cx, cy = self.runs[(999, 799)]["image_center_px"]
        self.assertEqual((cx, cy), (499.0, 399.0))

    def test_home_matches_true_centre_model_exactly(self):
        # HOME is an exact corner pixel of the square (no resampling), so its coordinate must equal the
        # documented model evaluated about the geometric centre, for every parity combination.
        x0, y0, x1, y1 = SQUARE
        for (w, h), r in self.runs.items():
            lat, lon = read_nav_rows(r["file"])[0]
            candidates = [ref.documented_model(x, y, ((w - 1) / 2, (h - 1) / 2), GEO_CENTER, MPP)
                          for x, y in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]]
            err = min(max(abs(lat - c[0]), abs(lon - c[1])) for c in candidates)
            self.assertLess(err, 1e-12, f"{w}x{h}: HOME deviates from the true-centre model by {err:.3e} deg")

    def test_resized_reference_is_centre_of_display_image(self):
        # 1200 x 900 resized to width 600 -> 600 x 450 display image
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "big.png")
            make_image(path, 1200, 900, SQUARE)
            r = run_pipeline(path, 600, tmp)
        self.assertEqual(tuple(r["image_center_px"]), ((600 - 1) / 2, (450 - 1) / 2))


if __name__ == '__main__':
    unittest.main()
