"""A run that does not produce a mission must not leave a mission file at the output path:
neither a partial file from this run nor a complete file from an earlier run that could be
mistaken for this run's result. Conversely, a run that cannot possibly produce a mission (invalid
arguments, unusable geographic reference/scale, unreadable image) must not destroy the earlier one."""
import os
import stat
import tempfile
import unittest

from tests import wpl_validator
from tests.test_resize_consistency import fake_select, make_image, run_pipeline

OUTPUT = os.path.join("data", "output", "enriched_drone_mission.waypoints")
STALE = "QGC WPL 110\n0\t1\t3\t16\t0\t0\t0\t0\t10.0\t10.0\t0\t1\n"


class TestStaleMission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, OUTPUT)
        os.makedirs(os.path.dirname(self.out))
        with open(self.out, "w") as f:
            f.write(STALE)
        self.flood = os.path.join(self.tmp.name, "flood.png")
        make_image(self.flood, 1000, 800, (200, 150, 499, 449))

    def tearDown(self):
        if os.path.isfile(self.out):
            os.chmod(self.out, stat.S_IREAD | stat.S_IWRITE)
        self.tmp.cleanup()

    def stale_intact(self):
        with open(self.out) as f:
            return f.read() == STALE

    # --- runs that generate (or try to generate) a mission -------------------------------------------

    def test_successful_run_replaces_stale_mission(self):
        run_pipeline(self.flood, 1600, self.tmp.name, redirect=False)
        self.assertFalse(self.stale_intact())
        findings, _ = wpl_validator.validate(self.out)
        self.assertEqual([f for f in findings if f.level == "ERROR"], [])

    def test_failed_generation_leaves_no_mission(self):
        # Valid reference 89.9999 N, but the route reaches ~2.5 km north of it: generation fails during
        # coordinate validation (pole crossing), after the earlier mission has been removed.
        run_pipeline(self.flood, 1600, self.tmp.name, lat=89.9999, redirect=False)
        self.assertFalse(os.path.exists(self.out), "an earlier run's mission survived a failed generation")

    def test_run_without_flood_regions_leaves_no_mission(self):
        dry = os.path.join(self.tmp.name, "dry.png")
        make_image(dry, 1000, 800, (0, 0, -1, -1))  # empty square: no red pixels
        with self.assertRaises(SystemExit):
            run_pipeline(dry, 1600, self.tmp.name, redirect=False)
        self.assertFalse(os.path.exists(self.out), "an earlier run's mission survived a run that produced none")

    def test_write_failure_leaves_no_partial_mission(self):
        def partial_then_fail(*args, filename=None, **kwargs):
            with open(filename, "w") as f:
                f.write("QGC WPL 110\n0\t1\t3\t16\t0\t0\t0\t0\t25.0\t82.0\t0\t1\n")
            raise OSError("simulated disk failure while writing")

        run_pipeline(self.flood, 1600, self.tmp.name, generate=partial_then_fail)
        self.assertFalse(os.path.exists(self.out), "a partially written mission was left behind")

    def test_unremovable_partial_mission_is_reported_not_crashed(self):
        def partial_read_only_then_fail(*args, filename=None, **kwargs):
            with open(filename, "w") as f:
                f.write("QGC WPL 110\n")
            os.chmod(filename, stat.S_IREAD)
            raise OSError("simulated disk failure while writing")

        with self.assertLogs(level="ERROR") as logs:
            run_pipeline(self.flood, 1600, self.tmp.name, generate=partial_read_only_then_fail)
        self.assertTrue(any("Could not remove mission file" in m for m in logs.output), logs.output)

    # --- runs that cannot replace the mission must not destroy it -----------------------------------

    def test_missing_image_preserves_previous_mission(self):
        with self.assertRaises(SystemExit) as cm:
            run_pipeline(os.path.join(self.tmp.name, "missing.png"), 1600, self.tmp.name, redirect=False)
        self.assertEqual(cm.exception.code, 1)
        self.assertTrue(self.stale_intact())

    def test_invalid_display_width_preserves_previous_mission(self):
        for width in (0, -5):
            with self.assertRaises(SystemExit) as cm:
                run_pipeline(self.flood, width, self.tmp.name, redirect=False)
            self.assertEqual(cm.exception.code, 2)
            self.assertTrue(self.stale_intact())

    def test_invalid_reference_or_scale_rejected_before_any_work(self):
        clicked = []

        def must_not_be_called(self_, image):
            clicked.append(True)
            return fake_select(self_, image)

        for kwargs in (dict(lat=95.0), dict(lon=-200.0), dict(lat=90.0), dict(mpp=0.0), dict(mpp=-2.0),
                       dict(mpp=float("nan"))):
            with self.subTest(**kwargs):
                with self.assertRaises(SystemExit) as cm:
                    run_pipeline(self.flood, 1600, self.tmp.name, redirect=False, select=must_not_be_called, **kwargs)
                self.assertEqual(cm.exception.code, 2)
                self.assertEqual(clicked, [], "operator was asked to click before the configuration was checked")
                self.assertTrue(self.stale_intact())

    def test_unremovable_previous_mission_stops_cleanly(self):
        # A read-only earlier mission cannot be removed; continuing would leave it looking like this run's
        # output, so the run must stop with a clear error (exit 1), not a PermissionError traceback.
        os.chmod(self.out, stat.S_IREAD)
        with self.assertRaises(SystemExit) as cm:
            run_pipeline(self.flood, 1600, self.tmp.name, redirect=False)
        self.assertEqual(cm.exception.code, 1)

    def test_directory_at_output_path_stops_cleanly(self):
        os.remove(self.out)
        os.makedirs(self.out)
        with self.assertRaises(SystemExit) as cm:
            run_pipeline(self.flood, 1600, self.tmp.name, redirect=False)
        self.assertEqual(cm.exception.code, 1)
        os.rmdir(self.out)

    def test_missing_output_directory_is_created(self):
        os.remove(self.out)
        os.rmdir(os.path.dirname(self.out))
        run_pipeline(self.flood, 1600, self.tmp.name, redirect=False)
        self.assertTrue(os.path.isfile(self.out))


if __name__ == '__main__':
    unittest.main()
