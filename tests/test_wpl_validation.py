"""Phase 10: missions from generate_mission_file pass the independent validator, and the validator
itself rejects known-bad missions (so a PASS is meaningful)."""
import os
import tempfile
import unittest
from src.mission.mission_output import generate_mission_file
from tests import wpl_validator as v

HOME, A, B = (375, 374), (420, 300), (300, 450)
ROUTE = {"full_path": [HOME, HOME, (390, 340), A, (360, 400), B, HOME],
         "drop_indices": [0, 1, 3, 5, 6], "home": HOME, "image_center_px": (375, 374),
         "geo_center": (25.3176, 82.9739), "meters_per_pixel": 2.592, "expected_drops": 3}


class TestWplValidation(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".waypoints")
        os.close(fd)
        generate_mission_file(ROUTE["full_path"], ROUTE["drop_indices"], ROUTE["home"], ROUTE["image_center_px"],
                              ROUTE["geo_center"], ROUTE["meters_per_pixel"], filename=self.path)

    def tearDown(self):
        os.remove(self.path)

    def errors(self, sidecar=ROUTE):
        findings, _ = v.validate(self.path, sidecar)
        return [f for f in findings if f.level == "ERROR"]

    def mutate(self, seq, col, value):
        with open(self.path) as f:
            lines = f.read().splitlines()
        fields = lines[seq + 1].split("\t")
        fields[col] = value
        lines[seq + 1] = "\t".join(fields)
        with open(self.path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def servo_seq(self):
        _, items = v.validate(self.path)
        return next(r["seq"] for r in items if r["cmd"] == v.DO_SET_SERVO)

    def test_generated_mission_passes(self):
        self.assertEqual(self.errors(), [])

    def test_detects_servo_params_in_wrong_slots(self):
        s = self.servo_seq()
        self.mutate(s, 4, "0")      # param1
        self.mutate(s, 5, "0")      # param2
        self.mutate(s, 6, "2000")   # param3  (the original serialization)
        self.assertTrue(any(f.check == "servo" for f in self.errors()))

    def test_detects_sequence_gap(self):
        self.mutate(3, 0, "99")
        self.assertTrue(any(f.check == "sequence" for f in self.errors()))

    def test_detects_invalid_coordinates(self):
        self.mutate(4, 8, "90.5")
        self.mutate(5, 9, "180.0163")
        checks = {f.check for f in self.errors()}
        self.assertIn("latitude", checks)
        self.assertIn("longitude", checks)

    def test_detects_coordinate_not_matching_pixel(self):
        self.mutate(4, 8, "25.40000000")
        self.assertTrue(any(f.check == "route" for f in self.errors()))

    def test_waypoint_on_reference_meridian_is_not_flagged(self):
        # px == cx gives exactly the reference longitude, written as the short decimal 82.9739
        _, items = v.validate(self.path)
        self.assertTrue(any(r["lon_s"] == "82.9739" for r in items))
        findings, _ = v.validate(self.path, ROUTE)
        self.assertFalse(any(f.check == "precision" for f in findings))

    def test_detects_truncated_coordinates(self):
        with open(self.path) as f:
            lines = f.read().splitlines()
        for k in range(1, len(lines)):
            fields = lines[k].split("\t")
            fields[8], fields[9] = f"{float(fields[8]):.4f}", f"{float(fields[9]):.4f}"
            lines[k] = "\t".join(fields)
        with open(self.path, "w") as f:
            f.write("\n".join(lines) + "\n")
        self.assertTrue(any(f.check == "route" for f in self.errors()))

    def test_detects_missing_drop(self):
        route = dict(ROUTE, expected_drops=4)
        self.assertTrue(any(f.check == "drops" for f in self.errors(route)))


if __name__ == '__main__':
    unittest.main()
