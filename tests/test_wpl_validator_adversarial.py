"""
Adversarial tests for tests/wpl_validator.py: every corruption of a known-good A.D.A.P.T. mission
must be reported as an ERROR by the check responsible for it. Corruptions the validator cannot detect
in principle without the route sidecar are asserted as such, so its blind spots are explicit.
"""
import os
import tempfile
import unittest

from src.mission.mission_output import generate_mission_file
from tests import wpl_validator as v

HOME = (375, 374)
ROUTE = {"full_path": [HOME, HOME, (390, 340), (420, 300), (360, 400), (300, 450), HOME],
         "drop_indices": [0, 1, 3, 5, 6], "home": HOME, "image_center_px": (374.5, 373.5),
         "geo_center": (25.3176, 82.9739), "meters_per_pixel": 2.592, "expected_drops": 3}
SPEC = dict(servo_channel=9, servo_pwm=2000)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "m.waypoints")
        generate_mission_file(ROUTE["full_path"], ROUTE["drop_indices"], ROUTE["home"], ROUTE["image_center_px"],
                              ROUTE["geo_center"], ROUTE["meters_per_pixel"], filename=self.path)
        with open(self.path) as f:
            self.lines = f.read().splitlines()

    def tearDown(self):
        self.tmp.cleanup()

    def reset(self):
        # fresh copy of the known-good mission for the next subtest (cleans up the previous one)
        self.tearDown()
        self.setUp()

    def rows(self):
        return [l.split("\t") for l in self.lines[1:]]

    def write(self, header=None, rows=None, raw=None):
        if raw is None:
            rows = self.rows() if rows is None else rows
            raw = "\n".join([self.lines[0] if header is None else header] + ["\t".join(r) for r in rows]) + "\n"
        with open(self.path, "w") as f:
            f.write(raw)

    def errors(self, sidecar=ROUTE, **spec):
        findings, _ = v.validate(self.path, sidecar, **(spec or SPEC))
        return {f.check for f in findings if f.level == "ERROR"}

    def seq_of(self, cmd, nth=0):
        return [i for i, r in enumerate(self.rows()) if r[3] == str(cmd)][nth]

    def set_field(self, row, col, value):
        rows = self.rows()
        rows[row][col] = value
        self.write(rows=rows)

    def renumber(self, rows):
        for i, r in enumerate(rows):
            r[0] = str(i)
        return rows


class TestBaseline(Base):
    def test_untouched_mission_passes_with_and_without_sidecar(self):
        self.assertEqual(self.errors(), set())
        self.assertEqual(self.errors(sidecar=None), set())


class TestStructure(Base):
    def test_missing_header(self):
        self.write(raw="\n".join(self.lines[1:]) + "\n")
        self.assertIn("header", self.errors())

    def test_wrong_header(self):
        self.write(header="QGC WPL 100")
        self.assertIn("header", self.errors())

    def test_wrong_column_count(self):
        rows = self.rows()
        rows[3] = rows[3][:-1]
        self.write(rows=rows)
        self.assertIn("columns", self.errors())

    def test_space_separated_fields(self):
        # The format specifies <tab> separators
        rows = self.rows()
        self.write(raw="\n".join([self.lines[0]] + ["\t".join(r) for r in rows[:3]] + [" ".join(rows[3])]
                                 + ["\t".join(r) for r in rows[4:]]) + "\n")
        self.assertIn("columns", self.errors())

    def test_duplicate_sequence(self):
        self.set_field(4, 0, "3")
        self.assertIn("sequence", self.errors())

    def test_skipped_sequence(self):
        rows = self.rows()
        for r in rows[5:]:
            r[0] = str(int(r[0]) + 1)
        self.write(rows=rows)
        self.assertIn("sequence", self.errors())

    def test_malformed_numbers(self):
        for col, bad in ((8, "25.3.1"), (9, ""), (4, "abc"), (0, "1.0"), (10, "1e")):
            with self.subTest(col=col, bad=bad):
                self.reset()
                self.set_field(3, col, bad)
                self.assertIn("parse", self.errors())

    def test_autocontinue_zero(self):
        self.set_field(3, 11, "0")
        self.assertIn("autocontinue", self.errors())


class TestFieldsAndCoordinates(Base):
    def test_invalid_frame_on_navigation_item(self):
        # Frame 0 = altitude above mean sea level: "100" would no longer mean 100 m above home
        for frame in ("0", "6", "99"):
            with self.subTest(frame=frame):
                self.reset()
                self.set_field(self.seq_of(16, 2), 2, frame)
                self.assertIn("frame", self.errors())

    def test_invalid_command(self):
        self.set_field(3, 3, "999")
        self.assertIn("command", self.errors())

    def test_invalid_coordinates(self):
        for col, bad, check in ((8, "90.0001", "latitude"), (8, "-91", "latitude"), (9, "180.0001", "longitude"),
                                (9, "-181", "longitude"), (8, "nan", "coordinate"), (9, "inf", "coordinate"),
                                (8, "-inf", "coordinate")):
            with self.subTest(col=col, bad=bad):
                self.reset()
                self.set_field(3, col, bad)
                self.assertIn(check, self.errors())

    def test_wrong_cruise_altitude(self):
        self.set_field(self.seq_of(16, 2), 10, "90")
        self.assertIn("altitude", self.errors(sidecar=None))

    def test_negative_altitude(self):
        self.set_field(self.seq_of(16, 2), 10, "-5")
        self.assertIn("altitude", self.errors(sidecar=None))


class TestHomeTakeoffReturn(Base):
    def test_home_wrong_command_or_altitude(self):
        for col, bad in ((3, "22"), (10, "50")):
            with self.subTest(col=col):
                self.reset()
                self.set_field(0, col, bad)
                self.assertIn("home", self.errors())

    def test_home_moved_detected_against_takeoff_and_land(self):
        self.set_field(0, 8, "25.4")
        e = self.errors(sidecar=None)
        self.assertTrue({"takeoff", "return"} <= e, e)

    def test_takeoff_wrong(self):
        for col, bad in ((10, "50"), (8, "25.4"), (3, "16")):
            with self.subTest(col=col):
                self.reset()
                self.set_field(1, col, bad)
                self.assertIn("takeoff", self.errors())

    def test_missing_rtl(self):
        rows = self.rows()
        del rows[-2]
        self.write(rows=self.renumber(rows))
        self.assertIn("return", self.errors())

    def test_missing_land(self):
        rows = self.rows()
        del rows[-1]
        self.write(rows=self.renumber(rows))
        self.assertIn("return", self.errors())

    def test_land_not_at_home(self):
        self.set_field(len(self.rows()) - 1, 8, "25.4")
        self.assertIn("return", self.errors(sidecar=None))


class TestDropsAndServo(Base):
    def drop_block_rows(self, nth=0):
        s = self.seq_of(183, nth)
        return s - 2, s + 1  # loiter .. ascend

    def test_missing_drop_block(self):
        rows = self.rows()
        a, b = self.drop_block_rows(1)
        del rows[a:b + 1]
        self.write(rows=self.renumber(rows))
        e = self.errors()
        self.assertTrue({"drops", "route"} <= e, e)

    def test_extra_drop_block(self):
        rows = self.rows()
        a, b = self.drop_block_rows(1)
        rows[b + 1:b + 1] = [list(r) for r in rows[a:b + 1]]
        self.write(rows=self.renumber(rows))
        e = self.errors()
        self.assertTrue({"drops", "route"} <= e, e)

    def test_servo_params_in_wrong_slots(self):
        s = self.seq_of(183)
        rows = self.rows()
        rows[s][4:8] = ["0", "0", "2000", "0"]
        self.write(rows=rows)
        self.assertIn("servo", self.errors())

    def test_malformed_servo_channel(self):
        for bad in ("9.5", "0", "-1"):
            with self.subTest(bad=bad):
                self.reset()
                self.set_field(self.seq_of(183), 4, bad)
                self.assertIn("servo", self.errors())

    def test_servo_channel_mismatch(self):
        self.set_field(self.seq_of(183), 4, "10")
        self.assertIn("servo", self.errors())

    def test_incorrect_pwm(self):
        for bad in ("1000", "3000", "0"):
            with self.subTest(bad=bad):
                self.reset()
                self.set_field(self.seq_of(183), 5, bad)
                self.assertIn("servo", self.errors())

    def test_wrong_drop_altitude(self):
        s = self.seq_of(183)
        self.set_field(s - 1, 10, "15")
        self.assertIn("drop", self.errors(sidecar=None))

    def test_servo_not_after_descent(self):
        # Swap descent waypoint and servo: the release would fire before reaching drop altitude
        s = self.seq_of(183)
        rows = self.rows()
        rows[s - 1], rows[s] = rows[s], rows[s - 1]
        self.write(rows=self.renumber(rows))
        self.assertIn("drop", self.errors(sidecar=None))


class TestRouteCorrespondence(Base):
    def test_shifted_waypoint(self):
        s = self.seq_of(16, 3)
        self.set_field(s, 8, f"{float(self.rows()[s][8]) + 1e-6:.12f}")  # ~11 cm
        self.assertIn("route", self.errors())

    def test_swapped_lat_lon(self):
        s = self.seq_of(16, 3)
        rows = self.rows()
        rows[s][8], rows[s][9] = rows[s][9], rows[s][8]
        self.write(rows=rows)
        self.assertIn("route", self.errors())

    def test_reordered_waypoints(self):
        rows = self.rows()
        a, b = self.seq_of(16, 3), self.seq_of(16, 4)
        rows[a][8:10], rows[b][8:10] = rows[b][8:10], rows[a][8:10]
        self.write(rows=rows)
        self.assertIn("route", self.errors())


class TestValidatorRobustness(Base):
    """Defects found in the validator itself by this attack."""

    def test_infinite_coordinate_reported_not_crashed(self):
        # Previously raised "math domain error" while summarising the route
        self.set_field(4, 9, "inf")
        self.assertIn("coordinate", self.errors())

    def test_nan_caught_by_route_check_itself(self):
        # max() silently skips NaN; the route comparison must flag it on its own
        self.set_field(self.seq_of(16, 4), 9, "nan")
        findings, _ = v.validate(self.path, ROUTE, **SPEC)
        self.assertTrue(any(f.check == "route" and f.level == "ERROR" for f in findings))

    def test_antimeridian_mission_is_not_a_false_positive(self):
        # A route that crosses 180 E has wrapped (negative) longitudes in the file; the independent model
        # gives values above +180. Compared on the circle they agree.
        route = dict(ROUTE, geo_center=(-17.0, 179.9995), meters_per_pixel=10.0)
        generate_mission_file(route["full_path"], route["drop_indices"], route["home"], route["image_center_px"],
                              route["geo_center"], route["meters_per_pixel"], filename=self.path)
        findings, items = v.validate(self.path, route, **SPEC)
        self.assertTrue(any(r["lon"] < 0 for r in items) and any(r["lon"] > 0 for r in items))
        self.assertEqual([f for f in findings if f.level == "ERROR"], [])


class TestDocumentedBlindSpots(Base):
    """Without the route sidecar these corruptions are internally consistent missions; only the
    sidecar (the pixel route) can reveal them. Asserted so the limitation stays visible."""

    def test_consistent_waypoint_shift_needs_sidecar(self):
        s = self.seq_of(16, 4)  # the transit waypoint (390, 340), not part of any drop block
        self.set_field(s, 8, f"{float(self.rows()[s][8]) + 1e-4:.12f}")
        self.assertEqual(self.errors(sidecar=None), set())
        self.assertIn("route", self.errors())

    def test_renumbered_missing_drop_needs_sidecar(self):
        rows = self.rows()
        s = self.seq_of(183, 1)
        del rows[s - 2:s + 2]
        self.write(rows=self.renumber(rows))
        self.assertEqual(self.errors(sidecar=None), set())
        self.assertIn("drops", self.errors())


if __name__ == '__main__':
    unittest.main()
