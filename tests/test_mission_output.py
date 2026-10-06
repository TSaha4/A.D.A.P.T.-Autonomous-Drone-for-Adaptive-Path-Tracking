import os
import tempfile
import unittest
from src.mission.mission_output import generate_mission_file


class TestMissionOutput(unittest.TestCase):
    def setUp(self):
        fd, self.filename = tempfile.mkstemp(suffix=".waypoints")
        os.close(fd)

    def tearDown(self):
        os.remove(self.filename)

    def write_mission(self, full_path, drop_indices, home, **kwargs):
        generate_mission_file(full_path, drop_indices, home, (100, 100), (25.0, 82.0), 2.0,
                              filename=self.filename, **kwargs)
        with open(self.filename) as f:
            return [line.rstrip("\n").split("\t") for line in f.readlines()[1:]]

    def servo_rows(self, rows):
        return [r for r in rows if r[3] == "183"]

    def test_home_cluster_gets_a_drop(self):
        # HOME is itself a cluster's safe point, so NN-TSP visits it again right after takeoff
        home, a = (100, 100), (150, 120)
        rows = self.write_mission([home, home, a, home], [0, 1, 2, 3], home)
        self.assertEqual(len(self.servo_rows(rows)), 2)

    def test_single_cluster_mission_has_a_drop(self):
        home = (100, 100)
        rows = self.write_mission([home, home, home], [0, 1, 2], home)
        self.assertEqual(len(self.servo_rows(rows)), 1)

    def test_no_drop_on_departure_or_return(self):
        # HOME not among the drop points: only the intermediate point is a drop
        home, a = (100, 100), (150, 120)
        rows = self.write_mission([home, a, home], [0, 1, 2], home)
        self.assertEqual(len(self.servo_rows(rows)), 1)

    def test_servo_command_params(self):
        home, a = (100, 100), (150, 120)
        rows = self.write_mission([home, a, home], [0, 1, 2], home)
        servo = self.servo_rows(rows)[0]
        self.assertEqual(servo[4], "9")     # param1: servo channel
        self.assertEqual(servo[5], "2000")  # param2: PWM

        rows = self.write_mission([home, a, home], [0, 1, 2], home, servo_channel=10, servo_pwm=1100)
        servo = self.servo_rows(rows)[0]
        self.assertEqual((servo[4], servo[5]), ("10", "1100"))

    def test_mission_sequence(self):
        home, a = (100, 100), (150, 120)
        rows = self.write_mission([home, a, home], [0, 1, 2], home)
        self.assertEqual([int(r[0]) for r in rows], list(range(len(rows))))
        commands = [r[3] for r in rows]
        self.assertEqual(commands[:2], ["16", "22"])   # home, takeoff
        self.assertEqual(commands[-2:], ["20", "21"])  # RTL, land


if __name__ == '__main__':
    unittest.main()
