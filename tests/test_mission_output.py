import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)

from src.mission.mission_output import display_path_on_map, generate_mission_file
from main import map_routed_drop_distances


class TestMissionOutput(unittest.TestCase):
    def setUp(self):
        fd, self.filename = tempfile.mkstemp(suffix=".waypoints")
        os.close(fd)

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def write_mission(self, full_path, drop_indices, home, **kwargs):
        generate_mission_file(full_path, drop_indices, home, (100, 100), (25.0, 82.0), 2.0,
                              filename=self.filename, **kwargs)
        with open(self.filename) as f:
            return [line.rstrip("\n").split("\t") for line in f.readlines()[1:]]

    def servo_rows(self, rows):
        return [r for r in rows if r[3] == "183"]

    def test_sorties_have_distinct_colors_and_base_reload_labels(self):
        image = np.zeros((40, 110, 3), dtype=np.uint8)
        path = [(5, 20), (20, 20), (30, 20), (45, 20), (100, 20)]
        with patch("src.mission.mission_output.plt.show"):
            with patch("src.mission.mission_output.cv2.putText",
                       wraps=__import__("cv2").putText) as put_text:
                rendered = display_path_on_map(
                    image, [], None, path, [1, 3], home=(5, 5),
                    reload_indices=[2])
        self.assertTupleEqual(tuple(rendered[20, 10]), (0, 0, 213))  # sortie 1: red #d50000
        self.assertTupleEqual(tuple(rendered[20, 65]), (126, 35, 26))  # sortie 2: navy #1a237e, inside an RTB dash
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("BASE", labels)
        self.assertIn("RELOAD", labels)

    def test_routed_distance_mapping_follows_drop_semantics_across_reload(self):
        class Route:
            points = [(0, 0), (10, 0), (0, 0), (0, 0), (20, 0), (0, 0)]
            drop_indices = [1, 4]

        mapped = map_routed_drop_distances(
            Route(), [(10, 0), (20, 0)], [10, 11, 0, 30, 31, 0], (0, 0))
        self.assertEqual(mapped[0], 10.0)
        self.assertEqual(mapped[1], 30.0)
        self.assertEqual(mapped[("home", 1)], 30.0)

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
