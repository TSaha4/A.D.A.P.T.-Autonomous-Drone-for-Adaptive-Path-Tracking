import unittest
from unittest.mock import patch

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)

from src.mission.mission_output import display_path_on_map
from main import map_routed_drop_distances


class TestMissionOutput(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
