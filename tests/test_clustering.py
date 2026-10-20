import unittest
import numpy as np
import cv2
from src.vision.clustering import cluster_contours


class TestClustering(unittest.TestCase):
    def test_centroid_of_square(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:41, 30:51] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers, edges = cluster_contours(contours)
        self.assertEqual(centers, [(40, 30)])
        self.assertEqual(edges[0].shape[1], 2)

    def test_zero_area_contour_falls_back_to_hull_vertex(self):
        line = np.array([[[10, 10]], [[20, 10]]], dtype=np.int32)
        centers, edges = cluster_contours([line])
        self.assertEqual(len(centers), 1)
        self.assertIn(tuple(centers[0]), [tuple(p) for p in edges[0]])


if __name__ == '__main__':
    unittest.main()
