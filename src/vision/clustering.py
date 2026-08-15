import numpy as np
import cv2

def cluster_contours(contours):
    # Compute cluster centers and convex hull edges for each contour.
    # Returns:
    #     cluster_centers: list of (x,y) tuples
    #     cluster_edges: list of arrays of contour points (edges)
    
    cluster_centers = []
    cluster_edges = []

    for cnt in contours:
        # Convex hull of the contour
        hull = cv2.convexHull(cnt)
        cluster_edges.append(hull.reshape(-1, 2))

        # Compute centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = hull[0][0], hull[0][1]
        cluster_centers.append((cx, cy))

    return cluster_centers, cluster_edges
