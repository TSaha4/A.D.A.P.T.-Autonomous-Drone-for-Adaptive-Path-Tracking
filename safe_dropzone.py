import numpy as np
import cv2

def point_line_distance(point, line_start, line_end):
    # Find the shortest distance from a point to a line segment.
    line_vec = line_end - line_start
    pnt_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(pnt_vec)

    line_unitvec = line_vec / line_len
    proj_length = np.dot(pnt_vec, line_unitvec)

    if proj_length < 0:
        return np.linalg.norm(point - line_start)
    elif proj_length > line_len:
        return np.linalg.norm(point - line_end)
    else:
        proj_point = line_start + proj_length * line_unitvec
        return np.linalg.norm(point - proj_point)


def find_safe_drop_points(clustered_contours, path_order, path_points):
    # Find safe drop points for each cluster.
    
    # Parameters:
    #     clustered_contours : list of np.ndarray
    #         List of clustered contours (each Nx2 points).
    #     path_order : list of int
    #         Order of clusters visited by pathfinding.
    #     path_points : list of (x, y)
    #         Actual points visited in order (e.g., hull vertices).
            
    # Returns:
    #     safe_points : list of (x, y)
    #         Safe drop points chosen for each cluster.
    safe_points = []
    path_points = [np.array(p, dtype=float) for p in path_points]

    for cluster_idx in path_order:
        contour = clustered_contours[cluster_idx]
        contour_points = np.squeeze(contour)

        if contour_points.ndim == 1:
            contour_points = np.expand_dims(contour_points, axis=0)

        min_dist = float('inf')
        best_pt = contour_points[0]

        for pt in contour_points:
            pt_np = np.array(pt, dtype=float)
            for i in range(len(path_points) - 1):
                dist = point_line_distance(pt_np, path_points[i], path_points[i + 1])
                if dist < min_dist:
                    min_dist = dist
                    best_pt = pt_np

        safe_points.append(tuple(best_pt))

    return safe_points
