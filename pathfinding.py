import numpy as np

def nearest_neighbor_tsp(points, home=None):
    # Simple Nearest Neighbor TSP with HOME start/return.
    # Args:
    #     points: list of (x,y) tuples
    #     home: (x,y) tuple, drone start/return
    # Returns:
    #     order: list of indices in visit order
    #     ordered_points: list of points in visit order
    if not points:
        return [], []

    pts = np.array(points)
    n = len(pts)
    visited = [False] * n
    order = []

    # Start at HOME if provided, else first point
    if home is not None:
        current = np.array(home)
    else:
        current = pts[0]

    for _ in range(n):
        dists = np.linalg.norm(pts - current, axis=1)
        dists = [d if not visited[i] else np.inf for i, d in enumerate(dists)]
        idx = int(np.argmin(dists))
        order.append(idx)
        visited[idx] = True
        current = pts[idx]

    # Return to HOME if specified
    ordered_points = [points[i] for i in order]
    if home is not None:
        ordered_points.insert(0, home)
        ordered_points.append(home)

    return order, ordered_points
