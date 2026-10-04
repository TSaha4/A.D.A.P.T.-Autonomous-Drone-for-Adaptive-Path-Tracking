import sys, os; sys.path.insert(0, os.path.abspath('.'))
import cv2, numpy as np, json
from src.routing.pathfinding import (
    polyline_intersections, build_routing_grid, cell_of, cell_center, locate_cell,
    compute_full_path, polyline_flood_pixels, nearest_neighbor_tsp, routed_distance_function
)
from src.routing.dstarlite import DStarLite
import sys; sys.path.insert(0, '.')
from src.mission.constraints import plan_mission_stops, sortie_battery_report
from main import refine_route, map_routed_drop_distances
from src.mission.multi_base import single_drop_reach
from src.mission.safe_dropzone import filter_drops_by_home_connectivity
from config import settings

# Load inputs from session 172629
bhopal_img = cv2.imread('data/input/bhopal.png')
geo_lat, geo_lon = 23.37, 78.03
cx, cy = bhopal_img.shape[1] // 2, bhopal_img.shape[0] // 2
m_per_px = 2.0

def latlon_to_pixel(lat, lon):
    dy_m = (lat - geo_lat) * 111320.0
    dx_m = (lon - geo_lon) * (111320.0 * np.cos(np.radians(geo_lat)))
    return int(round(cx + dx_m / m_per_px)), int(round(cy - dy_m / m_per_px))

def parse(w):
    drops = []
    home = None
    with open(w) as f:
        for line in f:
            if not line.strip() or line.startswith('QGC'): continue
            p = line.strip().split('\t')
            if p[3] == '16' and float(p[10]) == 0.0 and home is None:
                home = latlon_to_pixel(float(p[8]), float(p[9]))
            if p[3] == '183':
                drops.append(latlon_to_pixel(float(p[8]), float(p[9])))
    return home, drops

h1, d1 = parse('data/output/session_20261004_172629/base1.waypoints')
h2, d2 = parse('data/output/session_20261004_172629/base2.waypoints')

print(f"Base 1: home={h1}, drops={len(d1)}")
print(f"Base 2: home={h2}, drops={len(d2)}")

# Read the waypoints directly as polylines to verify the initial crossing
def read_wpl_polyline(path):
    pts = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('QGC'): continue
            p = line.strip().split('\t')
            if int(p[3]) in (16, 22):
                pts.append(latlon_to_pixel(float(p[8]), float(p[9])))
    return pts

p1 = read_wpl_polyline('data/output/session_20261004_172629/base1.waypoints')
p2 = read_wpl_polyline('data/output/session_20261004_172629/base2.waypoints')

initial_hits = polyline_intersections(p1, p2, return_segments=True)
print(f"Initial polyline intersections: {len(initial_hits)}")
