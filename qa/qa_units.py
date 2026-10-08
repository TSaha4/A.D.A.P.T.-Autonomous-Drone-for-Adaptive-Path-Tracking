"""Adversarial unit / property tests (QA suite; see qa/README.md and docs/QA_REPORT.md).

Each test states the behaviour the component SHOULD have; a failure is a finding to be diagnosed.
"""
import heapq
import math
import os
import tempfile

import cv2
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings as hsettings, strategies as st

import harness  # noqa: F401,E402 - puts the repository on sys.path, selects the Agg backend

from src.mission.constraints import leg_battery_fraction, plan_mission_stops, sortie_battery_report
from src.mission.coordinates import pixel_to_latlon
from src.mission.mission_output import generate_mission_file, validate_waypoints_file
from src.mission.multi_base import select_bases, single_drop_reach
from src.mission.safe_dropzone import filter_drops_by_home_connectivity, find_safe_drop_points
from src.routing.dstarlite import DStarLite
from src.routing.pathfinding import (build_routing_grid, compute_full_path, compute_routed_path,
                                     grid_distance_field, nearest_neighbor_tsp, polyline_flood_pixels,
                                     polyline_intersections, routed_distance_function)
from src.vision.clustering import cluster_contours
from src.weather.flood_spread import forecast_at, forecast_frames, obstacle_mask_at, predict_spread
from tests import wpl_validator

H = hsettings(deadline=None, max_examples=150, suppress_health_check=[HealthCheck.too_slow])
H_SLOW = hsettings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow])


# ----------------------------------------------------------------------------------------------- coordinates
@H
@given(lat0=st.floats(-89.5, 89.5), lon0=st.floats(-179.999, 179.999), mpp=st.floats(0.01, 500),
       px=st.integers(-2000, 2000), py=st.integers(-2000, 2000))
def test_coordinates_all_hemispheres_match_independent_formula_or_reject(lat0, lon0, mpp, px, py):
    try:
        lat, lon = pixel_to_latlon(px, py, (0, 0), (lat0, lon0), mpp)
    except ValueError:
        # Rejection is allowed only for the documented domain violations
        north = -py * mpp
        lat_exp = lat0 + north / 111320.0
        dlon = px * mpp / (111320.0 * math.cos(math.radians(lat0)))
        assert abs(lat_exp) > 90 or abs(dlon) >= 180
        return
    exp_lat = lat0 + (-py * mpp) / 111320.0
    exp_lon = lon0 + px * mpp / (111320.0 * math.cos(math.radians(lat0)))
    if exp_lon > 180:
        exp_lon -= 360
    if exp_lon < -180:
        exp_lon += 360
    assert -90 <= lat <= 90 and -180 <= lon <= 180
    assert abs(lat - exp_lat) < 1e-9
    assert abs(((lon - exp_lon + 180) % 360) - 180) < 1e-9


@pytest.mark.parametrize("geo", [(0.0, 0.0), (-33.9, 151.2), (51.5, -0.12), (-22.9, -43.2), (0.0, 179.99999),
                                 (0.0, -180.0), (89.0, 0.0), (-89.0, 0.0)])
def test_coordinates_special_references(geo):
    lat, lon = pixel_to_latlon(10, 10, (10, 10), geo, 2.0)
    assert (lat, lon) == pytest.approx(geo)


@pytest.mark.parametrize("bad", [(91, 0), (-91, 0), (90, 0), (0, 181), (float("nan"), 0), (0, float("inf"))])
def test_coordinates_invalid_reference_rejected(bad):
    with pytest.raises(ValueError):
        pixel_to_latlon(0, 0, (0, 0), bad, 2.0)


@pytest.mark.parametrize("mpp", [0, -1, float("nan"), float("inf")])
def test_coordinates_invalid_scale_rejected(mpp):
    with pytest.raises(ValueError):
        pixel_to_latlon(0, 0, (0, 0), (25, 82), mpp)


# ------------------------------------------------------------------------------------------------ clustering
def _contour(points):
    return np.array(points, dtype=np.int32).reshape(-1, 1, 2)


@pytest.mark.parametrize("pts", [[(5, 5)], [(5, 5), (9, 5)], [(1, 1), (2, 2), (3, 3), (4, 4)],
                                 [(5, 5), (5, 5), (5, 5)], [(0, 0), (10, 0), (10, 10), (0, 10)]])
def test_clustering_degenerate_contours(pts):
    centers, edges = cluster_contours([_contour(pts)])
    assert len(centers) == 1 and len(edges) == 1
    cx, cy = centers[0]
    assert isinstance(cx, int) and isinstance(cy, int)
    xs, ys = zip(*pts)
    assert min(xs) <= cx <= max(xs) and min(ys) <= cy <= max(ys)
    assert edges[0].ndim == 2 and edges[0].shape[1] == 2


def test_clustering_empty_and_many():
    assert cluster_contours([]) == ([], [])
    rng = np.random.default_rng(1)
    contours = [_contour(rng.integers(0, 500, size=(int(rng.integers(1, 30)), 2))) for _ in range(300)]
    centers, edges = cluster_contours(contours)
    assert len(centers) == len(edges) == 300


# ------------------------------------------------------------------------------------------------ drop points
@pytest.mark.parametrize("path", [[], [(3, 3)], [(0, 0), (50, 0)]])
def test_legacy_drop_selection_path_variants(path):
    hull = np.array([[10, 10], [30, 10], [30, 30], [10, 30]])
    pts = find_safe_drop_points([hull], [0], path)
    assert len(pts) == 1 and tuple(pts[0]) in {tuple(map(float, p)) for p in hull}


def test_legacy_drop_selection_empty_order():
    assert find_safe_drop_points([], [], []) == []


def blob_mask(seed, shape=(200, 260), n=8):
    rng = np.random.default_rng(seed)
    m = np.zeros(shape, np.uint8)
    for _ in range(n):
        cv2.ellipse(m, (int(rng.integers(0, shape[1])), int(rng.integers(0, shape[0]))),
                    (int(rng.integers(3, 60)), int(rng.integers(2, 40))), float(rng.integers(0, 180)), 0, 360, 255, -1)
    return m


@H_SLOW
@given(seed=st.integers(0, 10_000), grow=st.integers(0, 6))
def test_dry_drop_points_are_dry_near_their_region_and_separated(seed, grow):
    mask = blob_mask(seed)
    obstacle = cv2.dilate(mask, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) if grow else mask
    contours = [c for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if cv2.contourArea(c) >= 50]
    if not contours:
        return
    centers, edges = cluster_contours(contours)
    r_max = max(2, min(20, int(np.ceil(max(mask.shape) * 0.03))))
    for i, c in enumerate(contours):
        pts = find_safe_drop_points([edges[i]], [0], centers, obstacle_mask=obstacle, raw_contours=[c])
        filled = np.zeros_like(mask)
        cv2.fillPoly(filled, [c.reshape(-1, 1, 2)], 255)
        near = cv2.dilate(filled, np.ones((2 * r_max + 1, 2 * r_max + 1), np.uint8))
        for x, y in pts:
            assert 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]
            assert obstacle[y, x] == 0, "drop point is flooded"
            assert near[y, x] > 0, "drop point too far from its region"
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                assert np.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]) >= 40 - 1e-9


def test_dry_drop_points_fully_flooded_map_gives_none():
    mask = np.full((100, 100), 255, np.uint8)
    contour = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], np.int32).reshape(-1, 1, 2)
    assert find_safe_drop_points([contour.reshape(-1, 2)], [0], [(50, 50)], obstacle_mask=mask,
                                 raw_contours=[contour]) == []


def test_connectivity_filter_bad_inputs():
    with pytest.raises(ValueError):
        filter_drops_by_home_connectivity([(1, 1)], (0, 0), None)
    with pytest.raises(ValueError):
        filter_drops_by_home_connectivity([(1, 1)], (0, 0), np.zeros((10, 10)), downsample_factor=0)
    assert filter_drops_by_home_connectivity([], (0, 0), np.zeros((10, 10))) == ([], [])
    flooded = np.full((50, 50), 255, np.uint8)
    assert filter_drops_by_home_connectivity([(10, 10)], (5, 5), flooded) == ([], [0])


# ------------------------------------------------------------------------------------------------------- TSP
@H
@given(pts=st.lists(st.tuples(st.integers(0, 100), st.integers(0, 100)), max_size=40),
       home=st.one_of(st.none(), st.tuples(st.integers(0, 100), st.integers(0, 100))))
def test_tsp_is_permutation_with_home_ends(pts, home):
    order, ordered = nearest_neighbor_tsp(pts, home=home)
    assert sorted(order) == list(range(len(pts)))
    if pts:
        inner = ordered[1:-1] if home is not None else ordered
        assert inner == [pts[i] for i in order]
        if home is not None:
            assert ordered[0] == home and ordered[-1] == home


# --------------------------------------------------------------------------------------------------- D* Lite
def dijkstra(rows, cols, obstacles, s, g):
    dist = {s: 0.0}
    heap = [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == g:
            return d
        if d > dist[u]:
            continue
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                v = (u[0] + dx, u[1] + dy)
                if 0 <= v[0] < cols and 0 <= v[1] < rows and v not in obstacles:
                    nd = d + math.hypot(dx, dy)
                    if nd < dist.get(v, math.inf):
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
    return math.inf


@H_SLOW
@given(rows=st.integers(1, 25), cols=st.integers(1, 25), density=st.floats(0, 0.6), seed=st.integers(0, 10_000))
def test_dstar_matches_dijkstra_and_paths_are_valid(rows, cols, density, seed):
    rng = np.random.default_rng(seed)
    free = rng.random((rows, cols)) >= density
    obstacles = {(x, y) for y in range(rows) for x in range(cols) if not free[y, x]}
    s = (int(rng.integers(cols)), int(rng.integers(rows)))
    g = (int(rng.integers(cols)), int(rng.integers(rows)))
    obstacles -= {s, g}
    path = DStarLite((rows, cols), s, g).plan_path(obstacles)
    ref = dijkstra(rows, cols, obstacles, s, g)
    if math.isinf(ref):
        assert path == []
        return
    assert path and path[0] == s and path[-1] == g
    length = 0.0
    for a, b in zip(path, path[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 and b not in obstacles
        length += math.hypot(a[0] - b[0], a[1] - b[1])
    assert abs(length - ref) < 1e-9


def test_dstar_blocked_start_or_goal_and_tiny_grids():
    assert DStarLite((5, 5), (0, 0), (4, 4)).plan_path({(0, 0)}) == []
    assert DStarLite((5, 5), (0, 0), (4, 4)).plan_path({(4, 4)}) == []
    assert DStarLite((1, 1), (0, 0), (0, 0)).plan_path(set()) == [(0, 0)]
    assert DStarLite((1, 2), (0, 0), (1, 0)).plan_path(set()) == [(0, 0), (1, 0)]


# ------------------------------------------------------------------------------------------ legacy planner
@H_SLOW
@given(seed=st.integers(0, 10_000), n=st.integers(0, 8))
def test_legacy_full_path_stop_mapping(seed, n):
    rng = np.random.default_rng(seed)
    mask = blob_mask(seed, (120, 160), 5)
    pts = [(int(rng.integers(0, 160)), int(rng.integers(0, 120))) for _ in range(n)]
    full, drops = compute_full_path(pts, mask)
    assert len(drops) == len(pts)
    assert [full[i] for i in drops] == pts
    assert drops == sorted(drops)


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (4, 200), (200, 4)])
def test_legacy_full_path_tiny_masks(shape):
    mask = np.zeros(shape, np.uint8)
    pts = [(0, 0), (shape[1] - 1, shape[0] - 1), (0, 0)]
    full, drops = compute_full_path(pts, mask)
    assert [full[i] for i in drops] == pts


# ------------------------------------------------------------------------------------------ routed planner
@H_SLOW
@given(seed=st.integers(0, 10_000), n=st.integers(1, 6), mpp=st.floats(0.5, 10))
def test_routed_path_invariants(seed, n, mpp):
    rng = np.random.default_rng(seed)
    mask = blob_mask(seed, (120, 160), 6)
    dry = np.argwhere(mask == 0)
    pts = [tuple(int(v) for v in dry[rng.integers(len(dry))][::-1]) for _ in range(n + 1)]
    report = []
    full, stops, legs = compute_routed_path(pts, mask, meters_per_pixel=mpp, leg_report=report)
    assert [full[i] for i in stops] == pts
    assert len(legs) == len(pts)  # one entry per stop; the last is 0
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(full, full[1:])) * mpp
    assert abs(sum(legs) - total) < 1e-6 * max(1.0, total)
    for leg in report:
        if leg["status"] == "dstar":
            assert leg["route_flooded_px"] == 0, leg
        assert leg["status"] in ("dstar", "same_cell", "fallback")


def test_routed_path_degenerate_inputs():
    m = np.zeros((50, 50), np.uint8)
    assert compute_routed_path([], m) == ([], [], [])
    assert compute_routed_path([(1, 1)], m) == ([(1, 1)], [0], [])
    full, stops, legs = compute_routed_path([(5, 5), (5, 5), (5, 5)], m, meters_per_pixel=2.0)
    assert [full[i] for i in stops] == [(5, 5)] * 3 and sum(legs) == 0
    full, stops, legs = compute_routed_path([(-10, -10), (60, 60)], m, meters_per_pixel=1.0)
    assert full[0] == (-10, -10) and full[-1] == (60, 60)
    flooded = np.full((50, 50), 255, np.uint8)
    report = []
    compute_routed_path([(5, 5), (40, 40)], flooded, leg_report=report)
    assert report[0]["status"] == "fallback"


# ------------------------------------------------------------------------------------ routing-grid helpers
@H_SLOW
@given(seed=st.integers(0, 10_000))
def test_distance_field_finite_iff_same_component(seed):
    mask = blob_mask(seed, (100, 120), 7)
    grid = build_routing_grid(mask, 5)
    free = np.argwhere(grid.occupancy == 0)
    if len(free) == 0:
        return
    sy, sx = free[0]
    field = grid_distance_field(grid, (int(sx), int(sy)))
    same = grid.labels == grid.labels[sy, sx]
    assert np.all(np.isfinite(field) == (same & (grid.occupancy == 0)))


def brute_intersects(a, b):
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return (v > 0) - (v < 0)

    def on_seg(p, q, r):
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    for p1, p2 in zip(a, a[1:]):
        if p1 == p2:
            continue
        for q1, q2 in zip(b, b[1:]):
            if q1 == q2:
                continue
            o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
            if o1 != o2 and o3 != o4:
                return True
            if (o1 == 0 and on_seg(p1, q1, p2)) or (o2 == 0 and on_seg(p1, q2, p2)) or \
                    (o3 == 0 and on_seg(q1, p1, q2)) or (o4 == 0 and on_seg(q1, p2, q2)):
                return True
    return False


poly = st.lists(st.tuples(st.integers(0, 20), st.integers(0, 20)), min_size=0, max_size=7)


@H
@given(a=poly, b=poly)
def test_polyline_intersections_match_brute_force(a, b):
    got = bool(polyline_intersections(a, b))
    assert got == brute_intersects(a, b)
    assert got == bool(polyline_intersections(b, a))
    assert bool(polyline_intersections(a, b, return_segments=True)) == got


def test_polyline_flood_pixels_edges():
    m = np.zeros((10, 10), np.uint8)
    assert polyline_flood_pixels([], m) == 0 and polyline_flood_pixels([(1, 1)], m) == 0
    m[5, :] = 255
    assert polyline_flood_pixels([(0, 0), (0, 9)], m) == 1
    assert polyline_flood_pixels([(-50, -50), (100, 100)], m) >= 1


def test_routed_distance_function_symmetry_and_unreachable():
    mask = np.zeros((100, 100), np.uint8)
    mask[:, 48:53] = 255
    grid = build_routing_grid(mask, 5)
    d = routed_distance_function(grid, [(10, 10), (90, 90)], 2.0)
    assert d((20, 20), (10, 10)) == d((10, 10), (20, 20))
    assert math.isinf(d((10, 10), (90, 90)))


# --------------------------------------------------------------------------------------------- flood model
@pytest.mark.parametrize("weather", [{}, {"precipitation": -5, "wind_speed_10m": -3, "wind_direction_10m": 0},
                                     {"precipitation": 1e6, "wind_speed_10m": 0, "wind_direction_10m": 0},
                                     {"precipitation": 0, "wind_speed_10m": 50, "wind_direction_10m": 720}])
def test_predict_spread_weather_extremes(weather):
    m = np.zeros((40, 40), np.uint8)
    m[18:22, 18:22] = 255
    out = predict_spread(m, weather, 2.0)
    assert out.shape == m.shape and out.dtype == np.uint8 and np.all(out[m > 0] > 0)


@pytest.mark.xfail(strict=True, reason="QA-14 (latent): predict_spread itself does not validate weather; main sanitizes it first (QA-02)")
@pytest.mark.parametrize("value", [float("nan"), None, "3"])
def test_predict_spread_non_numeric_weather_fails_cleanly(value):
    """A non-numeric weather value must not crash with an obscure error deep in the model."""
    m = np.zeros((40, 40), np.uint8)
    m[18:22, 18:22] = 255
    try:
        predict_spread(m, {"precipitation": value, "wind_speed_10m": 0, "wind_direction_10m": 0}, 2.0)
    except (ValueError, TypeError) as e:
        pytest.fail(f"predict_spread raised {type(e).__name__}: {e}")


@pytest.mark.xfail(strict=True, reason="QA-14 (latent): predict_spread itself does not validate the horizon; main rejects it (QA-01)")
@pytest.mark.parametrize("horizon", [float("nan"), float("inf")])
def test_predict_spread_non_finite_horizon(horizon):
    m = np.zeros((20, 20), np.uint8)
    try:
        predict_spread(m, {"precipitation": 0, "wind_speed_10m": 0, "wind_direction_10m": 0}, horizon)
    except (ValueError, OverflowError) as e:
        pytest.fail(f"non-finite horizon raised {type(e).__name__}: {e}")


@H_SLOW
@given(seed=st.integers(0, 1000), horizon=st.floats(0.01, 6), p=st.floats(0, 30), v=st.floats(0, 60),
       d=st.floats(0, 360))
def test_forecast_frames_monotone_and_last_equals_prediction(seed, horizon, p, v, d):
    m = blob_mask(seed, (60, 80), 4)
    w = {"precipitation": p, "wind_speed_10m": v, "wind_direction_10m": d}
    frames = forecast_frames(m, w, horizon)
    times = sorted(frames)
    assert times[0] == 0.0 and times[-1] <= max(horizon, 1.0)
    for a, b in zip(times, times[1:]):
        assert not np.any((frames[a] > 0) & (frames[b] == 0))
    assert np.array_equal(frames[times[-1]], (predict_spread(m, w, horizon) > 0).astype(np.uint8) * 255)
    areas = [np.count_nonzero(obstacle_mask_at(frames, t)) for t in np.linspace(-1, horizon + 1, 9)]
    assert areas == sorted(areas)


def test_forecast_query_edge_times():
    m = np.zeros((10, 10), np.uint8)
    frames = {0.0: m, 1.0: np.full_like(m, 255)}
    assert not obstacle_mask_at(frames, -5).any()
    assert obstacle_mask_at(frames, 1e9).all()
    with pytest.raises(ValueError):
        forecast_at({}, 1.0, None)
    assert forecast_at({}, 1.0, m)[0].shape == m.shape


# ---------------------------------------------------------------------------------------------- battery
def test_battery_fraction_edges():
    assert leg_battery_fraction(0) == 0
    assert leg_battery_fraction(-100) == 0
    assert leg_battery_fraction(1000, -5, -5) == leg_battery_fraction(1000)
    assert math.isinf(leg_battery_fraction(math.inf))
    assert leg_battery_fraction(2100) == pytest.approx(1.0)  # 7 min at 5 m/s


@pytest.mark.xfail(strict=True, reason="QA-12 (latent): NaN distances cannot reach the planner (scale validated, distances finite or inf)")
def test_battery_fraction_nan_distance_is_not_free():
    """A NaN distance must not be treated as a free (zero-cost) leg."""
    v = leg_battery_fraction(float("nan"))
    assert not (v == 0.0), "NaN distance consumed no battery"


# ------------------------------------------------------------------------------------ sortie planning
@H_SLOW
@given(seed=st.integers(0, 10_000), n=st.integers(0, 40), mpp=st.floats(0.5, 12), wind=st.floats(0, 12),
       cap=st.sampled_from([0.25, 0.5, 1.0, 8.0]), reserve=st.sampled_from([0.0, 0.2, 0.5]))
def test_plan_mission_stops_invariants(seed, n, mpp, wind, cap, reserve):
    rng = np.random.default_rng(seed)
    home = (200, 200)
    drops = list({(int(rng.integers(0, 400)), int(rng.integers(0, 400))) for _ in range(n)} - {home})
    cur = np.zeros((400, 400), np.uint8)
    route = plan_mission_stops(home, drops, {0.0: cur}, cur, meters_per_pixel=mpp, headwind_mps=wind,
                               capacity=cap, per_drop_kg=0.25, reserve=reserve, initial_battery=1.0)
    pts = route.points
    assert pts[0] == home and pts[-1] == home
    visited = [pts[i] for i in route.drop_indices]
    unreachable = [drops[i] for i in route.unreachable]
    assert sorted(visited + unreachable) == sorted(drops), "drops lost or duplicated"
    assert len(set(visited)) == len(visited)
    assert all(pts[i] == home for i in route.reload_indices)
    # per sortie: payload within capacity, battery within the usable share
    report = sortie_battery_report(pts, route.drop_indices, route.reload_indices, meters_per_pixel=mpp,
                                   per_drop_kg=0.25, headwind_mps=wind)
    for s in report:
        assert s["drops"] * 0.25 <= cap + 1e-9
        assert s["battery_fraction"] <= 1 - reserve + 1e-9
    assert [round(s["battery_fraction"], 9) for s in report] == [round(b, 9) for b in route.sortie_battery]
    for i in route.unreachable:
        d = math.hypot(drops[i][0] - home[0], drops[i][1] - home[1]) * mpp
        assert leg_battery_fraction(d, 0.25, wind) + leg_battery_fraction(d, 0, wind) > 1 - reserve - 1e-12


@pytest.mark.xfail(strict=True, reason="QA-13 (latent): payload constants are not user-configurable")
def test_plan_mission_stops_capacity_below_one_drop():
    """Payload capacity smaller than one package: must not crash with AssertionError."""
    cur = np.zeros((50, 50), np.uint8)
    try:
        r = plan_mission_stops((0, 0), [(10, 10)], {0.0: cur}, cur, capacity=0.1, per_drop_kg=0.25)
    except AssertionError as e:
        pytest.fail(f"AssertionError: {e}")
    assert r.drop_indices == []


def test_plan_mission_stops_flooded_and_outside_drops_unreachable():
    cur = np.zeros((50, 50), np.uint8)
    cur[10, 10] = 255
    r = plan_mission_stops((0, 0), [(10, 10), (500, 500), (20, 20)], {0.0: cur}, cur, meters_per_pixel=1.0)
    assert sorted(r.unreachable) == [0, 1] and len(r.drop_indices) == 1


# ------------------------------------------------------------------------------------------ base selection
def straight(mpp):
    return lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1])) * mpp


@H_SLOW
@given(seed=st.integers(0, 10_000), nd=st.integers(0, 40), nc=st.integers(0, 40), mpp=st.floats(1, 20),
       k=st.integers(1, 6), sep=st.sampled_from([0.0, 50.0, 150.0]))
def test_select_bases_invariants(seed, nd, nc, mpp, k, sep):
    rng = np.random.default_rng(seed)
    drops = [(int(rng.integers(0, 800)), int(rng.integers(0, 800))) for _ in range(nd)]
    cands = [(int(rng.integers(0, 800)), int(rng.integers(0, 800))) for _ in range(nc)]
    dist = straight(mpp)
    res = select_bases(cands, drops, dist, np.zeros((800, 800), np.uint8), headwind_mps=0.0, wind_from_deg=None,
                       per_drop_kg=0.25, payload_capacity_kg=8.0, reserve=0.2, meters_per_pixel=mpp, max_bases=k,
                       min_separation_px=sep)
    bases, assign, unreach = res["bases"], res["assignment"], res["unreachable"]
    assert len(bases) <= k
    assert all(tuple(b) in set(map(tuple, cands)) for b in bases)
    assert set(assign) | set(unreach) == set(range(nd)) and not (set(assign) & set(unreach))
    if len(bases) > 1:
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                assert np.hypot(bases[i][0] - bases[j][0], bases[i][1] - bases[j][1]) >= sep - 1e-9
    reach = single_drop_reach(bases, drops, dist, headwind_mps=0.0, per_drop_kg=0.25, reserve=0.2)
    for d, b in assign.items():
        assert 0 <= b < len(bases) and d in reach[b]
        assert dist(bases[b], drops[d]) <= min(dist(bases[o], drops[d]) for o in range(len(bases)) if d in reach[o]) + 1e-9
    assert all(any(v == b for v in assign.values()) for b in range(len(bases))), "base without drops"
    all_reach = single_drop_reach(cands, drops, dist, headwind_mps=0.0, per_drop_kg=0.25, reserve=0.2)
    coverable = set().union(*all_reach) if all_reach else set()
    for d in unreach:
        if d not in coverable:
            assert "no flood-clear base site" in unreach[d]


def test_select_bases_duplicate_candidates_with_zero_separation():
    drops = [(0, 0), (1000, 0)]
    cands = [(500, 0), (500, 0), (0, 10), (1000, 10)]
    res = select_bases(cands, drops, straight(2.0), np.zeros((10, 10), np.uint8), headwind_mps=0.0,
                       wind_from_deg=None, per_drop_kg=0.25, payload_capacity_kg=8.0, reserve=0.2,
                       meters_per_pixel=2.0, max_bases=4, min_separation_px=0.0)
    assert len(set(map(tuple, res["bases"]))) == len(res["bases"]), "two bases at the same spot"


# -------------------------------------------------------------------------------------------- export
@H_SLOW
@given(seed=st.integers(0, 10_000), n=st.integers(1, 30), lat=st.floats(-60, 60), lon=st.floats(-179, 179),
       mpp=st.floats(0.1, 20))
def test_export_roundtrip_with_independent_validator(seed, n, lat, lon, mpp):
    rng = np.random.default_rng(seed)
    home = (100, 100)
    pts = [home] + [(int(rng.integers(0, 200)), int(rng.integers(0, 200))) for _ in range(n)] + [home]
    drops = sorted(set(int(i) for i in rng.integers(1, len(pts) - 1, size=max(1, n // 2))))
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "m.waypoints")
        generate_mission_file(pts, drops, home, (99.5, 99.5), (lat, lon), mpp, filename=f)
        assert validate_waypoints_file(f) == []
        side = {"full_path": pts, "drop_indices": drops, "home": list(home), "image_center_px": [99.5, 99.5],
                "geo_center": [lat, lon], "meters_per_pixel": mpp}
        findings, _ = wpl_validator.validate(f, side, servo_channel=9, servo_pwm=2000)
        assert [x for x in findings if x.level == "ERROR"] == []


def test_export_empty_path_is_still_structurally_valid():
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "m.waypoints")
        generate_mission_file([], [], (5, 5), (5, 5), (25, 82), 2.0, filename=f)
        assert validate_waypoints_file(f) == []


def test_runtime_validator_rejects_garbage():
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "m.waypoints")
        for content in ["", "QGC WPL 110\n", "garbage\n", "QGC WPL 110\n0\t1\t0\t16\tx\t0\t0\t0\t1\t1\t0\t1\n",
                        "QGC WPL 110\n0\t1\t0\t16\t0\t0\t0\t0\t95\t0\t0\t1\n"]:
            with open(f, "w") as fh:
                fh.write(content)
            assert validate_waypoints_file(f), repr(content)
