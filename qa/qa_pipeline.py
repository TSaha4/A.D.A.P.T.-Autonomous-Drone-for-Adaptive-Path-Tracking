"""Adversarial end-to-end tests of main.main() (QA suite; see qa/README.md and docs/QA_REPORT.md).

Acceptable outcomes: a valid mission, or a graceful exit (status 0/1/2 with a log message). Not acceptable:
an uncaught exception ("EXC"), an invalid mission file, or a stale/partial file.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import pytest

import harness as Hn
from tests import wpl_validator

PY = sys.executable
REPO = Hn.REPO
SINGLE_OUT = os.path.join("data", "output", "enriched_drone_mission.waypoints")


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory() as d:
        yield d


def single_ok(tmp, n_regions=None):
    out = os.path.join(tmp, SINGLE_OUT)
    assert os.path.exists(out), "no single-UAV mission"
    findings, items = wpl_validator.validate(out, servo_channel=9, servo_pwm=2000)
    assert [f for f in findings if f.level == "ERROR"] == []
    if n_regions is not None:
        assert sum(r["cmd"] == wpl_validator.DO_SET_SERVO for r in items) == n_regions
    return items


def multi(tmp, shapes, k=None, mpp=4.0, size=(750, 750), weather=Hn.CALM, extra=(), settings_patch=None, bg=(255, 255, 255)):
    img_path = os.path.join(tmp, "flood.png")
    img = Hn.make_map(img_path, shapes, size, bg)
    args = [img_path, "--mode", "multi", "--session-root", os.path.join(tmp, "s")]
    if k is not None:
        args += ["--max-bases", str(k)]
    code, exc = Hn.run_main(args, tmp, mpp=mpp, weather=weather, extra=extra, settings_patch=settings_patch)
    session = Hn.latest_session(os.path.join(tmp, "s"))
    return code, exc, session, img


def single(tmp, shapes, mpp=4.0, size=(750, 750), weather=Hn.CALM, args=(), bg=(255, 255, 255)):
    img_path = os.path.join(tmp, "flood.png")
    Hn.make_map(img_path, shapes, size, bg)
    return Hn.run_main([img_path] + list(args), tmp, mpp=mpp, weather=weather)


def assert_multi_valid(code, exc, session, img):
    assert exc is None, f"uncaught {type(exc).__name__}: {exc}"
    if code == 0 and session:
        v = Hn.check_multi_session(session, img)
        assert v == [], v
    if code == 2 and session:
        assert not glob.glob(os.path.join(session, "*.waypoints")), "infeasible run wrote missions"


# ------------------------------------------------------------------------------------------- CLI / files
@pytest.mark.parametrize("mode", ["single", "multi"])
@pytest.mark.parametrize("kind", ["missing", "empty", "corrupt", "directory", "text"])
def test_bad_image_files_fail_gracefully(tmp, mode, kind):
    p = os.path.join(tmp, "x.png")
    if kind == "empty":
        open(p, "wb").close()
    elif kind == "corrupt":
        open(p, "wb").write(b"\x89PNG\r\n\x1a\n" + os.urandom(500))
    elif kind == "directory":
        os.mkdir(p)
    elif kind == "text":
        open(p, "w").write("not an image")
    code, exc = Hn.run_main([p, "--mode", mode], tmp)
    assert exc is None and code == 1


@pytest.mark.parametrize("horizon", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("mode", ["single", "multi"])
def test_non_finite_horizon(tmp, horizon, mode):
    code, exc = single(tmp, [("rect", 100, 100, 150, 150)], args=["--horizon", horizon, "--mode", mode])
    assert exc is None, f"uncaught {type(exc).__name__}: {exc}"


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_negative_and_zero_horizon(tmp, mode):
    for h in ("0", "-3"):
        code, exc = single(tmp, [("rect", 100, 100, 150, 150)], args=["--horizon", h, "--mode", mode])
        assert exc is None and code == 0


@pytest.mark.parametrize("weather", [
    {"precipitation": None, "wind_speed_10m": 5.0, "wind_direction_10m": 90.0, "status": "success"},
    {"precipitation": 1.0, "wind_speed_10m": None, "wind_direction_10m": 90.0, "status": "success"},
    {"precipitation": 1.0, "wind_speed_10m": 5.0, "wind_direction_10m": None, "status": "success"},
    {"precipitation": float("nan"), "wind_speed_10m": 5.0, "wind_direction_10m": 90.0, "status": "success"},
    {"status": "success"},
])
@pytest.mark.parametrize("mode", ["single", "multi"])
def test_incomplete_weather_from_api(tmp, weather, mode):
    """Open-Meteo can return null for a field; the run must not crash with a TypeError."""
    code, exc = single(tmp, [("rect", 100, 100, 150, 150)], weather=weather, args=["--mode", mode])
    assert exc is None, f"uncaught {type(exc).__name__}: {exc}"


def test_extreme_precipitation_from_api(tmp):
    w = {"precipitation": 1e6, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0, "status": "success"}
    code, exc = single(tmp, [("rect", 100, 100, 150, 150)], weather=w)
    assert exc is None, f"uncaught {type(exc).__name__}: {exc}"


@pytest.mark.parametrize("content", ["{}", "{not json", '{"precipitation": 5}', "[]"])
def test_dummy_weather_file_variants(tmp, content):
    os.makedirs(os.path.join(tmp, "data", "input"))
    open(os.path.join(tmp, "data", "input", "dummy_weather.json"), "w").write(content)
    code, exc = single(tmp, [("rect", 100, 100, 150, 150)], args=["--dummy-weather"])
    assert exc is None, f"uncaught {type(exc).__name__}: {exc}"


@pytest.mark.parametrize("var,value", [("METERS_PER_PIXEL", "abc"), ("MAX_BASES", "four"), ("MAX_BASES", "2.5"),
                                       ("HOME_CLEARANCE_PX", "x")])
def test_invalid_environment_variables(var, value):
    env = dict(os.environ, **{var: value})
    r = subprocess.run([PY, "main.py", "--help"], cwd=REPO, env=env, capture_output=True, text=True)
    assert "Traceback" not in r.stderr, r.stderr[-400:]


def test_session_root_is_a_file(tmp):
    blocker = os.path.join(tmp, "blocker")
    open(blocker, "w").close()
    img = os.path.join(tmp, "f.png")
    Hn.make_map(img, [("rect", 100, 100, 150, 150)])
    code, exc = Hn.run_main([img, "--mode", "multi", "--session-root", blocker], tmp)
    assert exc is None, f"uncaught {type(exc).__name__}: {exc}"


def test_unicode_image_path(tmp):
    d = os.path.join(tmp, "बाढ़_flood")
    os.mkdir(d)
    img = os.path.join(d, "map.png")
    Hn.make_map(os.path.join(tmp, "plain.png"), [("rect", 100, 100, 150, 150)])
    os.replace(os.path.join(tmp, "plain.png"), img)
    code, exc = Hn.run_main([img], tmp)
    assert exc is None
    assert code == 0, "a valid image under a non-ASCII path could not be read"


# --------------------------------------------------------------------------------------- map geometry
GEOMETRIES = {
    "no_flood": [],
    "all_flooded": [("rect", 0, 0, 749, 749)],
    "tiny_region": [("rect", 300, 300, 314, 314)],
    "huge_region": [("rect", 50, 50, 700, 700)],
    "many_disconnected": [("rect", 20 + 70 * i, 20 + 70 * j, 50 + 70 * i, 50 + 70 * j) for i in range(10) for j in range(10)],
    "thin_lines": [("line", 0, 100 + 60 * i, 749, 120 + 60 * i, 3) for i in range(10)],
    "touching_borders": [("rect", 0, 0, 60, 60), ("rect", 690, 690, 749, 749), ("rect", 0, 690, 60, 749), ("rect", 330, 0, 400, 30)],
    "ring_enclosure": [("rect", 250, 250, 500, 262), ("rect", 250, 488, 500, 500), ("rect", 250, 250, 262, 500),
                       ("rect", 488, 250, 500, 500), ("rect", 360, 360, 390, 390)],
    "comb_corridors": [("rect", 100 + 40 * i, 0, 110 + 40 * i, 700) for i in range(14)],
}


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_geometry_single_mode(tmp, name):
    code, exc = single(tmp, GEOMETRIES[name])
    assert exc is None, f"uncaught {type(exc).__name__}: {exc}"
    if code == 0 and os.path.exists(os.path.join(tmp, SINGLE_OUT)):
        single_ok(tmp)


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
@pytest.mark.parametrize("k", [1, 4])
def test_geometry_multi_mode(tmp, name, k):
    code, exc, session, img = multi(tmp, GEOMETRIES[name], k=k)
    assert code in (0, 2), (code, exc)
    assert_multi_valid(code, exc, session, img)


def noisy_map(seed):
    rng = np.random.default_rng(seed)
    shapes = [("circle", int(rng.integers(0, 750)), int(rng.integers(0, 750)), int(rng.integers(1, 6)))
              for _ in range(600)]
    shapes += [("circle", int(rng.integers(0, 750)), int(rng.integers(0, 750)), int(rng.integers(15, 45)))
               for _ in range(6)]
    return shapes


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_noisy_segmentation(tmp, mode):
    if mode == "single":
        code, exc = single(tmp, noisy_map(3))
        assert exc is None and code == 0
        single_ok(tmp)
    else:
        code, exc, session, img = multi(tmp, noisy_map(3))
        assert_multi_valid(code, exc, session, img)


@pytest.mark.parametrize("size", [(1, 1), (2, 2), (5, 5), (12, 9), (40, 3), (3, 40)])
@pytest.mark.parametrize("mode", ["single", "multi"])
def test_tiny_images(tmp, size, mode):
    shapes = [("rect", 0, 0, max(0, size[0] // 2), max(0, size[1] // 2))]
    img_path = os.path.join(tmp, "f.png")
    Hn.make_map(img_path, shapes, size)
    code, exc = Hn.run_main([img_path, "--mode", mode, "--min-area", "0"], tmp)
    assert exc is None, f"uncaught {type(exc).__name__}: {exc}"


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_grayscale_and_alpha_images(tmp, mode):
    img = np.full((300, 300, 4), 255, np.uint8)
    img[100:150, 100:150] = (0, 0, 255, 255)
    p = os.path.join(tmp, "rgba.png")
    cv2.imwrite(p, img)
    code, exc = Hn.run_main([p, "--mode", mode], tmp)
    assert exc is None and code == 0
    g = np.full((300, 300), 255, np.uint8)
    p2 = os.path.join(tmp, "gray.png")
    cv2.imwrite(p2, g)
    code, exc = Hn.run_main([p2, "--mode", mode], tmp)
    assert exc is None


@pytest.mark.parametrize("width", ["1", "3", "100000"])
def test_display_width_extremes(tmp, width):
    code, exc = single(tmp, [("rect", 100, 100, 150, 150)], args=["--display-width", width])
    assert exc is None, f"uncaught {type(exc).__name__}: {exc}"


# ------------------------------------------------------------------------------------- single-UAV cases
def test_S1_one_region(tmp):
    code, exc = single(tmp, [("rect", 300, 300, 340, 340)])
    assert exc is None and code == 0
    single_ok(tmp, 1)


def test_S2_two_regions(tmp):
    code, exc = single(tmp, [("rect", 100, 100, 140, 140), ("rect", 600, 600, 640, 640)])
    assert exc is None and code == 0
    single_ok(tmp, 2)


def test_S4_large_mission(tmp):
    code, exc = single(tmp, GEOMETRIES["many_disconnected"])
    assert exc is None and code == 0
    single_ok(tmp, 100)


def test_S10_dynamic_weather(tmp):
    w = {"precipitation": 12.0, "wind_speed_10m": 35.0, "wind_direction_10m": 45.0, "status": "success"}
    code, exc = single(tmp, [("rect", 100, 100, 140, 140), ("rect", 400, 400, 460, 460)], weather=w, args=["--horizon", "4"])
    assert exc is None and code == 0
    single_ok(tmp, 2)


def test_S11_return_home_last(tmp):
    single(tmp, [("rect", 100, 100, 140, 140)])
    items = single_ok(tmp)
    assert [r["cmd"] for r in items[-2:]] == [20, 21]


def test_S12_empty_mission_leaves_no_file(tmp):
    code, exc = single(tmp, [])
    assert exc is None and code == 0
    assert not os.path.exists(os.path.join(tmp, SINGLE_OUT))


# -------------------------------------------------------------------------------------- multi-UAV cases
WEST = [("rect", 70, 330, 105, 365), ("rect", 120, 400, 150, 430), ("rect", 60, 440, 90, 470)]
EAST = [("rect", 620, 330, 655, 365), ("rect", 660, 400, 690, 430)]
SOUTH = [("rect", 350, 650, 385, 685), ("rect", 410, 640, 440, 670)]
NORTH = [("rect", 350, 40, 385, 75)]


def summary(session):
    return json.load(open(os.path.join(session, "run_summary.json")))


def test_M1_one_uav(tmp):
    code, exc, s, img = multi(tmp, WEST + EAST, k=1)
    assert_multi_valid(code, exc, s, img)
    assert len(summary(s)["bases"]) == 1


@pytest.mark.parametrize("case", ["two_clusters", "dominant", "uneven", "unreachable_region", "symmetric"])
def test_M2_two_uavs(tmp, case):
    shapes = {
        "two_clusters": WEST + EAST,
        "dominant": WEST + [("rect", 100 + 30 * i, 200, 115 + 30 * i, 215) for i in range(6)] + EAST[:1],
        "uneven": WEST + EAST[:1],
        "unreachable_region": WEST + EAST + [("rect", 600, 600, 749, 612), ("rect", 600, 600, 612, 749),
                                             ("rect", 690, 690, 710, 710)],
        "symmetric": [("rect", 100, 360, 130, 390), ("rect", 620, 360, 650, 390)],
    }[case]
    code, exc, s, img = multi(tmp, shapes, k=2)
    assert_multi_valid(code, exc, s, img)
    assert len(summary(s)["bases"]) <= 2


def test_M3_three_uavs(tmp):
    code, exc, s, img = multi(tmp, WEST + EAST + SOUTH, k=3)
    assert_multi_valid(code, exc, s, img)
    assert len(summary(s)["bases"]) == 3


def test_M4_M5_at_and_above_default_maximum(tmp):
    for k in (4, 5, 50):
        code, exc, s, img = multi(tmp, WEST + EAST + SOUTH + NORTH, k=k)
        assert_multi_valid(code, exc, s, img)
        assert len(summary(s)["bases"]) <= 4, "more bases than separated clusters need"


def test_M6_more_uavs_than_tasks(tmp):
    code, exc, s, img = multi(tmp, [("rect", 300, 300, 330, 330), ("rect", 360, 300, 390, 330)], k=10)
    assert_multi_valid(code, exc, s, img)
    sm = summary(s)
    assert len(sm["bases"]) == 1 and all(b["visited"] > 0 for b in sm["bases"])


def test_M7_many_waypoints_few_uavs(tmp):
    code, exc, s, img = multi(tmp, GEOMETRIES["many_disconnected"], k=3, mpp=2.0)
    assert_multi_valid(code, exc, s, img)
    assert len(summary(s)["drop_points"]) >= 100


def test_M10_no_feasible_base(tmp):
    code, exc, s, img = multi(tmp, WEST, k=4, mpp=200.0)
    assert exc is None and code == 2
    assert summary(s)["outcome"] == "NO_BASE_REACHES_ANY_DROP"
    assert not glob.glob(os.path.join(s, "*.waypoints"))


def test_M11_zero_separation_allows_no_colocated_bases(tmp):
    code, exc, s, img = multi(tmp, WEST + EAST + SOUTH, k=4, settings_patch={"MIN_BASE_SEPARATION_PX": 0.0})
    assert_multi_valid(code, exc, s, img)
    homes = [tuple(b["home"]) for b in summary(s)["bases"]]
    assert len(homes) == len(set(homes))


def test_M13_crossing_routes_are_repaired_or_rejected(tmp):
    # Two clusters interleaved diagonally: bases are likely to be placed so that sorties cross.
    shapes = [("rect", 100, 100, 130, 130), ("rect", 600, 600, 630, 630), ("rect", 600, 100, 630, 130),
              ("rect", 100, 600, 130, 630), ("rect", 350, 350, 380, 380)]
    code, exc, s, img = multi(tmp, shapes, k=4, mpp=3.0)
    assert_multi_valid(code, exc, s, img)


def test_M14_independent_routes_unchanged(tmp):
    code, exc, s, img = multi(tmp, WEST + EAST, k=4)
    assert_multi_valid(code, exc, s, img)
    assert summary(s)["overlap_repairs"] == []


# -------------------------------------------------------------------------------- determinism / state
def mission_hashes(session):
    return {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()
            for p in sorted(glob.glob(os.path.join(session, "*.waypoints")))}


def test_determinism_and_no_state_leak(tmp):
    shapes = WEST + EAST + SOUTH
    runs = []
    for i in range(3):
        sub = os.path.join(tmp, f"r{i}")
        os.mkdir(sub)
        code, exc, s, img = multi(sub, shapes, k=4)
        assert exc is None and code == 0
        runs.append(mission_hashes(s))
        if i == 1:  # an unrelated run in between must not change the next result
            other = os.path.join(tmp, "other")
            os.mkdir(other)
            multi(other, NORTH + SOUTH, k=1, mpp=7.0)
    assert runs[0] == runs[1] == runs[2]
    s1 = os.path.join(tmp, "single1")
    s2 = os.path.join(tmp, "single2")
    os.mkdir(s1), os.mkdir(s2)
    single(s1, shapes)
    single(s2, shapes)
    assert open(os.path.join(s1, SINGLE_OUT), "rb").read() == open(os.path.join(s2, SINGLE_OUT), "rb").read()


def test_single_and_multi_runs_do_not_touch_each_others_outputs(tmp):
    single(tmp, WEST)
    before = open(os.path.join(tmp, SINGLE_OUT), "rb").read()
    code, exc, s, img = multi(tmp, WEST + EAST)
    assert open(os.path.join(tmp, SINGLE_OUT), "rb").read() == before


@pytest.mark.xfail(strict=True, reason="QA-11 (latent): no production caller mutates the cached weather dict")
def test_api_weather_mutation_does_not_leak_through_cache():
    from unittest import mock
    from src.weather import weather_api
    weather_api.get_weather_data.cache_clear()
    with mock.patch("src.weather.weather_api.requests.get", side_effect=weather_api.requests.RequestException("x")):
        a = weather_api.get_weather_data(1.0, 2.0)
        a["precipitation"] = 999
        b = weather_api.get_weather_data(1.0, 2.0)
    weather_api.get_weather_data.cache_clear()
    assert b["precipitation"] == 0.0, "cached weather dict was mutated by a caller"
