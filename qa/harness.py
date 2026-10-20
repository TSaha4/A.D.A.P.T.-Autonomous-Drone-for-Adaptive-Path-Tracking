"""Shared helpers: run main.main() in-process on a synthetic or given image, and check multi-run invariants."""
import contextlib
import glob
import json
import os
import sys
from unittest import mock

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import main as M  # noqa: E402
from config import settings  # noqa: E402
from src.routing.pathfinding import polyline_intersections  # noqa: E402
from src.vision.image_processing import ImageProcessor  # noqa: E402
from tests import wpl_validator  # noqa: E402

CALM = {"precipitation": 0.0, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0, "status": "success"}


def make_map(path, shapes, size=(750, 750), bg=(255, 255, 255)):
    """shapes: list of ('rect', x0, y0, x1, y1) | ('circle', cx, cy, r) | ('line', x0, y0, x1, y1, t)."""
    w, h = size
    img = np.full((h, w, 3), bg, np.uint8)
    for s in shapes:
        if s[0] == "rect":
            cv2.rectangle(img, (s[1], s[2]), (s[3], s[4]), (0, 0, 255), -1)
        elif s[0] == "circle":
            cv2.circle(img, (s[1], s[2]), s[3], (0, 0, 255), -1)
        elif s[0] == "line":
            cv2.line(img, (s[1], s[2]), (s[3], s[4]), (0, 0, 255), s[5])
    cv2.imwrite(path, img)
    return img


def fake_select(self, image):
    self.image = image.copy()
    self.hsv_lower = np.array([0, 100, 100], np.uint8)
    self.hsv_upper = np.array([179, 255, 255], np.uint8)
    self.is_red_wrap = True
    return self.hsv_lower, self.hsv_upper


def run_main(args, cwd, mpp=4.0, weather=CALM, extra=(), settings_patch=None):
    """Runs main.main() with argv args inside cwd. Returns (exit_code, exception_or_None)."""
    argv = ["main.py"] + list(args)
    if "--lat" not in args:
        argv += ["--lat", "25.0", "--lon", "82.0"]
    with contextlib.ExitStack() as st:
        st.enter_context(contextlib.chdir(cwd))
        st.enter_context(mock.patch.object(sys, "argv", argv))
        st.enter_context(mock.patch.object(settings, "METERS_PER_PIXEL", mpp))
        for k, v in (settings_patch or {}).items():
            st.enter_context(mock.patch.object(settings, k, v))
        st.enter_context(mock.patch.object(ImageProcessor, "select_sample_points", fake_select))
        st.enter_context(mock.patch.object(M, "get_weather_data", return_value=weather))
        st.enter_context(mock.patch.object(M, "show_image_safe"))
        st.enter_context(mock.patch.object(M, "display_path_on_map"))
        st.enter_context(mock.patch("src.mission.mission_output.plt.show"))
        st.enter_context(mock.patch("sys.stderr"))
        for p in extra:
            st.enter_context(p)
        try:
            M.main()
            return 0, None
        except SystemExit as e:
            return e.code, None
        except BaseException as e:  # noqa: BLE001 - recorded as a finding by the caller
            return "EXC", e


def latest_session(root):
    s = sorted(glob.glob(os.path.join(root, "session_*")))
    return s[-1] if s else None


def check_multi_session(session, image_bgr=None, clearance_px=15.0, battery_limit=0.8, capacity=8.0, per_drop=0.25):
    """Invariant checks for one multi-base run. Returns a list of violations (empty = all good)."""
    v = []
    summ = json.load(open(os.path.join(session, "run_summary.json")))
    if "bases" not in summ:
        missions = glob.glob(os.path.join(session, "base*.waypoints"))
        if missions:
            v.append(f"infeasible outcome {summ.get('outcome')} but missions written: {missions}")
        return v
    drops = [tuple(p) for p in summ["drop_points"]]
    n = len(drops)
    assigned = {int(k): b for k, b in summ["assignment"].items()}
    unreach = {int(k) for k in summ["unreachable_by_any_base"]}
    if set(assigned) & unreach:
        v.append("drop both assigned and unreachable")
    if set(assigned) | unreach != set(range(n)):
        v.append(f"drops neither assigned nor reported: {sorted(set(range(n)) - set(assigned) - unreach)}")
    # Coincident drop points of two adjacent regions are possible by design (one drop per region); they are
    # checked below only for being split across two missions.
    bases_in_summary = {b["base"]: b for b in summ["bases"]}
    planned_bases = set(bases_in_summary)
    not_planned = {int(k) for k in summ.get("not_planned", {})}
    lost = [d for d, b in assigned.items() if b not in planned_bases and d not in not_planned]
    if lost:
        v.append(f"assigned drops of bases without a plan are not reported anywhere: {lost}")
    if not_planned - {d for d, b in assigned.items() if b not in planned_bases}:
        v.append("not_planned lists drops of a base that has a plan")
    accounted = (summ["drops_visited"] + len(unreach) + summ["unreachable"] + summ["connectivity_excluded"]
                 + len(not_planned))
    if accounted != n:
        v.append(f"drop accounting: visited+unreachable+excluded+not_planned = {accounted} != {n} drop points")
    paths, mission_drops = [], {}
    for b, info in bases_in_summary.items():
        if info["assigned"] != info["visited"] + info["unreachable"] + info["connectivity_excluded"]:
            v.append(f"base {b}: assigned {info['assigned']} != visited+unreachable+excluded")
        mine = sorted(drops[d] for d, owner in assigned.items() if owner == b)
        if len(mine) != info["assigned"]:
            v.append(f"base {b}: summary assigned {info['assigned']} but assignment lists {len(mine)}")
        wp = os.path.join(session, f"base{b}.waypoints")
        if not info.get("waypoint_file"):
            v.append(f"base {b}: no mission file ({info.get('waypoint_problems')})")
            continue
        side = json.load(open(os.path.join(session, f"base{b}.route.json")))
        findings, items = wpl_validator.validate(wp, side, servo_channel=9, servo_pwm=2000)
        errs = [f for f in findings if f.level == "ERROR"]
        if errs:
            v.append(f"base {b}: validator errors {errs[:3]}")
        fp = [tuple(map(float, p)) for p in side["full_path"]]
        paths.append(fp)
        md = [tuple(int(round(c)) for c in fp[i]) for i in side["drop_indices"]]
        mission_drops[b] = md
        if len(md) != info["visited"]:
            v.append(f"base {b}: {len(md)} drops in mission, summary visited {info['visited']}")
        if not set(md) <= set(mine):
            v.append(f"base {b}: mission drops not assigned to it: {sorted(set(md) - set(mine))[:5]}")
        if sum(r["cmd"] == wpl_validator.DO_SET_SERVO for r in items) != len(md):
            v.append(f"base {b}: servo count mismatch")
        if tuple(map(int, side["home"])) != tuple(info["home"]):
            v.append(f"base {b}: sidecar home differs from summary")
        for s in info["routed_sortie_battery"]:
            if s > battery_limit + 1e-9:
                v.append(f"base {b}: sortie battery {s:.3f} > {battery_limit}")
        # payload per sortie
        bounds = sorted(side["reload_indices"]) + [len(fp) - 1]
        start = 0
        for end in bounds:
            k = sum(1 for i in side["drop_indices"] if start <= i <= end)
            if k * per_drop > capacity + 1e-9:
                v.append(f"base {b}: sortie with {k} drops exceeds payload capacity")
            start = end + 1
        if image_bgr is not None:
            hx, hy = info["home"]
            red = _red(image_bgr)
            if red[hy, hx]:
                v.append(f"base {b} on flood")
            dist = cv2.distanceTransform((red == 0).astype(np.uint8), cv2.DIST_L2, 5)
            if dist[hy, hx] < clearance_px - 1.5:
                v.append(f"base {b} clearance {dist[hy, hx]:.1f} < {clearance_px}")
    owners = {}
    for b, md in mission_drops.items():
        for d in set(md):
            owners.setdefault(d, set()).add(b)
    shared = [d for d, bs in owners.items() if len(bs) > 1]
    if shared:
        v.append(f"a drop point is served by two missions: {shared[:3]}")
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if polyline_intersections(paths[i], paths[j]):
                v.append(f"routes {i} and {j} cross")
    if image_bgr is not None:
        red = _red(image_bgr)
        for x, y in drops:
            if 0 <= y < red.shape[0] and 0 <= x < red.shape[1] and red[y, x]:
                v.append(f"drop {x, y} on flood")
    if summ["legs"]["fallback"] == 0 and summ["route_flooded_px_current"] != 0:
        v.append("route crosses current flood without a fallback leg")
    visited = sum(b["visited"] for b in summ["bases"])
    if visited != summ["drops_visited"]:
        v.append("drops_visited inconsistent")
    return v


def _red(img):
    """The pipeline's own flood mask for these synthetic maps (thresholds of fake_select + morphology), so that
    checks are made against what the planner actually treats as flood, not raw specks that opening removes."""
    p = ImageProcessor()
    fake_select(p, img)
    with mock.patch("builtins.print"):
        return p.mask_flood_areas(img) > 0
