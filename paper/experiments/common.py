"""
Shared evaluation harness for the A.D.A.P.T. paper.

Runs the UNMODIFIED pipeline entry point `main.main()` and records every stage. The only substitutions
are for things that cannot run unattended or reproducibly, and each is stated explicitly:

  1. Interactive colour sampling (operator clicks) -> a deterministic *simulated operator*: clicks at the
     centroids of the `n_clicks` largest connected components of saturated red pixels
     ((H <= 5 or H >= 175) and S > 180 and V > 180) in the display image. The HSV thresholds are then
     computed by the production `compute_dynamic_hsv` (optionally with non-default margins).
  2. GUI windows (cv2.imshow / plt.show) -> suppressed; the arrays are captured instead.
  3. Mission file path -> redirected into the experiment's output folder (production writer unchanged).
  4. Optional ablations, applied by wrapping (never editing) production functions:
       - predict=False : predict_spread returns the current mask (no predicted-flood obstacles);
       - hue_margin / sv_margin : arguments passed to the production compute_dynamic_hsv;
       - mpp : value of settings.METERS_PER_PIXEL.
Weather: `--dummy-weather` (data/input/dummy_weather.json) unless `weather` is given, so runs are reproducible.
"""
import json
import logging
import os
import sys
import time
from contextlib import contextmanager

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import main as M  # noqa: E402
import src.vision.image_processing as IP  # noqa: E402
import src.mission.mission_output as MO  # noqa: E402
from config import settings  # noqa: E402

MAPS = ["varanasi", "kanpur", "assam"]


class _Counter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


@contextmanager
def _patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def simulated_clicks(image, n_clicks=8):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] <= 5) | (hsv[..., 0] >= 175)) & (hsv[..., 1] > 180) & (hsv[..., 2] > 180)).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(red)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:n_clicks] + 1
    return [tuple(int(v) for v in cents[i]) for i in order]


def run(map_name=None, image_path=None, out_dir=None, tag=None, n_clicks=8, predict=True, hue_margin=20,
        sv_margin=30, mpp=None, display_width=750, weather=None, keep_arrays=False):
    """Run main.main() once; returns a dict of stage outputs, counts and timings."""
    image_path = image_path or os.path.join("data", "input", f"{map_name}.png")
    tag = tag or map_name
    out_dir = out_dir or os.path.join(RESULTS, "runs")
    os.makedirs(out_dir, exist_ok=True)
    rec = {"tag": tag, "image": image_path, "n_clicks": n_clicks, "predict": predict, "hue_margin": hue_margin,
           "sv_margin": sv_margin, "display_width": display_width}
    times, arrays = {}, {}

    def timed(name, fn):
        def wrapper(*a, **k):
            t = time.perf_counter()
            r = fn(*a, **k)
            times[name] = times.get(name, 0.0) + time.perf_counter() - t
            return r
        return wrapper

    orig_compute = IP.ImageProcessor.compute_dynamic_hsv
    orig_mask = IP.ImageProcessor.mask_flood_areas
    orig_contours = IP.ImageProcessor.find_filtered_contours

    def select(self, image):
        self.image = image.copy()
        self.sample_points = simulated_clicks(image, n_clicks)
        t = time.perf_counter()
        orig_compute(self, hue_margin=hue_margin, sv_margin=sv_margin)
        times["hsv_thresholds"] = time.perf_counter() - t
        rec.update(clicks=self.sample_points, hsv_lower=self.hsv_lower.tolist(), hsv_upper=self.hsv_upper.tolist(),
                   red_wrap=bool(self.is_red_wrap), display_shape=list(image.shape[:2]))
        arrays["display_image"] = image.copy()
        return self.hsv_lower, self.hsv_upper

    orig_predict = M.predict_spread

    def predict_fn(mask, weather_data, horizon):
        out = orig_predict(mask, weather_data, horizon) if predict else mask.copy()
        arrays["mask"], arrays["pred_mask"] = mask.copy(), out.copy()
        return out

    def capture(name, fn, keyer):
        def wrapper(*a, **k):
            t = time.perf_counter()
            r = fn(*a, **k)
            times[name] = times.get(name, 0.0) + time.perf_counter() - t
            keyer(r, a)
            return r
        return wrapper

    def on_clusters(r, a):
        rec["regions"] = len(r[0])
    def on_safe(r, a):
        rec["safe_points"] = [list(map(float, p)) for p in r]
    def on_tsp(r, a):
        rec["ordered_points"] = [list(map(float, p)) for p in r[1]]
    def on_path(r, a):
        rec["full_path"] = [list(map(float, p)) for p in r[0]]
        rec["drop_indices"] = list(map(int, r[1]))
        arrays["obstacle_mask"] = a[1].copy()

    mission_path = os.path.join(out_dir, f"{tag}.waypoints")

    def gen(full_path, drop_indices, home, image_center_px, geo_center, meters_per_pixel, filename=None, **kw):
        t = time.perf_counter()
        MO.generate_mission_file(full_path, drop_indices, home, image_center_px, geo_center, meters_per_pixel,
                                 filename=mission_path, **kw)
        times["mission_generation"] = time.perf_counter() - t
        rec.update(home=list(map(float, home)), image_center_px=list(map(float, image_center_px)),
                   geo_center=list(map(float, geo_center)), meters_per_pixel=float(meters_per_pixel),
                   mission=mission_path)

    counter = _Counter()
    root = logging.getLogger()
    root.addHandler(counter)
    argv = ["main.py", image_path, "--display-width", str(display_width)]
    if weather is None:
        argv.append("--dummy-weather")
    cwd = os.getcwd()
    os.chdir(REPO)
    t0 = time.perf_counter()
    try:
        with _patched(sys, "argv", argv), \
                _patched(settings, "METERS_PER_PIXEL", settings.METERS_PER_PIXEL if mpp is None else mpp), \
                _patched(IP.ImageProcessor, "select_sample_points", select), \
                _patched(IP.ImageProcessor, "mask_flood_areas", timed("mask_and_morphology", orig_mask)), \
                _patched(IP.ImageProcessor, "find_filtered_contours", timed("contours", orig_contours)), \
                _patched(M, "predict_spread", timed("flood_prediction", lambda m, w, h: predict_fn(m, w, h))), \
                _patched(M, "cluster_contours", capture("hull_centroids", M.cluster_contours, on_clusters)), \
                _patched(M, "find_safe_drop_points", capture("safe_drop_points", M.find_safe_drop_points, on_safe)), \
                _patched(M, "nearest_neighbor_tsp", capture("route_ordering", M.nearest_neighbor_tsp, on_tsp)), \
                _patched(M, "compute_full_path", capture("path_planning", M.compute_full_path, on_path)), \
                _patched(M, "generate_mission_file", gen), \
                _patched(M, "show_image_safe", lambda *a, **k: None), \
                _patched(M, "display_path_on_map", lambda *a, **k: None), \
                _patched(M, "get_weather_data", (lambda lat, lon: dict(weather)) if weather is not None else M.get_weather_data):
            M.main()
        rec["exit"] = 0
    except SystemExit as e:
        rec["exit"] = e.code
    finally:
        rec["total_s"] = time.perf_counter() - t0
        os.chdir(cwd)
        root.removeHandler(counter)

    msgs = counter.records
    rec["no_path_legs"] = sum("No path found" in m for _, m in msgs)
    rec["unsnappable_endpoints"] = sum("No free cell found" in m for _, m in msgs)
    rec["snapped_endpoints"] = sum(m.startswith("Snapped") for _, m in msgs)
    rec["planned_legs"] = sum("D* Lite planning from" in m for _, m in msgs)
    rec["log_warnings"] = sum(l == "WARNING" for l, _ in msgs)
    rec["log_errors"] = sum(l == "ERROR" for l, _ in msgs)
    rec["stage_seconds"] = times
    if "mask" in arrays:
        rec["mask_fraction"] = float(np.count_nonzero(arrays["mask"])) / arrays["mask"].size
        rec["pred_fraction"] = float(np.count_nonzero(arrays["pred_mask"])) / arrays["pred_mask"].size
    if "obstacle_mask" in arrays:
        rec["obstacle_fraction"] = float(np.count_nonzero(arrays["obstacle_mask"])) / arrays["obstacle_mask"].size
    if "full_path" in rec:
        p = np.array(rec["full_path"])
        rec["route_length_px"] = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        rec["route_points"] = len(rec["full_path"])
    if rec.get("mission") and os.path.exists(rec["mission"]):
        with open(rec["mission"]) as f:
            rows = [l.split("\t") for l in f.read().splitlines()[1:]]
        rec["mission_items"] = len(rows)
        rec["mission_drops"] = sum(r[3] == "183" for r in rows)
        rec["route_length_m"] = _route_length_m(rows)
        lat = [float(r[8]) for r in rows if r[3] != "183"]
        lon = [float(r[9]) for r in rows if r[3] != "183"]
        rec["lat_bounds"], rec["lon_bounds"] = [min(lat), max(lat)], [min(lon), max(lon)]
        with open(os.path.join(out_dir, f"{tag}.sidecar.json"), "w") as f:
            json.dump({k: rec[k] for k in ("full_path", "drop_indices", "home", "image_center_px", "geo_center",
                                            "meters_per_pixel")} | {"expected_drops": rec.get("regions")}, f)
    rec["original_shape"] = list(cv2.imread(os.path.join(REPO, image_path)).shape[:2])
    if keep_arrays:
        rec["_arrays"] = arrays
    return rec


def _route_length_m(rows):
    import math
    nav = [(float(r[8]), float(r[9])) for r in rows if r[3] in ("16", "22", "21")]
    total = 0.0
    for (a, b), (c, d) in zip(nav, nav[1:]):
        p1, p2 = math.radians(a), math.radians(c)
        h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
        total += 2 * 6371008.8 * math.asin(min(1.0, math.sqrt(h)))
    return total


def save_json(obj, *parts):
    path = os.path.join(RESULTS, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return path
