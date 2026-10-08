"""
Equivalence of the integrated multi-base mode with the reference implementation it was ported from.

The reference is the multi-base planner of commit ca27215 (branch routing-algo). Export it first, outside the
repository, without touching the working tree:
    git archive ca27215 | tar -x -C <REFERENCE_DIR>
then run:
    python paper/experiments/multi_uav_equivalence.py <REFERENCE_DIR>

Both implementations are run on the same map with identical inputs: the same operator sample points, the same
weather, the same ground size of a display pixel (2 m; the reference uses METERS_PER_PIXEL directly per display
pixel, the integrated code uses METERS_PER_PIXEL / s, so METERS_PER_PIXEL is set to 2 s on that side) and, for
non-calm weather, the reference's own forecast frames (the two code bases use different spread rules). The plan
of every base is captured at the mission writer and compared: base positions, assignment, every route point,
drop and reload indices, and the overlap repairs performed.

Cases: calm weather with the simulated operator on Varanasi and Kanpur, and the moderate-weather Varanasi case
with the reference's automatic sample points, in which the reference performs two reassignment repairs.
Output: results/multi_uav/equivalence.json
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results", "multi_uav", "equivalence.json")
CASES = [("varanasi", "calm", "simulated"), ("kanpur", "calm", "simulated"), ("varanasi", "moderate", "reference_auto")]
WEATHER = {"calm": {"precipitation": 0.0, "wind_speed_10m": 0.0, "wind_direction_10m": 0.0, "status": "success"},
           "moderate": {"precipitation": 3.0, "wind_speed_10m": 12.0, "wind_direction_10m": 270.0, "status": "success"}}


def worker(side, root, reference, image_path, weather_name, sampling, out_json, work):
    """Runs one implementation (its own main.py) in this process and writes the captured plans."""
    import importlib.util
    from unittest import mock
    sys.path.insert(0, root)
    os.makedirs(work, exist_ok=True)
    os.chdir(work)
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    img = cv2.imread(image_path)
    s = 750 / img.shape[1] if img.shape[1] > 750 else 1.0
    disp = cv2.resize(img, (750, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    if sampling == "reference_auto":
        clicks = load("ref_ip", os.path.join(reference, "src", "vision", "image_processing.py")).ImageProcessor() \
            .auto_sample_points(disp)
    else:  # the paper's simulated operator (common.simulated_clicks)
        hsv = cv2.cvtColor(disp, cv2.COLOR_BGR2HSV)
        red = (((hsv[..., 0] <= 5) | (hsv[..., 0] >= 175)) & (hsv[..., 1] > 180) & (hsv[..., 2] > 180)).astype(np.uint8)
        _, _, stats, cents = cv2.connectedComponentsWithStats(red)
        order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:8] + 1
        clicks = [tuple(int(v) for v in cents[i]) for i in order]
    clicks = [tuple(map(int, p)) for p in clicks]

    import main as M
    from config import settings
    captured = []
    real_generate = M.generate_mission_file

    def capture(full_path, drop_indices, home, *args, filename=None, reload_indices=None, **kw):
        captured.append({"file": os.path.basename(str(filename)), "home": [int(v) for v in home],
                         "full_path": [[int(v) for v in p] for p in full_path],
                         "drop_indices": [int(i) for i in drop_indices],
                         "reload_indices": [int(i) for i in (reload_indices or [])]})
        return real_generate(full_path, drop_indices, home, *args, filename=filename, reload_indices=reload_indices,
                             **kw)

    weather = WEATHER[weather_name]
    patches = [mock.patch.object(M, "get_weather_data", lambda lat, lon: dict(weather)),
               mock.patch.object(M, "generate_mission_file", capture),
               mock.patch.object(settings, "MAX_BASES", 4),
               mock.patch("matplotlib.pyplot.show", lambda *a, **k: None)]
    lat, lon = "25.3176", "82.9739"
    if side == "reference":
        argv = ["main.py", "--image", image_path, "--no-gui", "--lat", lat, "--lon", lon,
                "--sample-points", ";".join(f"{x},{y}" for x, y in clicks)]
        patches.append(mock.patch.object(settings, "METERS_PER_PIXEL", 2.0))
        session_root = os.path.join(work, "data", "output")
    else:
        import src.vision.image_processing as IP

        def select(self, image):
            self.image = image.copy()
            self.sample_points = list(clicks)
            self.compute_dynamic_hsv()
            return self.hsv_lower, self.hsv_upper

        argv = ["main.py", image_path, "--mode", "multi", "--lat", lat, "--lon", lon, "--session-root", work]
        patches += [mock.patch.object(settings, "METERS_PER_PIXEL", 2.0 * s),
                    mock.patch.object(IP.ImageProcessor, "select_sample_points", select),
                    mock.patch.object(M, "show_image_safe", lambda *a, **k: None)]
        if weather_name != "calm":
            # Use the reference's forecast frames (its spread rule needs these settings constants).
            for name, value in (("FLOOD_GROWTH_PX_PER_5MM_HOUR", 1.0), ("FLOOD_WIND_SHIFT_PX_PER_10KMH_HOUR", 1.0),
                                ("FLOOD_PRECIP_REFERENCE_MM_PER_H", 5.0), ("FLOOD_WIND_REFERENCE_KMH", 10.0)):
                setattr(settings, name, value)
            ref_flood = load("ref_flood", os.path.join(reference, "src", "weather", "flood_spread.py"))
            patches.append(mock.patch.object(M, "forecast_frames",
                                             lambda mask, w, h, predictor=None: ref_flood.predict_spread(mask, w, h)))
        session_root = work
    code = 0
    with mock.patch.object(sys, "argv", argv):
        for p in patches:
            p.start()
        try:
            M.main()
        except SystemExit as e:
            code = e.code
    sessions = sorted(d for d in os.listdir(session_root) if d.startswith("session_"))
    with open(os.path.join(session_root, sessions[-1], "run_summary.json")) as f:
        summary = json.load(f)
    final = {}
    for c in captured:  # the reference also writes files for plans that are later replaced: keep the last one
        final[c["file"]] = c
    with open(out_json, "w") as f:
        json.dump({"exit": code, "clicks": clicks, "assignment": summary.get("assignment"),
                   "unreachable": sorted(summary.get("unreachable_by_any_base", {})),
                   "bases": [b["home"] for b in summary.get("bases", [])],
                   "repairs": summary.get("overlap_repairs"), "plans": [final[k] for k in sorted(final)]}, f)


def main():
    if sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
        return
    reference = os.path.abspath(sys.argv[1])
    if not os.path.exists(os.path.join(reference, "src", "mission", "multi_base.py")):
        raise SystemExit(f"{reference} is not an export of the reference implementation (see the module docstring)")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for map_name, weather_name, sampling in CASES:
            image = os.path.join(REPO, "data", "input", f"{map_name}.png")
            out = {}
            for side, root in (("reference", reference), ("integrated", REPO)):
                path = os.path.join(tmp, f"{map_name}_{weather_name}_{side}.json")
                subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", side, root, reference, image,
                                weather_name, sampling, path, os.path.join(tmp, f"work_{map_name}_{weather_name}_{side}")],
                               check=True, capture_output=True)
                out[side] = json.load(open(path))
            a, b = out["reference"], out["integrated"]
            plans_a = {p["file"]: p for p in a["plans"]}
            plans_b = {p["file"]: p for p in b["plans"]}
            same_plans = plans_a.keys() == plans_b.keys() and all(
                plans_a[f][k] == plans_b[f][k] for f in plans_a for k in ("home", "full_path", "drop_indices", "reload_indices"))
            row = {"map": map_name, "weather": weather_name, "sampling": sampling, "exit": [a["exit"], b["exit"]],
                   "bases": len(a["bases"]), "repairs": a["repairs"],
                   "route_points": sum(len(p["full_path"]) for p in a["plans"]),
                   "identical": {"samples": a["clicks"] == b["clicks"], "bases": a["bases"] == b["bases"],
                                 "assignment": a["assignment"] == b["assignment"],
                                 "unreachable": a["unreachable"] == b["unreachable"],
                                 "repairs": a["repairs"] == b["repairs"], "plans": same_plans}}
            row["all_identical"] = all(row["identical"].values())
            results.append(row)
            print(row)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
