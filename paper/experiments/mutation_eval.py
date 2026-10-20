"""E8 - Mutation testing of the geospatial / mission code and of the integrated multi-base (multi-UAV) mode.

Each mutant injects one plausible defect into a TEMPORARY COPY of the repository (production files are never
touched) and runs the full test suite; the mutant is "killed" if any test fails.
Output: results/mutation/mutation.csv
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

C, MO, MAIN = "src/mission/coordinates.py", "src/mission/mission_output.py", "main.py"
MUTANTS = [
    ("y sign flipped", C, "dy_px = cy - py", "dy_px = py - cy"),
    ("x sign flipped", C, "dx_px = px - cx", "dx_px = cx - px"),
    ("axes swapped", C, "dx_m = dx_px * meters_per_pixel", "dx_m = dy_px * meters_per_pixel"),
    ("latitude constant 111000", C, "dlat = dy_m / 111320.0", "dlat = dy_m / 111000.0"),
    ("cos of degrees", C, "math.cos(math.radians(geo_lat))", "math.cos(geo_lat)"),
    ("longitude scale without cos", C, "(111320.0 * math.cos(math.radians(geo_lat)))", "111320.0"),
    ("cos of result latitude", C, "math.cos(math.radians(geo_lat))", "math.cos(math.radians(geo_lat + dlat))"),
    ("no longitude wrap", C, "lon = (lon + 180.0) % 360.0 - 180.0", "pass"),
    ("always wrap", C, "if not -180.0 <= lon <= 180.0:\n", "if True:\n"),
    ("pole check removed", C, "if not -90.0 <= lat <= 90.0:", "if False:"),
    ("half-parallel check removed", C, "if not abs(dlon) < 180.0:", "if False:"),
    ("zero scale accepted", C, "if meters_per_pixel <= 0:", "if meters_per_pixel < 0:"),
    ("polar reference accepted", C, "-90.0 < geo_lat < 90.0", "-90.0 <= geo_lat <= 90.0"),
    ("finite check removed", C, "if not all(math.isfinite(v)", "if False and not all(math.isfinite(v)"),
    ("lat/lon swapped in WPL", MO, "# Fly to location at cruise altitude\n            f.write(f\"{seq}\\t0\\t3\\t16\\t0\\t0\\t0\\t0\\t{lat}\\t{lon}",
     "# Fly to location at cruise altitude\n            f.write(f\"{seq}\\t0\\t3\\t16\\t0\\t0\\t0\\t0\\t{lon}\\t{lat}"),
    ("servo parameters in old slots", MO, "183\\t{servo_channel}\\t{servo_pwm}\\t0\\t0", "183\\t0\\t0\\t{servo_pwm}\\t0"),
    ("HOME-region drop lost", MO, "0 < i < len(full_path) - 1", "0 < i < len(full_path) - 1 and pt != home"),
    ("conversion inside write loop", MO, "lat, lon = path_latlon[i]",
     "lat, lon = pixel_to_latlon(pt[0], pt[1], image_center_px, geo_center, meters_per_pixel)"),
    ("partial-file guard removed", MO,
     "path_latlon = [pixel_to_latlon(pt[0], pt[1], image_center_px, geo_center, meters_per_pixel) for pt in full_path]",
     "path_latlon = None"),
    ("centre W//2 (pre-fix)", MAIN, "((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)\n        geo_center",
     "(image.shape[1] // 2, image.shape[0] // 2)\n        geo_center"),
    ("resize not compensated", MAIN, "settings.METERS_PER_PIXEL / scale_factor\n", "settings.METERS_PER_PIXEL\n"),
    ("resize compensated twice", MAIN, "settings.METERS_PER_PIXEL / scale_factor\n", "settings.METERS_PER_PIXEL / scale_factor ** 2\n"),
    ("stale mission kept", MAIN, 'if args.mode == "single" and os.path.lexists(output_file):\n        if not remove_mission_file',
     "if False:\n        if not remove_mission_file"),
    ("preflight removed", MAIN, "pixel_to_latlon(0, 0, (0, 0), (args.lat, args.lon), settings.METERS_PER_PIXEL)", "pass"),
    # Multi-base (multi-UAV) mode, integrated from the MAIN branch
    ("multi: centre W//2", MAIN, "((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)\n    pred_mask",
     "(image.shape[1] // 2, image.shape[0] // 2)\n    pred_mask"),
    ("multi: resize not compensated", MAIN, "settings.METERS_PER_PIXEL / scale_factor  # metres per display pixel",
     "settings.METERS_PER_PIXEL  # metres per display pixel"),
    ("multi: overlap gate disabled", MAIN, "    if overlaps:\n        log_deviation(", "    if False:\n        log_deviation("),
    ("multi: reload stop not encoded", MO, "            if i in reload_set:\n", "            if False:\n"),
    ("multi: reload index unchecked", MO,
     "if not (0 < r < len(full_path) - 1) or tuple(full_path[r]) != tuple(home) or r in drop_indices:", "if False:"),
    ("multi: nudged reach kept stale", "src/mission/overlap_repair.py", "            refresh_reach(first)\n",
     "            pass\n"),
    ("multi: forecast frames stamped k", "src/weather/flood_spread.py", "frames[float(min(k, horizon_hours))] = frame",
     "frames[float(k)] = frame"),
    ("multi: legacy drop selection replaced", "src/mission/safe_dropzone.py", "    if obstacle_mask is not None:\n        return find_dry",
     "    if True:\n        return find_dry"),
    ("multi: snap default radius 15", "src/routing/pathfinding.py", "max_h, max_search_radius=10, target_component",
     "max_h, max_search_radius=15, target_component"),
]
PY = sys.executable


def copy_repo(dst):
    shutil.copytree(common.REPO, dst, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", ".pytest_cache",
                                                                     "output", "paper"))
    os.makedirs(os.path.join(dst, "data", "output"), exist_ok=True)


def apply(root, path, old, new):
    p = os.path.join(root, path)
    src = open(p, newline="").read()
    eol = "\r\n" if "\r\n" in src else "\n"
    o, n = old.replace("\n", eol), new.replace("\n", eol)
    if src.count(o) != 1:
        raise SystemExit(f"anchor found {src.count(o)} times in {path}: {old!r}")
    open(p, "w", newline="").write(src.replace(o, n))


def pytest(root):
    return subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=root, capture_output=True, text=True)


def main():
    rows = []
    with tempfile.TemporaryDirectory() as t:
        base = os.path.join(t, "base")
        copy_repo(base)
        b = pytest(base)
        print("baseline:", b.stdout.strip().splitlines()[-1])
        rows.append({"mutant": "(none - baseline)", "file": "", "status": "PASS" if b.returncode == 0 else "FAIL",
                     "first_failure": "", "seconds": ""})
        for name, path, old, new in MUTANTS:
            root = os.path.join(t, "m")
            shutil.rmtree(root, ignore_errors=True)
            copy_repo(root)
            apply(root, path, old, new)
            t0 = time.time()
            r = pytest(root)
            first = next((l for l in r.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))), "")
            status = "KILLED" if r.returncode != 0 else "SURVIVED"
            rows.append({"mutant": name, "file": path, "status": status, "first_failure": first[:160],
                         "seconds": round(time.time() - t0, 1)})
            print(f"{status:8} {name}")
    os.makedirs(os.path.join(common.RESULTS, "mutation"), exist_ok=True)
    with open(os.path.join(common.RESULTS, "mutation", "mutation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    killed = sum(r["status"] == "KILLED" for r in rows)
    print(f"{killed}/{len(MUTANTS)} killed")


if __name__ == "__main__":
    main()
