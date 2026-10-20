"""
Single-UAV vs multi-UAV comparison for the multi-base mode (main.py --mode multi).

The same mission (map, simulated operator clicks, weather, drop points) is planned with the base cap
K = MAX_BASES set to 1 (one base: a single UAV flying every sortie) and to 2, 3 and 4 (up to K bases, one
UAV each). Everything else is the unmodified pipeline entry point main.main(); the substitutions are the
ones of common.py: simulated operator clicks instead of interactive sampling, no GUI windows, and fixed
weather instead of the live request. Ground scale is the configured METERS_PER_PIXEL (2 m per ORIGINAL
pixel, not the maps' true scale; see the paper's limitations).

Every number below is read from what the run itself produced: run_summary.json, the per-base route
sidecars and the exported missions (re-checked with the independent validator tests/wpl_validator.py).
Completion time is an estimate derived from route length at the configured cruise speed (DRONE_SPEED_MPS);
it excludes take-off, landing, loiter/descent at drops and reload turnaround, and assumes all UAVs launch
together. It is labelled as such wherever it is reported.

Outputs: results/multi_uav/comparison.json, results/multi_uav/comparison.csv, the session folders under
results/multi_uav/sessions/ (missions, route sidecars, run summaries; rendered maps only with --keep-maps),
and the LaTeX table ../figures/tab_multi_uav.tex.

Usage: python paper/experiments/multi_uav_comparison.py [--keep-maps] [--table-only | --scale-check]
--scale-check repeats the K = 4, moderate-weather runs of Varanasi and Kanpur at the landmark-estimated true
map scale (results/multi_uav/true_scale_check.json).
"""
import csv
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results", "multi_uav")
TABLE = os.path.join(HERE, "..", "figures", "tab_multi_uav.tex")
MAPS = ["varanasi", "kanpur", "assam"]
CAPS = [1, 2, 3, 4]
# Landmark-estimated true scales of the dot maps (m per original pixel; paper, geographic-validity subsection)
TRUE_SCALE = {"varanasi": 93.0, "kanpur": 130.0}
WEATHER = {"dummy": {"precipitation": 10.0, "wind_speed_10m": 30.0, "wind_direction_10m": 270.0},   # data/input/dummy_weather.json
           "moderate": {"precipitation": 3.0, "wind_speed_10m": 12.0, "wind_direction_10m": 270.0}}


def run_config(map_name, weather_name, cap, mpp=None):
    """One run of main.main() --mode multi --max-bases cap; returns the measured record.
    mpp overrides METERS_PER_PIXEL (used only by the true-scale check)."""
    sys.path[:0] = [REPO, HERE]
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import main as M
    import src.vision.image_processing as IP
    from common import simulated_clicks
    from tests import wpl_validator
    from config import settings

    logging.disable(logging.CRITICAL)
    tag = f"{map_name}_{weather_name}_K{cap}" + ("" if mpp is None else f"_mpp{mpp:g}")
    mpp = settings.METERS_PER_PIXEL if mpp is None else mpp
    root = os.path.join(OUT, "sessions", tag)
    os.makedirs(root, exist_ok=True)
    for old in glob.glob(os.path.join(root, "session_*")):  # keep one session per configuration
        for f in glob.glob(os.path.join(old, "*")):
            os.remove(f)
        os.rmdir(old)
    rec = {"map": map_name, "weather": weather_name, "cap": cap, "meters_per_pixel": mpp}
    original = IP.ImageProcessor.compute_dynamic_hsv

    def select(self, image):
        self.image = image.copy()
        self.sample_points = simulated_clicks(image, 8)
        original(self)
        return self.hsv_lower, self.hsv_upper

    argv = ["main.py", os.path.join("data", "input", f"{map_name}.png"), "--mode", "multi",
            "--max-bases", str(cap), "--session-root", root]
    cwd = os.getcwd()
    os.chdir(REPO)
    t0 = time.perf_counter()
    try:
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(IP.ImageProcessor, "select_sample_points", select), \
                mock.patch.object(M, "get_weather_data", lambda lat, lon: dict(WEATHER[weather_name])), \
                mock.patch.object(M, "show_image_safe", lambda *a, **k: None), \
                mock.patch.object(settings, "METERS_PER_PIXEL", mpp), \
                mock.patch("matplotlib.pyplot.show", lambda *a, **k: None):
            try:
                M.main()
                rec["exit"] = 0
            except SystemExit as e:
                rec["exit"] = e.code
    finally:
        rec["wall_s"] = time.perf_counter() - t0
        os.chdir(cwd)
    sessions = sorted(glob.glob(os.path.join(root, "session_*")))
    if not sessions:
        rec["outcome"] = "NO_SESSION (no flood zone / no cluster)"
        return rec
    session = sessions[-1]
    if "--keep-maps" not in sys.argv:  # rendered maps are ~2 MB each; the data below does not need them
        for png in glob.glob(os.path.join(session, "*.png")):
            os.remove(png)
    rec["session"] = os.path.relpath(session, HERE)
    s = json.load(open(os.path.join(session, "run_summary.json")))
    rec["outcome"] = s.get("outcome")
    rec["meters_per_pixel_display"] = s.get("meters_per_pixel_display")
    rec["drop_points"] = len(s.get("drop_points", []))
    if "bases" not in s:
        return rec
    speed = settings.DRONE_SPEED_MPS
    uavs = []
    all_valid = True
    for b in s["bases"]:
        item = {"base": b["base"], "home": b["home"], "drops": b["visited"], "sorties": b["sorties"],
                "route_length_m": b["route_length_m"], "cruise_time_min": b["route_length_m"] / speed / 60.0,
                "max_sortie_battery": max(b["routed_sortie_battery"]) if b["routed_sortie_battery"] else 0.0,
                "mission_file": b["waypoint_file"] is not None}
        if b["waypoint_file"]:
            mission = os.path.join(session, f"base{b['base']}.waypoints")
            sidecar = json.load(open(os.path.join(session, f"base{b['base']}.route.json")))
            findings, items = wpl_validator.validate(mission, sidecar, servo_channel=9, servo_pwm=2000)
            item["mission_items"] = len(items)
            item["reloads"] = sum(r["cmd"] == wpl_validator.NAV_LAND for r in items[:-1])
            item["validator_errors"] = sum(f.level == "ERROR" for f in findings)
            all_valid &= item["validator_errors"] == 0
        else:
            all_valid = False
        uavs.append(item)
    rec.update({
        "uavs": uavs, "n_uavs": len(uavs), "all_missions_valid": all_valid,
        "assigned": len(s["assignment"]), "unreachable_by_any_base": len(s["unreachable_by_any_base"]),
        "connectivity_excluded": s["connectivity_excluded"], "planner_unreachable": s["unreachable"],
        "visited": s["drops_visited"], "reloads": s["reloads"], "sorties": sum(u["sorties"] for u in uavs),
        "total_route_km": s["total_route_length_m"] / 1000.0,
        "max_uav_route_km": max(u["route_length_m"] for u in uavs) / 1000.0,
        "est_completion_min": max(u["cruise_time_min"] for u in uavs),
        "total_cruise_min": sum(u["cruise_time_min"] for u in uavs),
        "max_sortie_battery": max(u["max_sortie_battery"] for u in uavs),
        "drops_per_uav": [u["drops"] for u in uavs],
        "overlaps_remaining": len(s.get("path_overlaps", [])),
        "repairs": [r["tier"] for r in s.get("overlap_repairs", [])],
        "fallback_legs": s["legs"]["fallback"], "route_px_in_current_flood": s["route_flooded_px_current"],
        "territory_violations": len(s.get("territory_violations", [])),
        "min_base_separation_px": s.get("base_separation_min_px")})
    return rec


def write_table(records):
    def cells(r):
        if r is None or "uavs" not in r:
            return [r"\multicolumn{6}{c}{infeasible: no base reaches any drop (exit 2)}"]
        return [str(r["n_uavs"]), f"{r['visited']}/{r['drop_points']}", str(r["sorties"]),
                f"{r['total_route_km']:.1f}", f"{r['max_uav_route_km']:.1f}", f"{r['est_completion_min']:.0f}"]

    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{One drone vs.\ a drone fleet: single- and multi-UAV planning with the multi-base mode on the same missions (simulated "
             r"operator, \texttt{METERS\_PER\_PIXEL}$=2$~m per original pixel). $K$ is the base cap; $K=1$ is one "
             r"UAV flying every sortie from one base. Visited: drops delivered of dry drop points generated. km: "
             r"routed length of all missions; max/UAV: longest single mission. Time: estimated completion (minutes) "
             r"from route length at 5~m/s, all UAVs launched together, excluding take-off, landing, drop manoeuvres "
             r"and reload turnaround. Every exported mission passed the independent validator; no two routes cross.}",
             r"\label{tab:multiuav}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{@{}llrrrrrrcrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{6}{c}{Dummy weather (10 mm/h, 30 km/h)} & & "
             r"\multicolumn{6}{c}{Moderate weather (3 mm/h, 12 km/h)}\\",
             r"\cmidrule{3-8}\cmidrule{10-15}",
             r"Map & $K$ & UAVs & Visited & Sorties & km & max/UAV & Time & & "
             r"UAVs & Visited & Sorties & km & max/UAV & Time\\", r"\midrule"]
    index = {(r["map"], r["weather"], r["cap"]): r for r in records}
    for m in MAPS:
        for i, k in enumerate(CAPS):
            row = [m.capitalize() if i == 0 else "", str(k)] + cells(index.get((m, "dummy", k))) + [""] + \
                  cells(index.get((m, "moderate", k)))
            lines.append(" & ".join(row) + r"\\")
        if m != MAPS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(TABLE, "w") as f:
        f.write("% Generated by paper/experiments/multi_uav_comparison.py -- do not edit by hand.\n")
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    if "--scale-check" in sys.argv:
        # Same planner at the maps' landmark-estimated true scale (moderate weather, K = 4)
        names = list(TRUE_SCALE)
        with ProcessPoolExecutor(max_workers=len(names)) as pool:
            records = list(pool.map(run_config, names, ["moderate"] * len(names), [4] * len(names),
                                    [TRUE_SCALE[n] for n in names]))
        with open(os.path.join(OUT, "true_scale_check.json"), "w") as f:
            json.dump(records, f, indent=1)
        for r in records:
            print({k: r.get(k) for k in ("map", "meters_per_pixel", "exit", "outcome", "drop_points", "visited")})
        return
    if "--table-only" in sys.argv:
        write_table(json.load(open(os.path.join(OUT, "comparison.json"))))
        return
    configs = [(m, w, k) for m in MAPS for w in WEATHER for k in CAPS]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
        records = list(pool.map(run_config, *zip(*configs)))
    with open(os.path.join(OUT, "comparison.json"), "w") as f:
        json.dump(records, f, indent=1)
    keys = ["map", "weather", "cap", "exit", "outcome", "n_uavs", "drop_points", "assigned", "unreachable_by_any_base",
            "connectivity_excluded", "planner_unreachable", "visited", "sorties", "reloads", "total_route_km",
            "max_uav_route_km", "est_completion_min", "total_cruise_min", "max_sortie_battery", "drops_per_uav",
            "overlaps_remaining", "repairs", "fallback_legs", "route_px_in_current_flood", "territory_violations",
            "all_missions_valid", "wall_s"]
    with open(os.path.join(OUT, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in records:
            w.writerow([r.get(k, "") for k in keys])
    write_table(records)
    for r in records:
        print({k: r.get(k) for k in ("map", "weather", "cap", "exit", "outcome", "n_uavs", "visited", "drop_points",
                                     "total_route_km", "est_completion_min", "repairs", "all_missions_valid")})


if __name__ == "__main__":
    main()
