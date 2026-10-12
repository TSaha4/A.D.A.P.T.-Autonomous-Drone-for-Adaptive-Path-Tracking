"""Repeat the unchanged single-UAV pipeline for varied simulated operator clicks.

Ten predefined seeds per map and three weather scenarios. Each seed samples 4-8
of the deterministic red-component centroids and perturbs clicks by at most
three display pixels. This is a stress test, not a human-operator study.
"""
import csv
import json
import os
import random
import statistics
from pathlib import Path

import common

OUT = Path(common.RESULTS) / "operator_robustness"
WEATHER = {"calm": {"precipitation": 0.0, "wind_speed_10m": 0.0,
                     "wind_direction_10m": 0.0},
           "moderate": {"precipitation": 3.0, "wind_speed_10m": 12.0,
                         "wind_direction_10m": 270.0},
           "heavy": {"precipitation": 10.0, "wind_speed_10m": 30.0,
                     "wind_direction_10m": 270.0}}


def run_once(map_name, scenario, seed):
    rng = random.Random(seed)
    base = common.simulated_clicks
    def clicks(image, n_clicks=8):
        pts = base(image, n_clicks)
        rng.shuffle(pts)
        count = min(len(pts), rng.randint(4, 8))
        h, w = image.shape[:2]
        return [(min(w - 1, max(0, x + rng.randint(-3, 3))),
                 min(h - 1, max(0, y + rng.randint(-3, 3)))) for x, y in pts[:count]]
    common.simulated_clicks = clicks
    try:
        r = common.run(map_name, out_dir=str(OUT / "runs"),
                       tag=f"{map_name}_{scenario}_s{seed}", weather=WEATHER[scenario])
    finally:
        common.simulated_clicks = base
    return {"map": map_name, "weather": scenario, "seed": seed,
            "clicks": len(r.get("clicks", [])),
            **{k: r.get(k) for k in ("exit", "mask_fraction", "regions", "no_path_legs",
                                     "route_length_px", "mission_items", "total_s")}}


def summarize(rows):
    out = []
    for map_name in common.MAPS:
        for weather in WEATHER:
            group = [r for r in rows if r["map"] == map_name and r["weather"] == weather]
            entry = {"map": map_name, "weather": weather, "n": len(group),
                     "successes": sum(r["exit"] == 0 for r in group)}
            for field in ("mask_fraction", "regions", "no_path_legs", "route_length_px", "mission_items", "total_s"):
                values = [r[field] for r in group if r[field] is not None]
                entry[field + "_mean"] = statistics.fmean(values) if values else None
                entry[field + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
            out.append(entry)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    records = (json.loads((OUT / "summary.json").read_text())["runs"]
               if os.environ.get("ADAPT_RESUME") == "1" and (OUT / "summary.json").exists() else [])
    completed = {(r["map"], r["weather"], r["seed"]) for r in records}
    for map_name in common.MAPS:
        for scenario in WEATHER:
            for seed in range(10):
                if (map_name, scenario, seed) in completed:
                    continue
                row = run_once(map_name, scenario, seed)
                records.append(row)
                print(row["map"], row["weather"], seed, row["exit"], flush=True)
    summary = summarize(records)
    (OUT / "summary.json").write_text(json.dumps({"runs": records, "summary": summary}, indent=2))
    with (OUT / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(json.dumps(summary, indent=2))
