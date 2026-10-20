"""E1 - End-to-end runs on the three repository maps (baseline configuration).

Outputs (paper/experiments/results/e2e/):
  <map>.json            stage outputs, counts, timings
  <map>.npz             display image, current/predicted/obstacle masks (for figures and E5)
  <map>.waypoints       generated QGC WPL 110 mission (production writer)
  <map>.sidecar.json    route given to the mission writer (for independent validation)
  <map>.validation.txt  independent validator report (tests/wpl_validator.py)
  summary.json          table used by the paper
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from tests import wpl_validator  # noqa: E402

OUT = os.path.join(common.RESULTS, "e2e")


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for m in common.MAPS:
        rec = common.run(m, out_dir=OUT, keep_arrays=True)
        arrays = rec.pop("_arrays")
        np.savez_compressed(os.path.join(OUT, f"{m}.npz"), **arrays)
        side = json.load(open(os.path.join(OUT, f"{m}.sidecar.json")))
        findings, items = wpl_validator.validate(rec["mission"], side, servo_channel=9, servo_pwm=2000)
        with open(os.path.join(OUT, f"{m}.validation.txt"), "w") as f:
            f.write("\n".join(map(str, findings)))
        rec["validator_errors"] = sum(x.level == "ERROR" for x in findings)
        rec["validator_warnings"] = sum(x.level == "WARN" for x in findings)
        rec["mission"] = os.path.relpath(rec["mission"], common.REPO)
        common.save_json(rec, "e2e", f"{m}.json")
        summary[m] = {k: rec.get(k) for k in (
            "original_shape", "display_shape", "meters_per_pixel", "geo_center", "image_center_px", "hsv_lower",
            "hsv_upper", "red_wrap", "mask_fraction", "pred_fraction", "obstacle_fraction", "regions",
            "planned_legs", "no_path_legs", "snapped_endpoints", "unsnappable_endpoints", "route_points",
            "route_length_px", "route_length_m", "mission_items", "mission_drops", "home", "lat_bounds", "lon_bounds",
            "log_warnings", "log_errors", "validator_errors", "validator_warnings", "stage_seconds", "total_s")}
        s = summary[m]
        print(f"{m}: mask {100 * s['mask_fraction']:.2f}% regions {s['regions']} legs {s['planned_legs']} "
              f"no-path {s['no_path_legs']} route pts {s['route_points']} items {s['mission_items']} drops "
              f"{s['mission_drops']} validator E{s['validator_errors']}/W{s['validator_warnings']} total {s['total_s']:.1f}s")
    common.save_json(summary, "e2e", "summary.json")


if __name__ == "__main__":
    main()
