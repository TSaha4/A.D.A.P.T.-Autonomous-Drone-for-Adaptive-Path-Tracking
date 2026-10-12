"""Full-pipeline fleet sensitivity to scale and battery-model assumptions.

Each row is a new run through main.main() with the same map, simulated clicks
and fixed moderate weather. Settings change by ±20% one at a time; the model is
uncalibrated, so this is sensitivity analysis rather than aircraft validation.
"""
import json
import sys
from pathlib import Path
from unittest import mock

import multi_uav_comparison as cmp
sys.path.insert(0, cmp.REPO)
from config import settings

OUT = Path(cmp.HERE) / "results" / "fleet_sensitivity"
PARAMS = ("DRONE_LOADED_ENDURANCE_MIN", "PAYLOAD_PER_DROP_KG", "WIND_PENALTY_PER_MPS")


def compact(r, variable, multiplier):
    return {"map": r["map"], "variable": variable, "multiplier": multiplier,
            "meters_per_pixel": r["meters_per_pixel"], "outcome": r["outcome"],
            "wall_s": r["wall_s"], "drop_points": r.get("drop_points"),
            "visited": r.get("visited", 0), "uavs": r.get("n_uavs", 0),
            "makespan_cruise_min": r.get("est_completion_min"),
            "max_sortie_battery": r.get("max_sortie_battery"),
            "mission_valid": r.get("all_missions_valid")}


def run_case(map_name, variable, multiplier, mpp=2.0):
    cmp.OUT = str(OUT / f"{variable}_{multiplier:g}")
    patch = (mock.patch.object(settings, variable, getattr(settings, variable) * multiplier)
             if variable in PARAMS else mock.patch.object(settings, "DRONE_SPEED_MPS", settings.DRONE_SPEED_MPS))
    with patch:
        record = cmp.run_config(map_name, "moderate", 4, mpp=mpp)
    return compact(record, variable, multiplier)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for map_name in ("varanasi", "kanpur"):
        for mpp in (2.0, 3.0, 5.0, 10.0, 20.0, 50.0,
                    93.0 if map_name == "varanasi" else 130.0):
            row = run_case(map_name, "scale", 1.0, mpp)
            rows.append(row)
            print(row, flush=True)
        for param in PARAMS:
            for mul in (0.8, 1.2):
                row = run_case(map_name, param, mul)
                rows.append(row)
                print(row, flush=True)
    (OUT / "sensitivity.json").write_text(json.dumps(rows, indent=2))
