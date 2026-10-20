"""Check an explicit flood-mask occupancy review gate on all supplied maps."""
import json
from pathlib import Path

import common

OUT = Path(common.RESULTS) / "assam_gate"
WEATHER = {"precipitation": 3.0, "wind_speed_10m": 12.0,
           "wind_direction_10m": 270.0}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name in common.MAPS:
        rec = common.run(name, out_dir=str(OUT), tag=name,
                         weather=WEATHER, max_mask_fraction=0.8)
        results.append({"map": name, "exit": rec["exit"],
                        "mission_exists": bool(rec.get("mission") and Path(rec["mission"]).exists()),
                        "gate_fraction": 0.8,
                        "baseline_mask_fraction": json.loads((Path(common.RESULTS) / "e2e" / f"{name}.json").read_text()).get("mask_fraction")})
        print(results[-1], flush=True)
    (OUT / "comparison.json").write_text(json.dumps(results, indent=2))
