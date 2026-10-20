"""Compare the pipeline's flat-Earth pixel conversion with GeoTIFF CRS coordinates.

Uses all usable official Sen1Floods11 test chips. Each chip's affine transform
and CRS are treated as the geographic reference. The pipeline conversion is
anchored at the exact raster centre and supplied with the local WGS84 average
pixel size; this isolates its remaining projection and scale error.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Geod, Transformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mission.coordinates import pixel_to_latlon  # noqa: E402

BENCH = ROOT / "paper/experiments/results/sen1floods11/test_metrics.json"
CACHE = ROOT / "paper/experiments/cache/sen1floods11/data/flood_events/HandLabeled/S1Hand"
OUT = ROOT / "paper/experiments/results/georeference"
GEOD = Geod(ellps="WGS84")


def truth(ds, x, y):
    mx, my = ds.transform * (x + 0.5, y + 0.5)
    lon, lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(mx, my)
    return lat, lon


def evaluate(chip):
    with rasterio.open(CACHE / f"{chip}_S1Hand.tif") as ds:
        h, w = ds.height, ds.width
        center = ((w - 1) / 2, (h - 1) / 2)
        clat, clon = truth(ds, *center)
        east = truth(ds, center[0] + 1, center[1])
        south = truth(ds, center[0], center[1] + 1)
        _, _, ew = GEOD.inv(clon, clat, east[1], east[0])
        _, _, ns = GEOD.inv(clon, clat, south[1], south[0])
        mpp = (abs(ew) + abs(ns)) / 2
        points = [(float(x), float(y)) for y in np.linspace(0, h - 1, 9)
                  for x in np.linspace(0, w - 1, 9)]
        errors = []
        for x, y in points:
            lat, lon = truth(ds, x, y)
            flat_lat, flat_lon = pixel_to_latlon(x, y, center, (clat, clon), mpp)
            _, _, d = GEOD.inv(lon, lat, flat_lon, flat_lat)
            errors.append(float(d))
        return {"chip": chip, "crs": str(ds.crs), "mpp_local": mpp,
                "max_error_m": max(errors), "median_error_m": float(np.median(errors)),
                "grid_error_m": errors, "width": w, "height": h}


if __name__ == "__main__":
    chips = [r["chip"] for r in json.loads(BENCH.read_text())["per_chip"]]
    records = [evaluate(chip) for chip in chips]
    maxes = [r["max_error_m"] for r in records]
    result = {"chips": len(records), "max_error_median": float(np.median(maxes)),
              "max_error_p95": float(np.percentile(maxes, 95)),
              "max_error_global": max(maxes), "per_chip": records}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "error_grid.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_chip"}, indent=2))
