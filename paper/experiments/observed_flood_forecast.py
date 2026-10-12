"""Compare the project's flood proxy with consecutive CEMS flood delineations.

EMSR504 AOI01 has an initial 19 March 2021 flood polygon map and monitoring
updates. Each pair is rasterised on its shared mapped AOI in UTM 56S at 30 m.
Open-Meteo archive weather at the initial acquisition hour is used as the
pipeline input. Persistence is the no-prediction baseline. Product footprints
are intersected to avoid counting areas not mapped at both times.
"""
import json
import math
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio.features
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import transform_geom

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.weather.flood_spread import predict_spread  # noqa: E402

CACHE = ROOT / "paper/experiments/cache/emsr504"
OUT = ROOT / "paper/experiments/results/forecast"
WEATHER = ROOT / "paper/experiments/reference/emsr504_weather.json"
PRODUCTS = [("initial", "2021-03-19T19:07"), ("monitor1", "2021-03-21T19:07"),
            ("monitor2", "2021-03-22T20:01")]
SOURCE = "https://mapping.emergency.copernicus.eu/activations/EMSR504/"
CRS = "EPSG:32756"
CELL_M = 30


def get_geojson(which, layer):
    with zipfile.ZipFile(CACHE / f"{which}.zip") as z:
        name = next(x for x in z.namelist() if layer in x and x.endswith(".json"))
        return json.loads(z.read(name))


def geometry(ds):
    return [transform_geom("EPSG:4326", CRS, f["geometry"])
            for f in ds["features"]]


def setup_grid():
    aois = [geometry(get_geojson(x, "areaOfInterestA")) for x, _ in PRODUCTS]
    # Grid spans the initial product AOI; per-pair valid pixels are the AOI intersection.
    bounds = [rasterio.features.bounds(g) for g in aois[0]]
    left = min(b[0] for b in bounds); bottom = min(b[1] for b in bounds)
    right = max(b[2] for b in bounds); top = max(b[3] for b in bounds)
    width = math.ceil((right - left) / CELL_M)
    height = math.ceil((top - bottom) / CELL_M)
    transform = from_origin(left, top, CELL_M, CELL_M)
    return (height, width), transform, aois


def raster(shapes, grid, transform):
    return rasterio.features.rasterize(shapes, out_shape=grid, transform=transform,
                                       fill=0, default_value=1, dtype="uint8").astype(bool)


def score(pred, obs, valid):
    tp = int(np.count_nonzero(pred & obs & valid))
    fp = int(np.count_nonzero(pred & ~obs & valid))
    fn = int(np.count_nonzero(~pred & obs & valid))
    return {"tp": tp, "fp": fp, "fn": fn,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def main():
    grid, transform, aois = setup_grid()
    masks = [raster(geometry(get_geojson(x, "observedEventA")), grid, transform)
             for x, _ in PRODUCTS]
    aoi_masks = [raster(x, grid, transform) for x in aois]
    w = json.loads(WEATHER.read_text())["hourly"]
    result = []
    for i in range(len(PRODUCTS) - 1):
        start, t0 = PRODUCTS[i]
        end, t1 = PRODUCTS[i + 1]
        valid = aoi_masks[i] & aoi_masks[i + 1]
        index = w["time"].index(t0[:13] + ":00")
        weather = {key: w[key][index] for key in ("precipitation", "wind_speed_10m", "wind_direction_10m")}
        horizon = (datetime.fromisoformat(t1) - datetime.fromisoformat(t0)).total_seconds() / 3600
        current = masks[i].astype(np.uint8) * 255
        predicted = predict_spread(current, weather, horizon) > 0
        row = {"start": start, "end": end, "start_acquisition_utc": t0,
               "end_acquisition_utc": t1, "horizon_hours": horizon,
               "weather_at_start": weather, "valid_pixels": int(valid.sum()),
               "observed_current_pixels": int((masks[i] & valid).sum()),
               "observed_next_pixels": int((masks[i + 1] & valid).sum()),
               "predicted_pixels": int((predicted & valid).sum()),
               "persistence": score(masks[i], masks[i + 1], valid),
               "proxy": score(predicted, masks[i + 1], valid)}
        result.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    output = {"source": SOURCE, "utm_crs": CRS, "cell_m": CELL_M, "grid": grid,
              "pairs": result}
    (OUT / "emsr504_aoi01.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
