"""Run the pushed planner on a georeferenced, hand-labelled flood annotation.

The hand label is rendered as a red annotation map because that is the input
contract of the existing HSV pipeline. This tests route/mission geography only:
it must never be counted as independent segmentation accuracy. The exported
script runs both the scalar-m/px and optional GeoTIFF converters and
compares both missions against the reference affine/CRS coordinates.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Geod, Transformer

import common
from src.mission.coordinates import pixel_to_latlon

ROOT = Path(common.REPO)
CACHE = ROOT / "paper/experiments/cache/sen1floods11/data/flood_events/HandLabeled"
OUT = ROOT / "paper/experiments/results/georeference"
CHIP = "Ghana_319168"
GEOD = Geod(ellps="WGS84")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(CACHE / "LabelHand" / f"{CHIP}_LabelHand.tif") as ds:
        label = ds.read(1)
    with rasterio.open(CACHE / "S1Hand" / f"{CHIP}_S1Hand.tif") as ds:
        transform = ds.transform
        crs = ds.crs
    # A saturated blue background avoids the production sampler's known
    # achromatic-white false positive; red remains a visual annotation layer.
    image = np.empty((*label.shape, 3), np.uint8)
    image[:] = (200, 70, 30)
    image[label == 1] = (0, 0, 255)
    image[label == -1] = (180, 180, 180)
    path = OUT / f"{CHIP}_annotation.png"
    cv2.imwrite(str(path), image)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    def geo(x, y):
        mx, my = transform * (x + 0.5, y + 0.5)
        lon, lat = to_wgs.transform(mx, my)
        return lat, lon
    h, w = label.shape
    center = ((w - 1) / 2, (h - 1) / 2)
    clat, clon = geo(*center)
    elat, elon = geo(center[0] + 1, center[1])
    slat, slon = geo(center[0], center[1] + 1)
    _, _, ew = GEOD.inv(clon, clat, elon, elat)
    _, _, ns = GEOD.inv(clon, clat, slon, slat)
    mpp = (abs(ew) + abs(ns)) / 2
    rec = common.run(image_path=str(path), out_dir=str(OUT), tag=CHIP,
                     display_width=w, mpp=mpp, geo_center=(clat, clon),
                     weather={"precipitation": 0.0, "wind_speed_10m": 0.0,
                              "wind_direction_10m": 0.0}, predict=False)
    georef_rec = common.run(image_path=str(path), out_dir=str(OUT), tag=CHIP + "_geotiff",
                     display_width=w, geotiff_reference=CACHE / "S1Hand" / f"{CHIP}_S1Hand.tif",
                     weather={"precipitation": 0.0, "wind_speed_10m": 0.0,
                              "wind_direction_10m": 0.0}, predict=False)
    (OUT / f"{CHIP}_scalar.json").write_text(json.dumps(rec, indent=2))
    (OUT / f"{CHIP}_geotiff.json").write_text(json.dumps(georef_rec, indent=2))
    errors = []
    for x, y in rec.get("full_path", []):
        tlat, tlon = geo(x, y)
        plat, plon = pixel_to_latlon(x, y, center, (clat, clon), mpp)
        _, _, d = GEOD.inv(tlon, tlat, plon, plat)
        errors.append(float(d))
    georef_errors = []
    # Compare the georeferenced mission's actual nav items with the exact
    # affine/CRS positions of their corresponding route points.
    rows = [line.split() for line in Path(georef_rec["mission"]).read_text().splitlines()[1:]]
    nav = [(float(r[8]), float(r[9])) for r in rows if r[3] == "16"]
    path_geo = [geo(x, y) for x, y in georef_rec["full_path"]]
    # HOME and repeated route points can yield extra navigation commands;
    # compare every command position with the closest ground-truth route point.
    for lat, lon in nav:
        georef_errors.append(min(GEOD.inv(lon, lat, p[1], p[0])[2] for p in path_geo))
    result = {"chip": CHIP, "input": "hand-labelled water mask rendered as red annotation",
              "note": "ground-truth label used as planner input; not a segmentation evaluation",
              "crs": str(crs), "center_wgs84": [clat, clon], "local_average_mpp": mpp,
              "route_points": len(errors), "drops": len(rec.get("safe_points", [])),
              "no_path_legs": rec.get("no_path_legs"), "exit": rec.get("exit"),
              "max_route_geographic_error_m": max(errors) if errors else None,
              "median_route_geographic_error_m": float(np.median(errors)) if errors else None,
              "geotiff_mission_exit": georef_rec.get("exit"),
              "geotiff_mission_items": georef_rec.get("mission_items"),
              "geotiff_max_nav_error_m": max(georef_errors) if georef_errors else None}
    (OUT / "mission_demo.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
