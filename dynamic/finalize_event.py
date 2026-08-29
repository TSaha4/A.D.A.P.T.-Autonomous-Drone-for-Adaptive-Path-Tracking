#!/usr/bin/env python3
"""Finalize one event after fetch_s1_cdse.py finished:
reproject -> mask/geo/style -> cleanup (>=4 ha blobs) -> resize for planner
-> mission (overlay incl. predicted-spread) -> TSP vs D* Lite comparison.
Usage: python finalize_event.py --name wayanad2024 --lat 11.755 --lon 76.08
                                --flood-date 2024-07-30 --dry-date 2024-07-08
                                [--sel-min 100] [--planner-min 25] [--top 3]
"""
import argparse, json, os, subprocess, sys

import cv2
import numpy as np
import rioxarray

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)          # repo root (this file lives in <repo>/dynamic/)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--flood-date", required=True)
    ap.add_argument("--dry-date", required=True)
    ap.add_argument("--sel-min", type=int, default=100)
    ap.add_argument("--planner-min", type=int, default=25)
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()

    out = os.path.join(HERE, "out", a.name)
    tif = os.path.join(out, "s1_flood_mask.tif")
    if not os.path.exists(tif):
        sys.exit(f"{tif} missing — fetch first")

    # ---- postprocess: WGS84, mask png, geo.json, style map ----
    ds = rioxarray.open_rasterio(tif, masked=True).squeeze()
    if ds.rio.crs and ds.rio.crs.to_epsg() != 4326:
        ds = ds.rio.reproject("EPSG:4326")
    vals = ds.values
    mask = ((vals > 0) & ~np.ma.getmaskarray(vals)).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(out, "flood_mask.png"), mask)
    t = ds.rio.transform(); h, w = mask.shape
    mpp = float(abs(t.e) * 111320.0)
    geo = {"lat_center": a.lat, "lon_center": a.lon,
           "lat_top": float(t.f), "lon_left": float(t.c),
           "lat_bottom": float(t.f - abs(t.e) * h), "lon_right": float(t.c + abs(t.a) * w),
           "meters_per_pixel": mpp, "flood_date": a.flood_date, "dry_date": a.dry_date,
           "vv_threshold_db": -22.0, "size_px": [int(w), int(h)], "crs": "EPSG:4326"}
    json.dump(geo, open(os.path.join(out, "geo.json"), "w"), indent=2)
    sys.path.insert(0, HERE)
    import fetch_s1_cdse as F
    F.style_map(mask, geo, os.path.join(out, "style_map.png"))
    km2 = float((mask > 0).sum()) * mpp * mpp / 1e6
    print(f"[{a.name}] raw new-water: {(mask>0).sum()} px = {km2:.2f} km2 @ {mpp:.1f} m/px")
    if (mask > 0).sum() == 0:
        sys.exit(f"[{a.name}] EMPTY mask — see fetch log for scene availability")

    # ---- cleanup: drop blobs < 4 ha ----
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask); kept = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 400:
            clean[labels == i] = 255; kept += 1
    cv2.imwrite(os.path.join(out, "flood_mask_clean.png"), clean)
    print(f"[{a.name}] blobs >= 4 ha: {kept}")

    # ---- resize for the planner ----
    run = os.path.join(out, "pipeline"); os.makedirs(run, exist_ok=True)
    W = 2000; H = int(h * W / w)
    small = cv2.resize(clean, (W, H), interpolation=cv2.INTER_AREA)
    small = (small > 63).astype(np.uint8) * 255
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(small, 8)
    outm = np.zeros_like(small); k2 = 0
    for i in range(1, n2):
        if st2[i, cv2.CC_STAT_AREA] >= a.planner_min:
            outm[lab2 == i] = 255; k2 += 1
    cv2.imwrite(os.path.join(run, "flood_mask.png"), outm)
    g2 = dict(geo); g2["meters_per_pixel"] = mpp * w / W; g2["size_px"] = [W, H]
    json.dump(g2, open(os.path.join(run, "geo.json"), "w"), indent=2)
    cv2.imwrite(os.path.join(run, "style_map.png"),
                cv2.resize(cv2.imread(os.path.join(out, "style_map.png")), (W, H)))

    # ---- mission (overlay incl. predicted-spread contours) ----
    venv_py = sys.executable
    r = subprocess.run([venv_py, os.path.join(HERE, "run_pipeline_on_mask.py"),
                        "--dir", run, "--min-area", str(a.planner_min)],
                       capture_output=True, text=True)
    print(r.stdout.strip()[-300:])

    # ---- TSP vs D* comparison on top-N zones ----
    r2 = subprocess.run([venv_py, os.path.join(HERE, "compare_tsp_dstar.py"),
                         "--name", a.name, "--sel-min", str(a.sel_min)],
                        capture_output=True, text=True)
    print("\n".join(l for l in r2.stdout.splitlines()
                    if l.startswith(("zone", "TSP", "D*"))))


if __name__ == "__main__":
    main()
