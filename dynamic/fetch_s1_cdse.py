#!/usr/bin/env python3
"""ACCURATE flood input via Sentinel-1 SAR — Copernicus Data Space (openEO).

One-time setup:
  1. Register (free): https://dataspace.copernicus.eu/  -> Create account
  2. pip install openeo rioxarray
  3. First run opens a browser (OIDC device flow) and caches the token.

Then, for ANY region/date:
  python fetch_s1_cdse.py --name nepal2026 --lat 28.05 --lon 85.15 --span 0.3 \
      --flood-date 2026-08-26 --dry-date 2026-08-10

Outputs (dynamic_input/out/<name>/):
  s1_flood_mask.tif  (georeferenced binary water: 1 = new floodwater)
  flood_mask.png     (binary mask, ready for run_pipeline_on_mask.py)
  style_map.png      (red-dot news style over OSM — feeds the unchanged HSV pipeline too)
  geo.json           (bounds/mpp for pixel_to_latlon — EXACT, from the SAR geotransform)

Method: Sentinel-1 GRD VV backscatter (10 m, cloud-penetrating).
  water(flood window, min-composite) < -22 dB  AND  NOT water(dry window)
  -> new floodwater. Tune --vv-threshold per region (-20..-25).
"""
import argparse, json, math, os, sys
from urllib.parse import quote
from urllib.request import Request, urlopen

import cv2
import numpy as np

UA = {"User-Agent": "ADAPT-dynamic-input/1.0 (research)"}
OSM = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


def ll_to_tile(lat, lon, z):
    n = 2 ** z
    return int((lon + 180.0) / 360.0 * n), int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)


def tile_bounds_frac(x0, y0, z0, x1, y1, z1):
    n = 2 ** z1
    lon1, lon2 = x0 / n * 360 - 180, x1 / n * 360 - 180
    lat2 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y0 / n))))
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y1 / n))))
    return lat1, lat2, lon1, lon2


def fetch(url, tries=3):
    for t in range(tries):
        try:
            with urlopen(Request(url, headers=UA), timeout=30) as r:
                return r.read()
        except Exception:
            if t == tries - 1:
                return None
    return None


def window(date, days=3):
    from datetime import date as D, timedelta
    d = D.fromisoformat(date)
    return (d - timedelta(days=days)).isoformat(), (d + timedelta(days=days)).isoformat()


def s1_water_mask(lat, lon, span, flood_date, dry_date, vv_thr, out_dir):
    import openeo
    conn = openeo.connect("openeo.dataspace.copernicus.eu")
    env = {}
    envf = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cdse.env")
    if os.path.exists(envf):
        for line in open(envf):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                env[k] = v
    if env.get("CDSE_CLIENT_ID") and env.get("CDSE_CLIENT_SECRET"):
        print("authenticating via CDSE client credentials...")
        conn.authenticate_oidc_client_credentials(
            client_id=env["CDSE_CLIENT_ID"], client_secret=env["CDSE_CLIENT_SECRET"])
    else:
        print("no .cdse.env found; authenticating via browser device flow...")
        conn.authenticate_oidc()

    dlat = span
    dlon = span / max(0.3, math.cos(math.radians(lat)))
    spatial = {"west": lon - dlon, "south": lat - dlat, "east": lon + dlon, "north": lat + dlat}

    def water_of(date, label):
        t0, t1 = window(date)
        print(f"  {label} window {t0}..{t1}")
        cube = (conn.load_collection("SENTINEL1_GRD", spatial_extent=spatial,
                                     temporal_extent=[t0, t1], bands=["VV"])
                .sar_backscatter(coefficient="sigma0-ellipsoid"))
        vv = cube.band("VV")
        # min-composite over time: pixel flooded on ANY pass in the window -> water
        vv_min = vv.reduce_dimension(dimension="t", reducer="min")
        # CDSE returns sigma0 in LINEAR power: convert the dB threshold
        lin_thr = 10 ** (vv_thr / 10.0)
        return vv_min < lin_thr         # operator builds the `lt` process

    print("computing flood-window water (server-side)...")
    w_flood = water_of(flood_date, "flood")
    print("computing dry-window water (server-side)...")
    w_dry = water_of(dry_date, "dry")

    # binary diff: 1 iff flood-water AND NOT dry-water (arithmetic merges cleanly)
    flood_new = (w_flood - w_dry) > 0
    tif = os.path.join(out_dir, "s1_flood_mask.tif")
    import time as _time
    import time as _t2
    for att in range(5):
        try:
            job = flood_new.send_job(title="ADAPT S1 flood mask")
            job.start_job()
            break
        except Exception as e:
            if "429" in str(e) and att < 4:
                w2 = 90 * (att + 1)
                print(f"429 on job creation; cooling down {w2}s...", flush=True)
                _t2.sleep(w2)
            else:
                raise
    print("job:", job.job_id)
    import time as _time
    while True:
        st = job.status()
        print(_time.strftime("%H:%M:%S"), st, flush=True)
        if st in ("finished", "error", "canceled"):
            break
        _time.sleep(20)
    if st != "finished":
        sys.exit(f"batch job ended with status {st}")
    res = job.get_results()
    assets = res.get_assets()
    assets = list(assets.values()) if isinstance(assets, dict) else assets
    if not assets:
        sys.exit("job produced no assets")
    assets[0].download(tif)
    print("downloaded:", tif, os.path.getsize(tif), "bytes")

    import rioxarray
    ds = rioxarray.open_rasterio(tif, masked=True).squeeze()
    vals = ds.values
    mask = ((vals > 0) & ~np.ma.getmaskarray(vals)).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(out_dir, "flood_mask.png"), mask)

    t = ds.rio.transform()
    lat_top, lon_left = t.f, t.c
    mpp_x, mpp_y = abs(t.a), abs(t.e)
    h, w = mask.shape
    geo = {"lat_center": lat, "lon_center": lon,
           "lat_top": float(lat_top), "lon_left": float(lon_left),
           "meters_per_pixel": float((mpp_x + mpp_y) / 2),
           "lat_bottom": float(lat_top - mpp_y * h), "lon_right": float(lon_left + mpp_x * w),
           "flood_date": flood_date, "dry_date": dry_date,
           "vv_threshold_db": vv_thr,
           "size_px": [int(w), int(h)],
           "crs": str(ds.rio.crs)}
    with open(os.path.join(out_dir, "geo.json"), "w") as f:
        json.dump(geo, f, indent=2)
    return mask, geo


def style_map(mask, geo, out_png):
    w, h = mask.shape[1], mask.shape[0]
    z = 9
    lat_top, lat_bot = geo.get("lat_top"), geo.get("lat_bottom")
    lon_l, lon_r = geo.get("lon_left"), geo.get("lon_right")
    if None in (lat_top, lat_bot, lon_l, lon_r):
        lat_c, lon_c = geo["lat_center"], geo["lon_center"]
        mpp = geo["meters_per_pixel"]
        dlat = h * mpp / 111320.0
        dlon = w * mpp / (111320.0 * math.cos(math.radians(lat_c)))
        lat_top, lat_bot, lon_l, lon_r = lat_c + dlat, lat_c - dlat, lon_c - dlon, lon_c + dlon
    x0, y0 = ll_to_tile(lat_top, lon_l, z)
    x1, y1 = ll_to_tile(lat_bot, lon_r, z)
    rows = []
    for y in range(y0, y1 + 1):
        row = []
        for x in range(x0, x1 + 1):
            raw = fetch(OSM.format(z=z, x=x, y=y))
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
            row.append(img if img is not None else np.full((256, 256, 3), 245, np.uint8))
        rows.append(np.hstack(row))
    base = np.vstack(rows)
    # crop tile mosaic to exact geo bounds
    tb1 = tile_bounds_frac(x0, y0, z, x1 + 1, y1 + 1, z)
    c_x0 = int((lon_l - tb1[2]) / (tb1[3] - tb1[2]) * base.shape[1])
    c_x1 = int((lon_r - tb1[2]) / (tb1[3] - tb1[2]) * base.shape[1])
    c_y0 = int((tb1[1] - lat_top) / (tb1[1] - tb1[0]) * base.shape[0])
    c_y1 = int((tb1[1] - lat_bot) / (tb1[1] - tb1[0]) * base.shape[0])
    base = base[max(c_y0,0):c_y1, max(c_x0,0):c_x1]
    base = cv2.resize(base, (w, h))
    style = cv2.addWeighted(base, 0.75, np.full_like(base, 255), 0.25, 0)
    style[mask > 0] = (0, 0, 230)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(style, cnts, -1, (0, 0, 160), 1)
    cv2.putText(style, f"Source: Copernicus Sentinel-1 {geo['flood_date']} vs {geo['dry_date']}; <{geo['vv_threshold_db']}dB change-det",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_png, style)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--span", type=float, default=0.3)
    ap.add_argument("--flood-date", required=True)
    ap.add_argument("--dry-date", required=True)
    ap.add_argument("--vv-threshold", type=float, default=-22.0)
    args = ap.parse_args()

    out = os.path.join(os.path.dirname(__file__), "out", args.name)
    os.makedirs(out, exist_ok=True)
    mask, geo = s1_water_mask(args.lat, args.lon, args.span,
                              args.flood_date, args.dry_date, args.vv_threshold, out)
    style_map(mask, geo, os.path.join(out, "style_map.png"))
    km2 = float((mask > 0).sum()) * geo["meters_per_pixel"] ** 2 / 1e6
    print(f"done -> {out} | new water: {(mask>0).sum()} px = {km2:.2f} km2 @ {geo['meters_per_pixel']:.1f} m/px")
    if (mask > 0).sum() == 0:
        print("WARNING: empty mask — checking scene availability via STAC...")
        import urllib.request as _u
        for label, date in (("flood", flood_date), ("dry", dry_date)):
            t0, t1 = window(date)
            body = json.dumps({"collections": ["SENTINEL1_GRD"],
                               "bbox": [spatial["west"], spatial["south"], spatial["east"], spatial["north"]],
                               "datetime": f"{t0}/{t1}", "limit": 100}).encode()
            req = _u.Request("https://stac.dataspace.copernicus.eu/v1/search", data=body,
                             headers={"Content-Type": "application/json"})
            try:
                n = len(json.loads(_u.urlopen(req, timeout=60).read()).get("features", []))
                print(f"  {label} window {t0}..{t1}: {n} S1 scenes")
            except Exception as e:
                print(f"  {label} window STAC check failed: {e}")


if __name__ == "__main__":
    main()
