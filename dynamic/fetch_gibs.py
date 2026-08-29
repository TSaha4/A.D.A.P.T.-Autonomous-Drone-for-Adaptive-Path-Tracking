#!/usr/bin/env python3
"""Dynamic A.D.A.P.T. input generator for ANY region — no API keys required.

NASA GIBS WMTS (daily MODIS Bands-721, open access)
  -> water change-detection vs a dry baseline date (water absorbs SWIR: dark
     in 721 regardless of sediment; missing tiles tracked and excluded)
  -> red-dot overlay on OSM basemap (India-Today news style)
  -> flood_mask.png + style_map.png + geo.json, ready for the A.D.A.P.T.
     pipeline (pixel_to_latlon gets real coordinates).

Usage:
  python fetch_gibs.py --name varanasi --lat 25.32 --lon 82.97 \
      --span 0.55 --flood-date 2025-07-28 --dry-date 2025-05-01
"""
import argparse, json, math, os, sys, time
import urllib.request

import cv2
import numpy as np

OVERPASS = "https://overpass-api.de/api/interpreter"
UA = {"User-Agent": "ADAPT-dynamic-input/1.0 (research)"}
GIBS = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
OSM = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
LAYER = "MODIS_Terra_CorrectedReflectance_Bands721"  # water is dark even if sediment-laden
Z = 9  # ~300 m/px — river-flood scale
MIN_COVER = 0.60   # min fraction of valid pixels for a usable scene
MAX_BLOB = 0.25    # drop blobs larger than this fraction of the image (artifacts)


def ll_to_tile(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def tile_bounds(x, y, z):
    n = 2 ** z
    lon1, lon2 = x / n * 360 - 180, (x + 1) / n * 360 - 180
    lat2 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lat1, lat2, lon1, lon2


def fetch(url, tries=3):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception:
            if t == tries - 1:
                return None
            time.sleep(1.5 * (t + 1))
    return None


def mosaic(bbox, date, cache):
    """Returns (image, valid_mask) or None; valid=0 where tiles are missing."""
    lat_min, lat_max, lon_min, lon_max = bbox
    x0, y0 = ll_to_tile(lat_max, lon_min, Z)
    x1, y1 = ll_to_tile(lat_min, lon_max, Z)
    imgs, valids = [], []
    for y in range(y0, y1 + 1):
        row, vrow = [], []
        for x in range(x0, x1 + 1):
            key = (date, Z, x, y)
            if key not in cache:
                raw = fetch(GIBS.format(layer=LAYER, date=date, z=Z, x=x, y=y))
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
                cache[key] = img
            img = cache[key]
            row.append(img if img is not None else np.zeros((256, 256, 3), np.uint8))
            vrow.append(np.full((256, 256), 255 if img is not None else 0, np.uint8))
        imgs.append(np.hstack(row))
        valids.append(np.hstack(vrow))
    return np.vstack(imgs), np.vstack(valids)


def water_mask(img, valid):
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
    m = ((v < 95) & (valid > 0)).astype(np.uint8) * 255
    return m


def new_water_by_diff(vd, vf, valid_d, valid_f, land):
    """Continuous change detection vs same-season baseline: pixel got notably
    darker AND is now very dark (open water is <~90 in 721, even sediment-laden;
    bare rock/veg after snowmelt rarely is). Land/valid masked."""
    new = ((vd - vf) > 35) & (vf < 100) & (land > 0) & (valid_d > 0) & (valid_f > 0)
    return (new.astype(np.uint8)) * 255


def cloud_fraction(img, valid):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, vl = hsv[..., 1], hsv[..., 2]
    white = (vl > 195) & (s < 35) & (valid > 0)
    return float(white.sum()) / max(1, float((valid > 0).sum()))


def river_corridor_mask(shape, bbox, px_per_m, width_m=600):
    """1 inside a ~width_m buffer around mapped rivers/streams (OSM Overpass, no auth)."""
    lat_min, lat_max, lon_min, lon_max = bbox
    q = (f'[out:json][timeout:60];(way["waterway"~"river|stream"]'
         f'({lat_min},{lon_min},{lat_max},{lon_max}););out geom;')
    try:
        from urllib.parse import quote
        raw = fetch(OVERPASS + "?data=" + quote(q), 2)
        import json as _json
        data = _json.loads(raw)
    except Exception:
        return np.ones(shape, np.uint8)  # fail-open
    mask = np.zeros(shape, np.uint8)
    for el in data.get("elements", []):
        g = el.get("geometry") or []
        pts = [(int((p["lon"] - lon_min) / (lon_max - lon_min) * shape[1]),
                int((lat_max - p["lat"]) / (lat_max - lat_min) * shape[0])) for p in g]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(mask, a, b, 255, 1)
    if not (mask > 0).any():
        return np.ones(shape, np.uint8)
    k = max(1, int(round((width_m / 2) / px_per_m)))
    if k > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return mask


def drop_huge_blobs(mask, area_px):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for c in cnts:
        if area_px * MAX_BLOB >= cv2.contourArea(c) >= 80:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


def scene_candidates(bbox, center_date, window, cache):
    from datetime import date as D, timedelta
    d0 = D.fromisoformat(center_date)
    out = []
    for off in range(-window, window + 1, 2):
        ds = (d0 + timedelta(days=off)).isoformat()
        r = mosaic(bbox, ds, cache)
        if r is None:
            continue
        img, valid = r
        cover = float((valid > 0).mean())
        cf = cloud_fraction(img, valid)
        print(f"  {ds}: coverage {cover*100:.0f}%, cloud {cf*100:.0f}%")
        if cover >= MIN_COVER:
            out.append((ds, cf, img, valid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--span", type=float, default=0.55)
    ap.add_argument("--flood-date", required=True)
    ap.add_argument("--dry-date", default=None,
                    help="baseline date; DEFAULT: auto same-season (12 days before flood date) — avoids snow/veg seasonal false positives. Pass explicitly for a winter baseline.")
    ap.add_argument("--coast-lon", type=float, default=None,
                    help="exclude ocean: ignore pixels east of this longitude")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(os.path.dirname(__file__), "out", args.name)
    os.makedirs(out, exist_ok=True)
    dlat, dlon = args.span, args.span / max(0.3, math.cos(math.radians(args.lat)))
    bbox = (args.lat - dlat, args.lat + dlat, args.lon - dlon, args.lon + dlon)
    cache = {}

    from datetime import date as _D, timedelta as _TD
    if args.dry_date is None:
        args.dry_date = (_D.fromisoformat(args.flood_date) - _TD(days=12)).isoformat()
        print(f"auto same-season baseline: {args.dry_date}")
    print("dry-date candidates:", flush=True)
    cands_d = scene_candidates(bbox, args.dry_date, 6, cache)
    if not cands_d:
        sys.exit("no usable dry-date scene")
    dry_date, _, dry_img, dry_valid = min(cands_d, key=lambda t: t[1])
    print(f"dry scene: {dry_date}")
    vd = cv2.cvtColor(dry_img, cv2.COLOR_BGR2HSV)[..., 2].astype(np.int16)
    land = np.ones(dry_img.shape[:2], np.uint8)
    if args.coast_lon is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        col = int((args.coast_lon - lon_min) / (lon_max - lon_min) * dry_img.shape[1])
        land[:, col:] = 0
    land &= (dry_valid > 0)

    print("flood-date candidates (scored by new-water px):", flush=True)
    cands_f = scene_candidates(bbox, args.flood_date, 6, cache)
    if not cands_f:
        sys.exit("no usable flood-date scene")
    px_per_m = 156543.03 * math.cos(math.radians(args.lat)) / (2 ** Z)
    corridor = river_corridor_mask(dry_img.shape[:2], bbox, px_per_m)
    print(f"river-corridor constraint: keeps {100*(corridor>0).mean():.0f}% of scene")
    best = None
    for ds, cf, img, valid in cands_f:
        if cf > 0.55:
            continue
        vf = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].astype(np.int16)
        nw = drop_huge_blobs(new_water_by_diff(vd, vf, dry_valid, valid, land), img.shape[0] * img.shape[1])
        nw = cv2.bitwise_and(nw, corridor)
        n = int((nw > 0).sum())
        print(f"  {ds}: new-water {n}px")
        if best is None or n > best[1]:
            best = (ds, n, img, valid, nw)
    if best is None or best[1] == 0:
        sys.exit("no flood signal detected in any scene (all cloudy or dry)")
    flood_date, _, flood_img, valid, flood_mask = best
    print(f"flood scene: {flood_date} ({(flood_mask>0).sum()} px new water)")

    # center-crop to requested aspect
    H, W = flood_img.shape[:2]
    target_ar = dlon / (2 * dlat) * 0.95
    cw = min(W, int(H * target_ar))
    x0c = (W - cw) // 2
    flood_img = flood_img[:, x0c:x0c + cw]
    dry_img = dry_img[:, x0c:x0c + cw]
    flood_mask = flood_mask[:, x0c:x0c + cw]
    if args.coast_lon is None and (corridor < 1).any():
        pass  # corridor already cropped below
    corridor_crop = corridor[:, x0c:x0c + cw]
    cv2.imwrite(os.path.join(out, "satellite.png"), flood_img)
    cv2.imwrite(os.path.join(out, "dry_scene.png"), dry_img)
    cv2.imwrite(os.path.join(out, "flood_mask.png"), flood_mask)
    km2 = float((flood_mask > 0).sum()) * (px_per_m ** 2) / 1e6
    print(f"flood area (corridor-constrained): {km2:.1f} km2")

    # OSM basemap + red-dot overlay (news style)
    x0t, y0t = ll_to_tile(bbox[1], bbox[2], Z)
    x1t, y1t = ll_to_tile(bbox[0], bbox[3], Z)
    rows = []
    for y in range(y0t, y1t + 1):
        row = []
        for x in range(x0t, x1t + 1):
            raw = fetch(OSM.format(z=Z, x=x, y=y))
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
            row.append(img if img is not None else np.full((256, 256, 3), 245, np.uint8))
        rows.append(np.hstack(row))
    base = np.vstack(rows)
    base = base[:, (base.shape[1] - cw) // 2:(base.shape[1] + cw) // 2]
    base = cv2.resize(base, (flood_img.shape[1], flood_img.shape[0]))

    style = cv2.addWeighted(base, 0.75, np.full_like(base, 255), 0.25, 0)
    style[flood_mask > 0] = (0, 0, 230)
    cnts, _ = cv2.findContours(flood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(style, cnts, -1, (0, 0, 160), 1)
    cv2.putText(style, f"Source: NASA GIBS MODIS 721 {flood_date} vs {dry_date}; A.D.A.P.T. water-det",
                (12, style.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out, "style_map.png"), style)

    geo = {"lat_center": args.lat, "lon_center": args.lon,
           "lat_top": tile_bounds(x0t, y0t, Z)[1], "lat_bottom": tile_bounds(x0t, y1t + 1, Z)[0],
           "lon_left": tile_bounds(x0t, y0t, Z)[2], "lon_right": tile_bounds(x1t + 1, y0t, Z)[3],
           "meters_per_pixel": 156543.03 * math.cos(math.radians(args.lat)) / (2 ** Z),
           "flood_date": flood_date, "dry_date": dry_date,
           "size_px": [int(flood_img.shape[1]), int(flood_img.shape[0])]}
    with open(os.path.join(out, "geo.json"), "w") as f:
        json.dump(geo, f, indent=2)
    print("done ->", out)


if __name__ == "__main__":
    main()
