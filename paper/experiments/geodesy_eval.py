"""E7 - Geospatial verification of the production pixel->lat/lon conversion (pixel_to_latlon).

Independent oracle: tests/geodesy_reference.py (WGS84 Vincenty; imports no production code; itself
validated against published values by tests/test_geodesy_reference.py).
  (a) synthetic ground truth: 1000x1000 px, 10 m/px, reference (25 N, 82 E) at the geometric centre;
  (b) randomised round trip through a tests-only inverse (100,000 trials);
  (c) model error vs WGS84 at the configured and at the landmark-estimated map extents;
  (d) mission-row correspondence for the E1 missions (display frame and original-image frame);
  (e) resize consistency: one synthetic map through nine display widths (unmodified main()).
Output: results/geodesy/*.json|csv
"""
import csv
import json
import math
import os
import random
import statistics
import sys
import tempfile

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from src.mission.coordinates import pixel_to_latlon  # noqa: E402
from tests import geodesy_reference as ref  # noqa: E402

OUT = os.path.join(common.RESULTS, "geodesy")


def synthetic():
    W = H = 1000
    C, GEO, MPP = ((W - 1) / 2, (H - 1) / 2), (25.0, 82.0), 10.0
    pts = {"centre": C, "top-centre": (C[0], 0), "bottom-centre": (C[0], H - 1), "left-centre": (0, C[1]),
           "right-centre": (W - 1, C[1]), "top-left": (0, 0), "top-right": (W - 1, 0), "bottom-left": (0, H - 1),
           "bottom-right": (W - 1, H - 1)}
    rows = []
    for name, (px, py) in pts.items():
        lat, lon = pixel_to_latlon(px, py, C, GEO, MPP)
        mlat, mlon = ref.documented_model(px, py, C, GEO, MPP)
        tlat, tlon = ref.physical_ground_truth(px, py, C, GEO, MPP)
        e, n = ref.pixel_offset_meters(px, py, C, MPP)
        err, _ = ref.vincenty_inverse(lat, lon, tlat, tlon)
        d = math.hypot(e, n)
        rows.append({"point": name, "px": px, "py": py, "east_m": e, "north_m": n, "lat": lat, "lon": lon,
                     "wgs84_lat": tlat, "wgs84_lon": tlon, "conformance_deg": max(abs(lat - mlat), abs(lon - mlon)),
                     "error_m": err, "error_pct": 100 * err / d if d else 0.0,
                     "bound_m": ref.model_error_bound_m(e, n, GEO[0])})
    with open(os.path.join(OUT, "synthetic_ground_truth.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return {"max_conformance_deg": max(r["conformance_deg"] for r in rows),
            "max_error_m": max(r["error_m"] for r in rows), "max_error_pct": max(r["error_pct"] for r in rows),
            "all_within_bound": all(r["error_m"] <= r["bound_m"] for r in rows)}


def round_trip(n=100000):
    rng = random.Random(20261006)
    sizes = [(750, 529), (750, 748), (1000, 800), (999, 799), (4000, 3000)]
    scales = [0.05, 1.0, 2.592, 10.0, 93.0, 633.0]
    regions = {"mid-latitudes": lambda: (rng.uniform(-60, 60), rng.uniform(-170, 170)),
               "near poles": lambda: (rng.choice([-1, 1]) * rng.uniform(85, 89.9), rng.uniform(-180, 180)),
               "near antimeridian": lambda: (rng.uniform(-60, 60), rng.choice([-1, 1]) * rng.uniform(179.5, 180)),
               "equator/prime meridian": lambda: (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5))}
    errs, rejected, by_region = [], 0, {k: [] for k in regions}
    for i in range(n):
        region = list(regions)[i % 4]
        W, H = rng.choice(sizes)
        C = ((W - 1) / 2, (H - 1) / 2)
        mpp, geo = rng.choice(scales), regions[region]()
        px, py = rng.uniform(-0.1 * W, 1.1 * W), rng.uniform(-0.1 * H, 1.1 * H)
        try:
            lat, lon = pixel_to_latlon(px, py, C, geo, mpp)
        except ValueError:
            rejected += 1
            continue
        rx, ry = ref.documented_model_inverse(lat, lon, C, geo, mpp)
        e = math.hypot(rx - px, ry - py)
        errs.append(e)
        by_region[region].append(e)
    return {"trials": n, "compared": len(errs), "rejected_unrepresentable": rejected, "max_px": max(errs),
            "mean_px": statistics.fmean(errs), "rms_px": math.sqrt(statistics.fmean(x * x for x in errs)),
            "max_px_by_region": {k: max(v) for k, v in by_region.items()}}


def model_error_at_extents():
    out = {}
    for name, W, H, mpp_orig, s in [("configured 2.0 m/px (Varanasi display)", 750, 748, 2.0, 750 / 972),
                                    ("Varanasi at estimated scale", 750, 748, 93.0, 750 / 972),
                                    ("Kanpur at estimated scale", 750, 742, 130.0, 750 / 976),
                                    ("Assam at estimated scale", 750, 529, 633.0, 750 / 1048)]:
        C, k = ((W - 1) / 2, (H - 1) / 2), mpp_orig / s
        worst = 0.0
        for px, py in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1), (C[0], 0), (0, C[1])]:
            lat, lon = pixel_to_latlon(px, py, C, (25.3176, 82.9739), k)
            t = ref.physical_ground_truth(px, py, C, (25.3176, 82.9739), k)
            worst = max(worst, ref.vincenty_inverse(lat, lon, *t)[0])
        out[name] = {"half_diagonal_km": math.hypot(W, H) / 2 * k / 1000, "worst_model_error_m": worst}
    return out


def correspondence():
    out = {}
    for m in common.MAPS:
        rec = json.load(open(os.path.join(common.RESULTS, "e2e", f"{m}.json")))
        side = json.load(open(os.path.join(common.RESULTS, "e2e", f"{m}.sidecar.json")))
        with open(os.path.join(common.REPO, rec["mission"])) as f:
            rows = [l.split("\t") for l in f.read().splitlines()[1:]]
        (H0, W0), (H1, W1) = rec["original_shape"], rec["display_shape"]
        sx, sy = W1 / W0, H1 / H0
        home, path, drops = tuple(side["home"]), [tuple(p) for p in side["full_path"]], set(side["drop_indices"])
        src = [home, home]
        for i, pt in enumerate(path):
            if i == 0 and pt == home:
                continue
            src.append(pt)
            if i in drops and 0 < i < len(path) - 1:
                src += [pt] * 4
        src += [home, home]
        assert len(src) == len(rows)
        da = db = 0.0
        for (x1, y1), r in zip(src, rows):
            lat, lon = float(r[8]), float(r[9])
            e = ref.documented_model(x1, y1, side["image_center_px"], side["geo_center"], side["meters_per_pixel"])
            da = max(da, abs(lat - e[0]), abs(ref.wrap_lon(lon - e[1])))
            x0, y0 = (x1 + 0.5) / sx - 0.5, (y1 + 0.5) / sy - 0.5
            o = ref.documented_model(x0, y0, ((W0 - 1) / 2, (H0 - 1) / 2), side["geo_center"], 2.0)
            db = max(db, ref.vincenty_inverse(lat, lon, *o)[0])
        out[m] = {"rows": len(rows), "display_frame_max_dev_deg": da, "original_frame_max_dev_m": db,
                  "sy_over_sx_minus_1_pct": 100 * (sy / sx - 1)}
    return out


def resize_attack():
    from tests.test_resize_consistency import run_pipeline
    W0, H0, MPP, GEO = 1200, 900, 10.0, (25.0, 82.0)
    C0 = ((W0 - 1) / 2, (H0 - 1) / 2)
    SQ = [(100, 100, 249, 249), (900, 150, 1049, 299), (500, 650, 649, 799)]
    truth = {(x, y): ref.physical_ground_truth(x, y, C0, GEO, MPP) for a, b, c, d in SQ
             for x, y in [(a, b), (c, b), (a, d), (c, d)]}
    model = {k: ref.documented_model(*k, C0, GEO, MPP) for k in truth}
    rows = []
    with tempfile.TemporaryDirectory() as t:
        img = np.full((H0, W0, 3), 255, np.uint8)
        for a, b, c, d in SQ:
            img[b:d + 1, a:c + 1] = (0, 0, 255)
        p = os.path.join(t, "synthetic.png")
        cv2.imwrite(p, img)
        for width in (300, 301, 600, 601, 900, 901, 1200, 1600, 2400):
            cap = run_pipeline(p, width, t)
            lines = open(cap["file"]).read().splitlines()[1:]
            drops = [(float(l.split("\t")[8]), float(l.split("\t")[9])) for l in lines if l.split("\t")[3] == "183"]
            keys = [min(truth, key=lambda k: ref.vincenty_inverse(*d, *truth[k])[0]) for d in drops]
            vs_model = max(ref.vincenty_inverse(*d, *model[k])[0] for d, k in zip(drops, keys))
            vs_truth = max(ref.vincenty_inverse(*d, *truth[k])[0] for d, k in zip(drops, keys))
            ratios = []
            for i in range(len(drops)):
                for j in range(i + 1, len(drops)):
                    ratios.append(ref.vincenty_inverse(*drops[i], *drops[j])[0] /
                                  ref.vincenty_inverse(*model[keys[i]], *model[keys[j]])[0] - 1)
            rows.append({"display_width": width, "metres_per_display_px": cap["meters_per_pixel"],
                         "drops": len(drops), "matched_corners": sorted(keys),
                         "max_drop_dev_from_model_m": vs_model, "max_drop_dev_from_wgs84_m": vs_truth,
                         "max_abs_spacing_error_pct": 100 * max(abs(r) for r in ratios)})
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    results = {"synthetic": synthetic(), "round_trip": round_trip(), "model_error_at_extents": model_error_at_extents(),
               "correspondence": correspondence()}
    print(json.dumps(results, indent=1, default=float))
    rs = resize_attack()
    results["resize"] = rs
    for r in rs:
        print(r)
    common.save_json(results, "geodesy", "geodesy_eval.json")
