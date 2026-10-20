"""Run fixed, ground-truth segmentation baselines on the official Sen1Floods11 test split.

The dataset files are cached outside Git. No label is used to tune a threshold or
choose a class. Water is the low-backscatter S1 cluster. The optional S2 checks
are run only where a corresponding optical chip exists. The production HSV
operator is reported as inapplicable when its red-click sampler finds no click.
"""
import csv
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from common import simulated_clicks  # noqa: E402
from src.vision.image_processing import ImageProcessor  # noqa: E402

BASE = "https://storage.googleapis.com/sen1floods11/v1.1"
CACHE = HERE / "cache" / "sen1floods11"
OUT = HERE / "results" / "sen1floods11"
SPLIT = "splits/flood_handlabeled/flood_test_data.csv"


def fetch(rel):
    target = CACHE / rel
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        data = urlopen(f"{BASE}/{rel}", timeout=90).read()
        target.write_bytes(data)
    return target


def task_files():
    rows = list(csv.reader(io.StringIO(fetch(SPLIT).read_text())))
    for s1_name, label_name in rows:
        stem = s1_name.removesuffix("_S1Hand.tif")
        prefix = "data/flood_events/HandLabeled"
        yield stem, (f"{prefix}/S1Hand/{s1_name}",
                     f"{prefix}/LabelHand/{label_name}",
                     f"{prefix}/S2Hand/{stem}_S2Hand.tif")


def stats(pred, truth, valid):
    p, t = pred[valid], truth[valid]
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    return tp, fp, fn


def scores(counts):
    tp, fp, fn = counts
    def ratio(a, b):
        return a / b if b else None
    return {"iou": ratio(tp, tp + fp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn)}


def optical_rgb(s2):
    # S2 bands B4, B3, B2 as an ordinary false-display input for the red-click
    # operator; reflectances are fixed-scaled without looking at the label.
    return np.clip(np.stack([s2[3], s2[2], s2[1]], axis=-1) / 3000 * 255,
                   0, 255).astype(np.uint8)[:, :, ::-1]


def run_chip(item):
    stem, rels = item
    s1_path, label_path = (fetch(x) for x in rels[:2])
    with rasterio.open(s1_path) as ds:
        s1 = ds.read()
        profile = {"crs": str(ds.crs), "bounds": list(ds.bounds), "transform": list(ds.transform)[:6]}
    with rasterio.open(label_path) as ds:
        label = ds.read(1)
    valid = (label == 0) | (label == 1)
    truth = label == 1
    vh = s1[1].astype(np.float32)
    finite = np.isfinite(vh) & valid
    if not finite.any():
        return {"chip": stem, "excluded": "no finite S1 pixels with valid labels"}
    lo, hi = np.percentile(vh[finite], [1, 99])
    clean_vh = np.nan_to_num(vh, nan=lo, posinf=hi, neginf=lo)
    gray = np.clip((clean_vh - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(gray[finite], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = gray < otsu_threshold
    samples = vh[finite].reshape(-1, 1)
    threshold = float(np.median(samples))
    initial = (samples >= threshold).astype(np.int32)
    if initial.min() == initial.max():
        initial = (samples >= (float(samples.min()) + float(samples.max())) / 2).astype(np.int32)
    km = np.zeros(vh.shape, dtype=bool)
    if initial.min() != initial.max():
        _, clusters, centers = cv2.kmeans(samples, 2, initial,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01),
            1, cv2.KMEANS_USE_INITIAL_LABELS)
        km[finite] = clusters.ravel() == int(np.argmin(centers))
    masks = {"S1_Otsu_VH": otsu, "S1_kmeans_VH": km}
    s2_present = False
    try:
        s2_path = fetch(rels[2])
        with rasterio.open(s2_path) as ds:
            s2 = ds.read()
        if s2.shape[1:] == label.shape:
            s2_present = True
            green, nir = s2[2].astype(float), s2[7].astype(float)
            ndwi = (green - nir) / (green + nir + 1e-9)
            masks["S2_NDWI_0"] = ndwi > 0
            rgb = optical_rgb(s2)
            lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
            luminance = lab[..., 0]
            valid_luminance = luminance[finite]
            cutoff, _ = cv2.threshold(valid_luminance, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masks["S2_RGB_Otsu_dark"] = luminance < cutoff
            fit = lab[::4, ::4].reshape(-1, 3).astype(np.float32)
            midpoint = float(np.median(fit[:, 0]))
            initial = (fit[:, 0] >= midpoint).astype(np.int32).reshape(-1, 1)
            if initial.min() != initial.max():
                _, _, centers = cv2.kmeans(fit, 2, initial,
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01),
                    1, cv2.KMEANS_USE_INITIAL_LABELS)
                dark = int(np.argmin(centers[:, 0]))
                distances = np.sum((lab.astype(np.float32)[:, :, None, :] -
                                    centers[None, None, :, :]) ** 2, axis=-1)
                masks["S2_Lab_kmeans_dark"] = np.argmin(distances, axis=-1) == dark
            clicks = simulated_clicks(rgb, 8)
            if clicks:
                operator = ImageProcessor()
                operator.image, operator.sample_points = rgb, clicks
                operator.compute_dynamic_hsv()
                masks["production_HSV_red_click"] = operator.mask_flood_areas(rgb) > 0
    except Exception as ex:
        if "404" not in str(ex):
            raise
    result = {"chip": stem, "location": stem.split("_")[0], "valid_px": int(valid.sum()),
              "water_px": int((truth & valid).sum()), "s2_present": s2_present,
              "georeference": profile,
              "confusion": {key: stats(mask, truth, valid & finite) for key, mask in masks.items()}}
    return result


def main():
    items = list(task_files())
    limit = int(os.environ.get("ADAPT_SEN1_LIMIT", "0"))
    if limit:
        items = items[:limit]
    with ThreadPoolExecutor(max_workers=6) as pool:
        all_rows = list(pool.map(run_chip, items))
    rows = [r for r in all_rows if "excluded" not in r]
    names = sorted(set().union(*(r["confusion"] for r in rows)))
    summary = {}
    bootstrap_rng = np.random.default_rng(2026)
    for name in names:
        group = [r for r in rows if name in r["confusion"]]
        total = [sum(r["confusion"][name][i] for r in group) for i in range(3)]
        macro = {k: float(np.mean([scores(r["confusion"][name])[k]
                                   for r in group if scores(r["confusion"][name])[k] is not None]))
                 for k in ("iou", "f1", "precision", "recall")}
        chip_ious = np.array([scores(r["confusion"][name])["iou"] for r in group
                              if scores(r["confusion"][name])["iou"] is not None], float)
        samples = bootstrap_rng.choice(chip_ious, size=(5000, len(chip_ious)), replace=True).mean(axis=1)
        summary[name] = {"chips": len(group), "micro": scores(total), "macro": macro,
                         "macro_iou_ci95": list(map(float, np.percentile(samples, [2.5, 97.5])))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_metrics.json").write_text(json.dumps({"split": SPLIT, "chips": len(rows),
        "excluded": [r for r in all_rows if "excluded" in r],
        "methods": summary, "per_chip": rows}, indent=2))
    print(json.dumps({"chips": len(rows), "excluded": len(all_rows) - len(rows),
                      "methods": summary}, indent=2))


if __name__ == "__main__":
    main()
