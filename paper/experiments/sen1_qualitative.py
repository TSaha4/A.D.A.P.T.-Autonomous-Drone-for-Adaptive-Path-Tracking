"""Save held-out visual examples with exact benchmark baselines and U-Net."""
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from torch.nn import functional as F

from sen1_unet import CACHE, OUT, UNet


def optical_rgb(s2):
    return np.clip(np.stack([s2[3], s2[2], s2[1]], axis=-1) / 3000 * 255,
                   0, 255).astype(np.uint8)[:, :, ::-1]

CHIPS = ("Sri-Lanka_713926", "Ghana_319168")


def main():
    model = UNet()
    model.load_state_dict(torch.load(CACHE / "unet_best_state.pt", map_location="cpu", weights_only=True))
    model.eval()
    root = CACHE / "data/flood_events/HandLabeled"
    for chip in CHIPS:
        with rasterio.open(root / "S1Hand" / f"{chip}_S1Hand.tif") as ds:
            s1 = ds.read().astype(np.float32)
        with rasterio.open(root / "S2Hand" / f"{chip}_S2Hand.tif") as ds:
            s2 = ds.read()
        with rasterio.open(root / "LabelHand" / f"{chip}_LabelHand.tif") as ds:
            truth = ds.read(1)
        valid = (truth == 0) | (truth == 1)
        vh = s1[1]
        finite = valid & np.isfinite(vh)
        lo, hi = np.percentile(vh[finite], [1, 99])
        gray = np.clip((np.nan_to_num(vh, nan=lo, posinf=hi, neginf=lo) - lo) /
                       max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        threshold, _ = cv2.threshold(gray[finite], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu = gray < threshold
        rgb = optical_rgb(s2)
        lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
        fit = lab[::4, ::4].reshape(-1, 3).astype(np.float32)
        initial = (fit[:, 0] >= np.median(fit[:, 0])).astype(np.int32).reshape(-1, 1)
        _, _, centers = cv2.kmeans(fit, 2, initial,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01),
            1, cv2.KMEANS_USE_INITIAL_LABELS)
        d = np.sum((lab.astype(np.float32)[:, :, None, :] - centers[None, None, :, :]) ** 2, axis=-1)
        lab_pred = np.argmin(d, axis=-1) == int(np.argmin(centers[:, 0]))
        ndwi = (s2[2].astype(float) - s2[7]) / (s2[2].astype(float) + s2[7] + 1e-9) > 0
        image = np.clip((np.nan_to_num(s1, nan=-30, posinf=5, neginf=-30) + 30) / 35, 0, 1)
        with torch.no_grad():
            x = F.interpolate(torch.from_numpy(image)[None], size=(128, 128), mode="area")
            logits = F.interpolate(model(x), size=truth.shape, mode="bilinear", align_corners=False)
            unet = logits.argmax(1)[0].numpy().astype(bool)
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"qualitative_{chip}.npz", rgb=rgb, truth=truth,
                            valid=valid, otsu=otsu, lab=lab_pred, ndwi=ndwi, unet=unet)
        print(chip, "saved", flush=True)


if __name__ == "__main__":
    main()
