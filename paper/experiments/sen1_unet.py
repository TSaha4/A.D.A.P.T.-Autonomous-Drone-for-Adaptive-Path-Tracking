"""Train a small U-Net on Sen1Floods11 train/valid; evaluate once on test.

Run with Python 3.11, CPU PyTorch and rasterio. The network uses only Sentinel-1
VV/VH, never S2 or labels as input. Train and validation select the epoch; the
official test split is used only after model selection. 128-pixel training inputs
are upsampled for full-resolution test metrics, which is stated in the result.
"""
import csv
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import rasterio
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "paper/experiments/cache/sen1floods11"
OUT = ROOT / "paper/experiments/results/sen1floods11"
BASE = "https://storage.googleapis.com/sen1floods11/v1.1"


def fetch(rel):
    path = CACHE / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urlopen(f"{BASE}/{rel}", timeout=90).read())
    return path


def split_rows(split):
    path = fetch(f"splits/flood_handlabeled/flood_{split}_data.csv")
    return list(csv.reader(io.StringIO(path.read_text())))


def files(row):
    base = "data/flood_events/HandLabeled"
    return (f"{base}/S1Hand/{row[0]}", f"{base}/LabelHand/{row[1]}")


def load_one(row):
    s1_file, label_file = files(row)
    with rasterio.open(fetch(s1_file)) as ds:
        bands = ds.read().astype(np.float32)
    with rasterio.open(fetch(label_file)) as ds:
        label = ds.read(1).astype(np.int64)
    valid = np.isfinite(bands).all(axis=0) & ((label == 0) | (label == 1))
    if not valid.any():
        return None
    bands = np.nan_to_num(bands, nan=-30.0, posinf=5.0, neginf=-30.0)
    image = np.clip((bands + 30.0) / 35.0, 0.0, 1.0)
    label[~valid] = 255
    return row[0].removesuffix("_S1Hand.tif"), torch.from_numpy(image), torch.from_numpy(label)


def dataset(split):
    rows = split_rows(split)
    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = list(pool.map(load_one, rows))
    return [x for x in loaded if x is not None], len(rows) - sum(x is not None for x in loaded)


class ConvBlock(nn.Module):
    def __init__(self, ins, outs):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(ins, outs, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(outs, outs, 3, padding=1), nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = ConvBlock(2, 16)
        self.e2 = ConvBlock(16, 32)
        self.mid = ConvBlock(32, 64)
        self.d2 = ConvBlock(96, 32)
        self.d1 = ConvBlock(48, 16)
        self.out = nn.Conv2d(16, 2, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.mid(F.max_pool2d(b, 2))
        d = self.d2(torch.cat((F.interpolate(c, size=b.shape[2:], mode="bilinear", align_corners=False), b), 1))
        d = self.d1(torch.cat((F.interpolate(d, size=a.shape[2:], mode="bilinear", align_corners=False), a), 1))
        return self.out(d)


def train_tensors(items):
    x = torch.stack([F.interpolate(item[1][None], size=(128, 128), mode="area")[0]
                     for item in items])
    y = torch.stack([F.interpolate(item[2][None, None].float(), size=(128, 128),
                                  mode="nearest")[0, 0].long() for item in items])
    return x, y


def confusion(model, items):
    model.eval()
    tp = fp = fn = 0
    per_chip = []
    with torch.no_grad():
        for name, im, label in items:
            small = F.interpolate(im[None], size=(128, 128), mode="area")
            logits = model(small)
            logits = F.interpolate(logits, size=label.shape, mode="bilinear", align_corners=False)
            p = logits.argmax(1)[0].numpy().astype(bool)
            t = label.numpy() == 1
            valid = label.numpy() != 255
            a = int(np.count_nonzero(p & t & valid))
            b = int(np.count_nonzero(p & ~t & valid))
            c = int(np.count_nonzero(~p & t & valid))
            tp += a; fp += b; fn += c
            per_chip.append({"chip": name, "tp": a, "fp": b, "fn": c,
                             "iou": a / (a + b + c) if a + b + c else None})
    return {"tp": tp, "fp": fp, "fn": fn,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "per_chip": per_chip}


def main():
    torch.manual_seed(2026)
    np.random.seed(2026)
    random.seed(2026)
    torch.set_num_threads(4)
    train, train_ex = dataset("train")
    valid, valid_ex = dataset("valid")
    test, test_ex = dataset("test")
    print("loaded", len(train), len(valid), len(test), flush=True)
    x, y = train_tensors(train)
    model = UNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    weights = torch.tensor([1.0, 4.0])
    best_iou, best_epoch, best_state = -1.0, 0, None
    history = []
    order_rng = torch.Generator().manual_seed(2026)
    for epoch in range(1, 13):
        model.train()
        losses = []
        for batch in torch.randperm(len(x), generator=order_rng).split(16):
            optimiser.zero_grad()
            logits = model(x[batch])
            loss = F.cross_entropy(logits, y[batch], weight=weights, ignore_index=255)
            loss.backward()
            optimiser.step()
            losses.append(float(loss))
        v = confusion(model, valid)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "valid_iou": v["iou"]})
        print(history[-1], flush=True)
        if v["iou"] > best_iou:
            best_iou, best_epoch = v["iou"], epoch
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_metrics = confusion(model, test)
    result = {"model": "small 3-level U-Net, 2-channel S1 input, 128x128 inference",
              "seed": 2026, "epochs": 12, "best_epoch": best_epoch,
              "splits": {"train": len(train), "valid": len(valid), "test": len(test)},
              "excluded": {"train": train_ex, "valid": valid_ex, "test": test_ex},
              "history": history, "test": test_metrics}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "unet_test.json").write_text(json.dumps(result, indent=2))
    torch.save(best_state, CACHE / "unet_best_state.pt")
    print("test", {k: v for k, v in test_metrics.items() if k != "per_chip"}, flush=True)


if __name__ == "__main__":
    main()
