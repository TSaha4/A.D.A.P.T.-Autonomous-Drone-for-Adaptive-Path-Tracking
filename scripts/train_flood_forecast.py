#!/usr/bin/env python3
"""
Train the weather-conditioned ConvLSTM flood forecaster (Task 1).

There is no public flood-sequence dataset shipped with this repo, so the DL
layer is bootstrapped on roll-outs of the physics (cellular-automata) baseline,
which doubles as the *fallback* predictor at runtime.  A ConvLSTM that learns
to reproduce the weather-conditioned spread dynamics from synthetic roll-outs
becomes a sequence forecaster that can later be fine-tuned on real inundation
sequences.

Usage
-----
    python scripts/train_flood_forecast.py [--grid 64] [--epochs 30] \\
        [--batch 8] [--out data/models/flood_convlstm.pt]

Requires PyTorch (``pip install torch``).  Without torch this script exits with
a clear message and the pipeline keeps using the CA baseline.
"""
import argparse
import os
import sys

import numpy as np

# Make `src` importable when run as a plain script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.weather.flood_predictor import weather_feature_vector, ca_sequence

WEATHER_KEYS = ["precipitation", "wind_speed_10m", "wind_direction_10m"]


def random_weather(rng: np.random.Generator) -> dict:
    return {
        "precipitation": float(rng.uniform(0.0, 15.0)),
        "wind_speed_10m": float(rng.uniform(0.0, 20.0)),
        "wind_direction_10m": float(rng.uniform(0.0, 360.0)),
    }


def random_blob_grid(size: int, rng: np.random.Generator) -> np.ndarray:
    """Random elliptical flood blobs used as synthetic initial conditions."""
    mask = np.zeros((size, size), dtype=np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        cx = int(rng.integers(8, size - 8))
        cy = int(rng.integers(8, size - 8))
        rx = int(rng.integers(3, max(4, size // 8)))
        ry = int(rng.integers(3, max(4, size // 8)))
        yy, xx = np.ogrid[:size, :size]
        ell = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        mask[ell] = 255
    return mask


def run_training(args) -> None:
    import torch
    import torch.nn as nn

    from src.weather.convlstm import FloodForecastNet

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = FloodForecastNet(weather_dim=3, hidden_channels=32, n_steps=args.n_steps).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    # The model emits sigmoid probabilities, so use binary cross-entropy on
    # [0,1] targets rather than raw-logit BCE.
    loss_fn = nn.BCELoss()

    times = [args.cadence_min * k for k in range(1, args.n_steps + 1)]
    rng = np.random.default_rng(0)
    n_batches = max(1, args.samples_per_epoch // args.batch)

    print(
        f"Training {args.samples_per_epoch * args.epochs} synthetic CA roll-outs "
        f"(grid {args.grid}x{args.grid}, horizon {args.n_steps * args.cadence_min:.0f} min, "
        f"cadence {args.cadence_min:.0f} min) on {device}..."
    )

    for epoch in range(1, args.epochs + 1):
        net.train()
        total = 0.0
        for _ in range(n_batches):
            obs, tgt, wvecs = [], [], []
            for _ in range(args.batch):
                base = random_blob_grid(args.grid, rng)
                w = random_weather(rng)
                frames, _ = ca_sequence(base, w, times)  # frames[0]=current
                obs.append(frames[0][None, None].astype(np.float32) / 255.0)
                tgt.append(np.stack(frames[1:])[:, None].astype(np.float32) / 255.0)
                wvecs.append(weather_feature_vector(w))
            xb = torch.from_numpy(np.stack(obs)).to(device)
            yb = torch.from_numpy(np.stack(tgt)).to(device)
            wb = torch.from_numpy(np.stack(wvecs)).to(device)

            logits = net(xb, wb)                     # (B, n_steps, 1, H, W)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())

        print(f"epoch {epoch:3d}/{args.epochs}  loss {total / n_batches:.4f}")

    net.eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "weather_dim": 3,
            "hidden_channels": 32,
            "n_steps": args.n_steps,
            "cadence_min": args.cadence_min,
            "internal_size": args.grid,
        },
        args.out,
    )
    print(f"Saved forecast weights to {args.out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=64, help="Internal grid resolution")
    parser.add_argument("--cadence-min", type=float, default=30.0)
    parser.add_argument("--n-steps", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="data/models/flood_convlstm.pt")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "PyTorch is not installed. Install it with `pip install torch`, or keep "
            "the CA baseline (backend='ca')."
        )
        return 1

    run_training(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
