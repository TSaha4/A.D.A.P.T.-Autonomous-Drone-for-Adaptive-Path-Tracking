"""
Weather-conditioned ConvLSTM flood forecaster (Task 1 DL layer).

Architecture
------------
A ConvLSTM encoder-decoder (Shi et al., *Convolutional LSTM Network: A Machine
Learning Approach for Precipitation Nowcasting*, NeurIPS 2015) that is
conditioned on scalar weather metadata through feature-wise linear modulation
(FiLM, Perez et al., 2018)::

    frame_t0 (1xHxW) + optional history frames
        --> ConvLSTM encoder (folds the observed sequence into a hidden state)
        --> decoder ConvLSTM unrolls N future steps
        --> each step decodes its hidden state to a flood mask via a 1x1 conv

    weather vector + elapsed time  --> MLP --> per-channel (gamma, beta)
        injected into the decoder's recurrent state every step (FiLM).

The decoder's N outputs are a *sequence* of predicted flood contours at an
internal cadence (default 30 min per step), so the planner gets time-indexed
contours rather than a single static forecast.  A requested timestamp that does
not land exactly on the cadence is rounded to the nearest available frame.

Weight file
-----------
``data/models/flood_convlstm.pt`` -- fit offline by
``scripts/train_flood_forecast.py`` on synthetic roll-outs of the
cellular-automata baseline (the CA model doubles as the training-data
generator).  Until weights exist the pipeline transparently uses the CA
baseline (see ``flood_predictor.predict_flood_sequence(backend="auto")``).

This module is *optional*: importing it never fails if PyTorch is missing --
``torch_available()`` returns False and the model raises an informative error.
"""
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when torch is installed
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    TORCH_AVAILABLE = False

DEFAULT_CADENCE_MIN = 30.0
DEFAULT_N_STEPS = 6  # 3 h horizon at 30-min cadence
DEFAULT_INTERNAL_SIZE = 64


def torch_available() -> bool:
    return bool(TORCH_AVAILABLE)


# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:  # class bodies below need nn.* at import time

    class _ConvLSTMCell(nn.Module):
        """Standard convolutional LSTM cell."""

        def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
            super().__init__()
            pad = kernel_size // 2
            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            self.conv = nn.Conv2d(
                in_channels + hidden_channels,
                4 * hidden_channels,
                kernel_size=kernel_size,
                padding=pad,
            )

        def forward(self, x, state):
            h, c = state
            gates = self.conv(torch.cat([x, h], dim=1))
            i, f, g, o = gates.chunk(4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)
            return h, (h, c)


    class FloodForecastNet(nn.Module):
        """Encoder-decoder ConvLSTM that emits a sequence of future flood masks.

        Parameters
        ----------
        weather_dim : size of the (normalised) weather feature vector.
        hidden_channels : width of the recurrent state.
        n_steps : number of future frames produced per call (decoder unroll).
        """

        def __init__(
            self,
            weather_dim: int = 3,
            hidden_channels: int = 32,
            n_steps: int = DEFAULT_N_STEPS,
        ):
            super().__init__()
            self.hidden_channels = hidden_channels
            self.n_steps = n_steps

            # Input encoder: project 1-channel observation into a feature map.
            self.in_conv = nn.Conv2d(1, hidden_channels, 3, padding=1)
            self.cell = _ConvLSTMCell(hidden_channels, hidden_channels)
            self.out_conv = nn.Conv2d(hidden_channels, 1, 1)

            # FiLM conditioning on weather + elapsed decoder step.
            self.film = nn.Sequential(
                nn.Linear(weather_dim + 1, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 2 * hidden_channels),
            )

        def _film_step(self, cond: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
            gamma, beta = self.film(cond).chunk(2, dim=-1)  # (B, 2*H) -> 2 x (B, H)
            gamma = gamma.view(-1, self.hidden_channels, 1, 1)
            beta = beta.view(-1, self.hidden_channels, 1, 1)
            return gamma * feature + beta

        def forward(self, obs: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
            """Forecast ``n_steps`` future frames.

            obs     : (B, T_in, 1, H, W) observed contour frames (>=1).
            weather : (B, weather_dim) normalised weather scalars.
            returns : (B, n_steps, 1, H, W) predicted frames in [0, 1].
            """
            B, T_in, _, H, W = obs.shape
            state = None
            h = None
            for t in range(T_in):  # fold any observed history through the ConvLSTM
                f = self.in_conv(obs[:, t])
                if state is None:
                    h = torch.zeros_like(f)
                    c = torch.zeros_like(f)
                    state = (h, c)
                h, state = self.cell(self._film_step(self._cond(weather, t), f), state)

            frames = []
            for k in range(self.n_steps):
                h, state = self.cell(self._film_step(self._cond(weather, T_in + k), h), state)
                frames.append(torch.sigmoid(self.out_conv(h)))
            return torch.stack(frames, dim=1)

        def _cond(self, weather: torch.Tensor, step: float) -> torch.Tensor:
            step_norm = torch.full(
                (weather.shape[0], 1), float(step) / 10.0, device=weather.device
            )
            return torch.cat([weather, step_norm], dim=1)


    class FloodConvLSTM:
        """High-level facade used by :mod:`src.weather.flood_predictor`."""

        def __init__(self, net: nn.Module, cadence_min: float = DEFAULT_CADENCE_MIN,
                     internal_size: int = DEFAULT_INTERNAL_SIZE):
            self.net = net
            self.net.eval()
            self.cadence_min = float(cadence_min)
            self.internal_size = int(internal_size)

        # ------------------------------------------------------------------ #
        @classmethod
        def load(cls, path: str, device: Optional[str] = None) -> "FloodConvLSTM":
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"No flood-forecast weights at {path}. Train one with "
                    "scripts/train_flood_forecast.py or use backend='ca'."
                )
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            state = torch.load(path, map_location=dev)
            net = FloodForecastNet(
                weather_dim=state.get("weather_dim", 3),
                hidden_channels=state.get("hidden_channels", 32),
                n_steps=state.get("n_steps", DEFAULT_N_STEPS),
            )
            net.load_state_dict(state["state_dict"])
            net.to(dev)
            return cls(
                net,
                cadence_min=state.get("cadence_min", DEFAULT_CADENCE_MIN),
                internal_size=state.get("internal_size", DEFAULT_INTERNAL_SIZE),
            )

        # ------------------------------------------------------------------ #
        def forecast(
            self,
            mask: np.ndarray,
            weather: Dict[str, float],
            future_minutes: Sequence[float],
        ) -> Tuple[List[np.ndarray], List[float]]:
            """Predict flood masks at the requested future minutes.

            Timestamps are mapped onto the model's internal cadence
            (rounded to the nearest available step).  Returns the predicted
            masks (uint8 {0, 255}) and the cadence times actually produced.
            """
            from src.weather.flood_predictor import weather_feature_vector

            cad = self.cadence_min
            steps = sorted(
                {int(round(float(t) / cad)) for t in future_minutes if t > 0}
            )
            if not steps:
                return [], []
            steps = [s for s in steps if s <= self.net.n_steps]
            if not steps:
                raise ValueError(
                    f"Requested minutes exceed the model horizon "
                    f"({self.net.n_steps * cad} min)."
                )

            base = np.asarray(mask)
            H, W = base.shape[:2]
            s = self.internal_size
            img = cv2_resize_mask(base, (s, s))
            obs = torch.from_numpy(img[None, None, None].astype(np.float32))  # (1,1,1,H,W)
            wvec = weather_feature_vector(weather)
            cond = torch.from_numpy(wvec[None].astype(np.float32))

            with torch.no_grad():
                pred = self.net(obs, cond)[0]  # (n_steps, 1, s, s)

            frames = []
            times = []
            for k in steps:
                m = (pred[k - 1, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255
                if (H, W) != (s, s):
                    m = cv2_resize_mask(m, (W, H))
                frames.append(m)
                times.append(float(k) * cad)
            return frames, times


    def cv2_resize_mask(mask: np.ndarray, dsize: Tuple[int, int]) -> np.ndarray:
        import cv2
        return cv2.resize(
            np.asarray(mask, dtype=np.uint8), dsize, interpolation=cv2.INTER_NEAREST
        )

else:  # torch not installed -> stubs that fail with a clear message

    class FloodConvLSTM:
        def __init__(self, *a, **k):
            raise RuntimeError(
                "PyTorch is not installed, so the ConvLSTM forecaster is unavailable. "
                "pip install torch, train weights with scripts/train_flood_forecast.py, "
                "or keep backend='ca'."
            )

        @classmethod
        def load(cls, *a, **k):
            FloodConvLSTM.__init__(cls)
