"""
Time-indexed flood-contour prediction.

The routing layer does not want a single static "predicted contour".  Instead it
wants to ask *"what does the flood look like at the time the drone will actually
be there?"*  This module provides that abstraction:

    * :class:`FloodTimeline` -- a sequence of flood masks indexed by time
      (minutes from mission start) with an ``obstacle_mask_at(t)`` query that
      rounds to / interpolates between the closest predicted frames.
    * :func:`predict_flood_sequence` -- produces such a timeline for a set of
      requested future timestamps.

Backends
--------
* ``dl``  -- deep-learning forecaster (:mod:`src.weather.convlstm`).  A
  weather-conditioned ConvLSTM that consumes the current contour (+ a short
  history) and emits a *sequence* of future contours.  Used when a trained
  weight file is available (see ``scripts/train_flood_forecast.py``).
* ``ca``  -- the original morphological cellular-automata model
  (:func:`src.weather.flood_spread.predict_spread`) kept as the *baseline /
  fallback* so the pipeline keeps working when no DL weights exist.  It also
  powers the data generator used to fit the DL model.

``backend="auto"`` tries the DL forecaster and transparently falls back to CA
when torch / weights are unavailable (logged, not silent).
"""
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from src.weather.flood_spread import predict_spread

logger = logging.getLogger(__name__)

# Weather scalar keys that condition both the CA baseline and the DL forecaster.
WEATHER_KEYS = ["precipitation", "wind_speed_10m", "wind_direction_10m"]


def _as_uint8_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalise any common mask dtype (bool / uint8 0-255) to uint8 {0,255}."""
    if mask is None:
        return None
    if mask.dtype == np.bool_:
        return (mask.astype(np.uint8) * 255)
    if mask.dtype != np.uint8:
        m, M = mask.min(), mask.max()
        if M - m > 1e-8:
            mask = ((mask - m) / (M - m) * 255.0)
        else:
            mask = (mask * 255.0)
    out = np.zeros(mask.shape, dtype=np.uint8)
    out[mask > 0] = 255
    return out


def weather_feature_vector(weather: Dict[str, float]) -> np.ndarray:
    """Small, normalised vector that conditions the DL forecaster.

    Order must match the model's weather embedding, so never reorder.
    """
    return np.asarray(
        [
            float(weather.get("precipitation", 0.0)) / 20.0,      # mm/h  (scale: heavy rain)
            float(weather.get("wind_speed_10m", 0.0)) / 20.0,     # m/s   (scale: gale)
            (float(weather.get("wind_direction_10m", 0.0)) % 360.0) / 360.0,
        ],
        dtype=np.float32,
    )


class FloodTimeline:
    """Sequence of flood masks indexed by mission time (minutes from t0).

    ``times_min[0]`` is normally ``0.0`` (the current observation) followed by
    predicted contours at the requested future timestamps.  The routing layer
    plans each flight leg against ``obstacle_mask_at(arrival_time)`` instead of
    against a single static snapshot.
    """

    def __init__(
        self,
        times_min: Sequence[float],
        masks: Sequence[np.ndarray],
        weather: Optional[Dict[str, float]] = None,
        source: str = "unknown",
    ):
        if len(times_min) == 0 or len(masks) == 0:
            raise ValueError("FloodTimeline needs at least one (time, mask) pair.")
        if len(times_min) != len(masks):
            raise ValueError("times_min and masks must have equal length.")
        self.times_min = np.asarray([float(t) for t in times_min], dtype=np.float64)
        if np.any(np.diff(self.times_min) <= 0):
            raise ValueError("times_min must be strictly increasing.")
        self.masks = [_as_uint8_mask(m) for m in masks]
        self.weather = dict(weather) if weather else {}
        self.source = source

    # ------------------------------------------------------------------ #
    def nearest_time_min(self, t_min: float) -> float:
        """Time of the closest available frame (ties round up to the later,
        safer frame).  Used to report which forecast slice a leg planned
        against."""
        t = float(t_min)
        ts = self.times_min
        if t <= ts[0]:
            return float(ts[0])
        if t >= ts[-1]:
            return float(ts[-1])
        hi = int(np.searchsorted(ts, t))
        lo = hi - 1
        if (t - ts[lo]) < (ts[hi] - t):
            return float(ts[lo])
        return float(ts[hi])

    def obstacle_mask_at(
        self, t_min: float, mode: str = "nearest", threshold: float = 0.5
    ) -> np.ndarray:
        """Return the obstacle (flood) mask valid at time ``t_min``.

        mode:
          * ``nearest``     -- round to the closest available predicted frame.
          * ``interpolate`` -- linear blend of the bracketing frames, kept
            binary by thresholding at ``threshold`` (value in [0,1]).
        """
        t = float(t_min)
        ts = self.times_min
        if t <= ts[0]:
            return self.masks[0].copy()
        if t >= ts[-1]:
            return self.masks[-1].copy()

        hi = int(np.searchsorted(ts, t))          # ts[hi-1] <= t < ts[hi]
        lo = hi - 1
        t_lo, t_hi = ts[lo], ts[hi]

        if mode == "nearest":
            return self.masks[hi if (t - t_lo) >= (t_hi - t) else lo].copy()

        if mode == "interpolate":
            frac_lo = (t_hi - t) / (t_hi - t_lo)
            frac_hi = (t - t_lo) / (t_hi - t_lo)
            blended = frac_lo * self.masks[lo].astype(np.float32) \
                + frac_hi * self.masks[hi].astype(np.float32)
            return (blended >= (threshold * 255.0)).astype(np.uint8) * 255

        raise ValueError(f"Unknown interpolation mode: {mode!r}")

    # ------------------------------------------------------------------ #
    def flooded(self, pixel: Tuple[int, int], t_min: float, mode: str = "nearest") -> bool:
        x, y = int(pixel[0]), int(pixel[1])
        m = self.obstacle_mask_at(t_min, mode=mode)
        if x < 0 or y < 0 or x >= m.shape[1] or y >= m.shape[0]:
            return False
        return bool(m[y, x] > 0)

    # ------------------------------------------------------------------ #
    def blocked_after_min(self, pixel: Tuple[int, int], mode: str = "nearest") -> float:
        """First time the given pixel is flooded, else ``inf``.

        Used to decide whether a drop zone will be swallowed by the predicted
        spread before the drone can reach it (Task 5).
        """
        x, y = int(pixel[0]), int(pixel[1])
        inf = float("inf")
        for t, m in zip(self.times_min, self.masks):
            if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x] > 0:
                return float(t)
        return inf

    # ------------------------------------------------------------------ #
    def n_predicted_frames(self) -> int:
        """Number of strictly-future frames (excludes the t=0 observation)."""
        return max(0, int(np.sum(self.times_min > 0)))

    def __len__(self) -> int:
        return len(self.masks)


# --------------------------------------------------------------------- #
#  CA (cellular-automata) baseline -- fallback + training-data generator
# --------------------------------------------------------------------- #
def _fractional_step(
    cur: np.ndarray, nxt: np.ndarray, frac_hours: float
) -> np.ndarray:
    """Keep only the fraction of a one-hour expansion that elapses in ``frac_hours``.

    A plain morphological step adds a whole ring at once; for sub-hour timestamps
    we threshold that ring by the distance-to-flood field so a half-hour step only
    floods the inner half of the hourly expansion (continuous, monotonic time).
    """
    dry = np.zeros(cur.shape, dtype=np.uint8)
    dry[cur == 0] = 255                       # flood=0 so distance measures dry->flood
    dist = cv2.distanceTransform(dry, cv2.DIST_L2, 5).astype(np.float64)
    new = (nxt > 0) & (cur == 0)              # pixels the 1h step would add
    ring = dist[new]
    if ring.size == 0:
        return cur.copy()
    reach = float(ring.max())
    keep = new & (dist <= max(0.0, frac_hours * reach))
    out = cur.copy()
    out[keep] = 255
    return out


def _ca_advance(mask: np.ndarray, weather: Dict[str, float], hours: float) -> np.ndarray:
    """Advance the physics model by ``hours`` (supports fractional hours)."""
    mask = _as_uint8_mask(mask)
    whole = int(hours)
    frac = hours - whole
    cur = mask
    for _ in range(whole):
        cur = predict_spread(cur, weather, 1.0)
    if frac > 1e-9:
        nxt = predict_spread(cur, weather, 1.0)
        cur = _fractional_step(cur, nxt, frac)
    return cur


def ca_sequence(
    mask: np.ndarray, weather: Dict[str, float], timestamps_min: Sequence[float]
) -> Tuple[List[np.ndarray], str]:
    """Physics-baseline timeline frames for the requested timestamps."""
    base = _as_uint8_mask(mask)
    times = sorted(float(t) for t in timestamps_min)
    if not times or times[0] != 0.0:
        times = [0.0] + [t for t in times if t > 0.0]
    frames = []
    for t in times:
        frames.append(_ca_advance(base, weather, t / 60.0) if t > 0.0 else base.copy())
    return frames, "ca-baseline"


# --------------------------------------------------------------------- #
#  DL forecaster integration (optional; see src/weather/convlstm.py)
# --------------------------------------------------------------------- #
def _dl_frames(
    mask: np.ndarray,
    weather: Dict[str, float],
    timestamps_min: Sequence[float],
    model_path: str,
    device: Optional[str] = None,
) -> Tuple[List[np.ndarray], str]:
    from src.weather.convlstm import FloodConvLSTM  # lazy: torch optional

    net = FloodConvLSTM.load(model_path, device=device)
    times = sorted(float(t) for t in timestamps_min)
    times = [0.0] + [t for t in times if t > 0.0]
    frames, _ = net.forecast(mask, weather, times[1:])
    return frames, "convlstm"


def predict_flood_sequence(
    mask: np.ndarray,
    weather: Dict[str, float],
    timestamps_min: Sequence[float],
    backend: str = "auto",
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> FloodTimeline:
    """Predict flood contours at multiple future timestamps.

    Returns a :class:`FloodTimeline` whose frame ``i`` is valid at
    ``timestamps_min[i]`` minutes from now.  Frame ``0`` is the current
    observation; the remaining frames are time-indexed predictions that the
    routing layer queries with ``obstacle_mask_at(leg_arrival_min)``.

    Backends
        * ``auto`` -- DL forecaster if torch + weights are available, else CA.
        * ``dl``   -- force DL (raises ``RuntimeError`` if unavailable).
        * ``ca``   -- force the physics baseline.
    """
    base = _as_uint8_mask(mask)
    if base is None:
        raise ValueError("A current flood mask is required.")

    times = sorted(float(t) for t in timestamps_min)
    if not times or times[0] != 0.0:
        times = [0.0] + [t for t in times if t > 0.0]

    if backend == "ca":
        frames, source = ca_sequence(base, weather, times)
    elif backend == "auto":
        try:
            frames, source = _dl_frames(base, weather, times, model_path or _default_model_path(), device)
            logger.info("Flood forecast backend: ConvLSTM (%s)", source)
        except Exception as exc:  # no torch / no weights / model error
            frames, source = ca_sequence(base, weather, times)
            logger.info("DL flood forecaster unavailable (%s). Falling back to CA baseline.", exc)
    elif backend == "dl":
        frames, source = _dl_frames(base, weather, times, model_path or _default_model_path(), device)
    else:
        raise ValueError(f"Unknown flood-prediction backend: {backend!r}")

    return FloodTimeline(times, frames, weather=weather, source=source)


def _default_model_path() -> str:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "data", "models", "flood_convlstm.pt")
