"""Lightweight temporal flood-contour extrapolator.

This is a deterministic, weather-conditioned image model, not a trained DL
model. It emits binary forecast frames at requested hours. A trained ConvLSTM
can replace the predictor behind this API when sequential labeled data exists.
"""
import math
import cv2
import numpy as np
from config import settings


class ForecastDict(dict):
    """Dictionary mapping hour floats to binary masks, with ndarray-like fallback for array ops."""

    def __array__(self, dtype=None):
        if not self:
            return np.zeros((0, 0), dtype=dtype or np.uint8)
        latest = self[max(self)]
        return np.asarray(latest, dtype=dtype)

    def __eq__(self, other):
        if isinstance(other, (np.ndarray, int, float)):
            return self.__array__() == other
        return super().__eq__(other)

    def __gt__(self, other):
        return self.__array__() > other

    def __ge__(self, other):
        return self.__array__() >= other

    def __lt__(self, other):
        return self.__array__() < other

    def __le__(self, other):
        return self.__array__() <= other

    @property
    def shape(self):
        return self.__array__().shape

    @property
    def dtype(self):
        return self.__array__().dtype

    @property
    def size(self):
        return self.__array__().size

    @property
    def ndim(self):
        return self.__array__().ndim


def predict_spread(mask: np.ndarray, weather_data: dict, horizon_hours: float,
                   timesteps=(1/12, 1/6, 0.5, 1.0, 2.0)) -> ForecastDict:
    """Return ``{hours_from_now: binary_mask}`` for future contour snapshots.

    Weather-conditioned dilation approximates uncertainty growth; wind shifts
    the contour downwind. The per-hour growth rates are explicit heuristics.
    Frames are cumulative: flooding never recedes inside the forecast horizon,
    so every frame contains all earlier frames.
    """
    if mask is None or mask.size == 0:
        return ForecastDict()
    src = (mask > 0).astype(np.uint8) * 255
    precip = max(0.0, float(weather_data.get("precipitation", 0.0)))
    wind = max(0.0, float(weather_data.get("wind_speed_10m", 0.0)))
    direction = np.deg2rad(float(weather_data.get("wind_direction_10m", 0.0)) + 180.0)
    diag = math.hypot(*src.shape[:2])

    def do_spread(base_img: np.ndarray, t: float) -> np.ndarray:
        growth = (precip / settings.FLOOD_PRECIP_REFERENCE_MM_PER_H) * t * settings.FLOOD_GROWTH_PX_PER_5MM_HOUR
        radius = int(np.floor(growth))
        fraction = growth - radius
        pred = base_img
        if radius >= diag:
            pred = np.full_like(src, 255) if np.any(src > 0) else src
        else:
            if radius > 0:
                size = 2 * radius + 1
                pred = cv2.dilate(pred, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
            if fraction > 1e-6:
                next_size = 2 * (radius + 1) + 1
                expanded = cv2.dilate(base_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (next_size, next_size)))
                pred = np.where(expanded > 0, np.maximum(pred, int(round(255 * fraction))), pred).astype(np.uint8)
        shift = (wind / settings.FLOOD_WIND_REFERENCE_KMH) * t * settings.FLOOD_WIND_SHIFT_PX_PER_10KMH_HOUR
        dx = int(round(np.sin(direction) * shift))
        dy = int(round(-np.cos(direction) * shift))
        if dx or dy:
            transform = np.float32([[1, 0, dx], [0, 1, dy]])
            pred = cv2.bitwise_or(pred, cv2.warpAffine(pred, transform, (pred.shape[1], pred.shape[0])))
        frame = (pred >= 128).astype(np.uint8) * 255
        return cv2.bitwise_or(frame, base_img)

    result = {0.0: src.copy()}
    sub_hours = sorted({float(t) for t in timesteps if 0 < float(t) <= min(1.0, horizon_hours)})
    if horizon_hours > 0 and not sub_hours and horizon_hours <= 1.0:
        sub_hours = [float(horizon_hours)]
    for t in sub_hours:
        result[t] = do_spread(src, t)

    last_hourly_frame = result.get(1.0, src)
    max_h = int(horizon_hours)
    for h in range(2, max_h + 1):
        last_hourly_frame = do_spread(last_hourly_frame, 1.0)
        result[float(h)] = last_hourly_frame
    if horizon_hours > max_h and horizon_hours > 1.0:
        rem = horizon_hours - max_h
        result[float(horizon_hours)] = do_spread(last_hourly_frame, rem)

    return ForecastDict(result)


def forecast_frames(mask: np.ndarray, weather_data: dict, horizon_hours: float, predictor=None) -> dict:
    """Time-indexed forecast ``{hours_from_now: binary mask}`` built from :func:`predict_spread`."""
    if mask is None or mask.size == 0:
        return {}
    predictor = predict_spread if predictor is None else predictor
    current = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if horizon_hours <= 0:
        predicted = (np.asarray(predictor(mask, weather_data, horizon_hours)) > 0).astype(np.uint8) * 255
        return {0.0: cv2.bitwise_or(current, predicted)}
    iterations = max(1, int(horizon_hours))
    frames = {0.0: current}
    previous = current
    if float(iterations) == float(horizon_hours):
        for k in range(1, iterations + 1):
            frame = (np.asarray(predictor(previous, weather_data, 1)) > 0).astype(np.uint8) * 255
            frame = cv2.bitwise_or(frame, previous)
            frames[float(k)] = frame
            previous = frame
    else:
        for k in range(1, iterations):
            frame = (np.asarray(predictor(previous, weather_data, 1)) > 0).astype(np.uint8) * 255
            frame = cv2.bitwise_or(frame, previous)
            frames[float(k)] = frame
            previous = frame
        last_key = float(min(iterations, horizon_hours))
        last = (np.asarray(predictor(mask, weather_data, horizon_hours)) > 0).astype(np.uint8) * 255
        frames[last_key] = cv2.bitwise_or(last, current)
    return frames


def _bracket(forecasts: dict, arrival_hours: float):
    """Return ``(t0, t1, alpha, source)`` for the frames around ``arrival_hours``."""
    times = sorted(forecasts)
    for t in times:
        if abs(t - arrival_hours) < 1e-9:
            return t, t, 0.0, "frame"
    if arrival_hours <= times[0]:
        return times[0], times[0], 0.0, "frame"
    if arrival_hours >= times[-1]:
        return times[-1], times[-1], 1.0, "frame_clamped"
    for t0, t1 in zip(times, times[1:]):
        if t0 <= arrival_hours <= t1:
            return t0, t1, float((arrival_hours - t0) / (t1 - t0)), "interpolated"
    return times[-1], times[-1], 1.0, "frame_fallback"


def interpolate_forecast_mask(forecasts: dict, arrival_hours: float, current_mask=None):
    """Linearly blend neighboring forecast frames into a uint8 grayscale mask.

    This soft blend is a diagnostic (per-pixel occupancy confidence). Binary
    obstacle queries use :func:`forecast_at`, which advances the flood front
    geometrically instead of thresholding this blend.
    """
    if not forecasts:
        if current_mask is None:
            raise ValueError("No flood forecasts or current mask available")
        return np.asarray(current_mask, dtype=np.uint8).copy(), {
            "requested_hours": float(arrival_hours), "lower_hours": None,
            "upper_hours": None, "alpha": None, "source": "current_mask"}
    t0, t1, alpha, source = _bracket(forecasts, arrival_hours)
    details = {"requested_hours": float(arrival_hours), "lower_hours": t0,
               "upper_hours": t1, "alpha": float(alpha), "source": source}
    if source != "interpolated":
        return np.asarray(forecasts[t0 if alpha == 0.0 else t1], dtype=np.uint8).copy(), details
    lower = np.asarray(forecasts[t0], dtype=np.float32)
    upper = np.asarray(forecasts[t1], dtype=np.float32)
    blended = np.clip(np.rint((1.0 - alpha) * lower + alpha * upper), 0, 255).astype(np.uint8)
    return blended, details


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Euclidean signed distance in pixels: negative inside flood, positive outside."""
    occupied = (np.asarray(mask) >= 128).astype(np.uint8)
    inside = cv2.distanceTransform(occupied, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(1 - occupied, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return outside - inside


def interpolate_flood_front(lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    """Binary mask with the flood front advanced ``alpha`` of the way from ``lower`` to ``upper``.

    Interpolates the two frames' signed distance fields, so each pixel between
    the fronts floods when the front reaches it: a pixel ``a`` px outside the
    lower front and ``b`` px inside the upper front floods at ``alpha = a/(a+b)``.
    The flooded area therefore grows progressively with alpha rather than
    switching all at once at a single threshold.
    """
    lower_bin = np.asarray(lower) >= 128
    upper_bin = np.asarray(upper) >= 128
    if alpha <= 0.0:
        return lower_bin.astype(np.uint8) * 255
    if alpha >= 1.0:
        return upper_bin.astype(np.uint8) * 255
    if not lower_bin.any() or not upper_bin.any() or lower_bin.all() or upper_bin.all():
        # No front to propagate from or to (distance fields are undefined):
        # pixels that differ switch at the temporal midpoint.
        return (lower_bin if alpha < 0.5 else upper_bin).astype(np.uint8) * 255
    blended = (1.0 - alpha) * signed_distance(lower) + alpha * signed_distance(upper)
    return (blended < 0).astype(np.uint8) * 255


def forecast_at(forecasts: dict, arrival_hours: float, current_mask=None):
    """Return a binary obstacle mask plus interpolation diagnostics."""
    if not forecasts:
        current, details = interpolate_forecast_mask(forecasts, arrival_hours, current_mask)
        return (current >= 128).astype(np.uint8) * 255, details
    t0, t1, alpha, source = _bracket(forecasts, arrival_hours)
    details = {"requested_hours": float(arrival_hours), "lower_hours": t0,
               "upper_hours": t1, "alpha": float(alpha), "source": source}
    if source != "interpolated":
        frame = forecasts[t0 if alpha == 0.0 else t1]
        return (np.asarray(frame) >= 128).astype(np.uint8) * 255, details
    return interpolate_flood_front(forecasts[t0], forecasts[t1], alpha), details


def obstacle_mask_at(forecasts: dict, arrival_hours: float, current_mask=None) -> np.ndarray:
    """Binary obstacles at ``arrival_hours``, advancing the front between frames."""
    return forecast_at(forecasts, arrival_hours, current_mask)[0]


def stretch_goal_convlstm_stub(mask: np.ndarray, weather_data: dict, horizon_hours: float) -> np.ndarray:
    """Placeholder for the stretch goal: ConvLSTM/U-Net based seq-to-image prediction.
    Currently falls back to the physics-informed morphological CA."""
    res = predict_spread(mask, weather_data, horizon_hours)
    if isinstance(res, dict):
        times = sorted(res.keys())
        return res[times[-1]] if times else mask
    return res
