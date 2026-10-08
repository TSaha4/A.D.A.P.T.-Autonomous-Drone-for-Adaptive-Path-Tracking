import cv2
import numpy as np
import math
from typing import List, Tuple

def predict_spread(mask: np.ndarray, weather_data: dict, horizon_hours: float) -> np.ndarray:
    """
    Predicts the spread of floods using a simple morphological cellular automata approach
    influenced by wind speed/direction and precipitation.

    Args:
        mask: The current flood binary mask (numpy array, 255 for flood, 0 for dry).
        weather_data: Dictionary containing 'precipitation', 'wind_speed_10m', 'wind_direction_10m'.
        horizon_hours: How many hours into the future to predict (each hour is one CA iteration).

    Returns:
        np.ndarray: The predicted flood mask.
    """
    if mask is None or mask.size == 0:
        return mask

    # Initialize predicted mask
    pred_mask = mask.copy()
    
    precip = weather_data.get("precipitation", 0.0)
    wind_speed = weather_data.get("wind_speed_10m", 0.0)
    # Meteorological wind direction is where the wind originates.
    # To get where it's blowing TO, we add 180 degrees.
    wind_dir_deg = weather_data.get("wind_direction_10m", 0.0) + 180.0
    wind_dir_rad = math.radians(wind_dir_deg)

    # Base spread from precipitation
    # 0 precip -> no isotropic spread. High precip -> more isotropic spread.
    # Arbitrary heuristic: 1 pixel of spread per 5mm of precip per hour
    base_spread_pixels = max(0, int(precip / 5.0))
    # Growth larger than the image diagonal already reaches every pixel; a larger element changes nothing but
    # its size, which for implausible precipitation values would exhaust memory.
    base_spread_pixels = min(base_spread_pixels, int(math.ceil(math.hypot(*mask.shape[:2]))) + 2)

    # Wind spread
    # Arbitrary heuristic: 1 pixel of directional spread per 10km/h of wind
    wind_spread_pixels = max(0, int(wind_speed / 10.0))
    
    dx = int(round(math.sin(wind_dir_rad) * wind_spread_pixels))
    # Image y-axis is inverted (0 at top, positive down)
    # North is negative y, South is positive y
    # cos(0) = 1 (North), so in image coords this should be -1.
    dy = int(round(-math.cos(wind_dir_rad) * wind_spread_pixels))

    iterations = max(1, int(horizon_hours))

    for _ in range(iterations):
        # 1. Isotropic spread based on precipitation
        if base_spread_pixels > 0:
            kernel_size = 2 * base_spread_pixels + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            pred_mask = cv2.dilate(pred_mask, kernel, iterations=1)
        
        # 2. Directional spread based on wind
        if wind_spread_pixels > 0:
            # Shift image by dx, dy
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(pred_mask, M, (pred_mask.shape[1], pred_mask.shape[0]))
            # Combine
            pred_mask = cv2.bitwise_or(pred_mask, shifted)

    return pred_mask


def forecast_frames(mask: np.ndarray, weather_data: dict, horizon_hours: float, predictor=None) -> dict:
    """Time-indexed forecast ``{hours_from_now: binary mask}`` built from :func:`predict_spread`.

    Used by the multi-base planner, which checks obstacles at the time a UAV arrives.
    ``predict_spread`` applies one spread iteration per hour, so the frame for ``k`` hours is
    ``predict_spread(mask, weather_data, k)``. It is computed as one further iteration of frame k-1, which
    gives the same mask with one iteration per frame instead of k. Frames are made for
    k = 1 .. max(1, int(horizon_hours)) and stamped ``min(k, horizon_hours)``; the last one is therefore
    exactly the mask ``predict_spread`` returns for the full horizon. The 0 h frame is the current mask.
    Frames are binary (0/255) and cumulative. Times between frames are queried with :func:`forecast_at`.
    ``predictor`` defaults to :func:`predict_spread`.
    """
    if mask is None or mask.size == 0:
        return {}
    predictor = predict_spread if predictor is None else predictor
    current = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if horizon_hours <= 0:
        # predict_spread still runs one iteration for a non-positive horizon; that spread is treated as present now
        predicted = (np.asarray(predictor(mask, weather_data, horizon_hours)) > 0).astype(np.uint8) * 255
        return {0.0: cv2.bitwise_or(current, predicted)}
    iterations = max(1, int(horizon_hours))
    frames = {0.0: current}
    previous = current
    for k in range(1, iterations + 1):
        frame = (np.asarray(predictor(previous, weather_data, 1)) > 0).astype(np.uint8) * 255
        frame = cv2.bitwise_or(frame, previous)
        frames[float(min(k, horizon_hours))] = frame
        previous = frame
    return frames


# --- Arrival-time queries over forecast frames (ported from the MAIN branch) ------------------------

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
    """
    Placeholder for the stretch goal: ConvLSTM/U-Net based seq-to-image prediction.
    Currently falls back to the physics-informed morphological CA.
    """
    return predict_spread(mask, weather_data, horizon_hours)
