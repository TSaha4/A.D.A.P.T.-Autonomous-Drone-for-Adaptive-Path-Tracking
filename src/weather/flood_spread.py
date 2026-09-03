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


def predict_spread_sequence(mask: np.ndarray, weather_data: dict,
                            timestamps_min, backend: str = "auto"):
    """
    Time-indexed flood prediction used by the mission planner.

    Replaces the old single static ``predict_spread`` output with a
    :class:`~src.weather.flood_predictor.FloodTimeline`: one flood contour per
    requested future timestamp so routing can query the obstacle state at the
    time the drone will actually arrive (not just the current forecast).

    ``backend`` selects the DL layer ("dl"/"auto") or the physics baseline
    ("ca"). The CA model is kept as a fallback baseline for comparison.
    """
    from src.weather.flood_predictor import predict_flood_sequence
    return predict_flood_sequence(mask, weather_data, timestamps_min, backend=backend)
