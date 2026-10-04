"""Throwaway direct diagnostic for flood forecast frames and interpolation.

Run from the repository root with:
    .venv/Scripts/python.exe debug_flood_spread.py
"""
from pathlib import Path

import cv2
import numpy as np

from src.vision.image_processing import ImageProcessor
from src.weather.flood_spread import (predict_spread, obstacle_mask_at, forecast_at,
                                      interpolate_forecast_mask)


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "data" / "input" / "varanasi.png"
SAMPLE_POINTS = [(400, 340), (462, 264), (546, 314)]
WEATHER = {
    "precipitation": 10.0,
    "wind_speed_10m": 30.0,
    "wind_direction_10m": 270.0,
}
HORIZON_HOURS = 2.0
QUERY_HOURS = [0.05, 0.15, 0.3, 0.5]


def report_mask(label, mask):
    values, counts = np.unique(mask, return_counts=True)
    print(f"{label}: shape={mask.shape}, dtype={mask.dtype}, "
          f"unique_counts={dict(zip(values.tolist(), counts.tolist()))}")


def sample_pixel_raw(mask, point_xy):
    """Replicate production's direct mask[y, x] access without conversion."""
    x, y = point_xy
    return mask[y, x]


def main():
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(IMAGE_PATH)
    height, width = image.shape[:2]
    if width > 750:
        image = cv2.resize(image, (750, int(round(height * 750 / width))),
                           interpolation=cv2.INTER_AREA)

    processor = ImageProcessor()
    processor.image = image.copy()
    processor.sample_points = list(SAMPLE_POINTS)
    processor.compute_dynamic_hsv()
    flood_mask = processor.mask_flood_areas(image)
    contours = processor.find_filtered_contours(flood_mask, min_area=200)
    print(f"Input image={IMAGE_PATH.name}, size={image.shape[1]}x{image.shape[0]}, "
          f"fixed_samples={SAMPLE_POINTS}, filtered_contours={len(contours)}")
    report_mask("current flood mask", flood_mask)

    forecasts = predict_spread(flood_mask, WEATHER, HORIZON_HOURS)
    print(f"weather={WEATHER}, horizon_hours={HORIZON_HOURS}")
    for timestamp, frame in forecasts.items():
        report_mask(f"forecast t=+{timestamp:.6f}h", frame)

    for arrival in QUERY_HOURS:
        blended = interpolate_forecast_mask(forecasts, arrival, flood_mask)[0]
        sampled = obstacle_mask_at(forecasts, arrival, flood_mask)
        _, details = forecast_at(forecasts, arrival, flood_mask)
        report_mask(f"linear_interpolation t=+{arrival:.3f}h details={details}", blended)
        report_mask(f"obstacle_mask_at t=+{arrival:.3f}h details={details}", sampled)

    known = np.zeros((3, 4), dtype=np.uint8)
    known[1, 2] = 173
    point = (2, 1)
    raw_value = sample_pixel_raw(known, point)
    print("raw pixel sample: "
          f"point_xy={point}, value={raw_value!r}, "
          f"type={type(raw_value).__name__}, "
          f"shape={getattr(raw_value, 'shape', None)}, "
          f"dtype={getattr(raw_value, 'dtype', None)}")


if __name__ == "__main__":
    main()
