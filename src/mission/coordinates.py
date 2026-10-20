import math

def pixel_to_latlon(px, py, image_center_px, geo_center, meters_per_pixel):
    """
    Convert pixel coordinates to lat/lon using a flat-earth approximation.
    
    Args:
        px, py: Target pixel coordinates (x is column, y is row).
        image_center_px: (cx, cy) pixel coordinates of the image center.
        geo_center: (lat, lon) coordinates of the image center.
        meters_per_pixel: Scale factor.
        
    Returns:
        (lat, lon) coordinates of the target pixel, longitude in [-180, 180].

    Raises:
        ValueError: if any input is non-finite, meters_per_pixel <= 0, the reference is not a valid
            non-polar position, the offset would cross a pole, or the east-west offset spans half a
            parallel or more.
    """
    cx, cy = image_center_px
    geo_lat, geo_lon = geo_center

    if not all(math.isfinite(v) for v in (px, py, cx, cy, geo_lat, geo_lon, meters_per_pixel)):
        raise ValueError("pixel_to_latlon inputs must be finite numbers.")
    if meters_per_pixel <= 0:
        raise ValueError(f"meters_per_pixel must be positive, got {meters_per_pixel}.")
    # cos(latitude) is zero at the poles, where the longitude scale below is undefined
    if not (-90.0 < geo_lat < 90.0 and -180.0 <= geo_lon <= 180.0):
        raise ValueError(f"Invalid geographic reference (lat, lon) = {geo_center}.")

    # Image y-axis is inverted (y=0 is at the top/North). 
    # Moving up (py < cy) means moving North (positive latitude shift).
    dy_px = cy - py 
    # Moving right (px > cx) means moving East (positive longitude shift).
    dx_px = px - cx 

    dx_m = dx_px * meters_per_pixel
    dy_m = dy_px * meters_per_pixel

    # Flat earth approximation constants
    # 1 degree of latitude is approx 111,320 meters
    dlat = dy_m / 111320.0
    # 1 degree of longitude is approx 111,320 * cos(latitude) meters
    dlon = dx_m / (111320.0 * math.cos(math.radians(geo_lat)))

    # An east-west offset of half a parallel or more wraps onto longitudes that other offsets also
    # produce, so no unique position exists in this model; this also rejects overflowed (inf) offsets.
    if not abs(dlon) < 180.0:
        raise ValueError(f"A {dx_m:.1f} m east-west offset at latitude {geo_lat} spans half a parallel or more.")

    lat = geo_lat + dlat
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"A {dy_m:.1f} m offset from latitude {geo_lat} crosses a pole.")
    lon = geo_lon + dlon
    # Longitude is periodic: bring results past the antimeridian back into [-180, 180)
    if not -180.0 <= lon <= 180.0:
        lon = (lon + 180.0) % 360.0 - 180.0

    return lat, lon
