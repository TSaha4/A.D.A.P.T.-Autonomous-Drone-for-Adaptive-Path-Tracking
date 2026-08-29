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
        (lat, lon) coordinates of the target pixel.
    """
    cx, cy = image_center_px
    geo_lat, geo_lon = geo_center

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

    return geo_lat + dlat, geo_lon + dlon
