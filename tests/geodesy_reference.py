"""
Independent geodetic reference used ONLY by the test-suite to validate
src.mission.coordinates.pixel_to_latlon. Nothing here is imported by production code,
and nothing here imports production code.

Contents
- WGS84 ellipsoid constants and principal radii of curvature.
- Vincenty (1975) direct and inverse geodesic solutions on the WGS84 ellipsoid.
- physical_ground_truth(): where a pixel *physically* lies if the image has a uniform
  scale of `meters_per_pixel`, is north-up, and `geo_center` is the ground position of
  `image_center_px`. Interpreted as an azimuthal-equidistant offset from the reference:
  E = (px - cx) * mpp metres east, N = (cy - py) * mpp metres north.
- documented_model(): an independent re-derivation of the model the production code
  documents (flat earth, 1 degree = 111,320 m, longitude scaled by cos(reference lat)),
  used for exact-conformance checks.
- documented_model_inverse(): test-only inverse of that model (no production inverse exists).
"""
import math

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# Constant used by the documented flat-earth model in src/mission/coordinates.py
MODEL_METERS_PER_DEGREE = 111320.0


def wrap_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def meridional_radius(lat_deg):
    """M(phi): radius of curvature in the meridian (north-south)."""
    s = math.sin(math.radians(lat_deg))
    return WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s * s) ** 1.5


def prime_vertical_radius(lat_deg):
    """N(phi): radius of curvature in the prime vertical; a parallel has radius N*cos(phi)."""
    s = math.sin(math.radians(lat_deg))
    return WGS84_A / math.sqrt(1.0 - WGS84_E2 * s * s)


def _vincenty_ab(cos2_alpha):
    u2 = cos2_alpha * (WGS84_A ** 2 - WGS84_B ** 2) / WGS84_B ** 2
    a = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    b = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    return a, b


def _delta_sigma(b, sin_s, cos_s, cos_2sm):
    return b * sin_s * (cos_2sm + b / 4 * (cos_s * (-1 + 2 * cos_2sm ** 2)
                                           - b / 6 * cos_2sm * (-3 + 4 * sin_s ** 2) * (-3 + 4 * cos_2sm ** 2)))


def vincenty_inverse(lat1, lon1, lat2, lon2, tol=1e-13, max_iter=1000):
    """Geodesic distance (m) and initial azimuth (deg, clockwise from north) on WGS84."""
    if lat1 == lat2 and wrap_lon(lon1) == wrap_lon(lon2):
        return 0.0, 0.0
    L = math.radians(wrap_lon(lon2 - lon1))
    U1 = math.atan((1 - WGS84_F) * math.tan(math.radians(lat1)))
    U2 = math.atan((1 - WGS84_F) * math.tan(math.radians(lat2)))
    sin_u1, cos_u1 = math.sin(U1), math.cos(U1)
    sin_u2, cos_u2 = math.sin(U2), math.cos(U2)
    lam = L
    for _ in range(max_iter):
        sin_l, cos_l = math.sin(lam), math.cos(lam)
        sin_s = math.hypot(cos_u2 * sin_l, cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_l)
        if sin_s == 0:
            return 0.0, 0.0
        cos_s = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_l
        sigma = math.atan2(sin_s, cos_s)
        sin_alpha = cos_u1 * cos_u2 * sin_l / sin_s
        cos2_alpha = 1 - sin_alpha ** 2
        cos_2sm = cos_s - 2 * sin_u1 * sin_u2 / cos2_alpha if cos2_alpha != 0 else 0.0
        c = WGS84_F / 16 * cos2_alpha * (4 + WGS84_F * (4 - 3 * cos2_alpha))
        lam_prev = lam
        lam = L + (1 - c) * WGS84_F * sin_alpha * (
            sigma + c * sin_s * (cos_2sm + c * cos_s * (-1 + 2 * cos_2sm ** 2)))
        if abs(lam - lam_prev) < tol:
            break
    else:
        raise RuntimeError("Vincenty inverse failed to converge (near-antipodal points)")
    a, b = _vincenty_ab(cos2_alpha)
    s = WGS84_B * a * (sigma - _delta_sigma(b, sin_s, cos_s, cos_2sm))
    az1 = math.degrees(math.atan2(cos_u2 * sin_l, cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_l)) % 360.0
    return s, az1


def vincenty_direct(lat1, lon1, az1_deg, distance_m, tol=1e-14, max_iter=1000):
    """Destination (lat, lon) after travelling distance_m along a geodesic with initial azimuth az1_deg."""
    if distance_m == 0:
        return lat1, lon1
    alpha1 = math.radians(az1_deg)
    sin_a1, cos_a1 = math.sin(alpha1), math.cos(alpha1)
    tan_u1 = (1 - WGS84_F) * math.tan(math.radians(lat1))
    cos_u1 = 1 / math.sqrt(1 + tan_u1 ** 2)
    sin_u1 = tan_u1 * cos_u1
    sigma1 = math.atan2(tan_u1, cos_a1)
    sin_alpha = cos_u1 * sin_a1
    cos2_alpha = 1 - sin_alpha ** 2
    a, b = _vincenty_ab(cos2_alpha)
    sigma = distance_m / (WGS84_B * a)
    for _ in range(max_iter):
        cos_2sm = math.cos(2 * sigma1 + sigma)
        sin_s, cos_s = math.sin(sigma), math.cos(sigma)
        sigma_prev = sigma
        sigma = distance_m / (WGS84_B * a) + _delta_sigma(b, sin_s, cos_s, cos_2sm)
        if abs(sigma - sigma_prev) < tol:
            break
    else:
        raise RuntimeError("Vincenty direct failed to converge")
    cos_2sm = math.cos(2 * sigma1 + sigma)
    sin_s, cos_s = math.sin(sigma), math.cos(sigma)
    tmp = sin_u1 * sin_s - cos_u1 * cos_s * cos_a1
    lat2 = math.atan2(sin_u1 * cos_s + cos_u1 * sin_s * cos_a1,
                      (1 - WGS84_F) * math.hypot(sin_alpha, tmp))
    lam = math.atan2(sin_s * sin_a1, cos_u1 * cos_s - sin_u1 * sin_s * cos_a1)
    c = WGS84_F / 16 * cos2_alpha * (4 + WGS84_F * (4 - 3 * cos2_alpha))
    L = lam - (1 - c) * WGS84_F * sin_alpha * (
        sigma + c * sin_s * (cos_2sm + c * cos_s * (-1 + 2 * cos_2sm ** 2)))
    return math.degrees(lat2), wrap_lon(lon1 + math.degrees(L))


def pixel_offset_meters(px, py, image_center_px, meters_per_pixel):
    """(east_m, north_m) of a pixel relative to the reference pixel; image rows grow southwards."""
    cx, cy = image_center_px
    return (px - cx) * meters_per_pixel, (cy - py) * meters_per_pixel


def physical_ground_truth(px, py, image_center_px, geo_center, meters_per_pixel):
    east, north = pixel_offset_meters(px, py, image_center_px, meters_per_pixel)
    distance = math.hypot(east, north)
    azimuth = math.degrees(math.atan2(east, north))
    return vincenty_direct(geo_center[0], geo_center[1], azimuth, distance)


def documented_model(px, py, image_center_px, geo_center, meters_per_pixel):
    """Independent re-statement of the documented equations, expressed via an equivalent sphere:
    R_model = 111,320 m/deg * 180/pi; lat = lat0 + N/R_model; lon = lon0 + E/(R_model cos lat0)."""
    east, north = pixel_offset_meters(px, py, image_center_px, meters_per_pixel)
    r_model = MODEL_METERS_PER_DEGREE * 180.0 / math.pi
    lat0 = math.radians(geo_center[0])
    lat = lat0 + north / r_model
    lon = math.radians(geo_center[1]) + east / (r_model * math.cos(lat0))
    return math.degrees(lat), math.degrees(lon)


def documented_model_inverse(lat, lon, image_center_px, geo_center, meters_per_pixel):
    """Test-only inverse of documented_model: (lat, lon) -> (px, py)."""
    cx, cy = image_center_px
    north = (lat - geo_center[0]) * MODEL_METERS_PER_DEGREE
    east = wrap_lon(lon - geo_center[1]) * MODEL_METERS_PER_DEGREE * math.cos(math.radians(geo_center[0]))
    return cx + east / meters_per_pixel, cy - north / meters_per_pixel


def model_error_bound_m(east, north, lat_deg):
    """
    Upper bound on |documented_model - physical_ground_truth| in metres.

    First-order (scale) terms: the model uses 111,320 m for one degree of latitude, whereas
    WGS84 has M(phi)*pi/180, between 110,574 m (equator) and 111,694 m (pole): relative error
    <= |M(phi)*pi/180 / 111320 - 1| on the north component. For longitude the model uses
    111,320*cos(phi) versus N(phi)*cos(phi)*pi/180: relative error <= |N(phi)*pi/180/111320 - 1|
    on the east component.
    Second-order (curvature) terms: a geodesic and the model's straight graticule offset
    diverge by O(d^2 * (|tan phi| + 1) / R); we allow d^2 * (|tan phi| + 1) / R_min.
    """
    k_n = abs(meridional_radius(lat_deg) * math.pi / 180.0 / MODEL_METERS_PER_DEGREE - 1.0)
    k_e = abs(prime_vertical_radius(lat_deg) * math.pi / 180.0 / MODEL_METERS_PER_DEGREE - 1.0)
    d = math.hypot(east, north)
    curvature = d * d * (abs(math.tan(math.radians(lat_deg))) + 1.0) / WGS84_B
    return k_n * abs(north) + k_e * abs(east) + curvature + 1e-3
