import os
from dotenv import load_dotenv

load_dotenv()


def _env(name, default, cast=float):
    """Numeric setting from the environment / .env; an unparsable value stops the program with a message
    naming the variable instead of a bare float()/int() traceback."""
    raw = os.getenv(name, default)
    try:
        return cast(raw)
    except (TypeError, ValueError):
        kind = "an integer" if cast is int else "a number"
        raise SystemExit(f"Configuration error: {name}={raw!r} (environment or .env) is not {kind}.") from None

WEATHER_API_BASE_URL = os.getenv("WEATHER_API_BASE_URL", "https://api.open-meteo.com/v1/forecast")
MAX_SAFE_WIND_SPEED = _env("MAX_SAFE_WIND_SPEED", "40.0")
MAX_SAFE_PRECIPITATION = _env("MAX_SAFE_PRECIPITATION", "15.0")
METERS_PER_PIXEL = _env("METERS_PER_PIXEL", "2.0")

# Mission aircraft reference: Garuda Aerospace Agri Kisan Drone (GA-AD).
# Manufacturer lists 8 kg rated payload, 5 m/s rated speed, and 7 min tested
# endurance with full payload. Values below are planning assumptions derived
# conservatively from that platform; calibrate with the deployed airframe.
DRONE_MODEL_REFERENCE = "Garuda Aerospace Agri Kisan Drone (GA-AD)"
DRONE_RATED_PAYLOAD_KG = 8.0
DRONE_LOADED_ENDURANCE_MIN = 7.0
DRONE_SPEED_MPS = 5.0
BATTERY_CAPACITY_WH = None  # Manufacturer does not publish this field.
PAYLOAD_CAPACITY_KG = 8.0
PAYLOAD_PER_DROP_KG = 0.25

# Time-budget battery model; payload and headwind increase estimated flight time.
PAYLOAD_TIME_FACTOR_PER_KG = 0.08
WIND_PENALTY_PER_MPS = 0.025
BATTERY_RESERVE_FRACTION = 0.20
INITIAL_BATTERY_FRACTION = 1.0

# Temporal flood growth assumptions, in image pixels per weather unit-hour.
FLOOD_GROWTH_PX_PER_5MM_HOUR = 1.0
FLOOD_WIND_SHIFT_PX_PER_10KMH_HOUR = 1.0
FLOOD_PRECIP_REFERENCE_MM_PER_H = 5.0
FLOOD_WIND_REFERENCE_KMH = 10.0

# Multi-point delivery parameters for flood contours:
# - Normal contours (< MODERATE_CONTOUR_AREA_THRESHOLD): 1 drop point
# - Moderate contours (>= MODERATE_CONTOUR_AREA_THRESHOLD): 2 drop points
# - Big contours (>= LARGE_CONTOUR_AREA_THRESHOLD): 3 drop points
# - Huge contours (>= HUGE_CONTOUR_AREA_THRESHOLD): up to 4 drop points (capped by MAX_DROPS_PER_CONTOUR)
MODERATE_CONTOUR_AREA_THRESHOLD = _env("MODERATE_CONTOUR_AREA_THRESHOLD", "2000.0", float)
LARGE_CONTOUR_AREA_THRESHOLD = _env("LARGE_CONTOUR_AREA_THRESHOLD", "8000.0", float)
HUGE_CONTOUR_AREA_THRESHOLD = _env("HUGE_CONTOUR_AREA_THRESHOLD", "25000.0", float)
LARGE_CONTOUR_AREA_STEP = _env("LARGE_CONTOUR_AREA_STEP", "3000.0", float)
MAX_DROPS_PER_CONTOUR = _env("MAX_DROPS_PER_CONTOUR", "4", int)
MIN_DROP_SEPARATION_PX = _env("MIN_DROP_SEPARATION_PX", "50.0", float)

# Payload release: MAV_CMD_DO_SET_SERVO param1 = servo output, param2 = PWM.
DROP_SERVO_CHANNEL = _env("DROP_SERVO_CHANNEL", "9", int)
DROP_SERVO_PWM = _env("DROP_SERVO_PWM", "2000", int)

# Flood-mask sanity bounds (fraction of image pixels). Outside these the HSV
# sampling almost certainly picked non-flood colours and the run is aborted.
MASK_MIN_COVERAGE_FRACTION = 0.001
MASK_MAX_COVERAGE_FRACTION = 0.50

# Geographic anchor (lat, lon of the IMAGE CENTRE) for the bundled test maps,
# used when --lat/--lon are not given. Estimated by hand from labelled towns
# on each map (approximately +/-5 km); approx_m_per_px is the map's true scale
# from the same estimate, recorded for reference only: the mission still uses
# METERS_PER_PIXEL (see Handoff.md, georeferencing limitation).
REGION_PRESETS = {
    "varanasi": {"lat": 25.14, "lon": 83.11, "region": "Varanasi / Chandauli districts, UP",
                 "approx_m_per_px": 100.0},
    "kanpur": {"lat": 26.45, "lon": 80.05, "region": "Kanpur Nagar / Kanpur Dehat districts, UP",
               "approx_m_per_px": 128.0},
    "bhopal": {"lat": 23.37, "lon": 78.03, "region": "Bhopal / Vidisha / Raisen districts, MP",
               "approx_m_per_px": 270.0},
}

# Minimum distance (display px) from any flood pixel (current or forecast) for a base.
HOME_CLEARANCE_PX = _env("HOME_CLEARANCE_PX", "15")

# Multi-base planning: when no single base can reach every coverable drop point
# within battery range, up to MAX_BASES bases are placed (fewest that cover all),
# each at least MIN_BASE_SEPARATION_PX (display px) from every other base.
# MAX_BASES=1 restricts the planner to a single base (one UAV).
MAX_BASES = _env("MAX_BASES", "4", int)
MIN_BASE_SEPARATION_PX = _env("MIN_BASE_SEPARATION_PX", "150")
