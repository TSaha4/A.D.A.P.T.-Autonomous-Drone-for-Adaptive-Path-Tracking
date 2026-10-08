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

# --- Multi-base / multi-UAV mode (main.py --mode multi); ported from the MAIN branch -------------

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

# Dry drop points (multi-base mode): large flood zones get several drop points on their boundary.
LARGE_CONTOUR_AREA_THRESHOLD = _env("LARGE_CONTOUR_AREA_THRESHOLD", "2500.0")
LARGE_CONTOUR_AREA_STEP = _env("LARGE_CONTOUR_AREA_STEP", "3000.0")
MAX_DROPS_PER_CONTOUR = _env("MAX_DROPS_PER_CONTOUR", "4", int)
MIN_DROP_SEPARATION_PX = _env("MIN_DROP_SEPARATION_PX", "40.0")

# Minimum distance (display px) from any flood pixel (current or forecast) for a base.
HOME_CLEARANCE_PX = _env("HOME_CLEARANCE_PX", "15")

# Multi-base planning: when no single base can reach every coverable drop point
# within battery range, up to MAX_BASES bases are placed (fewest that cover all),
# each at least MIN_BASE_SEPARATION_PX (display px) from every other base.
# MAX_BASES=1 restricts the planner to a single base (one UAV).
MAX_BASES = _env("MAX_BASES", "4", int)
MIN_BASE_SEPARATION_PX = _env("MIN_BASE_SEPARATION_PX", "150")
