import os
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_BASE_URL = os.getenv("WEATHER_API_BASE_URL", "https://api.open-meteo.com/v1/forecast")
MAX_SAFE_WIND_SPEED = float(os.getenv("MAX_SAFE_WIND_SPEED", "40.0"))
MAX_SAFE_PRECIPITATION = float(os.getenv("MAX_SAFE_PRECIPITATION", "15.0"))
METERS_PER_PIXEL = float(os.getenv("METERS_PER_PIXEL", "2.0"))

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

# Multi-point delivery parameters for large flood contours
LARGE_CONTOUR_AREA_THRESHOLD = float(os.getenv("LARGE_CONTOUR_AREA_THRESHOLD", "2500.0"))
LARGE_CONTOUR_AREA_STEP = float(os.getenv("LARGE_CONTOUR_AREA_STEP", "3000.0"))
MAX_DROPS_PER_CONTOUR = int(os.getenv("MAX_DROPS_PER_CONTOUR", "4"))
MIN_DROP_SEPARATION_PX = float(os.getenv("MIN_DROP_SEPARATION_PX", "40.0"))

# Payload release: MAV_CMD_DO_SET_SERVO param1 = servo output, param2 = PWM.
DROP_SERVO_CHANNEL = int(os.getenv("DROP_SERVO_CHANNEL", "9"))
DROP_SERVO_PWM = int(os.getenv("DROP_SERVO_PWM", "2000"))

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

# Minimum distance (px) from any flood pixel (current or forecast) for HOME.
HOME_CLEARANCE_PX = float(os.getenv("HOME_CLEARANCE_PX", "15"))

# Multi-base planning: when no single base can reach every coverable drop point
# within battery range, up to MAX_BASES bases are placed (fewest that cover all),
# each at least MIN_BASE_SEPARATION_PX from every other base.
MAX_BASES = int(os.getenv("MAX_BASES", "4"))
MIN_BASE_SEPARATION_PX = float(os.getenv("MIN_BASE_SEPARATION_PX", "150"))
