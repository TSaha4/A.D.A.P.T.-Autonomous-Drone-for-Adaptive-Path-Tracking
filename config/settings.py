import os
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_BASE_URL = os.getenv("WEATHER_API_BASE_URL", "https://api.open-meteo.com/v1/forecast")
MAX_SAFE_WIND_SPEED = float(os.getenv("MAX_SAFE_WIND_SPEED", "40.0"))
MAX_SAFE_PRECIPITATION = float(os.getenv("MAX_SAFE_PRECIPITATION", "15.0"))
METERS_PER_PIXEL = float(os.getenv("METERS_PER_PIXEL", "2.0"))

# ---- Mission-planner tuning (see src/mission/constraints.py for sources) ----
# Comma-separated future timestamps (minutes) the flood forecast is indexed by.
DEFAULT_FORECAST_MIN = [int(x) for x in os.getenv("FORECAST_MIN", "30,60,120").split(",") if x.strip()]
FLOOD_BACKEND = os.getenv("FLOOD_BACKEND", "auto")       # auto | dl | ca
HOME_CANDIDATES = int(os.getenv("HOME_CANDIDATES", "16"))
PATH_DOWNSAMPLE = int(os.getenv("PATH_DOWNSAMPLE", "5"))
DEFAULT_PACKAGE_KG = float(os.getenv("DEFAULT_PACKAGE_KG", "0.5"))

# Demo drone (DJI Mavic-class anchors; retune for your airframe).
DRONE_BATTERY_WH = float(os.getenv("DRONE_BATTERY_WH", "77.0"))
DRONE_ENDURANCE_MIN = float(os.getenv("DRONE_ENDURANCE_MIN", "46.0"))
DRONE_CRUISE_MPS = float(os.getenv("DRONE_CRUISE_MPS", "12.0"))
DRONE_PAYLOAD_CAPACITY_KG = float(os.getenv("DRONE_PAYLOAD_CAPACITY_KG", "2.0"))

