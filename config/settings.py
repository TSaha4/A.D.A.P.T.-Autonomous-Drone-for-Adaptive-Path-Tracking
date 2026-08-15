import os
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_BASE_URL = os.getenv("WEATHER_API_BASE_URL", "https://api.open-meteo.com/v1/forecast")
MAX_SAFE_WIND_SPEED = float(os.getenv("MAX_SAFE_WIND_SPEED", "40.0"))
MAX_SAFE_PRECIPITATION = float(os.getenv("MAX_SAFE_PRECIPITATION", "15.0"))
