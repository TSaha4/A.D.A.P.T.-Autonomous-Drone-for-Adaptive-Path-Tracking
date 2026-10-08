import requests
import logging
import math
from functools import lru_cache
from typing import Dict, Any

logger = logging.getLogger(__name__)

from config import settings

WEATHER_FIELDS = ("precipitation", "wind_speed_10m", "wind_direction_10m")


def sanitize_weather(raw, source="Weather data") -> Dict[str, Any]:
    """Copy of ``raw`` in which every field the pipeline uses is a finite float.

    The API can return null for a field, and a weather file can lack fields or not be an object at all. Such a
    field becomes 0.0 (calm, as for an unavailable API) and a warning names it; the status "success" becomes
    "partial". Other keys are kept.
    """
    data = dict(raw) if isinstance(raw, dict) else {}
    bad = []
    for key in WEATHER_FIELDS:
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            data[key] = float(value)
        else:
            bad.append(key)
            data[key] = 0.0
    if bad:
        logger.warning("%s: %s missing or not a finite number; using 0 (calm) instead, so flood spread and wind "
                       "effects may be under-estimated.", source, ", ".join(bad))
        if data.get("status") == "success":
            data["status"] = "partial"
    return data

# Cache the results to avoid redundant calls for the same coordinates within the same run.
# A maxsize of 32 is plenty for a typical drone mission over a single mapped area.
@lru_cache(maxsize=32)
def get_weather_data(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetches current weather data (wind, precipitation, visibility) from Open-Meteo.
    Uses in-memory caching to avoid redundant calls.
    
    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        
    Returns:
        A dictionary containing weather parameters. Fallbacks are provided if the API fails.
    """
    url = settings.WEATHER_API_BASE_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "precipitation,wind_speed_10m,wind_direction_10m,visibility",
        "timezone": "auto"
    }

    fallback_data = {
        "precipitation": 0.0,
        "wind_speed_10m": 0.0,
        "wind_direction_10m": 0.0,
        "visibility": 10000.0,
        "status": "fallback"
    }

    try:
        response = requests.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        if not current:
            logger.warning("Weather API returned no current data. Using fallbacks.")
            return fallback_data
            
        return sanitize_weather({
            "precipitation": current.get("precipitation", 0.0),
            "wind_speed_10m": current.get("wind_speed_10m", 0.0),
            "wind_direction_10m": current.get("wind_direction_10m", 0.0),
            "visibility": current.get("visibility", 10000.0),
            "status": "success"
        }, "Weather API")

    except requests.RequestException as e:
        logger.warning(f"Weather API request failed: {e}. Using fallback weather data.")
        return fallback_data
    except ValueError as e:
        logger.warning(f"Failed to parse Weather API response: {e}. Using fallback weather data.")
        return fallback_data
