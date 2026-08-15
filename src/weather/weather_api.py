import requests
import logging
from functools import lru_cache
from typing import Dict, Any

logger = logging.getLogger(__name__)

from config import settings

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
            
        return {
            "precipitation": current.get("precipitation", 0.0),
            "wind_speed_10m": current.get("wind_speed_10m", 0.0),
            "wind_direction_10m": current.get("wind_direction_10m", 0.0),
            "visibility": current.get("visibility", 10000.0),
            "status": "success"
        }

    except requests.RequestException as e:
        logger.warning(f"Weather API request failed: {e}. Using fallback weather data.")
        return fallback_data
    except ValueError as e:
        logger.warning(f"Failed to parse Weather API response: {e}. Using fallback weather data.")
        return fallback_data
