import unittest
from unittest.mock import patch, MagicMock
from src.weather.weather_api import get_weather_data

class TestWeatherAPI(unittest.TestCase):
    def setUp(self):
        # Clear cache before each test to ensure fresh state
        get_weather_data.cache_clear()

    @patch('src.weather.weather_api.requests.get')
    def test_get_weather_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "current": {
                "precipitation": 2.5,
                "wind_speed_10m": 15.0,
                "wind_direction_10m": 180,
                "visibility": 8000.0
            }
        }
        mock_get.return_value = mock_response

        data = get_weather_data(25.3176, 82.9739)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["precipitation"], 2.5)
        self.assertEqual(data["wind_speed_10m"], 15.0)
        
        # Test cache (mock should only be called once)
        get_weather_data(25.3176, 82.9739)
        mock_get.assert_called_once()

    @patch('src.weather.weather_api.requests.get')
    def test_get_weather_failure_fallback(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("Connection error")

        data = get_weather_data(25.3176, 82.9739)
        self.assertEqual(data["status"], "fallback")
        self.assertEqual(data["precipitation"], 0.0)
        self.assertEqual(data["wind_speed_10m"], 0.0)
        self.assertEqual(data["visibility"], 10000.0)

if __name__ == '__main__':
    unittest.main()
