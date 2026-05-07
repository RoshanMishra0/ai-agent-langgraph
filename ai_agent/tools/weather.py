import requests
from langchain.tools import tool
from config import OPENWEATHER_API_KEY

@tool
def get_weather(city: str) -> str:
    """Get real-time weather"""

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"

        params = {
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data.get("cod") != 200:
            return f"[WEATHER ERROR] {data.get('message')}"

        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]

        return f"{city}: {temp}°C, {desc}"

    except Exception as e:
        return str(e)
