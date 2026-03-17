"""Weather tool — free current conditions and forecast via Open-Meteo.

No API key required.  Geocoding via Open-Meteo's own geocoding endpoint.
Returns current conditions plus a 3-day high/low/rain-chance forecast.
"""

from __future__ import annotations

from typing import Any

import httpx

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

_GEO_URL     = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes → human description
_WMO: dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Rain showers", 81: "Heavy showers", 82: "Violent showers",
    85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Heavy thunderstorm",
}


def _geocode(city: str) -> tuple[float, float, str] | None:
    """Return (lat, lon, display_name) or None."""
    try:
        resp = httpx.get(
            _GEO_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None
        r = results[0]
        parts = [r.get("name", ""), r.get("admin1", ""), r.get("country", "")]
        display = ", ".join(p for p in parts if p)
        return r["latitude"], r["longitude"], display
    except Exception:
        return None


@ToolRegistry.register("weather")
class WeatherTool(BaseTool):
    """Fetch current weather and 3-day forecast for any city via Open-Meteo."""

    tool_id = "weather"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="weather",
            description=(
                "Get current weather conditions and a 3-day forecast for any city. "
                "Uses the free Open-Meteo API — no API key needed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g. 'London', 'New York', 'Tokyo').",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units (default: celsius).",
                    },
                },
                "required": ["city"],
            },
            category="utility",
        )

    def execute(self, **params: Any) -> ToolResult:
        city: str  = params.get("city", "").strip()
        units: str = params.get("units", "celsius")

        if not city:
            return ToolResult(
                tool_name="weather",
                content="Error: city is required.",
                success=False,
            )

        geo = _geocode(city)
        if geo is None:
            return ToolResult(
                tool_name="weather",
                content=f"Could not geocode '{city}'. Try a more specific city name.",
                success=False,
            )

        lat, lon, display_name = geo
        temp_unit = "fahrenheit" if units == "fahrenheit" else "celsius"
        temp_sym  = "°F" if units == "fahrenheit" else "°C"
        wind_unit = "mph" if units == "fahrenheit" else "kmh"

        try:
            resp = httpx.get(
                _WEATHER_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": [
                        "temperature_2m",
                        "apparent_temperature",
                        "relative_humidity_2m",
                        "weathercode",
                        "windspeed_10m",
                        "precipitation",
                    ],
                    "daily": [
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                        "weathercode",
                    ],
                    "temperature_unit": temp_unit,
                    "windspeed_unit": wind_unit,
                    "forecast_days": 4,
                    "timezone": "auto",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return ToolResult(
                tool_name="weather",
                content=f"Weather API error: {exc}",
                success=False,
            )

        cur  = data.get("current", {})
        day  = data.get("daily", {})

        temp        = cur.get("temperature_2m", "?")
        feels_like  = cur.get("apparent_temperature", "?")
        humidity    = cur.get("relative_humidity_2m", "?")
        wmo_code    = cur.get("weathercode", 0)
        wind        = cur.get("windspeed_10m", "?")
        precip      = cur.get("precipitation", 0)
        condition   = _WMO.get(int(wmo_code) if isinstance(wmo_code, (int, float)) else 0, "Unknown")

        lines = [
            f"**Weather for {display_name}**",
            f"Condition : {condition}",
            f"Temp      : {temp}{temp_sym}  (feels like {feels_like}{temp_sym})",
            f"Humidity  : {humidity}%",
            f"Wind      : {wind} {wind_unit}",
        ]
        if precip:
            lines.append(f"Precip    : {precip} mm")

        # 3-day forecast (skip today = index 0)
        dates   = day.get("time", [])
        hi_list = day.get("temperature_2m_max", [])
        lo_list = day.get("temperature_2m_min", [])
        pp_list = day.get("precipitation_probability_max", [])
        wc_list = day.get("weathercode", [])

        if len(dates) > 1:
            lines.append("\n**3-Day Forecast**")
            for i in range(1, min(4, len(dates))):
                date  = dates[i]
                hi    = hi_list[i] if i < len(hi_list) else "?"
                lo    = lo_list[i] if i < len(lo_list) else "?"
                pp    = pp_list[i] if i < len(pp_list) else "?"
                wc    = wc_list[i] if i < len(wc_list) else 0
                desc  = _WMO.get(int(wc) if isinstance(wc, (int, float)) else 0, "")
                lines.append(
                    f"{date}: {lo}{temp_sym}–{hi}{temp_sym}  {pp}% rain  {desc}"
                )

        return ToolResult(
            tool_name="weather",
            content="\n".join(lines),
            success=True,
            metadata={"city": display_name, "lat": lat, "lon": lon},
        )


__all__ = ["WeatherTool"]
