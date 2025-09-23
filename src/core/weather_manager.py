"""
Weather Manager - Fetch and process weather data for solar forecasting

Supports multiple weather APIs with fallback options:
- OpenWeatherMap (solar-specific API)
- Open-Meteo (free alternative)
- NREL NSRDB (historical solar data)
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import time
from dataclasses import dataclass
from enum import Enum

class WeatherProvider(Enum):
    """Supported weather API providers"""
    OPENWEATHER = "openweather"
    OPEN_METEO = "open_meteo"
    NREL = "nrel"

@dataclass
class WeatherData:
    """Standardized weather data structure"""
    timestamp: datetime
    temperature: float  # °C
    humidity: float  # %
    cloud_cover: float  # %
    visibility: float  # km
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    pressure: float  # hPa
    # Solar-specific data
    ghi: Optional[float] = None  # Global Horizontal Irradiance (W/m²)
    dni: Optional[float] = None  # Direct Normal Irradiance (W/m²)
    dhi: Optional[float] = None  # Diffuse Horizontal Irradiance (W/m²)
    uv_index: Optional[float] = None
    solar_elevation: Optional[float] = None  # degrees

class WeatherManager:
    """
    Unified weather data manager with multiple provider support
    """

    def __init__(self, cache_dir: str = "data/cache/weather"):
        """
        Initialize weather manager

        Args:
            cache_dir: Directory for caching weather data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load API keys from environment
        self.openweather_api_key = os.environ.get('OPENWEATHER_API_KEY')
        self.nrel_api_key = os.environ.get('NREL_API_KEY')

        # API endpoints
        self.openweather_base_url = "https://api.openweathermap.org/data/3.0"
        self.open_meteo_base_url = "https://api.open-meteo.com/v1"
        self.open_meteo_archive_url = "https://archive-api.open-meteo.com/v1"
        self.nrel_base_url = "https://developer.nrel.gov/api"

        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 1.0  # seconds between requests

    def get_weather_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 7,
        provider: WeatherProvider = WeatherProvider.OPEN_METEO
    ) -> List[WeatherData]:
        """
        Get weather forecast for specified location

        Args:
            latitude: Location latitude
            longitude: Location longitude
            days: Number of forecast days (1-15)
            provider: Weather API provider to use

        Returns:
            List of weather data points
        """

        if provider == WeatherProvider.OPENWEATHER and self.openweather_api_key:
            return self._get_openweather_forecast(latitude, longitude, days)
        elif provider == WeatherProvider.OPEN_METEO:
            return self._get_open_meteo_forecast(latitude, longitude, days)
        else:
            # Fallback to Open-Meteo if primary provider fails
            print(f"⚠️ {provider.value} not available, falling back to Open-Meteo")
            return self._get_open_meteo_forecast(latitude, longitude, days)

    def get_solar_irradiance_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 7
    ) -> pd.DataFrame:
        """
        Get solar irradiance forecast optimized for solar forecasting

        Args:
            latitude: Location latitude
            longitude: Location longitude
            days: Number of forecast days

        Returns:
            DataFrame with solar irradiance data
        """

        # Try OpenWeather solar API if available
        if self.openweather_api_key:
            try:
                return self._get_openweather_solar_forecast(latitude, longitude, days)
            except Exception as e:
                print(f"⚠️ OpenWeather solar API failed: {e}")

        # Fallback to Open-Meteo solar data
        return self._get_open_meteo_solar_forecast(latitude, longitude, days)

    def get_historical_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime,
        provider: WeatherProvider = WeatherProvider.OPEN_METEO
    ) -> List[WeatherData]:
        """
        Get historical weather data

        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date for historical data
            end_date: End date for historical data
            provider: Weather API provider

        Returns:
            List of historical weather data
        """

        cache_key = f"hist_{latitude}_{longitude}_{start_date.date()}_{end_date.date()}_{provider.value}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data

        try:
            if provider == WeatherProvider.OPEN_METEO:
                data = self._get_open_meteo_historical(latitude, longitude, start_date, end_date)
            elif provider == WeatherProvider.NREL and self.nrel_api_key:
                data = self._get_nrel_historical(latitude, longitude, start_date, end_date)
            else:
                # Fallback
                data = self._get_open_meteo_historical(latitude, longitude, start_date, end_date)

            self._save_to_cache(cache_key, data)
            return data

        except Exception as e:
            print(f"⚠️ Warning: Historical weather data unavailable: {e}")
            print("   This might be due to Open-Meteo archive limitations for recent dates")
            print("   Generating synthetic historical weather for demonstration...")

            # Generate synthetic weather data based on location and season
            return self._generate_synthetic_historical_weather(latitude, longitude, start_date, end_date)

    def _get_openweather_forecast(self, lat: float, lon: float, days: int) -> List[WeatherData]:
        """Get forecast from OpenWeatherMap"""
        if not self.openweather_api_key:
            raise ValueError("OpenWeatherMap API key required")

        self._rate_limit('openweather')

        url = f"{self.openweather_base_url}/onecall"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.openweather_api_key,
            'units': 'metric',
            'exclude': 'minutely,alerts'
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        weather_data = []

        # Process hourly forecasts
        for hour_data in data.get('hourly', [])[:days*24]:
            weather_data.append(WeatherData(
                timestamp=datetime.fromtimestamp(hour_data['dt']),
                temperature=hour_data['temp'],
                humidity=hour_data['humidity'],
                cloud_cover=hour_data['clouds'],
                visibility=hour_data.get('visibility', 10000) / 1000,  # Convert to km
                wind_speed=hour_data['wind_speed'],
                wind_direction=hour_data.get('wind_deg', 0),
                pressure=hour_data['pressure'],
                uv_index=hour_data.get('uvi', 0)
            ))

        return weather_data

    def _get_open_meteo_forecast(self, lat: float, lon: float, days: int) -> List[WeatherData]:
        """Get forecast from Open-Meteo (free)"""

        self._rate_limit('open_meteo')

        url = f"{self.open_meteo_base_url}/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': ','.join([
                'temperature_2m', 'relative_humidity_2m', 'cloud_cover',
                'visibility', 'wind_speed_10m', 'wind_direction_10m',
                'surface_pressure', 'shortwave_radiation', 'direct_radiation',
                'diffuse_radiation', 'uv_index'
            ]),
            'forecast_days': min(days, 16),  # Open-Meteo max is 16 days
            'timezone': 'auto'
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        weather_data = []
        hourly = data['hourly']

        for i in range(len(hourly['time'])):
            weather_data.append(WeatherData(
                timestamp=datetime.fromisoformat(hourly['time'][i].replace('T', ' ')),
                temperature=hourly['temperature_2m'][i] or 0,
                humidity=hourly['relative_humidity_2m'][i] or 0,
                cloud_cover=hourly['cloud_cover'][i] or 0,
                visibility=hourly['visibility'][i] / 1000 if hourly['visibility'][i] else 10,
                wind_speed=hourly['wind_speed_10m'][i] or 0,
                wind_direction=hourly['wind_direction_10m'][i] or 0,
                pressure=hourly['surface_pressure'][i] or 1013,
                ghi=hourly['shortwave_radiation'][i],  # Global Horizontal Irradiance
                dni=hourly['direct_radiation'][i],     # Direct Normal Irradiance
                dhi=hourly['diffuse_radiation'][i],    # Diffuse Horizontal Irradiance
                uv_index=hourly['uv_index'][i] or 0
            ))

        return weather_data

    def _get_openweather_solar_forecast(self, lat: float, lon: float, days: int) -> pd.DataFrame:
        """Get solar forecast from OpenWeatherMap Solar API"""
        # This would require the solar-specific OpenWeather subscription
        # For now, return empty DataFrame
        print("OpenWeather Solar API integration coming soon...")
        return pd.DataFrame()

    def _get_open_meteo_solar_forecast(self, lat: float, lon: float, days: int) -> pd.DataFrame:
        """Get solar irradiance forecast from Open-Meteo"""

        self._rate_limit('open_meteo_solar')

        url = f"{self.open_meteo_base_url}/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': ','.join([
                'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
                'direct_normal_irradiance', 'global_tilted_irradiance',
                'cloud_cover', 'temperature_2m'
            ]),
            'forecast_days': min(days, 16),
            'timezone': 'auto'
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Convert to DataFrame
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        # Rename columns for consistency
        df.rename(columns={
            'shortwave_radiation': 'ghi',
            'direct_radiation': 'dni',
            'diffuse_radiation': 'dhi',
            'direct_normal_irradiance': 'dni_normal',
            'global_tilted_irradiance': 'gti'
        }, inplace=True)

        return df

    def _get_open_meteo_historical(
        self, lat: float, lon: float, start_date: datetime, end_date: datetime
    ) -> List[WeatherData]:
        """Get historical weather from Open-Meteo"""

        self._rate_limit('open_meteo_historical')

        url = f"{self.open_meteo_archive_url}/archive"  # Use correct archive API URL
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'hourly': ','.join([
                'temperature_2m', 'relative_humidity_2m', 'cloud_cover',
                'wind_speed_10m', 'wind_direction_10m', 'surface_pressure',
                'shortwave_radiation', 'direct_radiation', 'diffuse_radiation'
            ]),
            'timezone': 'auto'
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        weather_data = []
        hourly = data['hourly']

        for i in range(len(hourly['time'])):
            weather_data.append(WeatherData(
                timestamp=datetime.fromisoformat(hourly['time'][i].replace('T', ' ')),
                temperature=hourly['temperature_2m'][i] or 0,
                humidity=hourly['relative_humidity_2m'][i] or 0,
                cloud_cover=hourly['cloud_cover'][i] or 0,
                visibility=10.0,  # Not available in historical data
                wind_speed=hourly['wind_speed_10m'][i] or 0,
                wind_direction=hourly['wind_direction_10m'][i] or 0,
                pressure=hourly['surface_pressure'][i] or 1013,
                ghi=hourly['shortwave_radiation'][i],
                dni=hourly['direct_radiation'][i],
                dhi=hourly['diffuse_radiation'][i]
            ))

        return weather_data

    def _get_nrel_historical(
        self, lat: float, lon: float, start_date: datetime, end_date: datetime
    ) -> List[WeatherData]:
        """Get historical solar data from NREL"""
        # NREL NSRDB integration would go here
        print("NREL historical data integration coming soon...")
        return []

    def _rate_limit(self, provider: str):
        """Apply rate limiting for API requests"""
        now = time.time()
        if provider in self.last_request_time:
            elapsed = now - self.last_request_time[provider]
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        self.last_request_time[provider] = time.time()

    def _load_from_cache(self, cache_key: str) -> Optional[List[WeatherData]]:
        """Load data from cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                # Convert back to WeatherData objects
                return [WeatherData(**item) for item in data]
            except Exception as e:
                print(f"⚠️ Cache load failed: {e}")
        return None

    def _save_to_cache(self, cache_key: str, data: List[WeatherData]):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            # Convert WeatherData objects to dicts for JSON serialization
            json_data = []
            for item in data:
                item_dict = item.__dict__.copy()
                item_dict['timestamp'] = item.timestamp.isoformat()
                json_data.append(item_dict)

            with open(cache_file, 'w') as f:
                json.dump(json_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")

    def get_weather_summary(self, latitude: float, longitude: float) -> Dict:
        """
        Get weather data summary for a location

        Args:
            latitude: Location latitude
            longitude: Location longitude

        Returns:
            Dictionary with weather capabilities and provider info
        """

        summary = {
            'location': {'latitude': latitude, 'longitude': longitude},
            'available_providers': [],
            'capabilities': {
                'forecast_days': 0,
                'historical_data': False,
                'solar_irradiance': False,
                'hourly_data': False
            }
        }

        # Check Open-Meteo (always available)
        summary['available_providers'].append({
            'name': 'Open-Meteo',
            'type': 'free',
            'forecast_days': 16,
            'solar_data': True
        })
        summary['capabilities']['forecast_days'] = 16
        summary['capabilities']['historical_data'] = True
        summary['capabilities']['solar_irradiance'] = True
        summary['capabilities']['hourly_data'] = True

        # Check OpenWeatherMap
        if self.openweather_api_key:
            summary['available_providers'].append({
                'name': 'OpenWeatherMap',
                'type': 'premium',
                'forecast_days': 8,
                'solar_data': True
            })

        # Check NREL
        if self.nrel_api_key:
            summary['available_providers'].append({
                'name': 'NREL',
                'type': 'government',
                'forecast_days': 0,
                'solar_data': True,
                'historical_only': True
            })

        return summary

    def _generate_synthetic_historical_weather(
        self, lat: float, lon: float, start_date: datetime, end_date: datetime
    ) -> List[WeatherData]:
        """Generate synthetic historical weather data for demonstration"""
        import numpy as np

        weather_data = []
        current_date = start_date

        while current_date <= end_date:
            # Generate realistic weather based on season and location
            day_of_year = current_date.timetuple().tm_yday

            # Seasonal temperature variation (simple model)
            base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            # Add some latitude effect
            if lat > 40:  # Northern locations
                base_temp -= 5
            elif lat < 25:  # Southern locations
                base_temp += 5

            # Add daily variation (24 hours)
            for hour in range(24):
                timestamp = current_date.replace(hour=hour)

                # Daily temperature cycle
                hour_temp_factor = -5 * np.cos(2 * np.pi * hour / 24)
                temperature = base_temp + hour_temp_factor + np.random.normal(0, 3)

                # Generate correlated weather parameters
                humidity = max(20, min(95, 60 + np.random.normal(0, 15)))
                cloud_cover = max(0, min(100, np.random.uniform(0, 100)))
                wind_speed = max(0, np.random.exponential(5))
                pressure = 1013 + np.random.normal(0, 10)

                # Solar irradiance (simplified model)
                sun_elevation = max(0, 60 * np.sin(2 * np.pi * hour / 24) * np.sin(2 * np.pi * day_of_year / 365))
                max_ghi = 1000 * np.sin(np.pi * sun_elevation / 90) if sun_elevation > 0 else 0
                ghi = max_ghi * (1 - cloud_cover / 100) + np.random.normal(0, 50)
                ghi = max(0, ghi)

                dni = ghi * 0.8 if ghi > 0 else 0
                dhi = ghi * 0.2 if ghi > 0 else 0

                weather_data.append(WeatherData(
                    timestamp=timestamp,
                    temperature=temperature,
                    humidity=humidity,
                    cloud_cover=cloud_cover,
                    visibility=10.0,
                    wind_speed=wind_speed,
                    wind_direction=np.random.uniform(0, 360),
                    pressure=pressure,
                    ghi=ghi,
                    dni=dni,
                    dhi=dhi,
                    uv_index=max(0, ghi / 100)
                ))

            current_date += timedelta(days=1)

        print(f"   Generated {len(weather_data)} synthetic weather data points")
        return weather_data


def create_weather_manager() -> WeatherManager:
    """
    Convenience function to create WeatherManager with default settings

    Returns:
        Configured WeatherManager instance
    """
    return WeatherManager()