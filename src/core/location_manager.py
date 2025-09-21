"""
Location Manager for Solar Energy Analysis

Handles location-based solar modeling including:
- Solar irradiance calculations based on latitude/longitude
- Seasonal adjustments for specific locations
- Weather pattern modeling
- Timezone handling
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, Optional, Union
import logging


class LocationManager:
    """
    Manages location-specific solar energy calculations and weather modeling
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        timezone_str: str = None,
        location_name: str = None,
    ):
        """
        Initialize location manager

        Args:
            latitude: Latitude in decimal degrees (-90 to 90)
            longitude: Longitude in decimal degrees (-180 to 180)
            timezone_str: Timezone string (e.g., 'America/New_York')
            location_name: Human-readable location name
        """
        self.latitude = self._validate_latitude(latitude)
        self.longitude = self._validate_longitude(longitude)
        self.timezone_str = timezone_str
        self.location_name = location_name or f"{latitude:.3f}, {longitude:.3f}"

        self.logger = logging.getLogger(__name__)

        # Solar constants
        self.SOLAR_CONSTANT = 1361  # W/m² - Solar irradiance at top of atmosphere
        self.EARTH_TILT = 23.45  # Earth's axial tilt in degrees

        # Location-specific factors (can be enhanced with real weather data)
        self.climate_factors = self._estimate_climate_factors()

        self.logger.info(f"LocationManager initialized for {self.location_name}")

    def _validate_latitude(self, lat: float) -> float:
        """Validate latitude is within valid range"""
        if not -90 <= lat <= 90:
            raise ValueError(f"Latitude {lat} must be between -90 and 90 degrees")
        return lat

    def _validate_longitude(self, lon: float) -> float:
        """Validate longitude is within valid range"""
        if not -180 <= lon <= 180:
            raise ValueError(f"Longitude {lon} must be between -180 and 180 degrees")
        return lon

    def _estimate_climate_factors(self) -> Dict[str, float]:
        """
        Estimate climate factors based on latitude
        These are rough estimates - real implementation would use weather APIs
        """
        abs_lat = abs(self.latitude)

        # Basic climate classification based on latitude
        if abs_lat < 23.5:  # Tropical
            cloud_factor = 0.7  # More clouds due to convection
            humidity_factor = 0.8
            seasonal_variation = 0.1  # Less seasonal variation
        elif abs_lat < 35:  # Subtropical
            cloud_factor = 0.8
            humidity_factor = 0.6
            seasonal_variation = 0.3
        elif abs_lat < 50:  # Temperate
            cloud_factor = 0.75
            humidity_factor = 0.5
            seasonal_variation = 0.5
        else:  # Polar/Subpolar
            cloud_factor = 0.6
            humidity_factor = 0.4
            seasonal_variation = 0.8  # High seasonal variation

        return {
            "cloud_factor": cloud_factor,
            "humidity_factor": humidity_factor,
            "seasonal_variation": seasonal_variation,
            "atmospheric_transmission": 0.75 - abs_lat * 0.002,  # Rough approximation
        }

    def get_solar_declination(self, day_of_year: int) -> float:
        """
        Calculate solar declination angle for given day of year

        Args:
            day_of_year: Day of year (1-366)

        Returns:
            Solar declination in degrees
        """
        # Solar declination varies sinusoidally throughout the year
        declination = self.EARTH_TILT * math.sin(
            math.radians(360 * (284 + day_of_year) / 365)
        )
        return declination

    def get_sunrise_sunset(self, date: datetime) -> Tuple[float, float]:
        """
        Calculate sunrise and sunset times for location on given date

        Args:
            date: Date to calculate for

        Returns:
            Tuple of (sunrise_hour, sunset_hour) in decimal hours
        """
        day_of_year = date.timetuple().tm_yday
        declination = math.radians(self.get_solar_declination(day_of_year))
        latitude_rad = math.radians(self.latitude)

        # Hour angle at sunrise/sunset
        try:
            cos_hour_angle = -math.tan(latitude_rad) * math.tan(declination)
            # Clamp to valid range to handle polar day/night
            cos_hour_angle = max(-1, min(1, cos_hour_angle))
            hour_angle = math.degrees(math.acos(cos_hour_angle))

            # Convert to hours from solar noon
            daylight_hours = 2 * hour_angle / 15  # 15 degrees per hour

            sunrise = 12 - daylight_hours / 2
            sunset = 12 + daylight_hours / 2

        except (ValueError, ZeroDivisionError):
            # Handle extreme latitudes (polar day/night)
            if abs(self.latitude) > 66.5:  # Arctic/Antarctic circle
                if abs(declination) > math.radians(90 - abs(self.latitude)):
                    # Polar day or night
                    if self.latitude * declination > 0:
                        sunrise, sunset = 0, 24  # Polar day
                    else:
                        sunrise, sunset = 12, 12  # Polar night
                else:
                    sunrise, sunset = 6, 18  # Default
            else:
                sunrise, sunset = 6, 18  # Default fallback

        return sunrise, sunset

    def get_solar_elevation(self, datetime_obj: datetime) -> float:
        """
        Calculate solar elevation angle for given datetime

        Args:
            datetime_obj: Datetime object

        Returns:
            Solar elevation angle in degrees
        """
        day_of_year = datetime_obj.timetuple().tm_yday
        declination = math.radians(self.get_solar_declination(day_of_year))
        latitude_rad = math.radians(self.latitude)

        # Solar time (simplified - doesn't account for equation of time)
        solar_time = datetime_obj.hour + datetime_obj.minute / 60.0
        hour_angle = math.radians(15 * (solar_time - 12))

        # Solar elevation
        sin_elevation = math.sin(latitude_rad) * math.sin(declination) + math.cos(
            latitude_rad
        ) * math.cos(declination) * math.cos(hour_angle)

        elevation = math.degrees(math.asin(max(-1, min(1, sin_elevation))))
        return max(0, elevation)  # Sun below horizon = 0

    def get_theoretical_solar_irradiance(self, datetime_obj: datetime) -> float:
        """
        Calculate theoretical solar irradiance for location and time

        Args:
            datetime_obj: Datetime object

        Returns:
            Theoretical solar irradiance in W/m²
        """
        elevation = self.get_solar_elevation(datetime_obj)

        if elevation <= 0:
            return 0

        # Air mass calculation (simplified)
        air_mass = 1 / math.sin(math.radians(elevation)) if elevation > 0 else 0
        air_mass = min(air_mass, 10)  # Cap extreme values

        # Atmospheric attenuation
        atmospheric_transmission = (
            self.climate_factors["atmospheric_transmission"] ** air_mass
        )

        # Direct normal irradiance
        dni = self.SOLAR_CONSTANT * atmospheric_transmission

        # Global horizontal irradiance (simplified model)
        ghi = dni * math.sin(math.radians(elevation))

        return max(0, ghi)

    def get_seasonal_adjustment_factor(self, date: datetime) -> float:
        """
        Get seasonal adjustment factor for solar production

        Args:
            date: Date to calculate for

        Returns:
            Adjustment factor (0.0 to 1.0+)
        """
        day_of_year = date.timetuple().tm_yday

        # Peak around summer solstice (day 172)
        seasonal_peak_day = (
            172 if self.latitude >= 0 else 355
        )  # Reverse for southern hemisphere

        # Calculate seasonal factor
        seasonal_angle = 2 * math.pi * (day_of_year - seasonal_peak_day) / 365
        base_seasonal = 1 + self.climate_factors["seasonal_variation"] * math.cos(
            seasonal_angle
        )

        return max(0.2, base_seasonal)  # Minimum 20% of peak

    def get_weather_adjustment_factor(
        self, date: datetime, base_randomness: bool = True
    ) -> float:
        """
        Get weather-based adjustment factor

        Args:
            date: Date to calculate for
            base_randomness: Whether to include random weather variation

        Returns:
            Weather adjustment factor (0.0 to 1.0)
        """
        # Seasonal cloud patterns
        day_of_year = date.timetuple().tm_yday

        # Many locations have more clouds in winter
        seasonal_cloud_factor = 0.8 + 0.2 * math.cos(
            2 * math.pi * (day_of_year - 172) / 365
        )

        base_factor = self.climate_factors["cloud_factor"] * seasonal_cloud_factor

        if base_randomness:
            # Add day-to-day weather variation
            np.random.seed(int(date.timestamp()) % 2**31)  # Deterministic randomness
            daily_variation = np.random.uniform(0.6, 1.0)

            # Occasional very cloudy days
            if np.random.random() < 0.15:  # 15% chance
                daily_variation *= np.random.uniform(0.1, 0.4)

            return base_factor * daily_variation
        else:
            return base_factor

    def calculate_location_solar_profile(self, date: datetime) -> Dict[str, float]:
        """
        Calculate complete solar profile for a given date

        Args:
            date: Date to calculate profile for

        Returns:
            Dictionary with solar profile data
        """
        sunrise, sunset = self.get_sunrise_sunset(date)
        seasonal_factor = self.get_seasonal_adjustment_factor(date)
        weather_factor = self.get_weather_adjustment_factor(date)

        # Calculate peak irradiance for the day
        noon_time = date.replace(hour=12, minute=0, second=0, microsecond=0)
        peak_irradiance = self.get_theoretical_solar_irradiance(noon_time)

        return {
            "sunrise_hour": sunrise,
            "sunset_hour": sunset,
            "daylight_hours": sunset - sunrise,
            "seasonal_factor": seasonal_factor,
            "weather_factor": weather_factor,
            "peak_irradiance": peak_irradiance,
            "daily_solar_factor": seasonal_factor * weather_factor,
        }

    def enhance_solar_data(self, solar_data: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance solar data with location-specific factors

        Args:
            solar_data: DataFrame with datetime index and solar data

        Returns:
            Enhanced DataFrame with location factors
        """
        enhanced_data = solar_data.copy()

        # Add location-based columns
        enhanced_data["solar_elevation"] = enhanced_data.index.map(
            self.get_solar_elevation
        )
        enhanced_data["theoretical_irradiance"] = enhanced_data.index.map(
            self.get_theoretical_solar_irradiance
        )
        enhanced_data["seasonal_factor"] = enhanced_data.index.map(
            lambda x: self.get_seasonal_adjustment_factor(x)
        )
        enhanced_data["weather_factor"] = enhanced_data.index.map(
            lambda x: self.get_weather_adjustment_factor(x)
        )

        # Add sunrise/sunset info
        enhanced_data["date"] = enhanced_data.index.date
        daily_profiles = {}

        for date in enhanced_data["date"].unique():
            daily_profiles[date] = self.calculate_location_solar_profile(
                datetime.combine(date, datetime.min.time())
            )

        enhanced_data["sunrise_hour"] = enhanced_data["date"].map(
            lambda x: daily_profiles[x]["sunrise_hour"]
        )
        enhanced_data["sunset_hour"] = enhanced_data["date"].map(
            lambda x: daily_profiles[x]["sunset_hour"]
        )
        enhanced_data["daylight_hours"] = enhanced_data["date"].map(
            lambda x: daily_profiles[x]["daylight_hours"]
        )

        # Clean up temporary column
        enhanced_data.drop("date", axis=1, inplace=True)

        # Add metadata
        enhanced_data.attrs.update(
            {
                "location_name": self.location_name,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "timezone": self.timezone_str,
                "enhanced_with_location": True,
            }
        )

        return enhanced_data

    def get_location_summary(self) -> Dict[str, Union[str, float]]:
        """Get summary of location characteristics"""
        # Calculate some example values
        sample_date = datetime(2024, 6, 21)  # Summer solstice
        summer_profile = self.calculate_location_solar_profile(sample_date)

        sample_date_winter = datetime(2024, 12, 21)  # Winter solstice
        winter_profile = self.calculate_location_solar_profile(sample_date_winter)

        return {
            "location_name": self.location_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone_str or "Not specified",
            "climate_type": self._get_climate_description(),
            "summer_daylight_hours": summer_profile["daylight_hours"],
            "winter_daylight_hours": winter_profile["daylight_hours"],
            "seasonal_variation": self.climate_factors["seasonal_variation"],
            "typical_cloud_factor": self.climate_factors["cloud_factor"],
            "atmospheric_clarity": self.climate_factors["atmospheric_transmission"],
        }

    def _get_climate_description(self) -> str:
        """Get human-readable climate description"""
        abs_lat = abs(self.latitude)

        if abs_lat < 23.5:
            return "Tropical"
        elif abs_lat < 35:
            return "Subtropical"
        elif abs_lat < 50:
            return "Temperate"
        elif abs_lat < 66.5:
            return "Subarctic/Subantarctic"
        else:
            return "Arctic/Antarctic"

    @classmethod
    def from_city(cls, city_name: str) -> "LocationManager":
        """
        Create LocationManager from city name using predefined coordinates

        Args:
            city_name: Name of the city

        Returns:
            LocationManager instance
        """
        # Common cities database (can be expanded)
        cities = {
            "new_york": (40.7128, -74.0060, "America/New_York", "New York, NY"),
            "los_angeles": (
                34.0522,
                -118.2437,
                "America/Los_Angeles",
                "Los Angeles, CA",
            ),
            "chicago": (41.8781, -87.6298, "America/Chicago", "Chicago, IL"),
            "denver": (39.7392, -104.9903, "America/Denver", "Denver, CO"),
            "miami": (25.7617, -80.1918, "America/New_York", "Miami, FL"),
            "seattle": (47.6062, -122.3321, "America/Los_Angeles", "Seattle, WA"),
            "phoenix": (33.4484, -112.0740, "America/Phoenix", "Phoenix, AZ"),
            "atlanta": (33.7490, -84.3880, "America/New_York", "Atlanta, GA"),
            "london": (51.5074, -0.1278, "Europe/London", "London, UK"),
            "berlin": (52.5200, 13.4050, "Europe/Berlin", "Berlin, Germany"),
            "tokyo": (35.6762, 139.6503, "Asia/Tokyo", "Tokyo, Japan"),
            "sydney": (-33.8688, 151.2093, "Australia/Sydney", "Sydney, Australia"),
        }

        city_key = city_name.lower().replace(" ", "_").replace(",", "")

        if city_key in cities:
            lat, lon, tz, display_name = cities[city_key]
            return cls(lat, lon, tz, display_name)
        else:
            raise ValueError(
                f"City '{city_name}' not found in database. Use custom coordinates instead."
            )

    def __str__(self) -> str:
        """String representation"""
        return f"LocationManager({self.location_name} at {self.latitude:.3f}°, {self.longitude:.3f}°)"

    def __repr__(self) -> str:
        """Detailed representation"""
        return (
            f"LocationManager(latitude={self.latitude}, longitude={self.longitude}, "
            f"timezone='{self.timezone_str}', location_name='{self.location_name}')"
        )
