"""
Location Feature Engineering - Extends existing LocationManager

Creates location-aware solar features using existing LocationManager infrastructure.
Extracts solar geometry and efficiency features found in notebook patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LocationFeatureEngineer:
    """
    Location-aware feature engineering that extends existing LocationManager.

    Creates solar geometry and location-specific features using the established
    LocationManager infrastructure for solar calculations.
    """

    def __init__(self, location_manager=None):
        """
        Initialize location feature engineer.

        Args:
            location_manager: Existing LocationManager instance
        """
        self.location_manager = location_manager

    def create_location_ml_features(
        self,
        data: pd.DataFrame,
        location_manager=None
    ) -> pd.DataFrame:
        """
        Create location-aware solar features for ML models.

        Leverages existing LocationManager to calculate solar geometry features
        found in notebook patterns.

        Args:
            data: DataFrame with datetime index
            location_manager: LocationManager instance (optional, uses self.location_manager if not provided)

        Returns:
            DataFrame with location features added
        """
        try:
            logger.info("Creating location-aware ML features")

            # Use provided location manager or instance variable
            loc_mgr = location_manager or self.location_manager

            if loc_mgr is None:
                logger.warning("No location manager provided - skipping location features")
                return data.copy()

            # Validate datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                logger.error("Data must have datetime index for location features")
                return data.copy()

            # Start with copy of data
            ml_data = data.copy()

            # Add solar geometry features
            ml_data = self._add_solar_geometry_features(ml_data, loc_mgr)
            ml_data = self._add_solar_efficiency_features(ml_data)
            ml_data = self._add_theoretical_features(ml_data, loc_mgr)

            logger.info(f"Created {len([c for c in ml_data.columns if c not in data.columns])} location features")
            return ml_data

        except Exception as e:
            logger.error(f"Error creating location ML features: {e}")
            return data.copy()

    def _add_solar_geometry_features(
        self,
        data: pd.DataFrame,
        location_manager
    ) -> pd.DataFrame:
        """
        Add solar geometry features using existing LocationManager.

        Extracted from notebook patterns found in 01c_baseline_ml_models.ipynb
        and other location-aware notebooks.
        """
        try:
            # Initialize feature columns
            daylight_hours = []
            solar_elevations = []
            theoretical_irradiances = []

            # Calculate for each date (following notebook pattern)
            for date_timestamp in data.index:
                try:
                    # Convert pandas timestamp to date (pattern from notebooks)
                    date_obj = date_timestamp.date()

                    # Calculate theoretical solar parameters (existing LocationManager methods)
                    sunrise, sunset = location_manager.get_sunrise_sunset(date_obj)
                    daylight_duration = sunset - sunrise
                    solar_elevation = location_manager.get_solar_elevation(date_timestamp.to_pydatetime())
                    theoretical_irradiance = location_manager.get_theoretical_solar_irradiance(date_timestamp.to_pydatetime())

                    daylight_hours.append(daylight_duration)
                    solar_elevations.append(max(0, solar_elevation))
                    theoretical_irradiances.append(max(0, theoretical_irradiance))

                except Exception as e:
                    logger.warning(f"Error calculating solar geometry for {date_timestamp}: {e}")
                    # Use reasonable defaults
                    daylight_hours.append(12.0)  # Average daylight
                    solar_elevations.append(45.0)  # Reasonable elevation
                    theoretical_irradiances.append(1000.0)  # Typical irradiance

            # Add features (consistent with notebook naming)
            data['theoretical_daylight_hours'] = daylight_hours
            data['max_solar_elevation'] = solar_elevations
            data['theoretical_irradiance'] = theoretical_irradiances

            return data

        except Exception as e:
            logger.error(f"Error adding solar geometry features: {e}")
            return data

    def _add_solar_efficiency_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add solar efficiency features based on location and production.

        Extracted from notebook patterns - efficiency calculations using location data.
        """
        try:
            if 'Production (kWh)' not in data.columns:
                logger.warning("Production column not found - skipping efficiency features")
                return data

            # Daylight utilization efficiency (found in ML notebooks)
            if 'theoretical_daylight_hours' in data.columns:
                # Avoid division by zero (pattern from notebooks)
                data['daylight_utilization'] = (
                    data['Production (kWh)'] / (data['theoretical_daylight_hours'] + 0.1)
                )

            # Irradiance efficiency (found in ML notebooks)
            if 'theoretical_irradiance' in data.columns:
                # Avoid division by zero (pattern from notebooks)
                data['irradiance_efficiency'] = (
                    data['Production (kWh)'] / (data['theoretical_irradiance'] + 0.1)
                )

            # Solar elevation efficiency
            if 'max_solar_elevation' in data.columns:
                data['elevation_efficiency'] = (
                    data['Production (kWh)'] / (data['max_solar_elevation'] + 0.1)
                )

            # Composite efficiency score
            if all(col in data.columns for col in ['daylight_utilization', 'irradiance_efficiency']):
                data['solar_efficiency_score'] = (
                    data['daylight_utilization'] * 0.5 + data['irradiance_efficiency'] * 0.5
                )

            return data

        except Exception as e:
            logger.error(f"Error adding solar efficiency features: {e}")
            return data

    def _add_theoretical_features(
        self,
        data: pd.DataFrame,
        location_manager
    ) -> pd.DataFrame:
        """
        Add theoretical solar features based on location characteristics.

        Creates features based on theoretical potential vs actual performance.
        """
        try:
            # Location characteristics (constant features)
            location_summary = location_manager.get_location_summary()

            data['latitude'] = location_manager.latitude
            data['longitude'] = location_manager.longitude
            data['climate_type_encoded'] = self._encode_climate_type(location_summary.get('climate_type', 'Temperate'))

            # Seasonal potential features
            if 'day_of_year' in data.columns:
                # Theoretical seasonal variation
                seasonal_variation = location_summary.get('seasonal_variation', 0.5)
                data['theoretical_seasonal_factor'] = 1 + seasonal_variation * np.cos(
                    2 * np.pi * (data['day_of_year'] - 172) / 365.25  # Peak at summer solstice
                )

            # Daylight variation features
            winter_hours = location_summary.get('winter_daylight_hours', 9)
            summer_hours = location_summary.get('summer_daylight_hours', 15)

            if 'theoretical_daylight_hours' in data.columns:
                # Daylight efficiency relative to location potential
                data['daylight_hours_normalized'] = (
                    (data['theoretical_daylight_hours'] - winter_hours) /
                    (summer_hours - winter_hours + 0.1)
                )

            # Location-based capacity factors
            if 'Production (kWh)' in data.columns and 'theoretical_irradiance' in data.columns:
                # Theoretical capacity utilization
                avg_production = data['Production (kWh)'].mean()
                avg_irradiance = data['theoretical_irradiance'].mean()

                if avg_irradiance > 0:
                    data['location_capacity_factor'] = avg_production / avg_irradiance
                else:
                    data['location_capacity_factor'] = 0.15  # Typical solar capacity factor

            return data

        except Exception as e:
            logger.error(f"Error adding theoretical features: {e}")
            return data

    def _encode_climate_type(self, climate_type: str) -> float:
        """
        Encode climate type as numeric feature.

        Maps climate types to numeric values based on solar production potential.
        """
        climate_mapping = {
            'Desert': 1.0,      # Highest solar potential
            'Arid': 0.9,
            'Subtropical': 0.8,
            'Temperate': 0.7,
            'Continental': 0.6,
            'Oceanic': 0.5,
            'Arctic': 0.3       # Lowest solar potential
        }

        return climate_mapping.get(climate_type, 0.7)  # Default to Temperate

    def create_location_summary_features(
        self,
        location_manager
    ) -> Dict[str, Any]:
        """
        Create location summary features for model metadata.

        Returns location characteristics as a dictionary for model context.
        """
        try:
            if location_manager is None:
                return {}

            location_summary = location_manager.get_location_summary()

            summary = {
                'latitude': location_manager.latitude,
                'longitude': location_manager.longitude,
                'location_name': getattr(location_manager, 'location_name', 'Unknown'),
                'climate_type': location_summary.get('climate_type', 'Unknown'),
                'climate_type_encoded': self._encode_climate_type(location_summary.get('climate_type', 'Temperate')),
                'seasonal_variation': location_summary.get('seasonal_variation', 0.5),
                'winter_daylight_hours': location_summary.get('winter_daylight_hours', 9),
                'summer_daylight_hours': location_summary.get('summer_daylight_hours', 15),
                'timezone': getattr(location_manager, 'timezone', 'UTC')
            }

            return summary

        except Exception as e:
            logger.error(f"Error creating location summary features: {e}")
            return {}

    def get_location_feature_names(self) -> List[str]:
        """
        Get list of location feature names created by this engineer.

        Returns:
            List of feature column names
        """
        base_features = [
            'theoretical_daylight_hours', 'max_solar_elevation', 'theoretical_irradiance',
            'daylight_utilization', 'irradiance_efficiency', 'elevation_efficiency',
            'solar_efficiency_score', 'latitude', 'longitude', 'climate_type_encoded',
            'theoretical_seasonal_factor', 'daylight_hours_normalized', 'location_capacity_factor'
        ]

        return base_features