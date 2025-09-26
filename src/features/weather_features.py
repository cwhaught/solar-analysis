"""
Weather Feature Engineering - Extends existing WeatherAnalyzer

Builds on the existing weather analysis infrastructure to provide ML-ready weather features.
Integrates with the established WeatherAnalyzer patterns and existing weather data pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging

from ..core.weather_analysis import WeatherAnalyzer
from ..core.weather_manager import WeatherData

logger = logging.getLogger(__name__)


class WeatherFeatureEngineer(WeatherAnalyzer):
    """
    Weather-specific feature engineering that extends existing WeatherAnalyzer.

    Builds on established weather correlation analysis to create ML-ready features
    for solar production prediction models.
    """

    def __init__(self):
        """Initialize weather feature engineer with existing analyzer infrastructure"""
        super().__init__()
        self.feature_cache = {}

    def create_weather_ml_features(
        self,
        daily_data: pd.DataFrame,
        weather_data: Union[pd.DataFrame, List[WeatherData]]
    ) -> pd.DataFrame:
        """
        Create comprehensive weather features for ML models.

        Leverages existing weather correlation analysis to generate features
        extracted from notebook patterns.

        Args:
            daily_data: Solar production data with datetime index
            weather_data: Weather data (DataFrame or WeatherData list)

        Returns:
            DataFrame with weather features added
        """
        try:
            logger.info("Creating weather ML features")

            # Convert weather data to DataFrame if needed (use existing patterns)
            if isinstance(weather_data, list):
                weather_df = self._convert_weather_data_to_df(weather_data)
            else:
                weather_df = weather_data.copy()

            # Start with copy of daily data
            ml_data = daily_data.copy()

            # Create core weather features (extracted from notebook patterns)
            ml_data = self._add_weather_efficiency_features(ml_data, weather_df)
            ml_data = self._add_weather_composite_features(ml_data, weather_df)
            ml_data = self._add_weather_lag_features(ml_data, weather_df)
            ml_data = self._add_weather_volatility_features(ml_data, weather_df)

            logger.info(f"Created {len([c for c in ml_data.columns if 'weather' in c.lower()])} weather features")
            return ml_data

        except Exception as e:
            logger.error(f"Error creating weather ML features: {e}")
            # Graceful fallback - return original data
            return daily_data.copy()

    def _add_weather_efficiency_features(
        self,
        data: pd.DataFrame,
        weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add weather efficiency features based on existing efficiency factor calculations.

        Builds on existing calculate_weather_efficiency_factors method.
        """
        try:
            # Merge weather data with production data
            merged_data = self._merge_weather_production_data(data, weather_df)

            if merged_data is None or len(merged_data) == 0:
                logger.warning("No weather data available for efficiency features")
                return data

            # Create efficiency features extracted from notebook patterns
            if 'sunshine_hours' in merged_data.columns:
                # Sunshine ratio (found in weather integration notebook)
                data['sunshine_ratio'] = merged_data['sunshine_hours'] / 12.0  # Max possible sunshine

            if 'cloudcover_mean_pct' in merged_data.columns:
                # Clear sky factor (found in weather integration notebook)
                data['clear_sky_factor'] = 100 - merged_data['cloudcover_mean_pct']

            if 'temp_max_c' in merged_data.columns and 'temp_min_c' in merged_data.columns:
                # Temperature range features (found in weather integration notebook)
                data['temp_range'] = merged_data['temp_max_c'] - merged_data['temp_min_c']

            # Solar weather score (composite metric from notebooks)
            data['solar_weather_score'] = self._calculate_solar_weather_score(merged_data)

            return data

        except Exception as e:
            logger.error(f"Error adding weather efficiency features: {e}")
            return data

    def _add_weather_composite_features(
        self,
        data: pd.DataFrame,
        weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add composite weather features that combine multiple weather variables.

        Based on patterns found in weather integration notebooks.
        """
        try:
            merged_data = self._merge_weather_production_data(data, weather_df)

            if merged_data is None:
                return data

            # Weather quality index (combination of sunshine, clouds, precipitation)
            if all(col in merged_data.columns for col in ['sunshine_hours', 'cloudcover_mean_pct', 'precipitation_mm']):
                sunshine_norm = merged_data['sunshine_hours'] / 12.0
                cloud_factor = (100 - merged_data['cloudcover_mean_pct']) / 100.0
                precip_factor = np.where(merged_data['precipitation_mm'] > 1, 0.5, 1.0)

                data['weather_quality_index'] = (sunshine_norm * 0.4 +
                                                cloud_factor * 0.4 +
                                                precip_factor * 0.2)

            # Optimal weather indicator (binary feature)
            if 'solar_weather_score' in data.columns:
                data['optimal_weather_day'] = (data['solar_weather_score'] >
                                              data['solar_weather_score'].quantile(0.8)).astype(int)

            return data

        except Exception as e:
            logger.error(f"Error adding composite weather features: {e}")
            return data

    def _add_weather_lag_features(
        self,
        data: pd.DataFrame,
        weather_df: pd.DataFrame,
        lags: List[int] = [1, 2, 3]
    ) -> pd.DataFrame:
        """
        Add lagged weather features for prediction models.

        Previous weather conditions can be predictive of production patterns.
        """
        try:
            merged_data = self._merge_weather_production_data(data, weather_df)

            if merged_data is None:
                return data

            # Key weather variables to lag
            weather_vars = ['sunshine_hours', 'cloudcover_mean_pct', 'temp_mean_c']
            available_vars = [var for var in weather_vars if var in merged_data.columns]

            for var in available_vars:
                for lag in lags:
                    lag_col = f'{var}_lag_{lag}d'
                    data[lag_col] = merged_data[var].shift(lag)

            return data

        except Exception as e:
            logger.error(f"Error adding weather lag features: {e}")
            return data

    def _add_weather_volatility_features(
        self,
        data: pd.DataFrame,
        weather_df: pd.DataFrame,
        windows: List[int] = [3, 7]
    ) -> pd.DataFrame:
        """
        Add weather volatility features (rolling standard deviations).

        Weather variability can impact production predictability.
        """
        try:
            merged_data = self._merge_weather_production_data(data, weather_df)

            if merged_data is None:
                return data

            # Weather variables for volatility
            weather_vars = ['sunshine_hours', 'cloudcover_mean_pct', 'temp_mean_c']
            available_vars = [var for var in weather_vars if var in merged_data.columns]

            for var in available_vars:
                for window in windows:
                    volatility_col = f'{var}_volatility_{window}d'
                    data[volatility_col] = merged_data[var].rolling(window=window, min_periods=1).std()

            return data

        except Exception as e:
            logger.error(f"Error adding weather volatility features: {e}")
            return data

    def _calculate_solar_weather_score(self, merged_data: pd.DataFrame) -> pd.Series:
        """
        Calculate composite solar weather score.

        Based on patterns found in weather integration notebooks.
        Combines multiple weather factors into a single production-predictive score.
        """
        try:
            # Initialize with neutral score
            score = pd.Series(50.0, index=merged_data.index)

            # Sunshine hours contribution (0-40 points)
            if 'sunshine_hours' in merged_data.columns:
                sunshine_norm = np.clip(merged_data['sunshine_hours'] / 12.0, 0, 1)
                score += sunshine_norm * 40

            # Cloud cover contribution (-30 to 0 points)
            if 'cloudcover_mean_pct' in merged_data.columns:
                cloud_penalty = -(merged_data['cloudcover_mean_pct'] / 100.0) * 30
                score += cloud_penalty

            # Temperature contribution (-10 to +10 points, optimal around 25°C)
            if 'temp_mean_c' in merged_data.columns:
                # Optimal temperature around 25°C for solar panels
                temp_factor = 1 - np.abs(merged_data['temp_mean_c'] - 25) / 25
                temp_factor = np.clip(temp_factor, -1, 1)
                score += temp_factor * 10

            # Precipitation penalty (-20 points)
            if 'precipitation_mm' in merged_data.columns:
                precip_penalty = -np.where(merged_data['precipitation_mm'] > 1, 20, 0)
                score += precip_penalty

            return np.clip(score, 0, 100)

        except Exception as e:
            logger.error(f"Error calculating solar weather score: {e}")
            return pd.Series(50.0, index=merged_data.index)

    def _merge_weather_production_data(
        self,
        production_data: pd.DataFrame,
        weather_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Merge production and weather data safely.

        Follows existing patterns for data merging with proper error handling.
        """
        try:
            # Ensure both have datetime indices
            if not isinstance(production_data.index, pd.DatetimeIndex):
                logger.error("Production data must have datetime index")
                return None

            if not isinstance(weather_df.index, pd.DatetimeIndex):
                # Try to convert if date column exists
                if 'date' in weather_df.columns:
                    weather_df = weather_df.set_index('date')
                elif 'Date/Time' in weather_df.columns:
                    weather_df = weather_df.set_index('Date/Time')
                else:
                    logger.error("Weather data must have datetime index or date column")
                    return None

            # Merge on datetime index
            merged = production_data.join(weather_df, how='left')

            if len(merged) == 0:
                logger.warning("No data after merging production and weather data")
                return None

            return merged

        except Exception as e:
            logger.error(f"Error merging weather and production data: {e}")
            return None

    def _convert_weather_data_to_df(self, weather_data: List[WeatherData]) -> pd.DataFrame:
        """
        Convert WeatherData list to DataFrame.

        Follows existing WeatherManager patterns for data conversion.
        """
        try:
            if not weather_data:
                return pd.DataFrame()

            # Extract data using existing WeatherData structure
            data_rows = []
            for wd in weather_data:
                row = {
                    'date': wd.timestamp,
                    'temp_mean_c': wd.temperature,
                    'sunshine_hours': getattr(wd, 'sunshine_duration', 0) / 3600,  # Convert to hours
                    'cloudcover_mean_pct': getattr(wd, 'cloud_cover', 50),
                    'precipitation_mm': getattr(wd, 'precipitation', 0),
                    'solar_radiation_mj': getattr(wd, 'solar_radiation', 0)
                }
                data_rows.append(row)

            df = pd.DataFrame(data_rows)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            return df

        except Exception as e:
            logger.error(f"Error converting weather data to DataFrame: {e}")
            return pd.DataFrame()

    def get_weather_feature_names(self) -> List[str]:
        """
        Get list of weather feature names created by this engineer.

        Returns:
            List of feature column names
        """
        base_features = [
            'sunshine_ratio', 'clear_sky_factor', 'temp_range', 'solar_weather_score',
            'weather_quality_index', 'optimal_weather_day'
        ]

        # Lag features
        weather_vars = ['sunshine_hours', 'cloudcover_mean_pct', 'temp_mean_c']
        lag_features = [f'{var}_lag_{lag}d' for var in weather_vars for lag in [1, 2, 3]]

        # Volatility features
        volatility_features = [f'{var}_volatility_{window}d'
                             for var in weather_vars for window in [3, 7]]

        return base_features + lag_features + volatility_features