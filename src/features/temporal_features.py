"""
Temporal Feature Engineering - Time-based and cyclical features

Extracts temporal patterns from notebook feature engineering code.
Creates time-based, cyclical, rolling, and lag features for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """
    Temporal feature engineering for solar energy time series data.

    Consolidates time-based feature engineering patterns found across notebooks:
    - Basic time features (day, month, quarter, weekday)
    - Cyclical encodings (sin/cos transformations)
    - Rolling statistics (mean, std, volatility)
    - Lag features for time series modeling
    """

    def __init__(self):
        """Initialize temporal feature engineer"""
        pass

    def create_temporal_ml_features(
        self,
        data: pd.DataFrame,
        target_col: str = 'Production (kWh)',
        include_sets: List[str] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive temporal features for ML models.

        Consolidates temporal feature patterns found across all notebooks.

        Args:
            data: DataFrame with datetime index
            target_col: Column name for target variable (for rolling/lag features)
            include_sets: List of feature sets to include ['time', 'cyclical', 'rolling', 'lag']

        Returns:
            DataFrame with temporal features added
        """
        try:
            logger.info("Creating temporal ML features")

            if include_sets is None:
                include_sets = ['time', 'cyclical', 'rolling', 'lag']

            # Validate datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                logger.error("Data must have datetime index for temporal features")
                return data.copy()

            # Start with copy of data
            ml_data = data.copy()

            # Add feature sets based on notebook patterns
            if 'time' in include_sets:
                ml_data = self._add_time_features(ml_data)

            if 'cyclical' in include_sets:
                ml_data = self._add_cyclical_features(ml_data)

            if 'rolling' in include_sets and target_col in ml_data.columns:
                ml_data = self._add_rolling_features(ml_data, target_col)

            if 'lag' in include_sets and target_col in ml_data.columns:
                ml_data = self._add_lag_features(ml_data, target_col)

            # Add advanced temporal features
            ml_data = self._add_temporal_volatility_features(ml_data, target_col)

            logger.info(f"Created {len([c for c in ml_data.columns if c not in data.columns])} temporal features")
            return ml_data

        except Exception as e:
            logger.error(f"Error creating temporal ML features: {e}")
            return data.copy()

    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic time-based features.

        Extracted from notebook patterns found in:
        - 01c_baseline_ml_models.ipynb
        - 02_weather_integration.ipynb
        - 01d_location_based_analysis.ipynb
        """
        try:
            # Basic time features (consistent across all notebooks)
            data['day_of_year'] = data.index.dayofyear
            data['day_of_week'] = data.index.dayofweek
            data['month'] = data.index.month
            data['quarter'] = data.index.quarter
            data['week_of_year'] = data.index.isocalendar().week

            # Binary features
            data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
            data['is_monday'] = (data.index.dayofweek == 0).astype(int)

            # Seasonal indicators
            data['is_winter'] = data['month'].isin([12, 1, 2]).astype(int)
            data['is_spring'] = data['month'].isin([3, 4, 5]).astype(int)
            data['is_summer'] = data['month'].isin([6, 7, 8]).astype(int)
            data['is_fall'] = data['month'].isin([9, 10, 11]).astype(int)

            return data

        except Exception as e:
            logger.error(f"Error adding time features: {e}")
            return data

    def _add_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical encodings for seasonal patterns.

        Extracted from notebook patterns - uses sin/cos encoding for cyclical time features.
        Found consistently across ML notebooks with 365.25 days per year for leap year accuracy.
        """
        try:
            # Daily cycle (found in all ML notebooks)
            data['day_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
            data['day_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)

            # Weekly cycle
            data['week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

            # Monthly cycle
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

            # Quarterly cycle
            data['quarter_sin'] = np.sin(2 * np.pi * data['quarter'] / 4)
            data['quarter_cos'] = np.cos(2 * np.pi * data['quarter'] / 4)

            return data

        except Exception as e:
            logger.error(f"Error adding cyclical features: {e}")
            return data

    def _add_rolling_features(
        self,
        data: pd.DataFrame,
        target_col: str,
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Add rolling statistics features.

        Extracted from notebook patterns - consistent window sizes across all ML notebooks:
        [7, 14, 30] day windows with min_periods=1 for edge cases.
        """
        try:
            if windows is None:
                windows = [7, 14, 30]  # Consistent across notebooks

            if target_col not in data.columns:
                logger.warning(f"Target column '{target_col}' not found for rolling features")
                return data

            # Rolling mean and std (found in all ML notebooks)
            for window in windows:
                data[f'rolling_mean_{window}d'] = data[target_col].rolling(
                    window=window, min_periods=1
                ).mean()

                data[f'rolling_std_{window}d'] = data[target_col].rolling(
                    window=window, min_periods=1
                ).std()

                # Rolling median (additional robust statistic)
                data[f'rolling_median_{window}d'] = data[target_col].rolling(
                    window=window, min_periods=1
                ).median()

                # Rolling min/max for range features
                data[f'rolling_min_{window}d'] = data[target_col].rolling(
                    window=window, min_periods=1
                ).min()

                data[f'rolling_max_{window}d'] = data[target_col].rolling(
                    window=window, min_periods=1
                ).max()

                # Rolling range
                data[f'rolling_range_{window}d'] = (
                    data[f'rolling_max_{window}d'] - data[f'rolling_min_{window}d']
                )

            return data

        except Exception as e:
            logger.error(f"Error adding rolling features: {e}")
            return data

    def _add_lag_features(
        self,
        data: pd.DataFrame,
        target_col: str,
        lags: List[int] = None
    ) -> pd.DataFrame:
        """
        Add lag features for time series modeling.

        Extracted from notebook patterns - consistent lag periods across all ML notebooks:
        [1, 2, 3, 7, 14] day lags.
        """
        try:
            if lags is None:
                lags = [1, 2, 3, 7, 14]  # Consistent across notebooks

            if target_col not in data.columns:
                logger.warning(f"Target column '{target_col}' not found for lag features")
                return data

            # Basic lag features (found in all ML notebooks)
            for lag in lags:
                data[f'{target_col.lower().replace(" ", "_").replace("(", "").replace(")", "")}_lag_{lag}d'] = (
                    data[target_col].shift(lag)
                )

            # Advanced lag features
            # Lag differences (change from previous periods)
            for lag in [1, 7]:
                lag_col = f'{target_col.lower().replace(" ", "_").replace("(", "").replace(")", "")}_lag_{lag}d'
                if lag_col in data.columns:
                    data[f'change_from_{lag}d_ago'] = data[target_col] - data[lag_col]
                    data[f'pct_change_from_{lag}d_ago'] = (
                        (data[target_col] - data[lag_col]) / (data[lag_col] + 0.01) * 100
                    )

            return data

        except Exception as e:
            logger.error(f"Error adding lag features: {e}")
            return data

    def _add_temporal_volatility_features(
        self,
        data: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """
        Add temporal volatility and pattern features.

        Based on patterns found in notebooks - production volatility and relative performance.
        """
        try:
            if target_col not in data.columns:
                return data

            # Production volatility (found in ML notebooks)
            data['production_volatility'] = data[target_col].rolling(7, min_periods=1).std()

            # Relative performance (found in ML notebooks)
            if 'rolling_mean_30d' in data.columns:
                data['relative_performance'] = data[target_col] / (data['rolling_mean_30d'] + 0.01)

            # Trend indicators
            # Short-term trend (7-day)
            if 'rolling_mean_7d' in data.columns:
                data['trend_7d'] = (
                    data['rolling_mean_7d'] - data['rolling_mean_7d'].shift(7)
                ) / (data['rolling_mean_7d'].shift(7) + 0.01)

            # Long-term trend (30-day)
            if 'rolling_mean_30d' in data.columns:
                data['trend_30d'] = (
                    data['rolling_mean_30d'] - data['rolling_mean_30d'].shift(30)
                ) / (data['rolling_mean_30d'].shift(30) + 0.01)

            # Seasonality indicators
            if 'day_of_year' in data.columns:
                # Distance from summer solstice (peak production time)
                summer_solstice = 172  # Approximately June 21
                data['days_from_summer_solstice'] = np.minimum(
                    np.abs(data['day_of_year'] - summer_solstice),
                    365 - np.abs(data['day_of_year'] - summer_solstice)
                )

                # Seasonal strength (closer to summer = higher)
                data['seasonal_strength'] = 1 - (data['days_from_summer_solstice'] / 182.5)

            return data

        except Exception as e:
            logger.error(f"Error adding temporal volatility features: {e}")
            return data

    def get_temporal_feature_names(self, include_sets: List[str] = None) -> List[str]:
        """
        Get list of temporal feature names created by this engineer.

        Args:
            include_sets: List of feature sets to include ['time', 'cyclical', 'rolling', 'lag']

        Returns:
            List of feature column names
        """
        if include_sets is None:
            include_sets = ['time', 'cyclical', 'rolling', 'lag']

        feature_names = []

        if 'time' in include_sets:
            time_features = [
                'day_of_year', 'day_of_week', 'month', 'quarter', 'week_of_year',
                'is_weekend', 'is_monday', 'is_winter', 'is_spring', 'is_summer', 'is_fall'
            ]
            feature_names.extend(time_features)

        if 'cyclical' in include_sets:
            cyclical_features = [
                'day_sin', 'day_cos', 'week_sin', 'week_cos',
                'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos'
            ]
            feature_names.extend(cyclical_features)

        if 'rolling' in include_sets:
            rolling_features = []
            for window in [7, 14, 30]:
                rolling_features.extend([
                    f'rolling_mean_{window}d', f'rolling_std_{window}d',
                    f'rolling_median_{window}d', f'rolling_min_{window}d',
                    f'rolling_max_{window}d', f'rolling_range_{window}d'
                ])
            feature_names.extend(rolling_features)

        if 'lag' in include_sets:
            lag_features = []
            for lag in [1, 2, 3, 7, 14]:
                lag_features.append(f'production_kwh_lag_{lag}d')
            # Change features
            lag_features.extend([
                'change_from_1d_ago', 'change_from_7d_ago',
                'pct_change_from_1d_ago', 'pct_change_from_7d_ago'
            ])
            feature_names.extend(lag_features)

        # Volatility features
        volatility_features = [
            'production_volatility', 'relative_performance', 'trend_7d', 'trend_30d',
            'days_from_summer_solstice', 'seasonal_strength'
        ]
        feature_names.extend(volatility_features)

        return feature_names