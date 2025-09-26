"""
Feature Pipeline - Main orchestration for solar energy feature engineering

Coordinates all feature engineering modules to create comprehensive ML-ready datasets.
Integrates with existing infrastructure (LocationManager, WeatherAnalyzer, etc.)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging

from ..core.location_loader import create_notebook_location_with_rates
from .temporal_features import TemporalFeatureEngineer
from .weather_features import WeatherFeatureEngineer
from .financial_features import FinancialFeatureEngineer
from .location_features import LocationFeatureEngineer

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Main feature engineering pipeline that orchestrates all feature engineers.

    Integrates with existing infrastructure to create comprehensive ML-ready datasets
    by coordinating temporal, weather, financial, and location-aware features.
    """

    def __init__(self, location_manager=None, config_path: str = None):
        """
        Initialize feature pipeline with existing infrastructure.

        Args:
            location_manager: Optional LocationManager instance
            config_path: Path to configuration file (defaults to .env)
        """
        try:
            # Follow existing configuration patterns
            if location_manager is None:
                self.location, self.electricity_rates, self.system_config = (
                    create_notebook_location_with_rates(config_path)
                )
            else:
                self.location = location_manager
                # Set defaults if rates not provided
                self.electricity_rates = {'residential_rate': 14.0, 'feed_in_tariff': 11.0}
                self.system_config = {'system_cost': 25000, 'federal_tax_credit': 7500}

            # Initialize feature engineers
            self.temporal_engineer = TemporalFeatureEngineer()
            self.weather_engineer = WeatherFeatureEngineer()
            self.financial_engineer = FinancialFeatureEngineer()
            self.location_engineer = LocationFeatureEngineer(self.location)

            logger.info(f"FeaturePipeline initialized for {getattr(self.location, 'location_name', 'Unknown location')}")

        except Exception as e:
            logger.error(f"Error initializing FeaturePipeline: {e}")
            # Set minimal defaults
            self.location = None
            self.electricity_rates = {'residential_rate': 14.0, 'feed_in_tariff': 11.0}
            self.system_config = {'system_cost': 25000, 'federal_tax_credit': 7500}

            # Initialize with defaults
            self.temporal_engineer = TemporalFeatureEngineer()
            self.weather_engineer = WeatherFeatureEngineer()
            self.financial_engineer = FinancialFeatureEngineer()
            self.location_engineer = LocationFeatureEngineer()

    def create_ml_dataset(
        self,
        daily_data: pd.DataFrame,
        feature_sets: List[str] = None,
        weather_data: Optional[pd.DataFrame] = None,
        target_col: str = 'Production (kWh)',
        clean_strategy: str = 'dropna'
    ) -> pd.DataFrame:
        """
        Create comprehensive ML-ready dataset with all selected features.

        Main method that orchestrates all feature engineering to replicate and improve
        upon notebook feature engineering patterns.

        Args:
            daily_data: Daily solar production data with datetime index
            feature_sets: List of feature sets to include
                         ['temporal', 'weather', 'financial', 'location']
            weather_data: Optional weather data DataFrame
            target_col: Target column name for rolling/lag features
            clean_strategy: Strategy for handling NaN values ('dropna', 'fillna')

        Returns:
            DataFrame with comprehensive features for ML models
        """
        try:
            logger.info("Creating comprehensive ML dataset")

            if feature_sets is None:
                feature_sets = ['temporal', 'financial', 'location']
                if weather_data is not None:
                    feature_sets.append('weather')

            # Validate input data
            if not isinstance(daily_data.index, pd.DatetimeIndex):
                logger.error("Daily data must have datetime index")
                return daily_data.copy()

            # Start with copy of daily data
            ml_dataset = daily_data.copy()
            initial_columns = set(ml_dataset.columns)

            # Add feature sets in optimal order
            if 'temporal' in feature_sets:
                logger.info("Adding temporal features")
                ml_dataset = self.temporal_engineer.create_temporal_ml_features(
                    ml_dataset, target_col=target_col
                )

            if 'location' in feature_sets and self.location is not None:
                logger.info("Adding location-aware features")
                ml_dataset = self.location_engineer.create_location_ml_features(
                    ml_dataset, location_manager=self.location
                )

            if 'financial' in feature_sets:
                logger.info("Adding financial features")
                ml_dataset = self.financial_engineer.create_financial_ml_features(
                    ml_dataset, self.electricity_rates, self.system_config
                )

            if 'weather' in feature_sets and weather_data is not None:
                logger.info("Adding weather features")
                ml_dataset = self.weather_engineer.create_weather_ml_features(
                    ml_dataset, weather_data
                )

            # Clean dataset
            ml_dataset = self._clean_dataset(ml_dataset, clean_strategy)

            # Log feature creation summary
            new_features = set(ml_dataset.columns) - initial_columns
            logger.info(f"Created {len(new_features)} features across {len(feature_sets)} feature sets")
            logger.info(f"Final dataset shape: {ml_dataset.shape}")

            return ml_dataset

        except Exception as e:
            logger.error(f"Error creating ML dataset: {e}")
            return daily_data.copy()

    def create_features_by_category(
        self,
        daily_data: pd.DataFrame,
        weather_data: Optional[pd.DataFrame] = None,
        target_col: str = 'Production (kWh)'
    ) -> Dict[str, pd.DataFrame]:
        """
        Create features by category for analysis and feature selection.

        Returns features organized by category for easier analysis.

        Args:
            daily_data: Daily solar production data
            weather_data: Optional weather data
            target_col: Target column for rolling/lag features

        Returns:
            Dictionary with feature categories as keys and DataFrames as values
        """
        try:
            feature_categories = {}

            # Temporal features
            temporal_data = self.temporal_engineer.create_temporal_ml_features(
                daily_data.copy(), target_col=target_col
            )
            feature_categories['temporal'] = self._extract_new_features(daily_data, temporal_data)

            # Location features
            if self.location is not None:
                location_data = self.location_engineer.create_location_ml_features(
                    daily_data.copy(), location_manager=self.location
                )
                feature_categories['location'] = self._extract_new_features(daily_data, location_data)

            # Financial features
            financial_data = self.financial_engineer.create_financial_ml_features(
                daily_data.copy(), self.electricity_rates, self.system_config
            )
            feature_categories['financial'] = self._extract_new_features(daily_data, financial_data)

            # Weather features
            if weather_data is not None:
                weather_feature_data = self.weather_engineer.create_weather_ml_features(
                    daily_data.copy(), weather_data
                )
                feature_categories['weather'] = self._extract_new_features(daily_data, weather_feature_data)

            return feature_categories

        except Exception as e:
            logger.error(f"Error creating features by category: {e}")
            return {}

    def get_feature_names_by_category(self) -> Dict[str, List[str]]:
        """
        Get feature names organized by category.

        Returns:
            Dictionary with feature categories and their feature names
        """
        return {
            'temporal': self.temporal_engineer.get_temporal_feature_names(),
            'weather': self.weather_engineer.get_weather_feature_names(),
            'financial': self.financial_engineer.get_financial_feature_names(),
            'location': self.location_engineer.get_location_feature_names()
        }

    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of all available features and configuration.

        Returns:
            Dictionary with feature engineering configuration and capabilities
        """
        feature_names = self.get_feature_names_by_category()

        return {
            'location_info': {
                'name': getattr(self.location, 'location_name', 'Unknown'),
                'latitude': getattr(self.location, 'latitude', None),
                'longitude': getattr(self.location, 'longitude', None),
                'configured': self.location is not None
            },
            'feature_counts': {
                category: len(features) for category, features in feature_names.items()
            },
            'total_features': sum(len(features) for features in feature_names.values()),
            'electricity_rates': self.electricity_rates,
            'system_config': self.system_config,
            'available_categories': list(feature_names.keys())
        }

    def validate_data_for_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data for feature engineering requirements.

        Args:
            data: Input DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }

        # Check datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            validation['errors'].append("Data must have datetime index")
            validation['valid'] = False

        # Check required columns
        required_cols = ['Production (kWh)']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            validation['errors'].append(f"Missing required columns: {missing_cols}")
            validation['valid'] = False

        # Check data sufficiency for rolling/lag features
        if len(data) < 30:
            validation['warnings'].append("Less than 30 days of data - some rolling features may be limited")

        if len(data) < 14:
            validation['warnings'].append("Less than 14 days of data - 14-day lag features will have many NaN values")

        # Check for missing values
        missing_pct = (data.isnull().sum() / len(data) * 100).max()
        if missing_pct > 10:
            validation['warnings'].append(f"High missing data percentage: {missing_pct:.1f}%")

        # Recommendations
        if 'Consumption (kWh)' not in data.columns:
            validation['recommendations'].append("Add consumption data for enhanced financial features")

        return validation

    def _clean_dataset(self, data: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """
        Clean dataset based on strategy.

        Args:
            data: DataFrame to clean
            strategy: Cleaning strategy ('dropna', 'fillna')

        Returns:
            Cleaned DataFrame
        """
        try:
            if strategy == 'dropna':
                # Drop rows with any NaN values
                cleaned = data.dropna()
                rows_dropped = len(data) - len(cleaned)
                if rows_dropped > 0:
                    logger.info(f"Dropped {rows_dropped} rows with NaN values")

            elif strategy == 'fillna':
                # Fill NaN values with appropriate strategies
                cleaned = data.copy()

                # Forward fill lag features
                lag_cols = [col for col in cleaned.columns if 'lag_' in col]
                if lag_cols:
                    cleaned[lag_cols] = cleaned[lag_cols].fillna(method='ffill')

                # Fill rolling features with mean
                rolling_cols = [col for col in cleaned.columns if 'rolling_' in col]
                if rolling_cols:
                    for col in rolling_cols:
                        cleaned[col] = cleaned[col].fillna(cleaned[col].mean())

                # Fill remaining with 0
                cleaned = cleaned.fillna(0)

            else:
                logger.warning(f"Unknown cleaning strategy: {strategy}, using dropna")
                cleaned = data.dropna()

            return cleaned

        except Exception as e:
            logger.error(f"Error cleaning dataset: {e}")
            return data

    def _extract_new_features(self, original_data: pd.DataFrame, enhanced_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only the new features from enhanced data.

        Args:
            original_data: Original DataFrame
            enhanced_data: DataFrame with new features

        Returns:
            DataFrame with only the new features
        """
        try:
            original_cols = set(original_data.columns)
            enhanced_cols = set(enhanced_data.columns)
            new_cols = list(enhanced_cols - original_cols)

            if new_cols:
                return enhanced_data[new_cols]
            else:
                return pd.DataFrame(index=enhanced_data.index)

        except Exception as e:
            logger.error(f"Error extracting new features: {e}")
            return pd.DataFrame(index=enhanced_data.index)