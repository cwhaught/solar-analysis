"""
Feature Engineering Module for Solar Energy Analysis

This module provides comprehensive feature engineering capabilities for solar energy
machine learning models, building on existing infrastructure:

- Extends existing WeatherAnalyzer for weather features
- Integrates with SolarFinancialCalculator for financial features
- Leverages LocationManager for location-aware features
- Orchestrates feature pipelines for ML model preparation

Key Components:
- FeaturePipeline: Main orchestration class
- WeatherFeatureEngineer: Weather-specific features (extends WeatherAnalyzer)
- FinancialFeatureEngineer: Financial efficiency features
- TemporalFeatureEngineer: Time-based and cyclical features
- LocationFeatureEngineer: Location-aware solar geometry features
"""

from .feature_pipeline import FeaturePipeline
from .weather_features import WeatherFeatureEngineer
from .financial_features import FinancialFeatureEngineer
from .temporal_features import TemporalFeatureEngineer
from .location_features import LocationFeatureEngineer

__all__ = [
    'FeaturePipeline',
    'WeatherFeatureEngineer',
    'FinancialFeatureEngineer',
    'TemporalFeatureEngineer',
    'LocationFeatureEngineer'
]

__version__ = '1.0.0'