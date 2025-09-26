"""
Tests for Feature Engineering Module

Comprehensive tests for the solar energy feature engineering pipeline,
including individual feature engineers and the main orchestration pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
import sys
sys.path.append('src')

from src.features.feature_pipeline import FeaturePipeline
from src.features.temporal_features import TemporalFeatureEngineer
from src.features.weather_features import WeatherFeatureEngineer
from src.features.financial_features import FinancialFeatureEngineer
from src.features.location_features import LocationFeatureEngineer


class TestTemporalFeatureEngineer:
    """Test temporal feature engineering"""

    def setup_method(self):
        """Set up test fixtures"""
        self.engineer = TemporalFeatureEngineer()

    def test_add_time_features(self, sample_daily_data):
        """Test basic time feature creation"""
        result = self.engineer._add_time_features(sample_daily_data)

        # Check that time features are added
        expected_features = ['day_of_year', 'day_of_week', 'month', 'quarter', 'is_weekend']
        for feature in expected_features:
            assert feature in result.columns

        # Check data types and ranges
        assert result['day_of_year'].min() >= 1
        assert result['day_of_year'].max() <= 366
        assert result['day_of_week'].min() >= 0
        assert result['day_of_week'].max() <= 6
        assert result['month'].min() >= 1
        assert result['month'].max() <= 12
        assert result['is_weekend'].dtype == int

    def test_add_cyclical_features(self, sample_daily_data):
        """Test cyclical encoding features"""
        # First add time features that cyclical features depend on
        data_with_time = self.engineer._add_time_features(sample_daily_data)
        result = self.engineer._add_cyclical_features(data_with_time)

        # Check that cyclical features are added
        expected_features = ['day_sin', 'day_cos', 'week_sin', 'week_cos']
        for feature in expected_features:
            assert feature in result.columns

        # Check that sin/cos values are in correct range
        assert result['day_sin'].min() >= -1
        assert result['day_sin'].max() <= 1
        assert result['day_cos'].min() >= -1
        assert result['day_cos'].max() <= 1

    def test_add_rolling_features(self, sample_daily_data):
        """Test rolling statistics features"""
        target_col = 'Production (kWh)'
        result = self.engineer._add_rolling_features(sample_daily_data, target_col)

        # Check that rolling features are added
        expected_features = ['rolling_mean_7d', 'rolling_std_7d', 'rolling_mean_14d', 'rolling_mean_30d']
        for feature in expected_features:
            assert feature in result.columns

        # Check that rolling means are reasonable
        assert not result['rolling_mean_7d'].isnull().all()
        assert result['rolling_mean_7d'].min() >= 0  # Production can't be negative

    def test_add_lag_features(self, sample_daily_data):
        """Test lag feature creation"""
        target_col = 'Production (kWh)'
        result = self.engineer._add_lag_features(sample_daily_data, target_col)

        # Check that lag features are added
        lag_features = [col for col in result.columns if 'lag_' in col]
        assert len(lag_features) > 0

        # Check that lag features have correct values
        prod_lag_1d = 'production_kwh_lag_1d'
        if prod_lag_1d in result.columns:
            # First row should be NaN for lag features
            assert pd.isna(result[prod_lag_1d].iloc[0])
            # Second row should equal first row of original data
            if len(result) > 1:
                assert result[prod_lag_1d].iloc[1] == sample_daily_data[target_col].iloc[0]

    def test_create_temporal_ml_features_integration(self, sample_daily_data):
        """Test full temporal feature creation"""
        result = self.engineer.create_temporal_ml_features(sample_daily_data)

        # Should have more columns than original
        assert len(result.columns) > len(sample_daily_data.columns)

        # Should maintain same number of rows
        assert len(result) <= len(sample_daily_data)  # May drop some due to NaN

        # Check that key feature types are present
        time_features = [col for col in result.columns if any(x in col for x in ['day_', 'month', 'quarter'])]
        rolling_features = [col for col in result.columns if 'rolling_' in col]
        lag_features = [col for col in result.columns if 'lag_' in col]

        assert len(time_features) > 0
        assert len(rolling_features) > 0
        assert len(lag_features) > 0

    def test_invalid_data_handling(self):
        """Test handling of invalid data"""
        # Test with non-datetime index
        invalid_data = pd.DataFrame({'Production (kWh)': [1, 2, 3]})
        result = self.engineer.create_temporal_ml_features(invalid_data)

        # Should return copy of original data
        assert len(result.columns) == len(invalid_data.columns)


class TestWeatherFeatureEngineer:
    """Test weather feature engineering"""

    def setup_method(self):
        """Set up test fixtures"""
        self.engineer = WeatherFeatureEngineer()

    def test_weather_efficiency_features(self, sample_daily_data, sample_weather_data):
        """Test weather efficiency feature creation"""
        result = self.engineer._add_weather_efficiency_features(
            sample_daily_data, sample_weather_data
        )

        # Check that weather features might be added (depends on weather data structure)
        # Should at least return the original data unchanged
        assert len(result) == len(sample_daily_data)

    def test_solar_weather_score_calculation(self, sample_weather_data):
        """Test solar weather score calculation"""
        # Create merged data format
        merged_data = sample_weather_data.copy()
        if 'sunshine_hours' not in merged_data.columns:
            merged_data['sunshine_hours'] = np.random.uniform(0, 12, len(merged_data))
        if 'cloudcover_mean_pct' not in merged_data.columns:
            merged_data['cloudcover_mean_pct'] = np.random.uniform(0, 100, len(merged_data))

        score = self.engineer._calculate_solar_weather_score(merged_data)

        # Score should be between 0 and 100
        assert score.min() >= 0
        assert score.max() <= 100
        assert not score.isnull().all()

    def test_weather_feature_names(self):
        """Test weather feature name retrieval"""
        feature_names = self.engineer.get_weather_feature_names()

        # Should return a list of strings
        assert isinstance(feature_names, list)
        assert all(isinstance(name, str) for name in feature_names)
        assert len(feature_names) > 0


class TestFinancialFeatureEngineer:
    """Test financial feature engineering"""

    def setup_method(self):
        """Set up test fixtures"""
        self.engineer = FinancialFeatureEngineer()

    def test_efficiency_features(self, sample_daily_data):
        """Test financial efficiency feature creation"""
        # Mock analysis results
        analysis_results = {
            'annual_metrics': {
                'capacity_factor': 20.0,
                'self_consumption_rate': 65.0,
                'grid_independence_rate': 70.0,
                'daily_average_kwh': 35.0
            }
        }

        result = self.engineer._add_efficiency_features(sample_daily_data, analysis_results)

        # Check that efficiency features are added
        expected_features = ['capacity_factor', 'self_consumption_rate', 'grid_independence']
        for feature in expected_features:
            assert feature in result.columns

        # Check that values are reasonable (converted to 0-1 scale)
        assert result['capacity_factor'].iloc[0] == 0.2  # 20% -> 0.2
        assert result['self_consumption_rate'].iloc[0] == 0.65  # 65% -> 0.65

    def test_financial_performance_features(self, sample_daily_data):
        """Test financial performance feature creation"""
        analysis_results = {'financial_benefits': {'total_annual_savings': 2000}}
        electricity_rates = {'residential_rate': 15.0, 'feed_in_tariff': 12.0}

        result = self.engineer._add_financial_performance_features(
            sample_daily_data, analysis_results, electricity_rates
        )

        # Should maintain original structure
        assert len(result) == len(sample_daily_data)

    def test_financial_feature_names(self):
        """Test financial feature name retrieval"""
        feature_names = self.engineer.get_financial_feature_names()

        # Should return a list of strings
        assert isinstance(feature_names, list)
        assert all(isinstance(name, str) for name in feature_names)
        assert len(feature_names) > 0


class TestLocationFeatureEngineer:
    """Test location feature engineering"""

    def setup_method(self):
        """Set up test fixtures"""
        self.engineer = LocationFeatureEngineer()

    def test_solar_geometry_features(self, sample_daily_data, mock_location_manager):
        """Test solar geometry feature creation"""
        result = self.engineer._add_solar_geometry_features(
            sample_daily_data, mock_location_manager
        )

        # Check that geometry features are added
        expected_features = ['theoretical_daylight_hours', 'max_solar_elevation', 'theoretical_irradiance']
        for feature in expected_features:
            assert feature in result.columns

        # Check reasonable ranges
        assert result['theoretical_daylight_hours'].min() >= 6  # Minimum reasonable daylight
        assert result['theoretical_daylight_hours'].max() <= 18  # Maximum reasonable daylight
        assert result['max_solar_elevation'].min() >= 0
        assert result['max_solar_elevation'].max() <= 90

    def test_solar_efficiency_features(self, sample_daily_data):
        """Test solar efficiency feature creation with location data"""
        # Add required location features first
        sample_daily_data['theoretical_daylight_hours'] = 12.0
        sample_daily_data['theoretical_irradiance'] = 1000.0

        result = self.engineer._add_solar_efficiency_features(sample_daily_data)

        # Check that efficiency features are added
        expected_features = ['daylight_utilization', 'irradiance_efficiency']
        for feature in expected_features:
            assert feature in result.columns

        # Check that calculations are reasonable
        assert (result['daylight_utilization'] >= 0).all()
        assert (result['irradiance_efficiency'] >= 0).all()

    def test_location_feature_names(self):
        """Test location feature name retrieval"""
        feature_names = self.engineer.get_location_feature_names()

        # Should return a list of strings
        assert isinstance(feature_names, list)
        assert all(isinstance(name, str) for name in feature_names)
        assert len(feature_names) > 0


class TestFeaturePipeline:
    """Test main feature pipeline orchestration"""

    def setup_method(self):
        """Set up test fixtures"""
        self.pipeline = FeaturePipeline()

    def test_create_ml_dataset_basic(self, sample_daily_data):
        """Test basic ML dataset creation"""
        result = self.pipeline.create_ml_dataset(
            sample_daily_data,
            feature_sets=['temporal']
        )

        # Should have more features than original
        assert len(result.columns) > len(sample_daily_data.columns)

        # Should maintain reasonable row count (some may be dropped due to NaN)
        assert len(result) <= len(sample_daily_data)
        assert len(result) >= len(sample_daily_data) * 0.4  # At least 40% retained (rolling features drop many)

    def test_create_ml_dataset_all_features(self, sample_daily_data):
        """Test ML dataset creation with all feature types"""
        result = self.pipeline.create_ml_dataset(
            sample_daily_data,
            feature_sets=['temporal', 'financial']  # Skip weather/location for simplicity
        )

        # Should have significantly more features
        assert len(result.columns) > len(sample_daily_data.columns) + 10

        # Check that different feature types are present
        temporal_features = [col for col in result.columns if any(x in col for x in ['day_', 'rolling_', 'lag_'])]
        financial_features = [col for col in result.columns if any(x in col for x in ['capacity_', 'efficiency', 'savings'])]

        assert len(temporal_features) > 0
        assert len(financial_features) > 0

    def test_feature_pipeline_validation(self, sample_daily_data):
        """Test data validation in feature pipeline"""
        validation = self.pipeline.validate_data_for_features(sample_daily_data)

        # Should pass basic validation
        assert isinstance(validation, dict)
        assert 'valid' in validation
        assert 'warnings' in validation
        assert 'errors' in validation

    def test_get_feature_names_by_category(self):
        """Test feature name organization by category"""
        feature_names = self.pipeline.get_feature_names_by_category()

        # Should return dictionary with categories
        assert isinstance(feature_names, dict)
        expected_categories = ['temporal', 'weather', 'financial', 'location']
        for category in expected_categories:
            assert category in feature_names
            assert isinstance(feature_names[category], list)

    def test_get_feature_summary(self):
        """Test feature summary generation"""
        summary = self.pipeline.get_feature_summary()

        # Should return comprehensive summary
        assert isinstance(summary, dict)
        expected_keys = ['location_info', 'feature_counts', 'total_features']
        for key in expected_keys:
            assert key in summary

        assert isinstance(summary['total_features'], int)
        assert summary['total_features'] > 0

    def test_invalid_data_handling(self):
        """Test handling of invalid input data"""
        # Test with non-datetime index
        invalid_data = pd.DataFrame({'Production (kWh)': [1, 2, 3]})

        result = self.pipeline.create_ml_dataset(invalid_data, feature_sets=['temporal'])

        # Should return original data gracefully
        assert len(result.columns) == len(invalid_data.columns)

    def test_error_handling_graceful_fallback(self, sample_daily_data):
        """Test graceful error handling in pipeline"""
        # Mock an error in feature creation
        with patch.object(self.pipeline.temporal_engineer, 'create_temporal_ml_features',
                         side_effect=Exception("Test error")):
            result = self.pipeline.create_ml_dataset(
                sample_daily_data,
                feature_sets=['temporal']
            )

            # Should still return data (graceful fallback)
            assert len(result) == len(sample_daily_data)


# Additional fixtures for feature engineering tests
@pytest.fixture
def sample_daily_data():
    """Generate sample daily solar data for feature testing"""
    dates = pd.date_range("2023-01-01", periods=60, freq="D")  # 2 months of data

    # Generate realistic production pattern
    day_of_year = dates.dayofyear
    seasonal_factor = 1 + 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365)  # Peak in summer
    base_production = 30 * seasonal_factor + np.random.normal(0, 5, len(dates))
    base_production = np.maximum(base_production, 0)  # No negative production

    consumption = 35 + np.random.normal(0, 8, len(dates))
    consumption = np.maximum(consumption, 5)  # Minimum consumption

    export = np.maximum(base_production - consumption, 0)
    import_energy = np.maximum(consumption - base_production, 0)

    df = pd.DataFrame({
        'Production (kWh)': base_production,
        'Consumption (kWh)': consumption,
        'Export (kWh)': export,
        'Import (kWh)': import_energy
    }, index=dates)

    return df


@pytest.fixture
def sample_weather_data():
    """Generate sample weather data for testing"""
    dates = pd.date_range("2023-01-01", periods=60, freq="D")

    df = pd.DataFrame({
        'sunshine_hours': np.random.uniform(4, 12, len(dates)),
        'cloudcover_mean_pct': np.random.uniform(10, 90, len(dates)),
        'temp_mean_c': np.random.uniform(15, 30, len(dates)),
        'temp_max_c': np.random.uniform(20, 35, len(dates)),
        'temp_min_c': np.random.uniform(10, 25, len(dates)),
        'precipitation_mm': np.random.exponential(2, len(dates)),
        'solar_radiation_mj': np.random.uniform(10, 30, len(dates))
    }, index=dates)

    return df


@pytest.fixture
def mock_location_manager():
    """Mock location manager for testing"""
    mock = Mock()
    mock.latitude = 35.663
    mock.longitude = -78.844
    mock.location_name = "Holly Springs, NC"

    # Mock methods
    mock.get_sunrise_sunset.return_value = (6.0, 18.0)  # 12 hours daylight
    mock.get_solar_elevation.return_value = 45.0
    mock.get_theoretical_solar_irradiance.return_value = 1000.0
    mock.get_location_summary.return_value = {
        'climate_type': 'Temperate',
        'seasonal_variation': 0.5,
        'winter_daylight_hours': 9.6,
        'summer_daylight_hours': 14.4
    }

    return mock