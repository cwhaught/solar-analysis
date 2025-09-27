"""
Tests for Data Processing Utilities

Tests for SolarDataProcessor that consolidates data processing patterns
from notebooks into reusable utilities.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
sys.path.append('src')

from src.data.processors import SolarDataProcessor


class TestSolarDataProcessor:
    """Test the SolarDataProcessor utility"""

    def setup_method(self):
        """Set up test fixtures"""
        self.processor = SolarDataProcessor()

        # Create sample daily data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        self.sample_daily_data = pd.DataFrame({
            'Production (kWh)': np.random.uniform(20, 40, 30),
            'Consumption (kWh)': np.random.uniform(25, 35, 30),
            'Export (kWh)': np.random.uniform(0, 15, 30),
            'Import (kWh)': np.random.uniform(0, 10, 30)
        }, index=dates)

        # Add metadata
        self.sample_daily_data.attrs = {
            'source': 'test',
            'granularity': '15min',
            'loaded_at': datetime.now()
        }

    def test_init(self):
        """Test processor initialization"""
        processor = SolarDataProcessor()
        assert processor.default_energy_columns == [
            'Production (kWh)', 'Consumption (kWh)',
            'Export (kWh)', 'Import (kWh)'
        ]

    def test_create_daily_summary_sum(self):
        """Test daily summary creation with sum aggregation"""
        # Create 15-minute data
        dates = pd.date_range('2023-01-01', periods=96, freq='15min')  # 1 day
        fifteen_min_data = pd.DataFrame({
            'Production (kWh)': np.ones(96) * 0.5,  # 0.5 kWh per 15 min
            'Consumption (kWh)': np.ones(96) * 0.3
        }, index=dates)

        result = self.processor.create_daily_summary(fifteen_min_data, method='sum')

        # Should aggregate to 1 day
        assert len(result) == 1
        assert result.index[0].date() == datetime(2023, 1, 1).date()

        # Should sum values correctly (allow for floating point precision)
        assert abs(result['Production (kWh)'].iloc[0] - 48.0) < 0.001  # 96 * 0.5
        assert abs(result['Consumption (kWh)'].iloc[0] - 28.8) < 0.001  # 96 * 0.3

        # Check metadata
        assert result.attrs['granularity'] == 'daily'
        assert result.attrs['aggregation_method'] == 'sum'

    def test_create_daily_summary_mean(self):
        """Test daily summary creation with mean aggregation"""
        dates = pd.date_range('2023-01-01', periods=96, freq='15min')
        fifteen_min_data = pd.DataFrame({
            'Production (kWh)': np.random.uniform(0, 2, 96),
            'Consumption (kWh)': np.random.uniform(0.5, 1.5, 96)
        }, index=dates)

        result = self.processor.create_daily_summary(fifteen_min_data, method='mean')

        # Should calculate means
        expected_prod_mean = fifteen_min_data['Production (kWh)'].mean()
        expected_cons_mean = fifteen_min_data['Consumption (kWh)'].mean()

        assert abs(result['Production (kWh)'].iloc[0] - expected_prod_mean) < 0.001
        assert abs(result['Consumption (kWh)'].iloc[0] - expected_cons_mean) < 0.001

    def test_create_monthly_summary(self):
        """Test monthly summary creation using non-deprecated frequency"""
        # Create 3 months of daily data
        dates = pd.date_range('2023-01-01', periods=90, freq='D')
        daily_data = pd.DataFrame({
            'Production (kWh)': np.random.uniform(20, 40, 90),
            'Consumption (kWh)': np.random.uniform(25, 35, 90)
        }, index=dates)

        result = self.processor.create_monthly_summary(daily_data, method='mean')

        # Should create 3 monthly records
        assert len(result) == 3

        # Check that it uses 'ME' (month end) frequency
        assert result.attrs['granularity'] == 'monthly'
        assert result.attrs['aggregation_method'] == 'mean'

    def test_calculate_net_energy(self):
        """Test net energy calculations"""
        result = self.processor.calculate_net_energy(self.sample_daily_data)

        # Check new columns are added
        assert 'Net Energy (kWh)' in result.columns
        assert 'Energy Surplus' in result.columns
        assert 'Self Sufficiency (%)' in result.columns

        # Check calculations
        for i in range(len(result)):
            expected_net = result['Production (kWh)'].iloc[i] - result['Consumption (kWh)'].iloc[i]
            assert abs(result['Net Energy (kWh)'].iloc[i] - expected_net) < 0.001

            expected_surplus = expected_net > 0
            assert result['Energy Surplus'].iloc[i] == expected_surplus

    def test_calculate_self_consumption_metrics(self):
        """Test self-consumption metrics calculation"""
        result = self.processor.calculate_self_consumption_metrics(self.sample_daily_data)

        # Check new columns are added
        expected_cols = ['Self Consumed (kWh)', 'Self Consumption Rate (%)', 'Grid Independence (%)']
        for col in expected_cols:
            assert col in result.columns

        # Check calculations
        for i in range(len(result)):
            expected_self_consumed = result['Production (kWh)'].iloc[i] - result['Export (kWh)'].iloc[i]
            assert abs(result['Self Consumed (kWh)'].iloc[i] - expected_self_consumed) < 0.001

            # Self consumption rate should be between 0 and 100
            assert 0 <= result['Self Consumption Rate (%)'].iloc[i] <= 100

            # Grid independence should be between 0 and 100
            assert 0 <= result['Grid Independence (%)'].iloc[i] <= 100

    def test_add_time_features(self):
        """Test time feature addition"""
        result = self.processor.add_time_features(self.sample_daily_data)

        # Check basic time features
        expected_features = [
            'day_of_year', 'day_of_week', 'month', 'quarter', 'week_of_year',
            'is_weekend', 'is_monday', 'is_winter', 'is_spring', 'is_summer', 'is_fall'
        ]
        for feature in expected_features:
            assert feature in result.columns

        # Check value ranges
        assert result['day_of_year'].min() >= 1
        assert result['day_of_year'].max() <= 366
        assert result['day_of_week'].min() >= 0
        assert result['day_of_week'].max() <= 6
        assert result['month'].min() >= 1
        assert result['month'].max() <= 12

        # Check binary features
        assert result['is_weekend'].dtype == int
        assert all(val in [0, 1] for val in result['is_weekend'])

    def test_add_time_features_with_cyclical(self):
        """Test time features with cyclical encoding"""
        result = self.processor.add_time_features(self.sample_daily_data, include_cyclical=True)

        # Check cyclical features are added
        cyclical_features = ['day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin', 'month_cos']
        for feature in cyclical_features:
            assert feature in result.columns

        # Check sin/cos ranges
        for feature in cyclical_features:
            assert result[feature].min() >= -1
            assert result[feature].max() <= 1

    def test_add_rolling_features(self):
        """Test rolling features addition"""
        result = self.processor.add_rolling_features(
            self.sample_daily_data,
            target_columns=['Production (kWh)'],
            windows=[7, 14],
            statistics=['mean', 'std']
        )

        # Check rolling features are added
        expected_features = [
            'production_rolling_mean_7d', 'production_rolling_std_7d',
            'production_rolling_mean_14d', 'production_rolling_std_14d'
        ]
        for feature in expected_features:
            assert feature in result.columns

        # Check that rolling means are reasonable
        assert result['production_rolling_mean_7d'].min() >= 0
        assert not result['production_rolling_mean_7d'].isnull().all()

    def test_add_rolling_features_default_columns(self):
        """Test rolling features with default energy columns"""
        result = self.processor.add_rolling_features(self.sample_daily_data)

        # Should create rolling features for all energy columns
        energy_cols = ['production', 'consumption', 'export', 'import']
        for col in energy_cols:
            mean_feature = f'{col}_rolling_mean_7d'
            assert mean_feature in result.columns

    def test_calculate_efficiency_metrics(self):
        """Test efficiency metrics calculation"""
        result = self.processor.calculate_efficiency_metrics(
            self.sample_daily_data,
            production_col='Production (kWh)'
        )

        # Check new columns
        assert 'Relative Performance' in result.columns
        assert 'Production Volatility' in result.columns
        assert 'Production CV' in result.columns

        # Check value ranges
        assert result['Relative Performance'].min() >= 0
        assert result['Production Volatility'].min() >= 0
        assert result['Production CV'].min() >= 0

    def test_prepare_ml_dataset(self):
        """Test comprehensive ML dataset preparation"""
        result = self.processor.prepare_ml_dataset(
            self.sample_daily_data,
            target_column='Production (kWh)'
        )

        # Should have significantly more columns than original
        assert len(result.columns) > len(self.sample_daily_data.columns)

        # Should include various feature types
        feature_types = {
            'net_energy': 'Net Energy (kWh)',
            'self_consumption': 'Self Consumed (kWh)',
            'time': 'day_of_year',
            'rolling': 'production_rolling_mean_7d',
            'efficiency': 'Relative Performance'
        }

        for feature_type, example_col in feature_types.items():
            assert example_col in result.columns, f"Missing {feature_type} features"

    def test_prepare_ml_dataset_dropna(self):
        """Test ML dataset preparation with NaN handling"""
        # Add some NaN values
        data_with_nan = self.sample_daily_data.copy()
        data_with_nan.loc[data_with_nan.index[5], 'Production (kWh)'] = np.nan

        result = self.processor.prepare_ml_dataset(data_with_nan, dropna=True)

        # Should remove rows with NaN
        assert len(result) < len(data_with_nan)
        assert not result.isnull().any().any()

    def test_prepare_ml_dataset_keep_na(self):
        """Test ML dataset preparation keeping NaN values"""
        data_with_nan = self.sample_daily_data.copy()
        data_with_nan.loc[data_with_nan.index[5], 'Production (kWh)'] = np.nan

        result = self.processor.prepare_ml_dataset(data_with_nan, dropna=False)

        # Should keep all rows
        assert len(result) == len(data_with_nan)

    def test_error_handling_missing_columns(self):
        """Test error handling for missing required columns"""
        incomplete_data = pd.DataFrame({
            'Production (kWh)': [1, 2, 3]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

        # Should raise error when trying to calculate net energy without consumption
        with pytest.raises(ValueError, match="Consumption column"):
            self.processor.calculate_net_energy(incomplete_data)

        # Should raise error for self-consumption without required columns
        with pytest.raises(ValueError, match="Missing required columns"):
            self.processor.calculate_self_consumption_metrics(incomplete_data)

    def test_error_handling_non_datetime_index(self):
        """Test error handling for non-datetime index"""
        data_wrong_index = pd.DataFrame({
            'Production (kWh)': [1, 2, 3],
            'Consumption (kWh)': [1, 1, 1]
        })  # No datetime index

        with pytest.raises(ValueError, match="DatetimeIndex"):
            self.processor.add_time_features(data_wrong_index)

    def test_metadata_preservation(self):
        """Test that metadata is preserved through processing"""
        result = self.processor.create_daily_summary(
            self.sample_daily_data,
            preserve_metadata=True
        )

        # Original metadata should be preserved
        assert 'source' in result.attrs
        assert result.attrs['source'] == 'test'

        # New metadata should be added
        assert result.attrs['granularity'] == 'daily'
        assert result.attrs['aggregation_method'] == 'sum'

    def test_unsupported_aggregation_method(self):
        """Test error handling for unsupported aggregation methods"""
        with pytest.raises(ValueError, match="Unsupported aggregation method"):
            self.processor.create_daily_summary(
                self.sample_daily_data,
                method='invalid_method'
            )

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames"""
        # Create empty DataFrame with datetime index
        empty_df = pd.DataFrame(
            columns=['Production (kWh)', 'Consumption (kWh)'],
            index=pd.DatetimeIndex([], name='datetime')
        )

        # Should handle empty data gracefully
        result = self.processor.create_daily_summary(empty_df)
        assert len(result) == 0
        assert list(result.columns) == list(empty_df.columns)


class TestIntegrationProcessing:
    """Integration tests for data processing utilities"""

    def test_full_processing_pipeline(self):
        """Test complete processing pipeline with realistic data"""
        # Create realistic solar data
        dates = pd.date_range('2023-01-01', periods=90, freq='D')  # 3 months

        # Simulate seasonal variation
        day_of_year = dates.dayofyear
        seasonal_factor = 1 + 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
        production = 30 * seasonal_factor + np.random.normal(0, 3, len(dates))
        production = np.maximum(production, 0)  # No negative production

        consumption = 35 + np.random.normal(0, 5, len(dates))
        consumption = np.maximum(consumption, 5)  # Minimum consumption

        export = np.maximum(production - consumption, 0)
        import_energy = np.maximum(consumption - production, 0)

        daily_data = pd.DataFrame({
            'Production (kWh)': production,
            'Consumption (kWh)': consumption,
            'Export (kWh)': export,
            'Import (kWh)': import_energy
        }, index=dates)

        processor = SolarDataProcessor()

        # Test complete pipeline
        ml_dataset = processor.prepare_ml_dataset(daily_data)

        # Should have comprehensive features
        assert len(ml_dataset.columns) > 20  # Many features added

        # Should maintain data integrity
        assert len(ml_dataset) <= len(daily_data)  # Some rows may be dropped due to rolling features

        # Check that energy balance is maintained
        if 'Net Energy (kWh)' in ml_dataset.columns:
            net_energy = ml_dataset['Net Energy (kWh)']
            expected_net = ml_dataset['Production (kWh)'] - ml_dataset['Consumption (kWh)']
            assert np.allclose(net_energy, expected_net, atol=0.001)

    def test_processing_performance(self):
        """Test processing performance with larger datasets"""
        import time

        # Create larger dataset
        dates = pd.date_range('2023-01-01', periods=365, freq='D')  # 1 year
        large_data = pd.DataFrame({
            'Production (kWh)': np.random.uniform(10, 50, 365),
            'Consumption (kWh)': np.random.uniform(20, 40, 365),
            'Export (kWh)': np.random.uniform(0, 20, 365),
            'Import (kWh)': np.random.uniform(0, 15, 365)
        }, index=dates)

        processor = SolarDataProcessor()

        start_time = time.time()
        result = processor.prepare_ml_dataset(large_data)
        processing_time = time.time() - start_time

        # Should process reasonably quickly (under 5 seconds for 1 year of daily data)
        assert processing_time < 5.0
        assert len(result) > 300  # Should keep most data after rolling features