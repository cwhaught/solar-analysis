"""
Tests for SolarDataManager class
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import tempfile
import os

from src.data_manager import SolarDataManager


class TestSolarDataManager:
    """Test cases for SolarDataManager"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary CSV file
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test_data.csv")
        self.cache_dir = os.path.join(self.temp_dir, "cache")

        # Create sample CSV data
        sample_data = {
            'Date/Time': ['2023-01-01 00:00:00', '2023-01-01 00:15:00', '2023-01-01 00:30:00'],
            'Production (Wh)': [1000, 1500, 2000],
            'Consumption (Wh)': [800, 1200, 1600],
            'Export (Wh)': [200, 300, 400],
            'Import (Wh)': [0, 0, 0]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(self.csv_path, index=False)

        # Mock EnphaseClient
        self.mock_client = Mock()

        # Initialize data manager
        self.data_manager = SolarDataManager(
            csv_path=self.csv_path,
            enphase_client=self.mock_client,
            cache_dir=self.cache_dir
        )

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test data manager initialization"""
        assert self.data_manager.csv_path == Path(self.csv_path)
        assert self.data_manager.client == self.mock_client
        assert self.data_manager.cache_dir == Path(self.cache_dir)
        assert self.data_manager.cache_dir.exists()

    def test_load_csv_data(self):
        """Test CSV data loading"""
        df = self.data_manager.load_csv_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ['Production (kWh)', 'Consumption (kWh)', 'Export (kWh)', 'Import (kWh)']
        assert df.index.name == 'Date/Time'
        assert df.attrs['source'] == 'csv'
        assert df.attrs['granularity'] == '15min'

        # Check data conversion from Wh to kWh
        assert df.iloc[0]['Production (kWh)'] == 1.0  # 1000 Wh -> 1.0 kWh

    def test_load_csv_data_caching(self):
        """Test CSV data caching behavior"""
        # First load
        df1 = self.data_manager.load_csv_data()

        # Second load (should use cache)
        df2 = self.data_manager.load_csv_data()

        # Should be the same object (cached)
        assert df1 is df2

    def test_load_csv_data_force_reload(self):
        """Test forcing CSV data reload"""
        # First load
        df1 = self.data_manager.load_csv_data()

        # Force reload
        df2 = self.data_manager.load_csv_data(force_reload=True)

        # Should be different objects but same data
        assert df1 is not df2
        assert df1.equals(df2)

    def test_load_csv_data_file_not_found(self):
        """Test CSV loading with missing file"""
        invalid_manager = SolarDataManager(
            csv_path="/nonexistent/file.csv",
            enphase_client=self.mock_client,
            cache_dir=self.cache_dir
        )

        with pytest.raises(FileNotFoundError):
            invalid_manager.load_csv_data()

    def test_load_api_data_caching(self):
        """Test API data caching"""
        # Mock API response
        mock_df = pd.DataFrame({
            'production': [100, 200, 300],
            'date': pd.date_range('2023-01-01', periods=3, freq='D')
        })
        self.mock_client.get_energy_lifetime.return_value = mock_df

        # First load
        df1 = self.data_manager.load_api_data()

        # Second load (should use cache)
        df2 = self.data_manager.load_api_data()

        # API should only be called once
        assert self.mock_client.get_energy_lifetime.call_count == 1
        assert df1 is df2

    def test_get_daily_production_csv_only(self):
        """Test daily production aggregation from CSV only"""
        # Load CSV data first
        self.data_manager.load_csv_data()

        daily_df = self.data_manager.get_daily_production()

        assert isinstance(daily_df, pd.DataFrame)
        assert len(daily_df) == 1  # Should aggregate to 1 day
        assert daily_df.index[0].date() == datetime(2023, 1, 1).date()

    @patch('src.data_manager.datetime')
    def test_cache_file_paths(self, mock_datetime):
        """Test cache file path generation"""
        mock_datetime.now.return_value = datetime(2023, 6, 15)

        expected_daily = Path(self.cache_dir) / "daily_production_combined.csv"
        expected_detailed = Path(self.cache_dir) / "detailed_production_recent.csv"

        assert self.data_manager.daily_cache_file == expected_daily
        assert self.data_manager.detailed_cache_file == expected_detailed

    def test_data_validation(self):
        """Test data validation and cleaning"""
        df = self.data_manager.load_csv_data()

        # Check that all values are non-negative after conversion
        assert (df >= 0).all().all()

        # Check datetime index
        assert pd.api.types.is_datetime64_any_dtype(df.index)

        # Check column names are standardized
        expected_cols = ['Production (kWh)', 'Consumption (kWh)', 'Export (kWh)', 'Import (kWh)']
        assert list(df.columns) == expected_cols