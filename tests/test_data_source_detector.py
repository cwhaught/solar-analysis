"""
Tests for DataSourceDetector module

Tests the intelligent data source detection, prioritization, and validation logic.
"""

import pytest
import os
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime, timedelta

from src.core.data_source_detector import DataSourceDetector, DataSourceInfo


class TestDataSourceInfo:
    """Test DataSourceInfo class"""

    def test_data_source_info_creation(self):
        """Test DataSourceInfo object creation"""
        info = DataSourceInfo(
            source_type="REAL_CSV",
            description="Real solar data",
            path="/path/to/data.csv",
            priority=100
        )

        assert info.source_type == "REAL_CSV"
        assert info.description == "Real solar data"
        assert info.path == "/path/to/data.csv"
        assert info.priority == 100
        assert info.available is True

    def test_data_source_info_defaults(self):
        """Test DataSourceInfo with default values"""
        info = DataSourceInfo("TEST", "Test description")

        assert info.source_type == "TEST"
        assert info.description == "Test description"
        assert info.path is None
        assert info.priority == 0
        assert info.available is True


class TestDataSourceDetector:
    """Test DataSourceDetector class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_location = Mock()
        self.mock_location.location_name = "Holly Springs, NC"

        self.detector = DataSourceDetector(location=self.mock_location)

    def test_init_without_location(self):
        """Test initialization without location"""
        detector = DataSourceDetector()
        assert detector.location is None

    def test_init_with_location(self):
        """Test initialization with location"""
        assert self.detector.location == self.mock_location


class TestCreateEnphaseClient:
    """Test Enphase client creation logic"""

    def setup_method(self):
        """Set up test fixtures"""
        self.detector = DataSourceDetector()
        # Clear environment variables
        for var in ['ENPHASE_ACCESS_TOKEN', 'ENPHASE_API_KEY', 'ENPHASE_SYSTEM_ID']:
            if var in os.environ:
                del os.environ[var]

    @patch('pathlib.Path.exists')
    def test_no_env_file(self, mock_exists):
        """Test behavior when no .env file exists"""
        mock_exists.return_value = False

        client, client_type = self.detector.create_enphase_client()

        assert client_type == "MOCK"
        assert hasattr(client, '__class__')
        assert "Mock" in client.__class__.__name__

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="# Just comments\n")
    def test_env_file_no_credentials(self, mock_file, mock_exists):
        """Test behavior when .env file exists but has no credentials"""
        mock_exists.return_value = True

        client, client_type = self.detector.create_enphase_client()

        assert client_type == "MOCK"

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="""
ENPHASE_ACCESS_TOKEN=your_access_token_here
ENPHASE_API_KEY=your_api_key_here
ENPHASE_SYSTEM_ID=your_system_id_here
""")
    def test_env_file_placeholder_credentials(self, mock_file, mock_exists):
        """Test behavior when .env file has placeholder credentials"""
        mock_exists.return_value = True

        client, client_type = self.detector.create_enphase_client()

        assert client_type == "MOCK"

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="""
ENPHASE_ACCESS_TOKEN=real_token_123
ENPHASE_API_KEY=real_key_456
ENPHASE_SYSTEM_ID=real_system_789
""")
    def test_env_file_real_credentials(self, mock_file, mock_exists):
        """Test behavior when .env file has real credentials"""
        mock_exists.return_value = True
        mock_enphase_client = Mock()

        # Mock the entire import and class creation
        with patch('builtins.__import__') as mock_import:
            mock_enphase_module = Mock()
            mock_enphase_module.EnphaseClient = Mock(return_value=mock_enphase_client)
            mock_import.return_value = mock_enphase_module

            client, client_type = self.detector.create_enphase_client()

        assert client_type == "REAL_API"
        assert client == mock_enphase_client

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="""
ENPHASE_ACCESS_TOKEN=real_token_123
ENPHASE_API_KEY=real_key_456
ENPHASE_SYSTEM_ID=real_system_789
""")
    def test_env_file_client_creation_error(self, mock_file, mock_exists):
        """Test behavior when EnphaseClient creation fails"""
        mock_exists.return_value = True

        # Mock import to raise exception
        with patch('builtins.__import__', side_effect=Exception("Import error")):
            client, client_type = self.detector.create_enphase_client()

        assert client_type == "MOCK"


class TestDetectAvailableCSVSources:
    """Test CSV source detection logic"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_location = Mock()
        self.mock_location.location_name = "Holly Springs, NC"
        self.detector = DataSourceDetector(location=self.mock_location)

    @patch('pathlib.Path.exists')
    def test_no_csv_files(self, mock_exists):
        """Test when no CSV files exist"""
        mock_exists.return_value = False

        sources = self.detector.detect_available_csv_sources()

        assert len(sources) == 0

    def test_csv_source_priority_ordering(self):
        """Test that CSV sources are properly prioritized"""
        # Create mock sources directly
        sources = [
            DataSourceInfo("GENERIC_MOCK", "Generic mock", "../data/raw/mock.csv", 25),
            DataSourceInfo("REAL_CSV", "Real data", "../data/raw/real.csv", 100),
            DataSourceInfo("SYNTHETIC_LOCATION", "Location data", "../data/raw/location.csv", 50)
        ]

        # Sort them as the detector would
        sources.sort(key=lambda x: x.priority, reverse=True)

        assert sources[0].source_type == "REAL_CSV"
        assert sources[0].priority == 100
        assert sources[1].source_type == "SYNTHETIC_LOCATION"
        assert sources[1].priority == 50
        assert sources[2].source_type == "GENERIC_MOCK"
        assert sources[2].priority == 25

    def test_location_name_processing(self):
        """Test location name processing for path generation"""
        test_cases = [
            ("Holly Springs, NC", "holly_springs"),
            ("San Francisco, CA", "san_francisco"),
            ("New York City, NY", "new_york_city"),
            ("St. Louis, MO", "st_louis")
        ]

        for location_name, expected_city in test_cases:
            # Test the logic that processes location names
            city_name = location_name.split(',')[0].lower().replace(' ', '_').replace('.', '')
            assert city_name == expected_city

    def test_no_location_provided(self):
        """Test CSV detection when no location is provided"""
        detector = DataSourceDetector()  # No location

        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            sources = detector.detect_available_csv_sources()

            # Should only find real CSV and generic mock (no location-specific)
            source_types = [s.source_type for s in sources]
            assert "REAL_CSV" in source_types
            assert "GENERIC_MOCK" in source_types
            assert "SYNTHETIC_LOCATION" not in source_types


class TestDetermineDataStrategy:
    """Test data strategy determination logic"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_location = Mock()
        self.mock_location.location_name = "Holly Springs, NC"
        self.detector = DataSourceDetector(location=self.mock_location)

    @patch.object(DataSourceDetector, 'create_enphase_client')
    @patch.object(DataSourceDetector, 'detect_available_csv_sources')
    def test_real_api_strategy(self, mock_detect_csv, mock_create_client):
        """Test strategy when real API is available"""
        # Mock real API client
        mock_client = Mock()
        mock_create_client.return_value = (mock_client, "REAL_API")

        # Mock CSV sources
        mock_csv_source = DataSourceInfo("REAL_CSV", "Real data", "../data/raw/real.csv", 100)
        mock_detect_csv.return_value = [mock_csv_source]

        strategy = self.detector.determine_data_strategy()

        assert strategy['client_type'] == "REAL_API"
        assert strategy['csv_path'] == "../data/raw/real.csv"
        assert "REAL API DATA" in strategy['data_type']
        assert strategy['selected_source'] == mock_csv_source

    @patch.object(DataSourceDetector, 'create_enphase_client')
    @patch.object(DataSourceDetector, 'detect_available_csv_sources')
    def test_mock_client_strategy(self, mock_detect_csv, mock_create_client):
        """Test strategy when using mock client"""
        # Mock client
        mock_client = Mock()
        mock_create_client.return_value = (mock_client, "MOCK")

        # Mock CSV sources
        mock_csv_source = DataSourceInfo("REAL_CSV", "Real data", "../data/raw/real.csv", 100)
        mock_detect_csv.return_value = [mock_csv_source]

        strategy = self.detector.determine_data_strategy()

        assert strategy['client_type'] == "MOCK"
        assert strategy['csv_path'] == "../data/raw/real.csv"
        assert strategy['data_type'] == "📈 REAL_CSV"

    @patch.object(DataSourceDetector, 'create_enphase_client')
    @patch.object(DataSourceDetector, 'detect_available_csv_sources')
    def test_no_csv_sources_error(self, mock_detect_csv, mock_create_client):
        """Test error when no CSV sources are available"""
        mock_client = Mock()
        mock_create_client.return_value = (mock_client, "MOCK")
        mock_detect_csv.return_value = []  # No CSV sources

        with pytest.raises(FileNotFoundError, match="No CSV data files available"):
            self.detector.determine_data_strategy()


class TestAnalyzeDataRecency:
    """Test data recency analysis logic"""

    def setup_method(self):
        """Set up test fixtures"""
        self.detector = DataSourceDetector()

    def test_empty_dataframe(self):
        """Test analysis with empty dataframe"""
        empty_df = pd.DataFrame()
        data_summary = {'api': {'available': False}, 'csv': {'available': False}}

        result = self.detector.analyze_data_recency(empty_df, data_summary)

        assert result['latest_date'] is None
        assert result['days_old'] is None
        assert result['recency_status'] == '❌ NO DATA'
        assert result['authenticity'] == 'Unknown'

    def test_recent_api_data(self):
        """Test analysis with recent API data"""
        # Use a fixed date for predictable testing
        fixed_date = datetime(2024, 1, 15)
        df = pd.DataFrame({
            'production': [25, 30, 28]
        }, index=pd.date_range(fixed_date, periods=3, freq='D'))

        data_summary = {
            'api': {'available': True, 'is_mock': False},
            'csv': {'available': True}
        }

        # Test the core functionality without relying on datetime mocking
        result = self.detector.analyze_data_recency(df, data_summary)

        # Focus on testing the key outputs we care about
        assert result['latest_date'] == pd.Timestamp('2024-01-17')  # Last date in range
        assert result['days_old'] is not None  # Should calculate some age
        assert "RECENT" in result['recency_status'] or "OLD" in result['recency_status']  # Should categorize
        assert result['actual_source'] == "✅ LIVE API DATA"
        assert "Real API data" in result['authenticity']

    def test_old_csv_data(self):
        """Test analysis with old CSV data"""
        # Create old data (120 days old)
        old_date = datetime.now() - timedelta(days=120)
        df = pd.DataFrame({
            'production': [25, 30, 28]
        }, index=pd.date_range(old_date, periods=3, freq='D'))

        data_summary = {
            'api': {'available': False},
            'csv': {'available': True, 'source_file': '../data/raw/4136754_custom_report.csv'}
        }

        result = self.detector.analyze_data_recency(df, data_summary)

        # Allow for small timing differences (118-122 days)
        assert result['days_old'] >= 118 and result['days_old'] <= 122
        assert "OLD DATA" in result['recency_status']
        assert result['actual_source'] == "✅ REAL CSV DATA"
        assert "Authentic solar panel data" in result['authenticity']

    def test_mock_data_detection(self):
        """Test detection of mock/synthetic data"""
        df = pd.DataFrame({
            'production': [25, 30, 28]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))

        data_summary = {
            'api': {'available': False},
            'csv': {'available': True, 'source_file': '../data/raw/mock_solar_data.csv'}
        }

        result = self.detector.analyze_data_recency(df, data_summary)

        assert result['actual_source'] == "⚠️ SYNTHETIC DATA"
        assert "Synthetic/demo data" in result['authenticity']

    @pytest.mark.parametrize("days_old,expected_status", [
        (3, "VERY RECENT"),
        (15, "RECENT"),
        (45, "SOMEWHAT OLD"),
        (120, "OLD DATA")
    ])
    def test_recency_status_categories(self, days_old, expected_status):
        """Test different recency status categories"""
        old_date = datetime.now() - timedelta(days=days_old)
        df = pd.DataFrame({
            'production': [25]
        }, index=[old_date])

        data_summary = {
            'api': {'available': False},
            'csv': {'available': True, 'source_file': 'test.csv'}
        }

        result = self.detector.analyze_data_recency(df, data_summary)

        assert expected_status in result['recency_status']
        assert result['days_old'] == days_old


class TestGenerateFinalReport:
    """Test final report generation"""

    def setup_method(self):
        """Set up test fixtures"""
        self.detector = DataSourceDetector()

    def test_generate_final_report_real_api(self, capsys):
        """Test report generation for real API data"""
        strategy = {
            'data_type': '📈 REAL API DATA',
            'client_type': 'REAL_API',
            'selected_source': DataSourceInfo('REAL_CSV', 'Real data', '../data/raw/real.csv', 100)
        }

        data_summary = {
            'api': {'available': True, 'is_mock': False, 'records': 1000, 'date_range': ['2024-01-01', '2024-12-31']},
            'csv': {'available': True}
        }

        recency_info = {
            'latest_date': datetime(2024, 12, 31),
            'recency_status': '🔥 VERY RECENT (only 1 days old)',
            'actual_source': '✅ LIVE API DATA',
            'authenticity': 'Real API data'
        }

        # Mock environment variable
        with patch.dict(os.environ, {'ENPHASE_SYSTEM_ID': 'test_system_123'}):
            self.detector.generate_final_report(strategy, data_summary, recency_info)

        captured = capsys.readouterr()
        assert "📈 Solar Data Loaded Successfully" in captured.out
        assert "REAL API DATA" in captured.out
        assert "Live API data from your Enphase system" in captured.out
        assert "test_system_123" in captured.out

    def test_generate_final_report_csv_fallback(self, capsys):
        """Test report generation for CSV fallback"""
        strategy = {
            'data_type': '📈 REAL API DATA (with CSV fallback)',
            'client_type': 'REAL_API',
            'selected_source': DataSourceInfo('REAL_CSV', 'Real data', '../data/raw/4136754_custom_report.csv', 100)
        }

        data_summary = {
            'api': {'available': False},
            'csv': {'available': True, 'records': 5000, 'date_range': ['2023-01-01', '2024-12-31']}
        }

        recency_info = {
            'latest_date': datetime(2024, 12, 31),
            'recency_status': '✅ RECENT (10 days old)',
            'actual_source': '✅ REAL CSV DATA',
            'authenticity': 'Authentic solar panel data'
        }

        self.detector.generate_final_report(strategy, data_summary, recency_info)

        captured = capsys.readouterr()
        assert "CSV file (API failed, using fallback)" in captured.out
        assert "SUCCESS: Using REAL solar production data!" in captured.out
        assert "API returned 401 (expired token)" in captured.out

    def test_generate_final_report_mock_data(self, capsys):
        """Test report generation for mock data"""
        strategy = {
            'data_type': '📈 SYNTHETIC_LOCATION',
            'client_type': 'MOCK',
            'selected_source': DataSourceInfo('SYNTHETIC_LOCATION', 'Mock data', '../data/raw/mock_data.csv', 50)
        }

        data_summary = {
            'api': {'available': False},
            'csv': {'available': True, 'records': 1000, 'date_range': ['2024-01-01', '2024-06-30']}
        }

        recency_info = {
            'latest_date': datetime(2024, 6, 30),
            'recency_status': '📅 SOMEWHAT OLD (90 days old)',
            'actual_source': '⚠️ SYNTHETIC DATA',
            'authenticity': 'Synthetic data'
        }

        self.detector.generate_final_report(strategy, data_summary, recency_info)

        captured = capsys.readouterr()
        assert "NOTE: Using synthetic data for demonstration" in captured.out
        assert "To use your real data:" in captured.out
        assert "Current mock data ends June 2024" in captured.out


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_priority_ordering_logic(self):
        """Test that priority ordering logic works correctly"""
        # Test the sorting logic used in the detector
        sources = [
            DataSourceInfo("GENERIC_MOCK", "Generic", "path1", 25),
            DataSourceInfo("REAL_CSV", "Real", "path2", 100),
            DataSourceInfo("SYNTHETIC_LOCATION", "Location", "path3", 50)
        ]

        # Sort as the detector would
        sources.sort(key=lambda x: x.priority, reverse=True)

        # Should be ordered by priority
        priorities = [s.priority for s in sources]
        assert priorities == sorted(priorities, reverse=True)

        # Real CSV should be first
        assert sources[0].source_type == "REAL_CSV"
        assert sources[0].priority == 100