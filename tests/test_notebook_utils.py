"""
Tests for notebook utilities module
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append('src')

from core.notebook_utils import NotebookEnvironment, quick_setup, load_with_analysis, load_with_features, print_notebook_header


class TestNotebookEnvironment:
    """Test the NotebookEnvironment class"""

    def test_init(self):
        """Test NotebookEnvironment initialization"""
        env = NotebookEnvironment()
        assert env.location is None
        assert env.detector is None
        assert env.strategy is None
        assert env.data_manager is None

    @patch('core.notebook_utils.create_notebook_location')
    @patch('core.notebook_utils.DataSourceDetector')
    @patch('core.notebook_utils.SolarDataManager')
    def test_setup_complete_environment(self, mock_data_manager, mock_detector, mock_location):
        """Test complete environment setup"""
        # Setup mocks
        mock_location_instance = Mock()
        mock_location_instance.location_name = "Test Location"
        mock_location.return_value = mock_location_instance

        mock_detector_instance = Mock()
        mock_strategy = {
            'csv_path': '/test/path.csv',
            'client': Mock(),
            'data_type': 'TEST_DATA'
        }
        mock_detector_instance.determine_data_strategy.return_value = mock_strategy
        mock_detector.return_value = mock_detector_instance

        mock_data_manager_instance = Mock()
        mock_data_manager.return_value = mock_data_manager_instance

        # Test setup
        env = NotebookEnvironment()
        result = env.setup_complete_environment()

        # Verify calls
        mock_location.assert_called_once()
        mock_detector.assert_called_once_with(location=mock_location_instance)
        mock_detector_instance.determine_data_strategy.assert_called_once()
        mock_data_manager.assert_called_once_with(
            csv_path='/test/path.csv',
            enphase_client=mock_strategy['client'],
            cache_dir="../data/processed"
        )

        # Verify result
        assert result['location'] == mock_location_instance
        assert result['detector'] == mock_detector_instance
        assert result['strategy'] == mock_strategy
        assert result['data_manager'] == mock_data_manager_instance

    def test_load_and_analyze_data_without_setup(self):
        """Test load_and_analyze_data raises error without setup"""
        env = NotebookEnvironment()

        with pytest.raises(RuntimeError, match="Must call setup_complete_environment"):
            env.load_and_analyze_data()

    @patch('core.notebook_utils.create_notebook_location')
    @patch('core.notebook_utils.DataSourceDetector')
    @patch('core.notebook_utils.SolarDataManager')
    def test_load_and_analyze_data(self, mock_data_manager, mock_detector, mock_location):
        """Test data loading and analysis"""
        # Setup environment
        env = NotebookEnvironment()
        env.data_manager = Mock()

        # Setup mock data
        mock_csv_data = pd.DataFrame({'Production (kWh)': [1, 2, 3]})
        mock_daily_data = pd.DataFrame({'Production (kWh)': [10, 20, 30]})
        mock_data_summary = {'csv': {'available': True}}
        mock_recency_info = {'latest_date': pd.Timestamp('2024-01-01')}

        env.data_manager.load_csv_data.return_value = mock_csv_data
        env.data_manager.get_daily_production.return_value = mock_daily_data
        env.data_manager.get_data_summary.return_value = mock_data_summary

        env.detector = Mock()
        env.detector.analyze_data_recency.return_value = mock_recency_info
        env.strategy = {'data_type': 'TEST'}

        # Test data loading
        result = env.load_and_analyze_data()

        # Verify calls
        env.data_manager.load_csv_data.assert_called_once()
        env.data_manager.get_daily_production.assert_called_once()
        env.data_manager.get_data_summary.assert_called_once()
        env.detector.analyze_data_recency.assert_called_once_with(mock_csv_data, mock_data_summary)
        env.detector.generate_final_report.assert_called_once()

        # Verify result
        assert 'csv_data' in result
        assert 'daily_data' in result
        assert 'data_summary' in result
        assert 'recency_info' in result

    def test_get_location_context_without_location(self):
        """Test get_location_context raises error without location"""
        env = NotebookEnvironment()

        with pytest.raises(RuntimeError, match="Must call setup_complete_environment"):
            env.get_location_context()

    def test_get_location_context(self):
        """Test location context retrieval"""
        env = NotebookEnvironment()
        env.location = Mock()
        env.location.location_name = "Test City"
        env.location.latitude = 40.0
        env.location.longitude = -75.0
        env.location.timezone_str = "America/New_York"

        mock_summary = {
            'climate_type': 'Temperate',
            'seasonal_variation': 0.5
        }
        env.location.get_location_summary.return_value = mock_summary

        result = env.get_location_context()

        env.location.get_location_summary.assert_called_once()
        assert result == mock_summary


class TestConvenienceFunctions:
    """Test convenience functions"""

    @patch('core.notebook_utils.NotebookEnvironment')
    def test_quick_setup(self, mock_env_class):
        """Test quick_setup function"""
        # Setup mock
        mock_env = Mock()
        mock_components = {
            'data_manager': Mock(),
            'detector': Mock(),
            'location': Mock(),
            'strategy': Mock()
        }
        mock_env.setup_complete_environment.return_value = mock_components
        mock_env_class.return_value = mock_env

        # Test function
        result = quick_setup()

        # Verify
        mock_env_class.assert_called_once()
        mock_env.setup_complete_environment.assert_called_once()

        assert result[0] == mock_components['data_manager']
        assert result[1] == mock_components['detector']
        assert result[2] == mock_components['location']
        assert result[3] == mock_components['strategy']

    @patch('core.notebook_utils.NotebookEnvironment')
    def test_load_with_analysis(self, mock_env_class):
        """Test load_with_analysis function"""
        # Setup mock
        mock_env = Mock()
        mock_components = {'data_manager': Mock()}
        mock_data_results = {'csv_data': Mock()}
        mock_location_context = {'climate_type': 'Temperate'}

        mock_env.setup_complete_environment.return_value = mock_components
        mock_env.load_and_analyze_data.return_value = mock_data_results
        mock_env.get_location_context.return_value = mock_location_context
        mock_env_class.return_value = mock_env

        # Test function
        result = load_with_analysis()

        # Verify calls
        mock_env.setup_complete_environment.assert_called_once()
        mock_env.load_and_analyze_data.assert_called_once()
        mock_env.get_location_context.assert_called_once()

        # Verify result contains all components
        assert 'data_manager' in result
        assert 'csv_data' in result
        assert 'location_context' in result

    def test_print_notebook_header(self, capsys):
        """Test notebook header printing"""
        print_notebook_header("Test Title", "Test description")

        captured = capsys.readouterr()
        assert "📊 Test Title" in captured.out
        assert "=" in captured.out
        assert "Test description" in captured.out

    def test_print_notebook_header_no_description(self, capsys):
        """Test notebook header printing without description"""
        print_notebook_header("Test Title")

        captured = capsys.readouterr()
        assert "📊 Test Title" in captured.out
        assert "=" in captured.out


class TestLoadWithFeatures:
    """Test the new load_with_features function"""

    @patch('core.notebook_utils.load_with_analysis')
    @patch('features.feature_pipeline.FeaturePipeline')
    def test_load_with_features_basic(self, mock_pipeline_class, mock_load_analysis):
        """Test basic load_with_features functionality"""
        # Mock base data
        mock_base_data = {
            'location': Mock(),
            'daily_data': pd.DataFrame({
                'Production (kWh)': [30, 35, 25, 40],
                'Consumption (kWh)': [35, 30, 40, 28]
            }, index=pd.date_range('2023-01-01', periods=4)),
            'data_summary': {}
        }
        mock_load_analysis.return_value = mock_base_data

        # Mock feature pipeline
        mock_pipeline = Mock()
        mock_pipeline.validate_data_for_features.return_value = {'valid': True, 'errors': [], 'warnings': []}
        mock_pipeline.create_ml_dataset.return_value = mock_base_data['daily_data'].copy()
        mock_pipeline.get_feature_summary.return_value = {'total_features': 25, 'feature_counts': {}}
        mock_pipeline_class.return_value = mock_pipeline

        # Test function
        result = load_with_features(feature_sets=['temporal', 'financial'])

        # Verify pipeline initialization
        mock_pipeline_class.assert_called_once_with(location_manager=mock_base_data['location'])

        # Verify pipeline calls
        mock_pipeline.validate_data_for_features.assert_called_once()
        mock_pipeline.create_ml_dataset.assert_called_once()
        mock_pipeline.get_feature_summary.assert_called_once()

        # Verify result structure
        assert 'ml_features' in result
        assert 'feature_pipeline' in result
        assert 'feature_summary' in result

    @patch('core.notebook_utils.load_with_analysis')
    @patch('features.feature_pipeline.FeaturePipeline')
    def test_load_with_features_validation_failure(self, mock_pipeline_class, mock_load_analysis):
        """Test load_with_features with validation failure"""
        # Mock base data
        mock_base_data = {
            'location': Mock(),
            'daily_data': pd.DataFrame({'invalid': [1, 2, 3]}),  # Missing required columns
            'data_summary': {}
        }
        mock_load_analysis.return_value = mock_base_data

        # Mock pipeline with validation failure
        mock_pipeline = Mock()
        mock_pipeline.validate_data_for_features.return_value = {
            'valid': False,
            'errors': ['Missing required columns'],
            'warnings': []
        }
        mock_pipeline_class.return_value = mock_pipeline

        # Test function
        result = load_with_features()

        # Should still return data with fallback
        assert 'ml_features' in result
        assert 'feature_pipeline' in result

    @patch('core.notebook_utils.load_with_analysis')
    @patch('features.feature_pipeline.FeaturePipeline')
    def test_load_with_features_error_handling(self, mock_pipeline_class, mock_load_analysis):
        """Test load_with_features error handling"""
        # Mock base data
        mock_base_data = {
            'location': Mock(),
            'daily_data': pd.DataFrame({
                'Production (kWh)': [30, 35, 25, 40]
            }, index=pd.date_range('2023-01-01', periods=4)),
        }
        mock_load_analysis.return_value = mock_base_data

        # Mock pipeline that raises exception
        mock_pipeline_class.side_effect = Exception("Pipeline error")

        # Test function - should not raise exception
        result = load_with_features()

        # Should gracefully fall back to base data
        assert 'ml_features' in result
        assert len(result['ml_features']) == len(mock_base_data['daily_data'])

    def test_load_with_features_default_parameters(self):
        """Test load_with_features with default parameters"""
        # This test will use real components but with mocked data
        with patch('core.notebook_utils.load_with_analysis') as mock_load_analysis:
            mock_base_data = {
                'location': None,  # No location
                'daily_data': pd.DataFrame({
                    'Production (kWh)': [30, 35, 25, 40]
                }, index=pd.date_range('2023-01-01', periods=4)),
                'data_summary': {}
            }
            mock_load_analysis.return_value = mock_base_data

            # Should not raise exception even with minimal data
            result = load_with_features()

            assert 'ml_features' in result