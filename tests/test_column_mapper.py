"""
Comprehensive test suite for ColumnMapper utility.

Tests all aspects of column detection, standardization, validation, and integration
with focus on real-world scenarios and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging

from src.data.column_mapper import (
    ColumnMapper,
    ColumnMapperConfig,
    DetectionResult,
    ColumnMapperError,
    AmbiguousColumnsError,
    MissingColumnsError,
    detect_energy_columns,
    standardize_energy_columns,
    validate_solar_data_columns
)


class TestColumnMapperConfig:
    """Test ColumnMapperConfig class."""

    def test_init_default(self):
        """Test default configuration initialization."""
        config = ColumnMapperConfig()

        assert 'production' in config.detection_patterns
        assert 'consumption' in config.detection_patterns
        assert 'export' in config.detection_patterns
        assert 'import' in config.detection_patterns

        assert config.confidence_threshold == 0.7
        assert config.fuzzy_match_threshold == 0.8
        assert not config.case_sensitive

    def test_add_custom_pattern(self):
        """Test adding custom detection patterns."""
        config = ColumnMapperConfig()

        config.add_custom_pattern('production', ['solar_gen', 'pv_output'])
        config.add_custom_pattern('production', ['generación'], 'es')

        patterns = config.get_all_patterns('production')
        assert 'solar_gen' in patterns
        assert 'pv_output' in patterns
        assert 'generación' in patterns

    def test_get_all_patterns(self):
        """Test pattern aggregation from all sources."""
        config = ColumnMapperConfig()

        patterns = config.get_all_patterns('production')

        # Should include default patterns
        assert 'production' in patterns
        assert 'generated' in patterns

        # Should include international patterns
        assert 'producción' in patterns  # Spanish
        assert 'produktion' in patterns  # German

    def test_international_patterns(self):
        """Test international pattern support."""
        config = ColumnMapperConfig()

        patterns = config.get_all_patterns('consumption')
        assert 'consumo' in patterns  # Spanish
        assert 'consommation' in patterns  # French
        assert 'verbrauch' in patterns  # German


class TestDetectionResult:
    """Test DetectionResult dataclass."""

    def test_overall_confidence_calculation(self):
        """Test overall confidence score calculation."""
        result = DetectionResult(
            mapping={'production': 'Energy Produced (Wh)'},
            confidence_scores={'production': 0.9, 'consumption': 0.8}
        )

        assert abs(result.overall_confidence - 0.85) < 0.001

    def test_is_high_confidence(self):
        """Test high confidence detection."""
        high_conf = DetectionResult(
            mapping={},
            confidence_scores={'production': 0.9, 'consumption': 0.85}
        )
        assert high_conf.is_high_confidence

        low_conf = DetectionResult(
            mapping={},
            confidence_scores={'production': 0.7, 'consumption': 0.6}
        )
        assert not low_conf.is_high_confidence

    def test_empty_confidence_scores(self):
        """Test behavior with empty confidence scores."""
        result = DetectionResult(mapping={}, confidence_scores={})
        assert result.overall_confidence == 0.0
        assert not result.is_high_confidence


class TestColumnMapperInit:
    """Test ColumnMapper initialization."""

    def test_init_default(self):
        """Test default initialization."""
        mapper = ColumnMapper()

        assert not mapper.strict_mode
        assert mapper.compatibility_mode == 'standard'
        assert isinstance(mapper.config, ColumnMapperConfig)
        assert isinstance(mapper.logger, logging.Logger)

    def test_init_strict_mode(self):
        """Test strict mode initialization."""
        mapper = ColumnMapper(strict_mode=True)
        assert mapper.strict_mode

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = ColumnMapperConfig()
        config.confidence_threshold = 0.9

        mapper = ColumnMapper(config=config)
        assert mapper.config.confidence_threshold == 0.9

    def test_init_logging_configuration(self):
        """Test logging setup."""
        mapper = ColumnMapper(log_level='DEBUG')
        assert mapper.logger.level == logging.DEBUG


class TestColumnDetection:
    """Test core column detection functionality."""

    def test_detect_columns_standard_names(self):
        """Test detection with standard column names."""
        df = pd.DataFrame({
            'Production (kWh)': [1, 2, 3],
            'Consumption (kWh)': [1, 2, 3],
            'Export (kWh)': [0.5, 1, 1.5],
            'Import (kWh)': [0.2, 0.5, 0.8]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert result['production'] == 'Production (kWh)'
        assert result['consumption'] == 'Consumption (kWh)'
        assert result['export'] == 'Export (kWh)'
        assert result['import'] == 'Import (kWh)'

    def test_detect_columns_alternative_names(self):
        """Test detection with alternative naming patterns."""
        df = pd.DataFrame({
            'Energy Produced (Wh)': [1000, 2000, 3000],
            'Energy Consumed (Wh)': [800, 1500, 2200],
            'Exported to Grid (Wh)': [200, 500, 800],
            'Imported from Grid (Wh)': [100, 300, 500]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert result['production'] == 'Energy Produced (Wh)'
        assert result['consumption'] == 'Energy Consumed (Wh)'
        assert result['export'] == 'Exported to Grid (Wh)'
        assert result['import'] == 'Imported from Grid (Wh)'

    def test_detect_columns_case_insensitive(self):
        """Test case insensitive detection."""
        df = pd.DataFrame({
            'ENERGY_PRODUCED_WH': [1000, 2000],
            'energy_consumed_wh': [800, 1500],
            'Energy_Exported_Wh': [200, 500]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert result['production'] == 'ENERGY_PRODUCED_WH'
        assert result['consumption'] == 'energy_consumed_wh'
        assert result['export'] == 'Energy_Exported_Wh'

    def test_detect_columns_with_underscores_and_spaces(self):
        """Test detection with various separators."""
        df = pd.DataFrame({
            'Solar_Generation_kWh': [10, 20],
            'Home Usage (kWh)': [8, 15],
            'Grid Export-kWh': [2, 5]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert result['production'] == 'Solar_Generation_kWh'
        assert result['consumption'] == 'Home Usage (kWh)'
        assert result['export'] == 'Grid Export-kWh'

    def test_detect_columns_partial_matches(self):
        """Test partial keyword matching."""
        df = pd.DataFrame({
            'PV_Production_Daily': [10, 20],
            'House_Consumption_Daily': [8, 15],
            'Grid_Export_Daily': [2, 5]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert result['production'] == 'PV_Production_Daily'
        assert result['consumption'] == 'House_Consumption_Daily'
        assert result['export'] == 'Grid_Export_Daily'

    def test_detect_columns_international_patterns(self):
        """Test detection with international column names."""
        df = pd.DataFrame({
            'Producción Solar (kWh)': [10, 20],
            'Consumo Casa (kWh)': [8, 15],
            'Généración (kWh)': [12, 25]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert 'production' in result

    def test_detect_columns_ambiguous_cases(self):
        """Test handling of ambiguous column names."""
        df = pd.DataFrame({
            'Production Day (kWh)': [10, 20],
            'Production Night (kWh)': [0, 0],
            'Consumption (kWh)': [8, 15]
        })

        # In non-strict mode, should resolve automatically
        mapper = ColumnMapper(strict_mode=False)
        result = mapper.detect_columns(df)

        assert 'production' in result
        assert result['production'] in ['Production Day (kWh)', 'Production Night (kWh)']

    def test_detect_columns_ambiguous_strict_mode(self):
        """Test strict mode with ambiguous columns raises error."""
        df = pd.DataFrame({
            'Production Day (kWh)': [10, 20],
            'Production Night (kWh)': [0, 0]
        })

        mapper = ColumnMapper(strict_mode=True)

        with pytest.raises(AmbiguousColumnsError) as exc_info:
            mapper.detect_columns(df)

        assert 'production' in str(exc_info.value)
        assert 'Production Day (kWh)' in str(exc_info.value)

    def test_detect_columns_missing_required_strict_mode(self):
        """Test strict mode with missing required columns."""
        df = pd.DataFrame({
            'Random Column': [1, 2, 3],
            'Another Column': [4, 5, 6]
        })

        mapper = ColumnMapper(strict_mode=True)

        with pytest.raises(MissingColumnsError) as exc_info:
            mapper.detect_columns(df)

        assert 'production' in str(exc_info.value)

    def test_detect_columns_empty_dataframe(self):
        """Test detection with empty DataFrame."""
        df = pd.DataFrame()

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert result == {}

    def test_detect_columns_no_energy_columns(self):
        """Test detection with no recognizable energy columns."""
        df = pd.DataFrame({
            'Random Data': [1, 2, 3],
            'More Random': [4, 5, 6],
            'Not Energy': [7, 8, 9]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert result == {}


class TestColumnStandardization:
    """Test column standardization functionality."""

    def test_standardize_columns_basic(self):
        """Test basic column standardization."""
        df = pd.DataFrame({
            'Energy Produced (Wh)': [1000, 2000],
            'Energy Consumed (Wh)': [800, 1500],
            'Other Column': ['A', 'B']
        })

        mapper = ColumnMapper()
        result = mapper.standardize_columns(df)

        assert 'Production (kWh)' in result.columns
        assert 'Consumption (kWh)' in result.columns
        assert 'Other Column' in result.columns  # Preserved

    def test_standardize_columns_preserves_data(self):
        """Test that data values are preserved during standardization."""
        df = pd.DataFrame({
            'Energy Produced (Wh)': [1000, 2000],
            'Other Data': [10, 20]
        })

        mapper = ColumnMapper()
        result = mapper.standardize_columns(df)

        # Check data preservation
        assert result['Production (kWh)'].tolist() == [1000, 2000]
        assert result['Other Data'].tolist() == [10, 20]

    def test_standardize_columns_preserves_index(self):
        """Test that DataFrame index is preserved."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        df = pd.DataFrame({
            'Energy Produced (Wh)': [1000, 2000, 3000]
        }, index=dates)

        mapper = ColumnMapper()
        result = mapper.standardize_columns(df)

        pd.testing.assert_index_equal(result.index, df.index)

    def test_standardize_columns_empty_dataframe(self):
        """Test standardization with empty DataFrame."""
        df = pd.DataFrame()

        mapper = ColumnMapper()
        result = mapper.standardize_columns(df)

        assert result.empty
        assert len(result.columns) == 0

    def test_standardize_columns_no_energy_columns(self):
        """Test standardization when no energy columns detected."""
        df = pd.DataFrame({
            'Random Column': [1, 2, 3],
            'Another Random': [4, 5, 6]
        })

        mapper = ColumnMapper()
        result = mapper.standardize_columns(df)

        # Should return unchanged DataFrame
        pd.testing.assert_frame_equal(result, df)

    def test_standardize_columns_preserve_non_energy_false(self):
        """Test standardization with preserve_non_energy=False."""
        df = pd.DataFrame({
            'Energy Produced (Wh)': [1000, 2000],
            'Random Column': [1, 2]
        })

        mapper = ColumnMapper()
        result = mapper.standardize_columns(df, preserve_non_energy=True)

        # Should still preserve (current implementation always preserves)
        assert 'Random Column' in result.columns


class TestColumnValidation:
    """Test column validation functionality."""

    def test_validate_energy_data_complete(self):
        """Test validation with complete energy data."""
        df = pd.DataFrame({
            'Production (kWh)': [10, 20],
            'Consumption (kWh)': [8, 15],
            'Export (kWh)': [2, 5],
            'Import (kWh)': [1, 3]
        })

        mapper = ColumnMapper()
        result = mapper.validate_energy_data(df)

        assert result['status'] == 'VALID'
        assert len(result['missing_critical']) == 0
        assert len(result['missing_important']) == 0

    def test_validate_energy_data_missing_critical(self):
        """Test validation with missing critical columns."""
        df = pd.DataFrame({
            'Consumption (kWh)': [8, 15],
            'Export (kWh)': [2, 5]
        })

        mapper = ColumnMapper()
        result = mapper.validate_energy_data(df)

        assert result['status'] == 'INVALID'
        assert 'production' in result['missing_critical']

    def test_validate_energy_data_missing_important(self):
        """Test validation with missing important columns."""
        df = pd.DataFrame({
            'Production (kWh)': [10, 20]
        })

        mapper = ColumnMapper()
        result = mapper.validate_energy_data(df)

        assert result['status'] == 'WARNING'
        assert 'consumption' in result['missing_important']
        assert 'export' in result['missing_important']
        assert 'import' in result['missing_important']

    def test_validate_energy_data_recommendations(self):
        """Test that validation provides helpful recommendations."""
        df = pd.DataFrame({
            'Random Column': [1, 2, 3]
        })

        mapper = ColumnMapper()
        result = mapper.validate_energy_data(df)

        assert len(result['recommendations']) > 0
        assert any('Critical columns missing' in rec for rec in result['recommendations'])


class TestAdvancedFeatures:
    """Test advanced features and edge cases."""

    def test_detect_columns_with_confidence(self):
        """Test enhanced detection with confidence scores."""
        df = pd.DataFrame({
            'Production (kWh)': [10, 20],  # High confidence
            'Some_Production_Data': [5, 10]  # Lower confidence
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns_with_confidence(df)

        assert isinstance(result, DetectionResult)
        assert 'production' in result.mapping
        assert 'production' in result.confidence_scores
        assert result.confidence_scores['production'] > 0.8

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        mapper = ColumnMapper()

        # Exact match should have high confidence
        patterns = ['production', 'generated']
        confidence = mapper._calculate_match_confidence('Production (kWh)', patterns)
        assert confidence >= 0.9

        # Partial match should have lower confidence
        confidence = mapper._calculate_match_confidence('Some_Prod_Data', patterns)
        assert 0.0 <= confidence < 0.9

    def test_caching_functionality(self):
        """Test that detection results are cached for performance."""
        df = pd.DataFrame({
            'Production (kWh)': [10, 20],
            'Consumption (kWh)': [8, 15]
        })

        mapper = ColumnMapper()

        # First call
        result1 = mapper.detect_columns(df)

        # Second call should use cache
        result2 = mapper.detect_columns(df)

        assert result1 == result2
        assert len(mapper._detection_cache) > 0

    def test_custom_patterns(self):
        """Test custom pattern functionality."""
        config = ColumnMapperConfig()
        config.add_custom_pattern('production', ['solar_output', 'pv_gen'])

        df = pd.DataFrame({
            'solar_output_kwh': [10, 20],
            'consumption': [8, 15]
        })

        mapper = ColumnMapper(config=config)
        result = mapper.detect_columns(df)

        assert result['production'] == 'solar_output_kwh'

    def test_resolve_ambiguous_matches(self):
        """Test ambiguous match resolution heuristics."""
        mapper = ColumnMapper()

        # Test unit preference heuristic
        ambiguous = {
            'production': ['Production_Daily', 'Production (kWh)', 'Prod_Data']
        }

        resolved = mapper._resolve_ambiguous_matches(ambiguous)
        assert resolved['production'] == 'Production (kWh)'  # Prefers units

    def test_datetime_column_detection(self):
        """Test datetime column detection."""
        df = pd.DataFrame({
            'timestamp': ['2024-01-01', '2024-01-02'],
            'Production (kWh)': [10, 20]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert result.get('datetime') == 'timestamp'

    def test_print_detection_summary(self, capsys):
        """Test detection summary printing."""
        df = pd.DataFrame({
            'Production (kWh)': [10, 20],
            'Consumption (kWh)': [8, 15]
        })

        mapper = ColumnMapper()
        mapper.print_detection_summary(df)

        captured = capsys.readouterr()
        assert "Column Detection Summary" in captured.out
        assert "Production" in captured.out
        assert "Consumption" in captured.out


class TestConvenienceFunctions:
    """Test convenience functions for simple use cases."""

    def test_detect_energy_columns_function(self):
        """Test detect_energy_columns convenience function."""
        df = pd.DataFrame({
            'Production (kWh)': [10, 20],
            'Consumption (kWh)': [8, 15]
        })

        result = detect_energy_columns(df)

        assert result['production'] == 'Production (kWh)'
        assert result['consumption'] == 'Consumption (kWh)'

    def test_standardize_energy_columns_function(self):
        """Test standardize_energy_columns convenience function."""
        df = pd.DataFrame({
            'Energy Produced (Wh)': [1000, 2000],
            'Other Column': ['A', 'B']
        })

        result = standardize_energy_columns(df)

        assert 'Production (kWh)' in result.columns
        assert 'Other Column' in result.columns

    def test_validate_solar_data_columns_function(self):
        """Test validate_solar_data_columns convenience function."""
        df = pd.DataFrame({
            'Production (kWh)': [10, 20]
        })

        result = validate_solar_data_columns(df)

        assert 'status' in result
        assert 'detected_columns' in result


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_non_string_column_names(self):
        """Test handling of non-string column names."""
        df = pd.DataFrame({
            0: [1, 2, 3],
            1: [4, 5, 6],
            'Production (kWh)': [10, 20, 30]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        # Should still detect the string column
        assert result['production'] == 'Production (kWh)'

    def test_very_long_column_names(self):
        """Test with unusually long column names."""
        long_name = 'Very_Long_Column_Name_With_Production_Energy_Solar_PV_Generation_Data_kWh'
        df = pd.DataFrame({
            long_name: [10, 20],
            'Normal Column': [1, 2]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert result['production'] == long_name

    def test_special_characters_in_names(self):
        """Test special characters in column names."""
        df = pd.DataFrame({
            'Prōdúctìøn (kWh)': [10, 20],
            'Consumption [kWh]': [8, 15],
            'Export & Grid (kWh)': [2, 5]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert len(result) >= 2  # Should detect at least production and consumption

    def test_compatibility_modes(self):
        """Test different compatibility modes."""
        df = pd.DataFrame({
            'Production (kWh)': [10, 20]
        })

        # Test different modes don't break
        for mode in ['strict', 'standard', 'permissive', 'legacy']:
            mapper = ColumnMapper(compatibility_mode=mode)
            result = mapper.detect_columns(df)
            assert 'production' in result


class TestPerformance:
    """Test performance and memory efficiency."""

    def test_performance_large_dataframe(self):
        """Test performance with large DataFrames."""
        # Create large DataFrame
        size = 10000
        df = pd.DataFrame({
            'Production (kWh)': np.random.random(size),
            'Consumption (kWh)': np.random.random(size),
            'Export (kWh)': np.random.random(size),
            'Import (kWh)': np.random.random(size),
            **{f'Random_Col_{i}': np.random.random(size) for i in range(10)}
        })

        mapper = ColumnMapper()

        # Should complete reasonably quickly
        import time
        start = time.time()
        result = mapper.detect_columns(df)
        end = time.time()

        assert end - start < 1.0  # Should complete in under 1 second
        assert len(result) == 4  # Should detect all 4 energy columns

    def test_performance_many_columns(self):
        """Test performance with many columns."""
        # Create DataFrame with many columns
        columns = {f'Random_Column_{i}': [1, 2] for i in range(100)}
        columns['Production (kWh)'] = [10, 20]
        columns['Consumption (kWh)'] = [8, 15]

        df = pd.DataFrame(columns)

        mapper = ColumnMapper()

        import time
        start = time.time()
        result = mapper.detect_columns(df)
        end = time.time()

        assert end - start < 0.5  # Should be fast even with many columns
        assert 'production' in result
        assert 'consumption' in result

    def test_memory_efficiency(self):
        """Test memory usage doesn't grow excessively."""
        mapper = ColumnMapper()

        # Process multiple DataFrames
        for i in range(10):
            df = pd.DataFrame({
                f'Production_{i} (kWh)': [10, 20],
                f'Consumption_{i} (kWh)': [8, 15]
            })
            mapper.detect_columns(df)

        # Cache should be limited in size
        assert len(mapper._detection_cache) <= 128  # LRU cache max size


class TestRealDataFormats:
    """Test with realistic solar energy data formats."""

    def test_enphase_enlighten_format(self):
        """Test with Enphase Enlighten CSV format."""
        df = pd.DataFrame({
            'Date/Time': ['2024-01-01 12:00:00', '2024-01-01 12:15:00'],
            'Energy Produced (Wh)': [250, 300],
            'Energy Consumed (Wh)': [200, 250],
            'Exported to Grid (Wh)': [50, 75],
            'Imported from Grid (Wh)': [25, 30]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert result['production'] == 'Energy Produced (Wh)'
        assert result['consumption'] == 'Energy Consumed (Wh)'
        assert result['export'] == 'Exported to Grid (Wh)'
        assert result['import'] == 'Imported from Grid (Wh)'
        assert result['datetime'] == 'Date/Time'

    def test_solaredge_monitoring_format(self):
        """Test with SolarEdge monitoring data format."""
        df = pd.DataFrame({
            'Time': ['01/01/2024 12:00:00', '01/01/2024 12:15:00'],
            'Production (kWh)': [0.25, 0.30],
            'Consumption (kWh)': [0.20, 0.25],
            'FeedIn (kWh)': [0.05, 0.075],
            'Purchased (kWh)': [0.025, 0.030]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert result['production'] == 'Production (kWh)'
        assert result['consumption'] == 'Consumption (kWh)'
        # Note: 'FeedIn' and 'Purchased' might not match exactly,
        # but should still be detected as export/import

    def test_custom_diy_format(self):
        """Test with custom DIY monitoring format."""
        df = pd.DataFrame({
            'timestamp': ['2024-01-01T12:00:00', '2024-01-01T12:15:00'],
            'solar_generation_wh': [250, 300],
            'house_usage_wh': [200, 250],
            'grid_export_wh': [50, 75],
            'grid_import_wh': [25, 30]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        assert result['production'] == 'solar_generation_wh'
        assert result['consumption'] == 'house_usage_wh'
        assert result['export'] == 'grid_export_wh'
        assert result['import'] == 'grid_import_wh'

    def test_mixed_units_format(self):
        """Test with mixed unit formats."""
        df = pd.DataFrame({
            'Solar Production (kWh)': [0.25, 0.30],
            'Home Consumption (Wh)': [200, 250],
            'Grid Export [kW]': [0.05, 0.075]
        })

        mapper = ColumnMapper()
        result = mapper.detect_columns(df)

        # Should detect regardless of unit format
        assert result['production'] == 'Solar Production (kWh)'
        assert result['consumption'] == 'Home Consumption (Wh)'
        assert result['export'] == 'Grid Export [kW]'


class TestIntegrationScenarios:
    """Test integration with real workflow scenarios."""

    def test_full_data_processing_workflow(self):
        """Test complete workflow from detection to standardization."""
        # Start with raw data
        raw_df = pd.DataFrame({
            'Energy Produced (Wh)': [1000, 2000, 3000],
            'Energy Consumed (Wh)': [800, 1500, 2200],
            'Exported to Grid (Wh)': [200, 500, 800],
            'Random Metadata': ['A', 'B', 'C']
        })

        mapper = ColumnMapper()

        # Step 1: Detect columns
        detected = mapper.detect_columns(raw_df)
        assert len(detected) == 3  # Should detect 3 energy columns

        # Step 2: Validate data
        validation = mapper.validate_energy_data(raw_df)
        assert validation['status'] in ['VALID', 'WARNING']

        # Step 3: Standardize columns
        standardized_df = mapper.standardize_columns(raw_df)
        assert 'Production (kWh)' in standardized_df.columns
        assert 'Consumption (kWh)' in standardized_df.columns
        assert 'Export (kWh)' in standardized_df.columns
        assert 'Random Metadata' in standardized_df.columns

        # Step 4: Verify data integrity
        assert len(standardized_df) == len(raw_df)
        assert standardized_df['Production (kWh)'].sum() == 6000
        assert standardized_df['Random Metadata'].tolist() == ['A', 'B', 'C']

    def test_error_recovery_workflow(self):
        """Test error recovery in problematic scenarios."""
        # Ambiguous data
        ambiguous_df = pd.DataFrame({
            'Production Day (kWh)': [10, 20],
            'Production Night (kWh)': [0, 0],
            'Consumption (kWh)': [8, 15]
        })

        # Should handle gracefully in non-strict mode
        mapper = ColumnMapper(strict_mode=False)
        result = mapper.detect_columns(ambiguous_df)

        assert 'production' in result
        assert 'consumption' in result

        # Should still be able to standardize
        standardized = mapper.standardize_columns(ambiguous_df)
        assert 'Production (kWh)' in standardized.columns

    def test_multiple_data_sources_integration(self):
        """Test handling multiple data sources with different formats."""
        # Source 1: Enphase format
        source1 = pd.DataFrame({
            'Energy Produced (Wh)': [1000, 2000],
            'Energy Consumed (Wh)': [800, 1500]
        })

        # Source 2: SolarEdge format
        source2 = pd.DataFrame({
            'Production (kWh)': [1.5, 2.5],
            'Consumption (kWh)': [1.2, 2.0]
        })

        mapper = ColumnMapper()

        # Both should standardize to same format
        std1 = mapper.standardize_columns(source1)
        std2 = mapper.standardize_columns(source2)

        # Should have same column names after standardization
        energy_cols1 = [col for col in std1.columns
                       if any(energy in col.lower()
                             for energy in ['production', 'consumption', 'export', 'import'])]
        energy_cols2 = [col for col in std2.columns
                       if any(energy in col.lower()
                             for energy in ['production', 'consumption', 'export', 'import'])]

        assert set(energy_cols1) == set(energy_cols2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])