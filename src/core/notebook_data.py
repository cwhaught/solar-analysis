"""
Notebook Data Utilities - High-level convenience functions

Provides ultra-simple data loading and setup functions for notebooks.
Eliminates boilerplate code and provides consistent interfaces across
all solar energy analysis notebooks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

try:
    from ..data.loaders import QuickDataLoader, StandardizedCSVLoader
    from ..data.processors import SolarDataProcessor
    from ..data.quality import DataQualityChecker
    from ..data.column_mapper import ColumnMapper, detect_energy_columns, standardize_energy_columns
except ImportError:
    # For notebook environment
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.loaders import QuickDataLoader, StandardizedCSVLoader
    from data.processors import SolarDataProcessor
    from data.quality import DataQualityChecker
    from data.column_mapper import ColumnMapper, detect_energy_columns, standardize_energy_columns
from .data_manager import SolarDataManager
from .data_source_detector import DataSourceDetector
from .location_loader import create_notebook_location

logger = logging.getLogger(__name__)


def quick_load_solar_data(
    source: str = 'auto',
    include_quality_report: bool = False,
    include_daily: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], Dict[str, Any]]:
    """
    Ultra-simple one-line data loading for notebooks.

    Replaces the common notebook pattern:
    ```python
    df = pd.read_csv('../data/raw/4136754_custom_report.csv')
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df.set_index('Date/Time', inplace=True)
    df_kwh = df / 1000
    daily_data = df_kwh.resample('D').sum()
    ```

    With simple:
    ```python
    fifteen_min_data, daily_data = quick_load_solar_data()
    ```

    Args:
        source: Data source ('auto', file path, or 'smart' for intelligent detection)
        include_quality_report: Include data quality analysis
        include_daily: Return daily aggregated data as well

    Returns:
        DataFrame, tuple of DataFrames, or dict with data and quality info
    """
    logger.info("Quick loading solar data")

    try:
        if source == 'smart':
            # Use intelligent data source detection
            return _smart_load_solar_data(include_quality_report, include_daily)
        else:
            # Use simple direct loading
            result = QuickDataLoader.load_solar_data(source, include_daily)

            if include_quality_report:
                # Add quality analysis
                data_for_quality = result[0] if include_daily else result
                quality_checker = DataQualityChecker()
                quality_report = quality_checker.generate_quality_report(data_for_quality)

                return {
                    'data': result,
                    'quality_report': quality_report,
                    'load_method': 'direct'
                }

            return result

    except Exception as e:
        logger.error(f"Error in quick_load_solar_data: {e}")
        raise


def load_with_quality_report(source: str = 'auto') -> Dict[str, Any]:
    """
    Load data with comprehensive quality analysis and reporting.

    Provides detailed quality metrics commonly needed in notebook analysis.

    Args:
        source: Data source identifier

    Returns:
        Dictionary with data, quality metrics, and formatted report
    """
    logger.info("Loading data with quality analysis")

    # Load data
    fifteen_min_data, daily_data = quick_load_solar_data(source, include_daily=True)

    # Comprehensive quality analysis
    quality_checker = DataQualityChecker()

    analysis = {
        'fifteen_min_data': fifteen_min_data,
        'daily_data': daily_data,
        'quality_analysis': {
            'completeness': quality_checker.check_completeness(fifteen_min_data),
            'missing_values': quality_checker.check_missing_values(fifteen_min_data),
            'integrity': quality_checker.check_data_integrity(fifteen_min_data)
        },
        'quality_report': quality_checker.generate_quality_report(fifteen_min_data),
        'daily_quality_report': quality_checker.generate_quality_report(daily_data)
    }

    return analysis


def setup_standard_datasets(
    source: str = 'auto',
    include_ml_features: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Setup standard datasets used across multiple notebooks.

    Creates commonly needed dataset variations:
    - 15-minute granular data
    - Daily aggregated data
    - Monthly summaries
    - Optionally: ML-ready dataset with features

    Args:
        source: Data source identifier
        include_ml_features: Include ML feature engineering

    Returns:
        Dictionary with standard dataset variations
    """
    logger.info("Setting up standard datasets")

    # Load base data
    fifteen_min_data, daily_data = quick_load_solar_data(source, include_daily=True)

    # Create processor for additional datasets
    processor = SolarDataProcessor()

    datasets = {
        'fifteen_min': fifteen_min_data,
        'daily': daily_data,
        'monthly': processor.create_monthly_summary(daily_data),
        'daily_with_metrics': processor.calculate_net_energy(
            processor.calculate_self_consumption_metrics(daily_data)
        )
    }

    # Add ML dataset if requested
    if include_ml_features:
        try:
            datasets['ml_ready'] = processor.prepare_ml_dataset(daily_data)
            logger.info(f"Created ML dataset with {datasets['ml_ready'].shape[1]} features")
        except Exception as e:
            logger.warning(f"Could not create ML dataset: {e}")
            datasets['ml_ready'] = daily_data

    return datasets


def _smart_load_solar_data(
    include_quality_report: bool = False,
    include_daily: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame], Dict[str, Any]]:
    """
    Intelligent data loading using DataSourceDetector and SolarDataManager.

    Replicates the smart detection pattern from modern notebooks.
    """
    try:
        # Setup intelligent data detection
        location = create_notebook_location()
        detector = DataSourceDetector(location=location)
        strategy = detector.determine_data_strategy()

        # Initialize enhanced data manager
        data_manager = SolarDataManager(
            csv_path=strategy['csv_path'],
            enphase_client=strategy['client'],
            cache_dir="../data/processed"
        )

        # Load data using enhanced manager
        fifteen_min_data = data_manager.load_csv_data(validate_quality=True)
        daily_data = data_manager.get_daily_production() if include_daily else None

        # Prepare result
        if include_daily:
            result = (fifteen_min_data, daily_data)
        else:
            result = fifteen_min_data

        if include_quality_report:
            quality_report = data_manager.get_data_quality_report('csv')
            enhanced_summary = data_manager.get_enhanced_data_summary()

            return {
                'data': result,
                'quality_report': quality_report,
                'data_summary': enhanced_summary,
                'load_method': 'smart',
                'strategy': strategy
            }

        return result

    except Exception as e:
        logger.warning(f"Smart loading failed, falling back to simple loading: {e}")
        # Fallback to simple loading
        return QuickDataLoader.load_solar_data('auto', include_daily)


# Legacy compatibility functions

def load_with_analysis() -> Dict[str, Any]:
    """
    Legacy compatibility function for existing notebook_utils.load_with_analysis().

    Maintains backward compatibility while using enhanced utilities internally.

    Returns:
        Dictionary containing all components and loaded data (legacy format)
    """
    logger.info("Loading with analysis (legacy compatibility mode)")

    try:
        # Use smart loading
        smart_result = _smart_load_solar_data(include_quality_report=True, include_daily=True)

        if isinstance(smart_result, dict):
            # Extract data from smart result
            fifteen_min_data, daily_data = smart_result['data']
            data_summary = smart_result['data_summary']
            strategy = smart_result['strategy']

            # Setup location
            location = create_notebook_location()

            # Create legacy-compatible result
            return {
                'location': location,
                'detector': DataSourceDetector(location=location),
                'strategy': strategy,
                'data_manager': SolarDataManager(
                    csv_path=strategy['csv_path'],
                    enphase_client=strategy['client']
                ),
                'csv_data': fifteen_min_data,
                'daily_data': daily_data,
                'data_summary': data_summary,
                'recency_info': {'latest_date': fifteen_min_data.index.max()},
                'location_context': location.get_location_summary()
            }

    except Exception as e:
        logger.error(f"Enhanced load_with_analysis failed: {e}")
        # Import and use original function as fallback
        from .notebook_utils import load_with_analysis as original_load_with_analysis
        return original_load_with_analysis()


# Specialized convenience functions

def quick_ml_setup(
    source: str = 'auto',
    target_column: str = 'Production (kWh)',
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Quick ML dataset setup for model training notebooks.

    Consolidates ML setup patterns found across model training notebooks.

    Args:
        source: Data source identifier
        target_column: Target variable for ML
        test_size: Fraction of data for testing

    Returns:
        Dictionary with train/test splits and feature information
    """
    from sklearn.model_selection import train_test_split

    # Load and prepare ML dataset
    datasets = setup_standard_datasets(source, include_ml_features=True)
    ml_data = datasets['ml_ready']

    # Prepare features and target
    feature_columns = [col for col in ml_data.columns if col != target_column]
    X = ml_data[feature_columns]
    y = ml_data[target_column]

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )

    return {
        'datasets': datasets,
        'ml_data': ml_data,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_columns': feature_columns,
        'target_column': target_column
    }


def print_data_overview(data: pd.DataFrame, title: str = "Dataset Overview") -> None:
    """
    Print standardized data overview for notebooks.

    Replaces manual data inspection patterns with consistent formatting.

    Args:
        data: DataFrame to analyze
        title: Title for the overview
    """
    print(f"\n📊 {title}")
    print("=" * (len(title) + 4))

    print(f"\n📈 Basic Info:")
    print(f"  • Shape: {data.shape}")
    print(f"  • Columns: {list(data.columns)}")

    if isinstance(data.index, pd.DatetimeIndex):
        print(f"  • Date range: {data.index.min().date()} to {data.index.max().date()}")
        print(f"  • Duration: {(data.index.max() - data.index.min()).days} days")

    print(f"\n📋 Data Quality:")
    missing_values = data.isnull().sum()
    if missing_values.any():
        print(f"  • Missing values: {missing_values.sum()}")
        for col, missing in missing_values.items():
            if missing > 0:
                pct = (missing / len(data)) * 100
                print(f"    - {col}: {missing} ({pct:.1f}%)")
    else:
        print("  • No missing values ✅")

    print(f"\n📊 Summary Statistics:")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary = data[numeric_cols].describe()
        print(summary.round(2))
    else:
        print("  • No numeric columns found")


def quick_column_detection(df: pd.DataFrame, print_summary: bool = True) -> Dict[str, str]:
    """
    Quick column detection and validation for notebook use.

    Eliminates the manual column detection pattern found in every notebook:
    ```python
    # OLD PATTERN (15+ lines per notebook)
    production_col = None
    consumption_col = None
    for col in daily_data.columns:
        if 'production' in col.lower() or 'produced' in col.lower():
            production_col = col
        elif 'consumption' in col.lower() or 'consumed' in col.lower():
            consumption_col = col
    ```

    With simple:
    ```python
    # NEW PATTERN (1 line)
    energy_cols = quick_column_detection(daily_data)
    production_col = energy_cols.get('production')
    ```

    Args:
        df: DataFrame to analyze
        print_summary: Print detection summary for notebook output

    Returns:
        Dict mapping energy type to column name
    """
    # Use WARNING level to suppress INFO messages in notebooks
    mapper = ColumnMapper(strict_mode=False, log_level='WARNING')

    if print_summary:
        mapper.print_detection_summary(df)

    return mapper.detect_columns(df)


def standardize_dataframe_columns(df: pd.DataFrame,
                                print_changes: bool = True) -> pd.DataFrame:
    """
    Standardize DataFrame column names for consistent analysis.

    Args:
        df: DataFrame to standardize
        print_changes: Print what changes were made

    Returns:
        DataFrame with standardized column names
    """
    if print_changes:
        original_cols = set(df.columns)

    # Use quiet mapper for standardization
    quiet_mapper = ColumnMapper(strict_mode=False, log_level='WARNING')
    result = quiet_mapper.standardize_columns(df)

    if print_changes:
        new_cols = set(result.columns)
        if original_cols != new_cols:
            print("✅ Column standardization applied:")
            detected = quiet_mapper.detect_columns(df)
            for energy_type, detected_col in detected.items():
                if detected_col in original_cols:
                    standard_name = quiet_mapper.config.standard_columns.get(energy_type)
                    if standard_name and standard_name in result.columns:
                        print(f"  '{detected_col}' → '{standard_name}'")
        else:
            print("✅ No column changes needed - already standardized")

    return result


def validate_energy_dataframe(df: pd.DataFrame,
                            print_report: bool = True) -> Dict[str, Any]:
    """
    Validate that DataFrame has expected energy columns.

    Args:
        df: DataFrame to validate
        print_report: Print validation report

    Returns:
        Validation results
    """
    validation = validate_solar_data_columns(df)

    if print_report:
        print(f"\n📊 Energy Data Validation")
        print("=" * 30)
        print(f"Status: {validation['status']}")
        print(f"Energy columns detected: {validation['column_count']}")

        if validation['detected_columns']:
            print(f"\n✅ Detected columns:")
            for energy_type, col_name in validation['detected_columns'].items():
                print(f"  • {energy_type.title()}: '{col_name}'")

        if validation['missing_critical'] or validation['missing_important']:
            print(f"\n⚠️  Missing columns:")
            if validation['missing_critical']:
                print(f"  • Critical: {validation['missing_critical']}")
            if validation['missing_important']:
                print(f"  • Important: {validation['missing_important']}")

        print(f"\n💡 Recommendations:")
        for rec in validation['recommendations']:
            print(f"  • {rec}")

    return validation


def deprecated_load_enphase_data(filename: str = "4136754_custom_report.csv") -> pd.DataFrame:
    """
    Deprecated: Use quick_load_solar_data() instead.

    Maintains compatibility with legacy notebooks while providing deprecation warning.
    """
    import warnings
    warnings.warn(
        "deprecated_load_enphase_data() is deprecated. Use quick_load_solar_data() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Use new utilities with legacy interface
    try:
        project_root = Path(__file__).parent.parent.parent
        file_path = project_root / "data" / "raw" / filename
        return QuickDataLoader.load_solar_data(str(file_path), include_daily=False)
    except Exception as e:
        logger.error(f"Legacy load failed: {e}")
        raise