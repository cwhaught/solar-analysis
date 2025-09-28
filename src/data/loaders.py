"""
Standardized Data Loading Utilities

Consolidates duplicate CSV loading patterns found across 9 notebooks
into reusable, tested utilities. Provides standardized preprocessing,
validation, and format handling for solar energy data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import warnings

# Import ColumnMapper for intelligent column detection
try:
    from .column_mapper import ColumnMapper
except ImportError:
    # Fallback for development/testing
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from column_mapper import ColumnMapper

logger = logging.getLogger(__name__)


class StandardizedCSVLoader:
    """
    Standardized CSV loading with automatic preprocessing and validation.

    Consolidates patterns found in 9 notebooks that repeatedly implement:
    - CSV loading with datetime conversion
    - Wh to kWh unit conversion
    - Column standardization
    - Basic data validation
    """

    def __init__(self, default_csv_path: Optional[str] = None):
        """
        Initialize loader with optional default file path.

        Args:
            default_csv_path: Default CSV file path for convenience
        """
        self.default_csv_path = default_csv_path

    def load_solar_csv(
        self,
        file_path: Optional[str] = None,
        datetime_col: str = 'Date/Time',
        auto_convert_units: bool = True,
        validate_data: bool = True,
        add_metadata: bool = True,
        standardize_columns: bool = True
    ) -> pd.DataFrame:
        """
        Load solar CSV data with standardized preprocessing.

        Consolidates the duplicate pattern found in notebooks:
        ```python
        df = pd.read_csv(file_path)
        df['Date/Time'] = pd.to_datetime(df['Date/Time'])
        df.set_index('Date/Time', inplace=True)
        df_kwh = df / 1000  # Convert Wh to kWh
        df_kwh.columns = ['Production (kWh)', 'Consumption (kWh)', 'Export (kWh)', 'Import (kWh)']
        ```

        Args:
            file_path: Path to CSV file (uses default if None)
            datetime_col: Name of datetime column
            auto_convert_units: Automatically convert Wh to kWh
            validate_data: Perform data validation
            add_metadata: Add metadata attributes to DataFrame
            standardize_columns: Apply intelligent column name standardization

        Returns:
            Processed DataFrame with datetime index and standardized columns

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        # Determine file path
        csv_path = file_path or self.default_csv_path
        if not csv_path:
            raise ValueError("No file_path provided and no default_csv_path set")

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info(f"Loading CSV data from {csv_path}")

        try:
            # Load CSV data
            df = pd.read_csv(csv_path)

            # Convert datetime and set index
            if datetime_col not in df.columns:
                raise ValueError(f"Datetime column '{datetime_col}' not found in CSV")

            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df.set_index(datetime_col, inplace=True)

            # Auto-convert units if requested
            if auto_convert_units:
                df = self.convert_to_kwh(df)

            # Validate data if requested
            if validate_data:
                validation_results = self.validate_solar_data(df)
                if not validation_results['is_valid']:
                    warnings.warn(f"Data validation issues: {validation_results['warnings']}")

            # Standardize column names if requested
            if standardize_columns:
                try:
                    column_mapper = ColumnMapper(strict_mode=False, log_level='WARNING')
                    df = column_mapper.standardize_columns(df)
                    logger.info("Applied intelligent column standardization")
                except Exception as e:
                    logger.warning(f"Column standardization failed: {e}")
                    # Continue without standardization

            # Add metadata
            if add_metadata:
                df.attrs.update({
                    'source_file': str(csv_path),
                    'loaded_at': datetime.now(),
                    'granularity': self._detect_granularity(df),
                    'loader_version': '1.0.0',
                    'auto_converted_units': auto_convert_units
                })

            logger.info(f"Successfully loaded {len(df)} records from {csv_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise

    def convert_to_kwh(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Wh columns to kWh with standardized column naming.

        Implements the standard pattern found in all notebooks:
        ```python
        df_kwh = df / 1000
        df_kwh.columns = ['Production (kWh)', 'Consumption (kWh)', 'Export (kWh)', 'Import (kWh)']
        ```

        Args:
            df: DataFrame with Wh columns

        Returns:
            DataFrame with kWh values and standardized column names
        """
        # Identify numeric columns that likely contain Wh data
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Convert to kWh (divide by 1000)
        df_kwh = df.copy()
        df_kwh[numeric_cols] = df[numeric_cols] / 1000

        # Standardize column names based on common patterns
        column_mapping = self._create_column_mapping(df.columns)
        df_kwh = df_kwh.rename(columns=column_mapping)

        # Add metadata about conversion
        df_kwh.attrs['unit_conversion'] = 'Wh_to_kWh'
        df_kwh.attrs['conversion_factor'] = 1000

        logger.debug(f"Converted {len(numeric_cols)} columns from Wh to kWh")
        return df_kwh

    def validate_solar_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data validation for solar energy data.

        Performs checks commonly done manually in notebooks:
        - Missing values analysis
        - Date range validation
        - Data completeness assessment
        - Basic integrity checks

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {}
        }

        try:
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                validation_results['warnings'].append(f"Missing values found: {missing_values.to_dict()}")

            # Check date range continuity
            if hasattr(df.index, 'freq') or len(df) > 1:
                date_gaps = self._check_date_continuity(df)
                if date_gaps:
                    validation_results['warnings'].append(f"Date gaps detected: {len(date_gaps)} gaps")

            # Check for negative production values (usually indicates data issues)
            if 'Production (kWh)' in df.columns:
                negative_production = (df['Production (kWh)'] < 0).sum()
                if negative_production > 0:
                    validation_results['warnings'].append(
                        f"Negative production values: {negative_production} records"
                    )

            # Calculate data completeness metrics
            metrics = self._calculate_completeness_metrics(df)
            validation_results['metrics'] = metrics

            # Determine overall validity
            validation_results['is_valid'] = len(validation_results['errors']) == 0

        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False

        return validation_results

    def _create_column_mapping(self, columns: List[str]) -> Dict[str, str]:
        """Create mapping from original column names to standardized names."""
        mapping = {}

        for col in columns:
            col_lower = col.lower()
            if 'production' in col_lower and 'wh' in col_lower:
                mapping[col] = 'Production (kWh)'
            elif 'consumption' in col_lower and 'wh' in col_lower:
                mapping[col] = 'Consumption (kWh)'
            elif 'export' in col_lower and 'wh' in col_lower:
                mapping[col] = 'Export (kWh)'
            elif 'import' in col_lower and 'wh' in col_lower:
                mapping[col] = 'Import (kWh)'

        return mapping

    def _detect_granularity(self, df: pd.DataFrame) -> str:
        """Detect data granularity based on index frequency."""
        if len(df) < 2:
            return 'unknown'

        time_diff = df.index[1] - df.index[0]

        if time_diff == timedelta(minutes=15):
            return '15min'
        elif time_diff == timedelta(hours=1):
            return 'hourly'
        elif time_diff == timedelta(days=1):
            return 'daily'
        else:
            return f'{time_diff.total_seconds()}s'

    def _check_date_continuity(self, df: pd.DataFrame) -> List[Tuple[datetime, datetime]]:
        """Check for gaps in date continuity."""
        gaps = []
        if len(df) < 2:
            return gaps

        # Expected frequency based on first interval
        expected_freq = df.index[1] - df.index[0]

        for i in range(1, len(df)):
            actual_diff = df.index[i] - df.index[i-1]
            if actual_diff > expected_freq * 1.5:  # Allow some tolerance
                gaps.append((df.index[i-1], df.index[i]))

        return gaps

    def _calculate_completeness_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data completeness metrics."""
        metrics = {}

        if len(df) > 0:
            # Overall completeness
            total_cells = df.size
            non_null_cells = df.count().sum()
            metrics['completeness_pct'] = (non_null_cells / total_cells) * 100

            # Date range metrics
            date_range = df.index.max() - df.index.min()
            metrics['date_range_days'] = date_range.days

            # Expected vs actual records (for time series data)
            if hasattr(df.index, 'freq') and df.index.freq:
                expected_records = len(pd.date_range(
                    start=df.index.min(),
                    end=df.index.max(),
                    freq=df.index.freq
                ))
                metrics['record_completeness_pct'] = (len(df) / expected_records) * 100

        return metrics


class QuickDataLoader:
    """
    Simple one-function data loading for notebook convenience.

    Provides ultra-simple interface for common notebook patterns:
    - Load data with one function call
    - Return both 15-minute and daily data
    - Handle common data sources automatically
    """

    @staticmethod
    def load_solar_data(
        source: Union[str, Path] = 'auto',
        include_daily: bool = True
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        One-line data loading for notebooks.

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
        fifteen_min_data, daily_data = QuickDataLoader.load_solar_data()
        ```

        Args:
            source: Data source ('auto', file path, or 'default')
            include_daily: Return daily aggregated data as well

        Returns:
            DataFrame or tuple of (15min_data, daily_data)
        """
        # Auto-detect source if needed
        if source == 'auto' or source == 'default':
            # Try common locations
            possible_paths = [
                '../data/raw/4136754_custom_report.csv',
                'data/raw/4136754_custom_report.csv',
                Path(__file__).parent.parent.parent / 'data' / 'raw' / '4136754_custom_report.csv'
            ]

            csv_path = None
            for path in possible_paths:
                if Path(path).exists():
                    csv_path = path
                    break

            if not csv_path:
                raise FileNotFoundError(
                    "Could not auto-detect CSV file. Please provide explicit path."
                )
        else:
            csv_path = source

        # Load data using StandardizedCSVLoader
        loader = StandardizedCSVLoader()
        fifteen_min_data = loader.load_solar_csv(csv_path)

        if include_daily:
            # Create daily summary using same patterns from notebooks
            daily_data = fifteen_min_data.resample('D').sum()
            daily_data.attrs = fifteen_min_data.attrs.copy()
            daily_data.attrs['granularity'] = 'daily'
            daily_data.attrs['aggregation_method'] = 'sum'

            return fifteen_min_data, daily_data
        else:
            return fifteen_min_data