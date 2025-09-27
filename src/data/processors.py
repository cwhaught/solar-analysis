"""
Solar Data Processing Utilities

Standard data processing operations extracted from notebook patterns.
Handles aggregation, calculations, and transformations consistently
across all solar energy analyses.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class SolarDataProcessor:
    """
    Standard solar data processing operations.

    Consolidates processing patterns found across multiple notebooks:
    - Daily/monthly aggregation with proper frequency handling
    - Net energy calculations (production - consumption)
    - Self-consumption rate calculations
    - Time-based feature additions
    """

    def __init__(self):
        """Initialize processor with default settings."""
        self.default_energy_columns = [
            'Production (kWh)', 'Consumption (kWh)',
            'Export (kWh)', 'Import (kWh)'
        ]

    def create_daily_summary(
        self,
        df: pd.DataFrame,
        method: str = 'sum',
        preserve_metadata: bool = True
    ) -> pd.DataFrame:
        """
        Create daily summary data with standardized aggregation.

        Replaces the common notebook pattern:
        ```python
        daily_data = df_kwh.resample('D').sum()
        ```

        Args:
            df: Input DataFrame with datetime index
            method: Aggregation method ('sum', 'mean', 'max', 'min')
            preserve_metadata: Copy metadata from original DataFrame

        Returns:
            Daily aggregated DataFrame with metadata
        """
        logger.debug(f"Creating daily summary using {method} aggregation")

        try:
            # Perform aggregation based on method
            if method == 'sum':
                daily_data = df.resample('D').sum()
            elif method == 'mean':
                daily_data = df.resample('D').mean()
            elif method == 'max':
                daily_data = df.resample('D').max()
            elif method == 'min':
                daily_data = df.resample('D').min()
            else:
                raise ValueError(f"Unsupported aggregation method: {method}")

            # Preserve metadata if requested
            if preserve_metadata and hasattr(df, 'attrs'):
                daily_data.attrs = df.attrs.copy()
                daily_data.attrs['granularity'] = 'daily'
                daily_data.attrs['aggregation_method'] = method
                daily_data.attrs['created_at'] = datetime.now()

            logger.info(f"Created daily summary: {len(daily_data)} days from {len(df)} records")
            return daily_data

        except Exception as e:
            logger.error(f"Error creating daily summary: {e}")
            raise

    def create_monthly_summary(
        self,
        df: pd.DataFrame,
        method: str = 'sum',
        preserve_metadata: bool = True
    ) -> pd.DataFrame:
        """
        Create monthly summary data using non-deprecated frequency.

        Replaces the problematic notebook pattern:
        ```python
        monthly_data = df_kwh.resample('M').mean()  # Deprecated
        ```

        With correct:
        ```python
        monthly_data = df_kwh.resample('ME').mean()  # Month End
        ```

        Args:
            df: Input DataFrame with datetime index
            method: Aggregation method ('sum', 'mean', 'max', 'min')
            preserve_metadata: Copy metadata from original DataFrame

        Returns:
            Monthly aggregated DataFrame
        """
        logger.debug(f"Creating monthly summary using {method} aggregation")

        try:
            # Use 'ME' (Month End) instead of deprecated 'M'
            if method == 'sum':
                monthly_data = df.resample('ME').sum()
            elif method == 'mean':
                monthly_data = df.resample('ME').mean()
            elif method == 'max':
                monthly_data = df.resample('ME').max()
            elif method == 'min':
                monthly_data = df.resample('ME').min()
            else:
                raise ValueError(f"Unsupported aggregation method: {method}")

            # Preserve metadata if requested
            if preserve_metadata and hasattr(df, 'attrs'):
                monthly_data.attrs = df.attrs.copy()
                monthly_data.attrs['granularity'] = 'monthly'
                monthly_data.attrs['aggregation_method'] = method
                monthly_data.attrs['created_at'] = datetime.now()

            logger.info(f"Created monthly summary: {len(monthly_data)} months from {len(df)} records")
            return monthly_data

        except Exception as e:
            logger.error(f"Error creating monthly summary: {e}")
            raise

    def calculate_net_energy(
        self,
        df: pd.DataFrame,
        production_col: str = 'Production (kWh)',
        consumption_col: str = 'Consumption (kWh)'
    ) -> pd.DataFrame:
        """
        Add net energy calculations to DataFrame.

        Implements the common notebook pattern:
        ```python
        df['Net Energy (kWh)'] = df['Production (kWh)'] - df['Consumption (kWh)']
        df['Energy Surplus'] = df['Net Energy (kWh)'] > 0
        ```

        Args:
            df: Input DataFrame with energy columns
            production_col: Name of production column
            consumption_col: Name of consumption column

        Returns:
            DataFrame with additional net energy columns
        """
        result_df = df.copy()

        if production_col not in df.columns:
            raise ValueError(f"Production column '{production_col}' not found")
        if consumption_col not in df.columns:
            raise ValueError(f"Consumption column '{consumption_col}' not found")

        # Calculate net energy
        result_df['Net Energy (kWh)'] = result_df[production_col] - result_df[consumption_col]

        # Add surplus indicator
        result_df['Energy Surplus'] = result_df['Net Energy (kWh)'] > 0

        # Add energy self-sufficiency percentage
        result_df['Self Sufficiency (%)'] = np.minimum(
            100,
            (result_df[production_col] / result_df[consumption_col]) * 100
        )

        logger.debug("Added net energy calculations")
        return result_df

    def calculate_self_consumption_metrics(
        self,
        df: pd.DataFrame,
        production_col: str = 'Production (kWh)',
        consumption_col: str = 'Consumption (kWh)',
        export_col: str = 'Export (kWh)'
    ) -> pd.DataFrame:
        """
        Calculate self-consumption rate and related metrics.

        Implements the pattern found in financial analysis notebooks:
        ```python
        self_consumed = production - export
        self_consumption_rate = (self_consumed / production) * 100
        ```

        Args:
            df: Input DataFrame with energy columns
            production_col: Name of production column
            consumption_col: Name of consumption column
            export_col: Name of export column

        Returns:
            DataFrame with self-consumption metrics
        """
        result_df = df.copy()

        # Validate required columns
        required_cols = [production_col, consumption_col, export_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Calculate self-consumed energy
        result_df['Self Consumed (kWh)'] = result_df[production_col] - result_df[export_col]

        # Calculate self-consumption rate (percentage of production used locally)
        with np.errstate(divide='ignore', invalid='ignore'):
            self_consumption_rate = (result_df['Self Consumed (kWh)'] / result_df[production_col]) * 100
            # Handle division by zero (when production is 0)
            self_consumption_rate = np.where(
                result_df[production_col] == 0,
                0,
                self_consumption_rate
            )
        result_df['Self Consumption Rate (%)'] = np.clip(self_consumption_rate, 0, 100)

        # Calculate grid independence rate (percentage of consumption met by solar)
        with np.errstate(divide='ignore', invalid='ignore'):
            grid_independence = (result_df['Self Consumed (kWh)'] / result_df[consumption_col]) * 100
            grid_independence = np.where(
                result_df[consumption_col] == 0,
                0,
                grid_independence
            )
        result_df['Grid Independence (%)'] = np.clip(grid_independence, 0, 100)

        logger.debug("Added self-consumption metrics")
        return result_df

    def add_time_features(
        self,
        df: pd.DataFrame,
        include_cyclical: bool = False
    ) -> pd.DataFrame:
        """
        Add basic time-based features commonly used in analysis.

        Implements the common notebook pattern:
        ```python
        df['day_of_year'] = df.index.dayofyear
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        ```

        Args:
            df: Input DataFrame with datetime index
            include_cyclical: Include sin/cos cyclical encodings

        Returns:
            DataFrame with additional time features
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for time features")

        result_df = df.copy()

        # Basic time features
        result_df['day_of_year'] = result_df.index.dayofyear
        result_df['day_of_week'] = result_df.index.dayofweek
        result_df['month'] = result_df.index.month
        result_df['quarter'] = result_df.index.quarter
        result_df['week_of_year'] = result_df.index.isocalendar().week

        # Binary features
        result_df['is_weekend'] = (result_df.index.dayofweek >= 5).astype(int)
        result_df['is_monday'] = (result_df.index.dayofweek == 0).astype(int)

        # Seasonal indicators
        result_df['is_winter'] = result_df['month'].isin([12, 1, 2]).astype(int)
        result_df['is_spring'] = result_df['month'].isin([3, 4, 5]).astype(int)
        result_df['is_summer'] = result_df['month'].isin([6, 7, 8]).astype(int)
        result_df['is_fall'] = result_df['month'].isin([9, 10, 11]).astype(int)

        # Cyclical encodings if requested
        if include_cyclical:
            # Daily cycle
            result_df['day_sin'] = np.sin(2 * np.pi * result_df['day_of_year'] / 365.25)
            result_df['day_cos'] = np.cos(2 * np.pi * result_df['day_of_year'] / 365.25)

            # Weekly cycle
            result_df['week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
            result_df['week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)

            # Monthly cycle
            result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
            result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)

        logger.debug(f"Added time features (cyclical: {include_cyclical})")
        return result_df

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        target_columns: Optional[List[str]] = None,
        windows: List[int] = [7, 14, 30],
        statistics: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
        """
        Add rolling statistical features commonly used in ML models.

        Implements the notebook pattern:
        ```python
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}d'] = df['Production (kWh)'].rolling(window=window).mean()
            df[f'rolling_std_{window}d'] = df['Production (kWh)'].rolling(window=window).std()
        ```

        Args:
            df: Input DataFrame
            target_columns: Columns to create rolling features for (default: energy columns)
            windows: Window sizes in periods
            statistics: Statistical functions to compute ('mean', 'std', 'min', 'max')

        Returns:
            DataFrame with additional rolling features
        """
        result_df = df.copy()

        # Default to energy columns if not specified
        if target_columns is None:
            target_columns = [col for col in self.default_energy_columns if col in df.columns]

        if not target_columns:
            logger.warning("No target columns found for rolling features")
            return result_df

        # Create rolling features
        for col in target_columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping rolling features")
                continue

            col_clean = col.replace(' (kWh)', '').replace(' ', '_').lower()

            for window in windows:
                for stat in statistics:
                    feature_name = f'{col_clean}_rolling_{stat}_{window}d'

                    if stat == 'mean':
                        result_df[feature_name] = df[col].rolling(window=window, min_periods=1).mean()
                    elif stat == 'std':
                        result_df[feature_name] = df[col].rolling(window=window, min_periods=1).std()
                    elif stat == 'min':
                        result_df[feature_name] = df[col].rolling(window=window, min_periods=1).min()
                    elif stat == 'max':
                        result_df[feature_name] = df[col].rolling(window=window, min_periods=1).max()
                    else:
                        logger.warning(f"Unsupported statistic: {stat}")

        logger.debug(f"Added rolling features for {len(target_columns)} columns")
        return result_df

    def calculate_efficiency_metrics(
        self,
        df: pd.DataFrame,
        production_col: str = 'Production (kWh)',
        include_volatility: bool = True,
        rolling_window: int = 30
    ) -> pd.DataFrame:
        """
        Calculate production efficiency and volatility metrics.

        Implements patterns found in ML model notebooks:
        ```python
        df['production_volatility'] = df['Production (kWh)'].rolling(7).std()
        df['relative_performance'] = df['Production (kWh)'] / df['rolling_mean_30d']
        ```

        Args:
            df: Input DataFrame
            production_col: Name of production column
            include_volatility: Include volatility calculations
            rolling_window: Window for rolling calculations

        Returns:
            DataFrame with efficiency metrics
        """
        if production_col not in df.columns:
            raise ValueError(f"Production column '{production_col}' not found")

        result_df = df.copy()

        # Rolling baseline for relative performance
        rolling_mean = df[production_col].rolling(window=rolling_window, min_periods=1).mean()

        # Relative performance (current vs historical average)
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_performance = df[production_col] / rolling_mean
            relative_performance = np.where(rolling_mean == 0, 1.0, relative_performance)
        result_df['Relative Performance'] = relative_performance

        # Production volatility if requested
        if include_volatility:
            volatility_window = min(7, rolling_window)  # Use shorter window for volatility
            result_df['Production Volatility'] = df[production_col].rolling(
                window=volatility_window, min_periods=1
            ).std()

            # Coefficient of variation (normalized volatility)
            with np.errstate(divide='ignore', invalid='ignore'):
                cv = result_df['Production Volatility'] / rolling_mean
                cv = np.where(rolling_mean == 0, 0, cv)
            result_df['Production CV'] = cv

        logger.debug("Added efficiency metrics")
        return result_df

    def prepare_ml_dataset(
        self,
        df: pd.DataFrame,
        target_column: str = 'Production (kWh)',
        include_time_features: bool = True,
        include_rolling_features: bool = True,
        include_efficiency_metrics: bool = True,
        dropna: bool = True
    ) -> pd.DataFrame:
        """
        Prepare comprehensive ML dataset with all common features.

        One-stop function that applies multiple processing steps commonly
        used together in ML model notebooks.

        Args:
            df: Input DataFrame
            target_column: Target variable for ML
            include_time_features: Add time-based features
            include_rolling_features: Add rolling statistics
            include_efficiency_metrics: Add efficiency calculations
            dropna: Remove rows with NaN values

        Returns:
            ML-ready DataFrame with comprehensive features
        """
        logger.info("Preparing comprehensive ML dataset")

        result_df = df.copy()

        # Add net energy calculations
        if all(col in df.columns for col in ['Production (kWh)', 'Consumption (kWh)']):
            result_df = self.calculate_net_energy(result_df)

        # Add self-consumption metrics
        if all(col in df.columns for col in ['Production (kWh)', 'Consumption (kWh)', 'Export (kWh)']):
            result_df = self.calculate_self_consumption_metrics(result_df)

        # Add time features
        if include_time_features:
            result_df = self.add_time_features(result_df, include_cyclical=True)

        # Add rolling features
        if include_rolling_features:
            result_df = self.add_rolling_features(result_df)

        # Add efficiency metrics
        if include_efficiency_metrics and target_column in result_df.columns:
            result_df = self.calculate_efficiency_metrics(result_df, target_column)

        # Remove NaN values if requested
        if dropna:
            initial_len = len(result_df)
            result_df = result_df.dropna()
            dropped_rows = initial_len - len(result_df)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with NaN values")

        logger.info(f"Created ML dataset with {result_df.shape[1]} features and {len(result_df)} samples")
        return result_df