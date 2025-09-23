"""
Solar Data Manager - Hybrid CSV/API System

Manages solar production data from multiple sources:
- Historical CSV data (15-minute granularity, 2+ years)
- Live API data (daily updates via energy_lifetime)
- Real-time monitoring (15-minute intervals via rgm_stats)

Provides unified interface for ML models and analysis.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union, Dict, Tuple
import logging


class SolarDataManager:
    """
    Unified solar data management combining CSV historical data with live API updates
    """

    def __init__(
        self, csv_path: str, enphase_client, cache_dir: str = "data/processed"
    ):
        """
        Initialize data manager

        Args:
            csv_path: Path to historical CSV file
            enphase_client: EnphaseClient instance for API access
            cache_dir: Directory for processed data cache
        """
        self.csv_path = Path(csv_path)
        self.client = enphase_client

        # Always resolve cache_dir relative to project root, not current working directory
        if not os.path.isabs(cache_dir):
            # Find project root by looking for pyproject.toml
            current_dir = Path(__file__).resolve().parent
            while current_dir.parent != current_dir:
                if (current_dir / "pyproject.toml").exists():
                    cache_dir = current_dir / cache_dir
                    break
                current_dir = current_dir.parent
            else:
                # Fallback: use relative to this file's directory
                cache_dir = Path(__file__).resolve().parent.parent.parent / cache_dir

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache files
        self.daily_cache_file = self.cache_dir / "daily_production_combined.csv"
        self.detailed_cache_file = self.cache_dir / "detailed_production_recent.csv"

        # Data containers
        self._csv_data = None
        self._api_data = None
        self._combined_daily = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Detect if using mock client
        self._is_mock_client = (
            hasattr(enphase_client, "__class__")
            and "Mock" in enphase_client.__class__.__name__
        )

    def load_csv_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load and process historical CSV data

        Args:
            force_reload: Force reload even if already cached

        Returns:
            DataFrame with processed CSV data
        """
        if self._csv_data is not None and not force_reload:
            return self._csv_data

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.logger.info(f"Loading CSV data from {self.csv_path}")

        # Load CSV data
        df = pd.read_csv(self.csv_path)
        df["Date/Time"] = pd.to_datetime(df["Date/Time"])
        df.set_index("Date/Time", inplace=True)

        # Convert to standard units (kWh)
        df_kwh = df / 1000
        df_kwh.columns = [
            "Production (kWh)",
            "Consumption (kWh)",
            "Export (kWh)",
            "Import (kWh)",
        ]

        # Add metadata
        df_kwh.attrs["source"] = "csv"
        df_kwh.attrs["granularity"] = "15min"
        df_kwh.attrs["loaded_at"] = datetime.now()

        self._csv_data = df_kwh
        self.logger.info(
            f"Loaded {len(df_kwh)} CSV records from {df_kwh.index.min()} to {df_kwh.index.max()}"
        )

        return self._csv_data

    def load_api_data(
        self, days_back: int = 30, force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Load recent data from API

        Args:
            days_back: Number of days to fetch from API
            force_reload: Force reload even if already cached

        Returns:
            DataFrame with API data
        """
        if self._api_data is not None and not force_reload:
            return self._api_data

        if self._is_mock_client:
            self.logger.info("Using mock client - skipping API data retrieval")
            self._api_data = pd.DataFrame()
            return self._api_data

        self.logger.info(f"Loading API data for last {days_back} days")

        try:
            # Get recent data from API
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            api_data = self.client.get_energy_lifetime(start_date, end_date)

            if not api_data.empty:
                # Standardize column names
                api_data = api_data.rename(
                    columns={"daily_energy_kwh": "Production (kWh)"}
                )

                # Add metadata
                api_data.attrs["source"] = "api"
                api_data.attrs["granularity"] = "daily"
                api_data.attrs["loaded_at"] = datetime.now()

                self._api_data = api_data
                self.logger.info(
                    f"Loaded {len(api_data)} API records from {api_data.index.min()} to {api_data.index.max()}"
                )
            else:
                self.logger.warning("No API data retrieved")
                self._api_data = pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error loading API data: {e}")
            self._api_data = pd.DataFrame()

        return self._api_data

    def get_daily_production(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source_priority: str = "csv_first",
    ) -> pd.DataFrame:
        """
        Get daily production data using hybrid approach

        Args:
            start_date: Start date for data
            end_date: End date for data
            source_priority: 'csv_first', 'api_first', or 'csv_only'

        Returns:
            DataFrame with daily production data
        """
        # Load both data sources
        csv_data = self.load_csv_data()
        api_data = self.load_api_data()

        # Convert CSV to daily if needed
        if not csv_data.empty and csv_data.attrs.get("granularity") == "15min":
            csv_daily = csv_data.resample("D").sum()
        else:
            csv_daily = csv_data.copy()

        # Determine data strategy
        if source_priority == "csv_only":
            combined = csv_daily.copy()

        elif source_priority == "csv_first":
            # Use CSV as primary, fill gaps with API
            combined = csv_daily.copy()

            if not api_data.empty:
                # Fill missing dates with API data
                api_only_dates = api_data.index.difference(csv_daily.index)
                if len(api_only_dates) > 0:
                    api_supplement = api_data.loc[api_only_dates, ["Production (kWh)"]]
                    combined = pd.concat([combined, api_supplement]).sort_index()

        elif source_priority == "api_first":
            # Use API as primary, fill gaps with CSV
            combined = api_data.copy() if not api_data.empty else pd.DataFrame()

            if not csv_daily.empty:
                csv_only_dates = csv_daily.index.difference(combined.index)
                if len(csv_only_dates) > 0:
                    csv_supplement = csv_daily.loc[csv_only_dates, ["Production (kWh)"]]
                    combined = pd.concat([combined, csv_supplement]).sort_index()

        # Apply date filtering
        if start_date:
            combined = combined[combined.index >= start_date]
        if end_date:
            combined = combined[combined.index <= end_date]

        # Add metadata about sources
        if not combined.empty:
            combined.attrs["source_priority"] = source_priority
            combined.attrs["csv_records"] = len(csv_daily)
            combined.attrs["api_records"] = len(api_data)
            combined.attrs["combined_records"] = len(combined)

        return combined

    def get_detailed_intervals(
        self, target_date: datetime, source: str = "auto"
    ) -> pd.DataFrame:
        """
        Get 15-minute interval data for a specific date

        Args:
            target_date: Date to get detailed data for
            source: 'csv', 'api', or 'auto'

        Returns:
            DataFrame with 15-minute intervals
        """
        if source == "csv" or source == "auto":
            # Try CSV first
            csv_data = self.load_csv_data()

            if not csv_data.empty:
                target_str = target_date.strftime("%Y-%m-%d")
                day_data = csv_data[csv_data.index.date == target_date.date()]

                if not day_data.empty:
                    day_data.attrs["source"] = "csv"
                    return day_data

        if source == "api" or (source == "auto" and day_data.empty):
            # Try API for recent dates
            try:
                api_intervals = self.client.get_rgm_stats(target_date, target_date)

                if not api_intervals.empty:
                    # Convert to standard format
                    standardized = pd.DataFrame(
                        {
                            "Production (kWh)": api_intervals.groupby(
                                api_intervals.index
                            )["energy_delivered_kwh"].sum()
                        }
                    )
                    standardized.attrs["source"] = "api"
                    return standardized

            except Exception as e:
                self.logger.warning(
                    f"Could not get API intervals for {target_date}: {e}"
                )

        return pd.DataFrame()

    def update_from_api(self, save_cache: bool = True) -> Dict[str, int]:
        """
        Update dataset with latest API data

        Args:
            save_cache: Save updated data to cache files

        Returns:
            Dictionary with update statistics
        """
        self.logger.info("Updating dataset from API")

        # Get existing daily data
        current_daily = self.get_daily_production(source_priority="csv_first")

        # Determine update window
        if not current_daily.empty:
            last_date = current_daily.index.max()
            days_to_update = (
                datetime.now().date() - last_date.date()
            ).days + 7  # Extra buffer
        else:
            days_to_update = 30

        # Get fresh API data
        fresh_api = self.load_api_data(days_back=days_to_update, force_reload=True)

        stats = {
            "existing_records": len(current_daily),
            "api_records": len(fresh_api),
            "new_records": 0,
            "updated_records": 0,
        }

        if not fresh_api.empty:
            # Find new dates
            if not current_daily.empty:
                new_dates = fresh_api.index.difference(current_daily.index)
                updated_dates = fresh_api.index.intersection(current_daily.index)

                stats["new_records"] = len(new_dates)
                stats["updated_records"] = len(updated_dates)

                # Combine data
                updated_daily = current_daily.copy()

                # Add new records
                if len(new_dates) > 0:
                    new_data = fresh_api.loc[new_dates, ["Production (kWh)"]]
                    updated_daily = pd.concat([updated_daily, new_data]).sort_index()

                # Update existing records with API data (if preferred)
                # This is optional - you might want to keep CSV data as authoritative

            else:
                updated_daily = fresh_api[["Production (kWh)"]].copy()
                stats["new_records"] = len(updated_daily)

            # Save cache if requested
            if save_cache and not updated_daily.empty:
                updated_daily.to_csv(self.daily_cache_file)
                self.logger.info(f"Saved updated daily data to {self.daily_cache_file}")

            # Update internal cache
            self._combined_daily = updated_daily

        self.logger.info(f"Update complete: {stats}")
        return stats

    def get_data_summary(self) -> Dict:
        """Get summary of available data sources"""
        csv_data = self.load_csv_data()
        api_data = self.load_api_data()

        summary = {
            "csv": {
                "available": not csv_data.empty,
                "records": len(csv_data) if not csv_data.empty else 0,
                "date_range": (
                    (
                        csv_data.index.min().strftime("%Y-%m-%d"),
                        csv_data.index.max().strftime("%Y-%m-%d"),
                    )
                    if not csv_data.empty
                    else None
                ),
                "granularity": (
                    csv_data.attrs.get("granularity", "unknown")
                    if not csv_data.empty
                    else None
                ),
            },
            "api": {
                "available": not api_data.empty,
                "is_mock": self._is_mock_client,
                "records": len(api_data) if not api_data.empty else 0,
                "date_range": (
                    (
                        api_data.index.min().strftime("%Y-%m-%d"),
                        api_data.index.max().strftime("%Y-%m-%d"),
                    )
                    if not api_data.empty
                    else None
                ),
                "granularity": (
                    api_data.attrs.get("granularity", "unknown")
                    if not api_data.empty
                    else None
                ),
            },
        }

        return summary

    def export_combined_dataset(
        self, filename: str, source_priority: str = "csv_first"
    ) -> bool:
        """
        Export the complete combined dataset

        Args:
            filename: Output filename
            source_priority: Data source priority

        Returns:
            True if export successful
        """
        try:
            combined_data = self.get_daily_production(source_priority=source_priority)

            if not combined_data.empty:
                combined_data.to_csv(filename)
                self.logger.info(f"Exported {len(combined_data)} records to {filename}")
                return True
            else:
                self.logger.warning("No data to export")
                return False

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False
