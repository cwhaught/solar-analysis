"""
Data Summary and Statistics - Standardized data analysis and reporting

Provides comprehensive data summarization, quality checks, and
statistical analysis functions for solar energy data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta


class SolarDataSummary:
    """
    Comprehensive data summary and statistics generator
    """

    def __init__(self):
        self.expected_intervals_per_day = 96  # 15-minute intervals

    def generate_dataset_overview(self, data: pd.DataFrame, data_type: str = "Solar Data") -> Dict[str, Any]:
        """
        Generate comprehensive dataset overview

        Args:
            data: Solar data DataFrame with datetime index
            data_type: Description of the data type

        Returns:
            Dictionary with dataset overview information
        """
        if data.empty:
            return {
                'data_type': data_type,
                'status': 'empty',
                'shape': (0, 0),
                'message': 'No data available'
            }

        # Basic dataset info
        shape = data.shape
        date_range = (data.index.min(), data.index.max())
        duration_days = (date_range[1] - date_range[0]).days + 1

        # Data granularity analysis
        granularity = self._detect_data_granularity(data)

        # Expected vs actual records
        if granularity == '15min':
            expected_records = duration_days * self.expected_intervals_per_day
        elif granularity == 'hourly':
            expected_records = duration_days * 24
        elif granularity == 'daily':
            expected_records = duration_days
        else:
            expected_records = None

        completeness = (len(data) / expected_records * 100) if expected_records else None

        return {
            'data_type': data_type,
            'shape': shape,
            'records': len(data),
            'columns': list(data.columns),
            'date_range': {
                'start': date_range[0].strftime('%Y-%m-%d %H:%M:%S'),
                'end': date_range[1].strftime('%Y-%m-%d %H:%M:%S'),
                'duration_days': duration_days
            },
            'granularity': granularity,
            'completeness': {
                'expected_records': expected_records,
                'actual_records': len(data),
                'completeness_pct': completeness
            }
        }

    def analyze_production_statistics(self, daily_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive production statistics

        Args:
            daily_data: DataFrame with daily production data

        Returns:
            Dictionary with production statistics
        """
        if 'Production (kWh)' not in daily_data.columns:
            return {'error': 'Production (kWh) column not found'}

        production = daily_data['Production (kWh)']

        # Basic statistics
        basic_stats = {
            'total_production': production.sum(),
            'average_daily': production.mean(),
            'median_daily': production.median(),
            'std_deviation': production.std(),
            'min_daily': production.min(),
            'max_daily': production.max(),
            'quartile_25': production.quantile(0.25),
            'quartile_75': production.quantile(0.75)
        }

        # Production consistency metrics
        coefficient_of_variation = production.std() / production.mean() if production.mean() > 0 else 0

        # Days with significant production (>10% of average)
        avg_production = production.mean()
        productive_days = (production > avg_production * 0.1).sum()
        zero_production_days = (production == 0).sum()

        consistency_metrics = {
            'coefficient_of_variation': coefficient_of_variation,
            'productive_days': productive_days,
            'zero_production_days': zero_production_days,
            'productive_days_pct': (productive_days / len(production)) * 100
        }

        return {
            'basic_statistics': basic_stats,
            'consistency_metrics': consistency_metrics,
            'analysis_period': len(daily_data)
        }

    def analyze_consumption_patterns(self, daily_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze energy consumption patterns

        Args:
            daily_data: DataFrame with daily consumption data

        Returns:
            Dictionary with consumption analysis
        """
        if 'Consumption (kWh)' not in daily_data.columns:
            return {'error': 'Consumption (kWh) column not found'}

        consumption = daily_data['Consumption (kWh)']

        # Basic consumption statistics
        consumption_stats = {
            'total_consumption': consumption.sum(),
            'average_daily': consumption.mean(),
            'min_daily': consumption.min(),
            'max_daily': consumption.max(),
            'std_deviation': consumption.std()
        }

        # Grid interaction analysis
        grid_analysis = {}
        if all(col in daily_data.columns for col in ['Export (kWh)', 'Import (kWh)', 'Production (kWh)']):
            total_export = daily_data['Export (kWh)'].sum()
            total_import = daily_data['Import (kWh)'].sum()
            total_production = daily_data['Production (kWh)'].sum()

            grid_analysis = {
                'total_export': total_export,
                'total_import': total_import,
                'net_export': total_export - total_import,
                'self_consumption_kwh': total_production - total_export,
                'self_consumption_rate': ((total_production - total_export) / total_production) * 100 if total_production > 0 else 0,
                'grid_independence_rate': (1 - total_import / consumption.sum()) * 100 if consumption.sum() > 0 else 0
            }

        return {
            'consumption_statistics': consumption_stats,
            'grid_interaction': grid_analysis
        }

    def analyze_seasonal_patterns(self, daily_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze seasonal production patterns

        Args:
            daily_data: DataFrame with daily production data

        Returns:
            Dictionary with seasonal analysis
        """
        if 'Production (kWh)' not in daily_data.columns:
            return {'error': 'Production (kWh) column not found'}

        # Monthly analysis
        monthly_stats = daily_data.groupby(daily_data.index.month)['Production (kWh)'].agg([
            'mean', 'sum', 'std', 'min', 'max', 'count'
        ]).round(2)

        # Find best and worst months
        best_month = monthly_stats['mean'].idxmax()
        worst_month = monthly_stats['mean'].idxmin()
        seasonal_variation = monthly_stats['mean'].max() / monthly_stats['mean'].min()

        # Quarterly analysis
        quarterly_stats = daily_data.groupby(daily_data.index.quarter)['Production (kWh)'].mean()

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        season_names = ['Winter', 'Spring', 'Summer', 'Fall']

        return {
            'monthly_statistics': {
                'data': monthly_stats.to_dict(),
                'best_month': {'number': best_month, 'name': month_names[best_month-1], 'avg_production': monthly_stats.loc[best_month, 'mean']},
                'worst_month': {'number': worst_month, 'name': month_names[worst_month-1], 'avg_production': monthly_stats.loc[worst_month, 'mean']},
                'seasonal_variation': seasonal_variation
            },
            'quarterly_statistics': {
                'data': quarterly_stats.to_dict(),
                'season_names': season_names
            }
        }

    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment

        Args:
            data: Solar data DataFrame

        Returns:
            Dictionary with data quality metrics
        """
        quality_issues = []
        warnings = []

        # Missing values analysis
        missing_values = data.isnull().sum()
        total_missing = missing_values.sum()

        if total_missing > 0:
            quality_issues.append(f"Found {total_missing} missing values")

        # Duplicate index check
        duplicate_indices = data.index.duplicated().sum()
        if duplicate_indices > 0:
            quality_issues.append(f"Found {duplicate_indices} duplicate timestamps")

        # Negative values check (where inappropriate)
        energy_columns = [col for col in data.columns if 'kWh' in col]
        negative_values = {}
        for col in energy_columns:
            if col in data.columns:
                neg_count = (data[col] < 0).sum()
                if neg_count > 0:
                    negative_values[col] = neg_count
                    if col == 'Production (kWh)':
                        quality_issues.append(f"Found {neg_count} negative production values")

        # Unrealistic values check
        unrealistic_values = {}
        if 'Production (kWh)' in data.columns:
            # Check for extremely high production values (>2kW per 15min = 0.5kWh)
            high_production = (data['Production (kWh)'] > 0.5).sum()
            if high_production > 0:
                warnings.append(f"Found {high_production} potentially high production values (>0.5kWh per 15min)")

        # Data continuity check
        if hasattr(data.index, 'freq') and data.index.freq is None:
            time_gaps = self._detect_time_gaps(data)
            if time_gaps > 0:
                warnings.append(f"Detected {time_gaps} potential time gaps in data")

        return {
            'total_records': len(data),
            'missing_values': missing_values.to_dict(),
            'total_missing': int(total_missing),
            'duplicate_timestamps': int(duplicate_indices),
            'negative_values': negative_values,
            'quality_issues': quality_issues,
            'warnings': warnings,
            'overall_quality': 'Good' if len(quality_issues) == 0 else 'Issues Found'
        }

    def _detect_data_granularity(self, data: pd.DataFrame) -> str:
        """
        Detect the time granularity of the data

        Args:
            data: DataFrame with datetime index

        Returns:
            String describing the granularity
        """
        if len(data) < 2:
            return 'insufficient_data'

        # Calculate time differences
        time_diffs = data.index[1:] - data.index[:-1]
        most_common_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs[0]

        if most_common_diff == timedelta(minutes=15):
            return '15min'
        elif most_common_diff == timedelta(hours=1):
            return 'hourly'
        elif most_common_diff == timedelta(days=1):
            return 'daily'
        else:
            return f'custom_{most_common_diff}'

    def _detect_time_gaps(self, data: pd.DataFrame) -> int:
        """
        Detect gaps in time series data

        Args:
            data: DataFrame with datetime index

        Returns:
            Number of detected gaps
        """
        if len(data) < 2:
            return 0

        time_diffs = data.index[1:] - data.index[:-1]
        expected_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else timedelta(minutes=15)

        # Count differences that are significantly larger than expected
        gaps = (time_diffs > expected_diff * 1.5).sum()
        return int(gaps)

    def print_comprehensive_summary(self, data: pd.DataFrame, daily_data: pd.DataFrame,
                                  data_type: str = "Solar Data") -> None:
        """
        Print comprehensive data summary

        Args:
            data: Raw solar data (15-minute intervals)
            daily_data: Daily aggregated data
            data_type: Description of the data
        """
        # Dataset overview
        overview = self.generate_dataset_overview(data, data_type)

        print(f"📊 {data_type} Comprehensive Summary")
        print("=" * 50)

        print(f"\n📈 Dataset Overview:")
        print(f"  Records: {overview['records']:,} ({overview['granularity']} intervals)")
        print(f"  Date range: {overview['date_range']['start']} to {overview['date_range']['end']}")
        print(f"  Duration: {overview['date_range']['duration_days']} days")

        if overview['completeness']['completeness_pct']:
            print(f"  Data completeness: {overview['completeness']['completeness_pct']:.1f}%")

        # Production statistics
        if not daily_data.empty:
            prod_stats = self.analyze_production_statistics(daily_data)
            if 'basic_statistics' in prod_stats:
                stats = prod_stats['basic_statistics']
                print(f"\n⚡ Production Statistics:")
                print(f"  Total production: {stats['total_production']:,.0f} kWh")
                print(f"  Average daily: {stats['average_daily']:.1f} kWh")
                print(f"  Peak day: {stats['max_daily']:.1f} kWh")
                print(f"  Minimum day: {stats['min_daily']:.1f} kWh")

                consistency = prod_stats['consistency_metrics']
                print(f"  Productive days: {consistency['productive_days_pct']:.1f}%")

            # Consumption analysis
            consumption_analysis = self.analyze_consumption_patterns(daily_data)
            if 'consumption_statistics' in consumption_analysis:
                cons_stats = consumption_analysis['consumption_statistics']
                print(f"\n🏠 Consumption Statistics:")
                print(f"  Total consumption: {cons_stats['total_consumption']:,.0f} kWh")
                print(f"  Average daily: {cons_stats['average_daily']:.1f} kWh")

                grid_stats = consumption_analysis.get('grid_interaction', {})
                if 'self_consumption_rate' in grid_stats:
                    print(f"  Self-consumption rate: {grid_stats['self_consumption_rate']:.1f}%")
                    print(f"  Grid independence: {grid_stats['grid_independence_rate']:.1f}%")

        # Data quality
        quality = self.check_data_quality(data)
        print(f"\n🔍 Data Quality:")
        print(f"  Overall quality: {quality['overall_quality']}")
        print(f"  Missing values: {quality['total_missing']:,}")

        if quality['quality_issues']:
            print(f"  Issues found: {len(quality['quality_issues'])}")
            for issue in quality['quality_issues'][:3]:  # Show first 3 issues
                print(f"    • {issue}")

        if quality['warnings']:
            print(f"  Warnings: {len(quality['warnings'])}")


# Convenience functions

def quick_data_summary(data: pd.DataFrame, daily_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Generate quick summary statistics

    Args:
        data: Raw solar data
        daily_data: Optional daily aggregated data

    Returns:
        Summary statistics dictionary
    """
    summarizer = SolarDataSummary()

    overview = summarizer.generate_dataset_overview(data)
    quality = summarizer.check_data_quality(data)

    result = {
        'overview': overview,
        'quality': quality
    }

    if daily_data is not None and not daily_data.empty:
        result['production'] = summarizer.analyze_production_statistics(daily_data)
        result['consumption'] = summarizer.analyze_consumption_patterns(daily_data)

    return result


def print_quick_stats(data: pd.DataFrame, title: str = "Solar Data") -> None:
    """
    Print quick data statistics

    Args:
        data: Solar data DataFrame
        title: Title for the summary
    """
    if data.empty:
        print(f"{title}: No data available")
        return

    print(f"📊 {title} Quick Stats:")
    print(f"  Shape: {data.shape}")
    print(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"  Duration: {(data.index.max() - data.index.min()).days} days")

    if 'Production (kWh)' in data.columns:
        production = data['Production (kWh)'].sum()
        print(f"  Total production: {production:,.1f} kWh")

    missing = data.isnull().sum().sum()
    if missing > 0:
        print(f"  Missing values: {missing:,}")