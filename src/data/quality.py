"""
Data Quality Analysis and Validation

Standardizes data quality checks and reporting found across notebooks.
Provides consistent validation methodology and professional reporting
for solar energy data analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    Comprehensive data quality analysis for solar energy data.

    Consolidates quality check patterns found across notebooks:
    - Missing value analysis
    - Data completeness assessment
    - Outlier detection
    - Integrity validation
    - Professional reporting
    """

    def __init__(self, energy_columns: Optional[List[str]] = None):
        """
        Initialize quality checker with default energy columns.

        Args:
            energy_columns: List of energy-related columns to focus on
        """
        self.energy_columns = energy_columns or [
            'Production (kWh)', 'Consumption (kWh)',
            'Export (kWh)', 'Import (kWh)'
        ]

    def check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive data completeness metrics.

        Replaces the notebook pattern:
        ```python
        print("=== Missing values ===")
        print(csv_data.isnull().sum())
        expected_intervals = (csv_data.index.max() - csv_data.index.min()).days * 96
        completeness = len(csv_data) / expected_intervals * 100
        ```

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with completeness metrics
        """
        logger.debug("Analyzing data completeness")

        completeness = {
            'overall_metrics': {},
            'column_metrics': {},
            'temporal_metrics': {}
        }

        try:
            # Overall completeness metrics
            total_cells = df.size
            non_null_cells = df.count().sum()
            completeness['overall_metrics'] = {
                'total_cells': total_cells,
                'non_null_cells': non_null_cells,
                'completeness_pct': (non_null_cells / total_cells * 100) if total_cells > 0 else 0,
                'total_records': len(df),
                'total_columns': len(df.columns)
            }

            # Per-column completeness
            for col in df.columns:
                col_completeness = {
                    'missing_count': df[col].isnull().sum(),
                    'present_count': df[col].count(),
                    'completeness_pct': (df[col].count() / len(df) * 100) if len(df) > 0 else 0
                }
                completeness['column_metrics'][col] = col_completeness

            # Temporal completeness (for time series data)
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
                temporal_metrics = self._analyze_temporal_completeness(df)
                completeness['temporal_metrics'] = temporal_metrics

        except Exception as e:
            logger.error(f"Error analyzing completeness: {e}")
            completeness['error'] = str(e)

        return completeness

    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detailed missing value analysis.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with missing value analysis
        """
        logger.debug("Analyzing missing values")

        missing_analysis = {
            'summary': {},
            'patterns': {},
            'recommendations': []
        }

        try:
            # Basic missing value summary
            missing_counts = df.isnull().sum()
            missing_pcts = (missing_counts / len(df) * 100)

            missing_analysis['summary'] = {
                'columns_with_missing': (missing_counts > 0).sum(),
                'total_missing_values': missing_counts.sum(),
                'worst_column': missing_counts.idxmax() if missing_counts.max() > 0 else None,
                'worst_column_pct': missing_pcts.max() if len(missing_pcts) > 0 else 0
            }

            # Missing value patterns
            missing_analysis['patterns'] = {
                'by_column': missing_counts.to_dict(),
                'by_percentage': missing_pcts.to_dict()
            }

            # Generate recommendations
            recommendations = []
            for col, pct in missing_pcts.items():
                if pct > 50:
                    recommendations.append(f"Consider dropping column '{col}' ({pct:.1f}% missing)")
                elif pct > 10:
                    recommendations.append(f"Investigate column '{col}' ({pct:.1f}% missing)")

            missing_analysis['recommendations'] = recommendations

        except Exception as e:
            logger.error(f"Error analyzing missing values: {e}")
            missing_analysis['error'] = str(e)

        return missing_analysis

    def check_data_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data integrity and identify potential issues.

        Validates common solar data integrity rules:
        - Production values should be non-negative
        - Export should not exceed production
        - Import + Export should roughly equal |Production - Consumption|

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with integrity check results
        """
        logger.debug("Checking data integrity")

        integrity_results = {
            'issues': [],
            'warnings': [],
            'summary': {}
        }

        try:
            issue_count = 0

            # Check for negative production values
            if 'Production (kWh)' in df.columns:
                negative_production = (df['Production (kWh)'] < 0).sum()
                if negative_production > 0:
                    integrity_results['issues'].append({
                        'type': 'negative_production',
                        'count': negative_production,
                        'description': f'{negative_production} records with negative production'
                    })
                    issue_count += negative_production

            # Check for export exceeding production
            if all(col in df.columns for col in ['Production (kWh)', 'Export (kWh)']):
                excess_export = (df['Export (kWh)'] > df['Production (kWh)']).sum()
                if excess_export > 0:
                    integrity_results['issues'].append({
                        'type': 'excess_export',
                        'count': excess_export,
                        'description': f'{excess_export} records where export exceeds production'
                    })
                    issue_count += excess_export

            # Check energy balance (Production - Consumption should equal Export - Import)
            if all(col in df.columns for col in self.energy_columns):
                df_clean = df[self.energy_columns].dropna()
                if len(df_clean) > 0:
                    net_energy = df_clean['Production (kWh)'] - df_clean['Consumption (kWh)']
                    net_flow = df_clean['Export (kWh)'] - df_clean['Import (kWh)']
                    balance_diff = abs(net_energy - net_flow)

                    # Allow small tolerance for rounding errors
                    tolerance = 0.1  # kWh
                    balance_issues = (balance_diff > tolerance).sum()

                    if balance_issues > 0:
                        integrity_results['warnings'].append({
                            'type': 'energy_balance',
                            'count': balance_issues,
                            'description': f'{balance_issues} records with energy balance discrepancies > {tolerance} kWh'
                        })

            # Check for extreme outliers
            for col in self.energy_columns:
                if col in df.columns:
                    outliers = self._detect_outliers(df[col])
                    if outliers['count'] > 0:
                        integrity_results['warnings'].append({
                            'type': 'outliers',
                            'column': col,
                            'count': outliers['count'],
                            'description': f'{outliers["count"]} potential outliers in {col}'
                        })

            integrity_results['summary'] = {
                'total_issues': len(integrity_results['issues']),
                'total_warnings': len(integrity_results['warnings']),
                'data_quality_score': self._calculate_quality_score(df, integrity_results)
            }

        except Exception as e:
            logger.error(f"Error checking data integrity: {e}")
            integrity_results['error'] = str(e)

        return integrity_results

    def generate_quality_report(
        self,
        df: pd.DataFrame,
        include_recommendations: bool = True
    ) -> str:
        """
        Generate comprehensive formatted quality report for notebooks.

        Replaces the manual quality reporting found in notebooks with
        standardized, professional output.

        Args:
            df: DataFrame to analyze
            include_recommendations: Include actionable recommendations

        Returns:
            Formatted quality report string
        """
        logger.info("Generating comprehensive quality report")

        try:
            # Run all quality checks
            completeness = self.check_completeness(df)
            missing_values = self.check_missing_values(df)
            integrity = self.check_data_integrity(df)

            # Build formatted report
            report_lines = []
            report_lines.append("📊 DATA QUALITY REPORT")
            report_lines.append("=" * 50)

            # Dataset overview
            report_lines.append(f"\n📈 Dataset Overview:")
            report_lines.append(f"  • Records: {len(df):,}")
            report_lines.append(f"  • Columns: {len(df.columns)}")
            if isinstance(df.index, pd.DatetimeIndex):
                report_lines.append(f"  • Date range: {df.index.min().date()} to {df.index.max().date()}")

            # Completeness summary
            overall = completeness.get('overall_metrics', {})
            report_lines.append(f"\n📋 Data Completeness:")
            report_lines.append(f"  • Overall: {overall.get('completeness_pct', 0):.1f}%")
            report_lines.append(f"  • Total cells: {overall.get('total_cells', 0):,}")
            report_lines.append(f"  • Non-null cells: {overall.get('non_null_cells', 0):,}")

            # Missing values summary
            missing_summary = missing_values.get('summary', {})
            if missing_summary.get('total_missing_values', 0) > 0:
                report_lines.append(f"\n⚠️ Missing Values:")
                report_lines.append(f"  • Columns affected: {missing_summary.get('columns_with_missing', 0)}")
                report_lines.append(f"  • Total missing: {missing_summary.get('total_missing_values', 0):,}")
                worst_col = missing_summary.get('worst_column')
                worst_pct = missing_summary.get('worst_column_pct', 0)
                if worst_col:
                    report_lines.append(f"  • Worst column: {worst_col} ({worst_pct:.1f}% missing)")

            # Data integrity summary
            integrity_summary = integrity.get('summary', {})
            quality_score = integrity_summary.get('data_quality_score', 0)
            report_lines.append(f"\n🔍 Data Integrity:")
            report_lines.append(f"  • Quality score: {quality_score:.1f}/100")
            report_lines.append(f"  • Issues found: {integrity_summary.get('total_issues', 0)}")
            report_lines.append(f"  • Warnings: {integrity_summary.get('total_warnings', 0)}")

            # List specific issues
            if integrity.get('issues'):
                report_lines.append(f"\n❌ Critical Issues:")
                for issue in integrity['issues']:
                    report_lines.append(f"  • {issue.get('description', 'Unknown issue')}")

            if integrity.get('warnings'):
                report_lines.append(f"\n⚠️ Warnings:")
                for warning in integrity['warnings']:
                    report_lines.append(f"  • {warning.get('description', 'Unknown warning')}")

            # Recommendations
            if include_recommendations:
                recommendations = []

                # Add missing value recommendations
                recommendations.extend(missing_values.get('recommendations', []))

                # Add integrity recommendations
                if quality_score < 80:
                    recommendations.append("Review data collection process - quality score below 80")

                if recommendations:
                    report_lines.append(f"\n💡 Recommendations:")
                    for rec in recommendations:
                        report_lines.append(f"  • {rec}")

            # Quality status
            report_lines.append(f"\n✅ Overall Assessment:")
            if quality_score >= 90:
                report_lines.append("  🟢 Excellent data quality - ready for analysis")
            elif quality_score >= 75:
                report_lines.append("  🟡 Good data quality - minor issues to address")
            elif quality_score >= 60:
                report_lines.append("  🟠 Fair data quality - some concerns need attention")
            else:
                report_lines.append("  🔴 Poor data quality - significant issues require fixing")

            return "\n".join(report_lines)

        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return f"Error generating quality report: {e}"

    def _analyze_temporal_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal data completeness for time series."""
        temporal_metrics = {}

        try:
            if len(df) < 2:
                return temporal_metrics

            # Detect expected frequency
            time_diffs = df.index[1:] - df.index[:-1]
            # Use value_counts() instead of mode() for TimedeltaIndex compatibility
            freq_counts = time_diffs.value_counts()
            expected_freq = freq_counts.index[0] if len(freq_counts) > 0 else None

            if expected_freq:
                # Calculate expected vs actual records
                total_span = df.index.max() - df.index.min()
                expected_records = int(total_span / expected_freq) + 1
                temporal_completeness = (len(df) / expected_records * 100)

                temporal_metrics = {
                    'expected_frequency': str(expected_freq),
                    'expected_records': expected_records,
                    'actual_records': len(df),
                    'temporal_completeness_pct': temporal_completeness,
                    'missing_intervals': expected_records - len(df)
                }

                # Detect gaps
                gaps = self._detect_time_gaps(df, expected_freq)
                temporal_metrics['gaps_detected'] = len(gaps)
                if gaps:
                    temporal_metrics['largest_gap'] = str(max(gaps, key=lambda x: x[1] - x[0]))

        except Exception as e:
            logger.error(f"Error analyzing temporal completeness: {e}")
            temporal_metrics['error'] = str(e)

        return temporal_metrics

    def _detect_time_gaps(self, df: pd.DataFrame, expected_freq: timedelta) -> List[Tuple[datetime, datetime]]:
        """Detect gaps in time series data."""
        gaps = []
        tolerance = expected_freq * 1.5  # Allow some tolerance

        for i in range(1, len(df)):
            actual_diff = df.index[i] - df.index[i-1]
            if actual_diff > tolerance:
                gaps.append((df.index[i-1], df.index[i]))

        return gaps

    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        outlier_info = {'count': 0, 'indices': []}

        try:
            if len(series.dropna()) < 4:  # Need at least 4 values for IQR
                return outlier_info

            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (series < lower_bound) | (series > upper_bound)
            outlier_info['count'] = outliers.sum()
            outlier_info['indices'] = series[outliers].index.tolist()

        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")

        return outlier_info

    def _calculate_quality_score(self, df: pd.DataFrame, integrity_results: Dict) -> float:
        """Calculate overall data quality score (0-100)."""
        try:
            score = 100.0

            # Penalize for missing data
            missing_pct = (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0
            score -= missing_pct * 0.5  # 0.5 point per % missing

            # Penalize for integrity issues
            issues = len(integrity_results.get('issues', []))
            warnings = len(integrity_results.get('warnings', []))

            score -= issues * 10  # 10 points per critical issue
            score -= warnings * 2  # 2 points per warning

            return max(0.0, min(100.0, score))

        except Exception:
            return 0.0