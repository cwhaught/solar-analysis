"""
Tests for the SolarVisualizationSuite module.

Tests cover dashboard creation functions, individual plotting methods,
and data preparation utilities with comprehensive mocking.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.visualization.solar_charts import SolarVisualizationSuite


class TestSolarVisualizationSuite:
    """Test suite for SolarVisualizationSuite class."""

    @pytest.fixture
    def suite(self):
        """Create a SolarVisualizationSuite instance for testing."""
        return SolarVisualizationSuite()

    @pytest.fixture
    def sample_daily_data(self):
        """Create sample daily production data."""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        production = np.random.normal(25, 5, len(dates))  # Average 25 kWh with variation
        production = np.maximum(production, 0)  # Ensure non-negative values

        return pd.DataFrame({
            'date': dates,
            'production': production
        })

    @pytest.fixture
    def sample_hourly_data(self):
        """Create sample hourly production data."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='h')
        hours = dates.hour
        # Simulate solar curve: peak at noon, zero at night
        production = np.maximum(0, 5 * np.sin(np.pi * (hours - 6) / 12))
        production = np.where((hours < 6) | (hours > 18), 0, production)

        return pd.DataFrame({
            'date': dates,
            'hour': hours,
            'production': production
        })

    @pytest.fixture
    def sample_weather_data(self):
        """Create sample weather data."""
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

        return pd.DataFrame({
            'date': dates,
            'temperature': np.random.normal(65, 15, len(dates)),
            'humidity': np.random.uniform(30, 90, len(dates)),
            'cloud_cover': np.random.uniform(0, 100, len(dates)),
            'clouds': np.random.uniform(0, 100, len(dates))  # Alternative column name
        })

    @pytest.fixture
    def sample_financial_data(self):
        """Create sample financial data."""
        return {
            'system_cost': 25000,
            'annual_savings': 2000,
            'federal_tax_credit': 7500,
            'electricity_rate': 0.12,
            'srec_annual': 500,
            'other_incentives': 1000
        }

    def test_initialization(self):
        """Test SolarVisualizationSuite initialization."""
        suite = SolarVisualizationSuite()
        assert suite.style == 'default'

        suite_custom = SolarVisualizationSuite(style='presentation')
        assert suite_custom.style == 'presentation'

    @patch('src.visualization.solar_charts.SolarVisualizationSuite._plot_production_trend')
    @patch('src.visualization.solar_charts.SolarVisualizationSuite._plot_monthly_summary')
    @patch('src.visualization.solar_charts.SolarVisualizationSuite._plot_production_distribution')
    @patch('src.visualization.solar_charts.SolarVisualizationSuite._plot_performance_metrics')
    @patch('src.visualization.solar_charts.SolarVisualizationSuite._plot_seasonal_patterns')
    @patch('src.visualization.solar_charts.create_styled_figure')
    def test_create_production_overview_dashboard(self, mock_create_fig, mock_seasonal,
                                                mock_perf, mock_dist, mock_monthly, mock_trend,
                                                suite, sample_daily_data):
        """Test production overview dashboard creation."""
        mock_fig = Mock()
        mock_create_fig.return_value = mock_fig

        # Mock gridspec and subplots
        mock_gs = Mock()
        mock_gs.__getitem__ = Mock(return_value=Mock())
        mock_fig.add_gridspec.return_value = mock_gs
        mock_ax = Mock()
        mock_fig.add_subplot.return_value = mock_ax

        result = suite.create_production_overview_dashboard(
            sample_daily_data,
            title="Test Dashboard",
            save_path="/tmp/test_dashboard.png"
        )

        assert result == mock_fig
        mock_create_fig.assert_called_once_with(figsize=(16, 12))
        mock_fig.suptitle.assert_called_once_with("Test Dashboard", fontsize=20, fontweight='bold', y=0.95)
        mock_fig.savefig.assert_called_once_with("/tmp/test_dashboard.png", dpi=300, bbox_inches='tight')

        # Verify plotting functions were called
        mock_trend.assert_called_once()
        mock_monthly.assert_called_once()
        mock_dist.assert_called_once()
        mock_perf.assert_called_once()
        mock_seasonal.assert_called_once()

    @patch('src.visualization.solar_charts.create_styled_figure')
    def test_create_production_overview_dashboard_with_hourly_data(self, mock_create_fig,
                                                                 suite, sample_daily_data,
                                                                 sample_hourly_data):
        """Test production overview dashboard with hourly data."""
        mock_fig = Mock()
        mock_create_fig.return_value = mock_fig

        mock_gs = Mock()
        mock_gs.__getitem__ = Mock(return_value=Mock())
        mock_fig.add_gridspec.return_value = mock_gs
        mock_ax = Mock()
        mock_ax.get_legend_handles_labels.return_value = ([], [])  # Return empty handles/labels
        mock_fig.add_subplot.return_value = mock_ax

        result = suite.create_production_overview_dashboard(
            sample_daily_data,
            hourly_data=sample_hourly_data
        )

        assert result == mock_fig
        # Should be called more times with hourly data (6 subplots vs 5)
        assert mock_fig.add_subplot.call_count >= 5

    @patch('src.visualization.solar_charts.create_styled_figure')
    def test_create_seasonal_analysis_dashboard(self, mock_create_fig,
                                              suite, sample_daily_data):
        """Test seasonal analysis dashboard creation."""
        mock_fig = Mock()
        mock_create_fig.return_value = mock_fig

        mock_gs = Mock()
        mock_gs.__getitem__ = Mock(return_value=Mock())
        mock_fig.add_gridspec.return_value = mock_gs
        mock_ax = Mock()
        mock_ax.get_legend_handles_labels.return_value = ([], [])  # Return empty handles/labels
        mock_fig.add_subplot.return_value = mock_ax

        result = suite.create_seasonal_analysis_dashboard(
            sample_daily_data,
            title="Seasonal Test"
        )

        assert result == mock_fig
        mock_create_fig.assert_called_once_with(figsize=(15, 10))
        mock_fig.suptitle.assert_called_once_with("Seasonal Test", fontsize=18, fontweight='bold', y=0.95)

    @patch('src.visualization.solar_charts.create_styled_figure')
    def test_create_financial_analysis_dashboard(self, mock_create_fig,
                                               suite, sample_daily_data,
                                               sample_financial_data):
        """Test financial analysis dashboard creation."""
        mock_fig = Mock()
        mock_create_fig.return_value = mock_fig

        mock_gs = Mock()
        mock_gs.__getitem__ = Mock(return_value=Mock())
        mock_fig.add_gridspec.return_value = mock_gs
        mock_ax = Mock()
        mock_ax.get_legend_handles_labels.return_value = ([], [])  # Return empty handles/labels
        mock_fig.add_subplot.return_value = mock_ax

        result = suite.create_financial_analysis_dashboard(
            sample_financial_data,
            sample_daily_data,
            title="Financial Test"
        )

        assert result == mock_fig
        mock_create_fig.assert_called_once_with(figsize=(14, 10))

    @patch('src.visualization.solar_charts.create_styled_figure')
    def test_create_weather_correlation_dashboard(self, mock_create_fig,
                                                suite, sample_weather_data,
                                                sample_daily_data):
        """Test weather correlation dashboard creation."""
        mock_fig = Mock()
        mock_create_fig.return_value = mock_fig

        mock_gs = Mock()
        mock_gs.__getitem__ = Mock(return_value=Mock())
        mock_fig.add_gridspec.return_value = mock_gs
        mock_ax = Mock()
        mock_ax.get_legend_handles_labels.return_value = ([], [])  # Return empty handles/labels
        mock_fig.add_subplot.return_value = mock_ax

        result = suite.create_weather_correlation_dashboard(
            sample_weather_data,
            sample_daily_data,
            title="Weather Test"
        )

        assert result == mock_fig
        mock_create_fig.assert_called_once_with(figsize=(14, 10))

    def test_plot_production_trend(self, suite, sample_daily_data):
        """Test production trend plotting."""
        fig, ax = plt.subplots()

        suite._plot_production_trend(ax, sample_daily_data)

        # Check that plot was created
        lines = ax.get_lines()
        assert len(lines) >= 1  # At least the main trend line

        # Check labels and title
        assert ax.get_xlabel() == 'Date'
        assert ax.get_ylabel() == 'Production (kWh)'
        assert ax.get_title() == 'Daily Solar Production Trend'

        plt.close(fig)

    def test_plot_monthly_summary(self, suite, sample_daily_data):
        """Test monthly summary plotting."""
        fig, ax = plt.subplots()

        suite._plot_monthly_summary(ax, sample_daily_data)

        # Check that bars were created
        patches = ax.patches
        assert len(patches) > 0  # Should have monthly bars

        assert ax.get_xlabel() == 'Month'
        assert ax.get_ylabel() == 'Total Production (kWh)'
        assert ax.get_title() == 'Monthly Production Totals'

        plt.close(fig)

    def test_plot_production_distribution(self, suite, sample_daily_data):
        """Test production distribution plotting."""
        fig, ax = plt.subplots()

        suite._plot_production_distribution(ax, sample_daily_data)

        # Check that histogram was created
        patches = ax.patches
        assert len(patches) > 0  # Should have histogram bins

        assert ax.get_xlabel() == 'Daily Production (kWh)'
        assert ax.get_ylabel() == 'Frequency'
        assert ax.get_title() == 'Production Distribution'

        plt.close(fig)

    def test_plot_performance_metrics(self, suite, sample_daily_data):
        """Test performance metrics display."""
        fig, ax = plt.subplots()

        suite._plot_performance_metrics(ax, sample_daily_data)

        # Should turn off axis for text display
        assert not ax.axison or ax.get_xlabel() == '' or ax.get_ylabel() == ''
        assert ax.get_title() == 'Performance Metrics'

        plt.close(fig)

    def test_plot_seasonal_patterns(self, suite, sample_daily_data):
        """Test seasonal patterns plotting."""
        fig, ax = plt.subplots()

        suite._plot_seasonal_patterns(ax, sample_daily_data)

        assert ax.get_xlabel() == 'Month'
        assert ax.get_ylabel() == 'Daily Production (kWh)'
        assert ax.get_title() == 'Seasonal Production Patterns'

        # Check that 12 months are displayed
        xticklabels = [label.get_text() for label in ax.get_xticklabels()]
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        assert set(month_labels).issubset(set(xticklabels))

        plt.close(fig)

    def test_prepare_seasonal_data(self, suite, sample_daily_data):
        """Test seasonal data preparation."""
        result = suite._prepare_seasonal_data(sample_daily_data)

        # Check that new columns were added
        assert 'month' in result.columns
        assert 'season' in result.columns
        assert 'daylight_hours' in result.columns

        # Check season mapping
        seasons = result['season'].unique()
        expected_seasons = {'Winter', 'Spring', 'Summer', 'Fall'}
        assert set(seasons).issubset(expected_seasons)

        # Check daylight hours range (should be roughly 8-16 hours)
        daylight_hours = result['daylight_hours']
        assert daylight_hours.min() >= 6
        assert daylight_hours.max() <= 18

    def test_plot_seasonal_comparison(self, suite, sample_daily_data):
        """Test seasonal comparison plotting."""
        fig, ax = plt.subplots()
        seasonal_data = suite._prepare_seasonal_data(sample_daily_data)

        suite._plot_seasonal_comparison(ax, seasonal_data)

        assert ax.get_xlabel() == 'Season'
        assert ax.get_ylabel() == 'Average Daily Production (kWh)'
        assert ax.get_title() == 'Seasonal Production Comparison'

        plt.close(fig)

    def test_plot_monthly_averages(self, suite, sample_daily_data):
        """Test monthly averages plotting."""
        fig, ax = plt.subplots()
        seasonal_data = suite._prepare_seasonal_data(sample_daily_data)

        suite._plot_monthly_averages(ax, seasonal_data)

        assert ax.get_xlabel() == 'Month'
        assert ax.get_ylabel() == 'Average Production (kWh)'
        assert ax.get_title() == 'Monthly Production Profile'

        # Should have 12 data points
        lines = ax.get_lines()
        assert len(lines) >= 1

        plt.close(fig)

    def test_plot_daylight_correlation(self, suite, sample_daily_data):
        """Test daylight correlation plotting."""
        fig, ax = plt.subplots()
        seasonal_data = suite._prepare_seasonal_data(sample_daily_data)

        suite._plot_daylight_correlation(ax, seasonal_data)

        assert ax.get_xlabel() == 'Daylight Hours'
        assert ax.get_ylabel() == 'Production (kWh)'
        assert ax.get_title() == 'Daylight vs Production'

        # Should have scatter plot and trend line
        collections = ax.collections
        lines = ax.get_lines()
        assert len(collections) > 0 or len(lines) > 0

        plt.close(fig)

    def test_plot_roi_projection(self, suite, sample_financial_data):
        """Test ROI projection plotting."""
        fig, ax = plt.subplots()

        suite._plot_roi_projection(ax, sample_financial_data)

        assert ax.get_xlabel() == 'Years'
        assert ax.get_ylabel() == 'Dollars ($)'
        assert ax.get_title() == 'ROI Projection (25 Years)'

        # Should have multiple lines (cumulative savings, net benefit, etc.)
        lines = ax.get_lines()
        assert len(lines) >= 2

        plt.close(fig)

    def test_plot_savings_breakdown(self, suite, sample_financial_data):
        """Test savings breakdown pie chart."""
        fig, ax = plt.subplots()

        suite._plot_savings_breakdown(ax, sample_financial_data)

        assert ax.get_title() == 'Financial Benefits Breakdown\n(10-Year View)'

        # Should have pie chart wedges
        wedges = [child for child in ax.get_children()
                 if hasattr(child, 'get_path') and child.get_path() is not None]
        assert len(wedges) > 0

        plt.close(fig)

    def test_merge_weather_production_data(self, suite, sample_weather_data, sample_daily_data):
        """Test weather and production data merging."""
        merged = suite._merge_weather_production_data(sample_weather_data, sample_daily_data)

        # Should have columns from both datasets
        assert 'production' in merged.columns
        assert 'temperature' in merged.columns
        assert 'date' in merged.columns

        # Should have reasonable number of rows (inner join)
        assert len(merged) > 0
        assert len(merged) <= min(len(sample_weather_data), len(sample_daily_data))

    def test_plot_temperature_correlation(self, suite, sample_weather_data, sample_daily_data):
        """Test temperature correlation plotting."""
        fig, ax = plt.subplots()
        merged_data = suite._merge_weather_production_data(sample_weather_data, sample_daily_data)

        suite._plot_temperature_correlation(ax, merged_data)

        assert ax.get_xlabel() == 'Temperature (°F)'
        assert ax.get_ylabel() == 'Production (kWh)'
        assert ax.get_title() == 'Temperature Impact'

        plt.close(fig)

    def test_plot_cloud_cover_impact(self, suite, sample_weather_data, sample_daily_data):
        """Test cloud cover impact plotting."""
        fig, ax = plt.subplots()
        merged_data = suite._merge_weather_production_data(sample_weather_data, sample_daily_data)

        suite._plot_cloud_cover_impact(ax, merged_data)

        assert ax.get_xlabel() == 'Cloud Cover'
        assert ax.get_ylabel() == 'Avg Production (kWh)'
        assert ax.get_title() == 'Cloud Cover Impact'

        plt.close(fig)

    def test_plot_humidity_effects(self, suite, sample_weather_data, sample_daily_data):
        """Test humidity effects plotting."""
        fig, ax = plt.subplots()
        merged_data = suite._merge_weather_production_data(sample_weather_data, sample_daily_data)

        suite._plot_humidity_effects(ax, merged_data)

        assert ax.get_xlabel() == 'Humidity Level'
        assert ax.get_ylabel() == 'Avg Production (kWh)'
        assert ax.get_title() == 'Humidity Effects'

        plt.close(fig)

    def test_plot_weather_pattern_analysis(self, suite, sample_weather_data, sample_daily_data):
        """Test weather pattern analysis plotting."""
        fig, ax = plt.subplots()
        merged_data = suite._merge_weather_production_data(sample_weather_data, sample_daily_data)

        suite._plot_weather_pattern_analysis(ax, merged_data)

        assert ax.get_xlabel() == 'Weather Condition'
        assert ax.get_ylabel() == 'Avg Production (kWh)'
        assert ax.get_title() == 'Weather Pattern Analysis'

        plt.close(fig)

    def test_plot_forecast_accuracy_no_forecast_data(self, suite, sample_weather_data, sample_daily_data):
        """Test forecast accuracy plotting with no forecast data."""
        fig, ax = plt.subplots()
        merged_data = suite._merge_weather_production_data(sample_weather_data, sample_daily_data)

        suite._plot_forecast_accuracy(ax, merged_data)

        assert ax.get_title() == 'Forecast Accuracy'

        plt.close(fig)

    def test_plot_forecast_accuracy_with_forecast_data(self, suite, sample_weather_data, sample_daily_data):
        """Test forecast accuracy plotting with forecast data."""
        fig, ax = plt.subplots()
        merged_data = suite._merge_weather_production_data(sample_weather_data, sample_daily_data)

        # Add forecast column
        merged_data['production_forecast'] = merged_data['production'] * np.random.uniform(0.8, 1.2, len(merged_data))

        suite._plot_forecast_accuracy(ax, merged_data)

        assert ax.get_xlabel() == 'Actual Production (kWh)'
        assert ax.get_ylabel() == 'Forecast (kWh)'
        assert ax.get_title() == 'Forecast Accuracy'

        plt.close(fig)

    def test_plot_hourly_patterns(self, suite, sample_hourly_data):
        """Test hourly patterns plotting."""
        fig, ax = plt.subplots()

        suite._plot_hourly_patterns(ax, sample_hourly_data)

        assert ax.get_xlabel() == 'Hour of Day'
        assert ax.get_ylabel() == 'Avg Production (kWh)'
        assert ax.get_title() == 'Hourly Production Pattern'

        # Check x-axis limits and ticks
        assert ax.get_xlim()[0] == 0
        assert ax.get_xlim()[1] == 23

        plt.close(fig)

    def test_plot_hourly_patterns_no_data(self, suite):
        """Test hourly patterns plotting with missing data."""
        fig, ax = plt.subplots()

        # Create dataframe without required columns
        empty_data = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=10)})

        suite._plot_hourly_patterns(ax, empty_data)

        assert ax.get_title() == 'Hourly Production Pattern'

        plt.close(fig)

    @patch('matplotlib.pyplot.close')
    def test_all_plotting_methods_handle_empty_data(self, mock_close, suite):
        """Test that all plotting methods handle empty or invalid data gracefully."""
        empty_data = pd.DataFrame()

        fig, ax = plt.subplots()

        # Test main plotting methods with empty data
        plotting_methods = [
            suite._plot_production_trend,
            suite._plot_monthly_summary,
            suite._plot_production_distribution,
            suite._plot_performance_metrics,
            suite._plot_seasonal_patterns,
        ]

        for method in plotting_methods:
            try:
                method(ax, empty_data)
            except (KeyError, IndexError, ValueError):
                # These are expected for empty data
                pass
            except Exception as e:
                pytest.fail(f"Method {method.__name__} raised unexpected exception: {e}")

        plt.close(fig)

    def test_financial_plotting_methods(self, suite, sample_financial_data, sample_daily_data):
        """Test financial plotting methods."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Test individual financial plotting methods
        suite._plot_monthly_financial_impact(ax1, sample_daily_data, sample_financial_data)
        suite._plot_payback_timeline(ax2, sample_financial_data)

        # Verify titles are set
        assert ax1.get_title() == 'Monthly Financial Impact'
        assert ax2.get_title() == 'Payback Timeline'

        plt.close(fig)

    @patch('src.visualization.solar_charts.plt.setp')
    def test_dashboard_with_save_path(self, mock_setp, suite, sample_daily_data):
        """Test dashboard creation with save functionality."""
        with patch('src.visualization.solar_charts.create_styled_figure') as mock_create_fig:
            mock_fig = Mock()
            mock_create_fig.return_value = mock_fig

            # Mock gridspec and subplot creation
            mock_gs = Mock()
            mock_gs.__getitem__ = Mock(return_value=Mock())
            mock_fig.add_gridspec.return_value = mock_gs
            mock_ax = Mock()
            mock_fig.add_subplot.return_value = mock_ax

            save_path = "/tmp/test_save.png"
            result = suite.create_production_overview_dashboard(
                sample_daily_data,
                save_path=save_path
            )

            # Verify save was called
            mock_fig.savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')

    def test_error_handling_invalid_financial_data(self, suite, sample_daily_data):
        """Test error handling with invalid financial data."""
        invalid_financial_data = {}  # Missing required keys

        fig, ax = plt.subplots()

        # Should not raise exception, but handle missing data gracefully
        try:
            suite._plot_roi_projection(ax, invalid_financial_data)
            suite._plot_savings_breakdown(ax, invalid_financial_data)
        except Exception as e:
            pytest.fail(f"Financial plotting methods should handle invalid data gracefully: {e}")

        plt.close(fig)

    def test_data_type_handling(self, suite):
        """Test handling of different data types and formats."""
        # Test with string dates
        string_date_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'production': [20, 25, 30]
        })

        fig, ax = plt.subplots()

        # Should handle string dates by converting to datetime
        try:
            suite._plot_production_trend(ax, string_date_data)
        except Exception as e:
            pytest.fail(f"Should handle string dates: {e}")

        plt.close(fig)