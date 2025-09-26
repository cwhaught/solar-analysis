"""
SolarVisualizationSuite - Comprehensive dashboard and chart functions for solar energy analysis.

This module provides standardized visualization functions to eliminate duplicate plotting code
across notebooks and create consistent, professional solar energy dashboards.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
from pathlib import Path

from .plot_config import (
    SOLAR_COLORS, SOLAR_COLOR_PALETTE, FIGURE_STYLES,
    apply_solar_style, create_styled_figure, format_solar_axes
)


class SolarVisualizationSuite:
    """
    Comprehensive visualization suite for solar energy data analysis.

    Provides standardized dashboard functions and individual chart creation
    methods with consistent styling and professional appearance.
    """

    def __init__(self, style: str = 'default'):
        """Initialize the visualization suite with specified style."""
        self.style = style
        apply_solar_style(style)

    def create_production_overview_dashboard(self,
                                           daily_data: pd.DataFrame,
                                           hourly_data: Optional[pd.DataFrame] = None,
                                           title: str = "Solar Production Overview",
                                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive production overview dashboard.

        Args:
            daily_data: Daily solar production data with 'production', 'date' columns
            hourly_data: Optional hourly data for detailed views
            title: Dashboard title
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig = create_styled_figure(figsize=(16, 12))

        # Create subplot grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Main production trend (spans top row)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_production_trend(ax1, daily_data)

        # Monthly summary
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_monthly_summary(ax2, daily_data)

        # Production distribution
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_production_distribution(ax3, daily_data)

        # Performance metrics
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_performance_metrics(ax4, daily_data)

        # Weather correlation (if hourly data available)
        if hourly_data is not None:
            ax5 = fig.add_subplot(gs[2, :2])
            self._plot_weather_correlation(ax5, hourly_data)

            ax6 = fig.add_subplot(gs[2, 2])
            self._plot_hourly_patterns(ax6, hourly_data)
        else:
            # Seasonal patterns
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_seasonal_patterns(ax5, daily_data)

        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_seasonal_analysis_dashboard(self,
                                         daily_data: pd.DataFrame,
                                         title: str = "Seasonal Solar Analysis",
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive seasonal analysis dashboard.

        Args:
            daily_data: Daily solar data with date and production columns
            title: Dashboard title
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig = create_styled_figure(figsize=(15, 10))

        # Prepare seasonal data
        seasonal_data = self._prepare_seasonal_data(daily_data)

        # Create subplot layout
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Seasonal comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_seasonal_comparison(ax1, seasonal_data)

        # Monthly averages
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_monthly_averages(ax2, seasonal_data)

        # Day length correlation
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_daylight_correlation(ax3, seasonal_data)

        # Production variance
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_production_variance(ax4, seasonal_data)

        # Efficiency trends
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_efficiency_trends(ax5, seasonal_data)

        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.95)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_financial_analysis_dashboard(self,
                                          financial_data: Dict[str, Any],
                                          daily_data: pd.DataFrame,
                                          title: str = "Solar Financial Analysis",
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive financial analysis dashboard.

        Args:
            financial_data: Dictionary containing financial metrics
            daily_data: Daily production data for calculations
            title: Dashboard title
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig = create_styled_figure(figsize=(14, 10))

        # Create subplot layout
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # ROI projection
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_roi_projection(ax1, financial_data)

        # Savings breakdown
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_savings_breakdown(ax2, financial_data)

        # Monthly financial impact
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_monthly_financial_impact(ax3, daily_data, financial_data)

        # Payback timeline
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_payback_timeline(ax4, financial_data)

        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.95)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_weather_correlation_dashboard(self,
                                           weather_data: pd.DataFrame,
                                           production_data: pd.DataFrame,
                                           title: str = "Weather Impact Analysis",
                                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive weather correlation dashboard.

        Args:
            weather_data: Weather data with temperature, humidity, etc.
            production_data: Solar production data
            title: Dashboard title
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig = create_styled_figure(figsize=(14, 10))

        # Merge weather and production data
        merged_data = self._merge_weather_production_data(weather_data, production_data)

        # Create subplot layout
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Temperature correlation
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_temperature_correlation(ax1, merged_data)

        # Cloud cover impact
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_cloud_cover_impact(ax2, merged_data)

        # Humidity effects
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_humidity_effects(ax3, merged_data)

        # Weather pattern analysis
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_weather_pattern_analysis(ax4, merged_data)

        # Forecast accuracy
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_forecast_accuracy(ax5, merged_data)

        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.95)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    # Individual plotting functions

    def _plot_production_trend(self, ax: plt.Axes, daily_data: pd.DataFrame):
        """Plot main production trend over time."""
        dates = pd.to_datetime(daily_data['date'])
        production = daily_data['production']

        ax.plot(dates, production, color=SOLAR_COLORS['production'],
                linewidth=2, alpha=0.8, label='Daily Production')

        # Add 7-day moving average
        ma_7 = production.rolling(window=7, center=True).mean()
        ax.plot(dates, ma_7, color=SOLAR_COLORS['production'],
                linewidth=3, alpha=0.6, label='7-Day Average')

        format_solar_axes(ax, 'Date', 'Production (kWh)')
        ax.legend(loc='upper left')
        ax.set_title('Daily Solar Production Trend', fontweight='bold')

        # Format x-axis dates
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_monthly_summary(self, ax: plt.Axes, daily_data: pd.DataFrame):
        """Plot monthly production summary."""
        daily_data = daily_data.copy()
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data['month'] = daily_data['date'].dt.strftime('%b %Y')

        monthly_stats = daily_data.groupby('month')['production'].agg(['mean', 'sum']).reset_index()

        bars = ax.bar(range(len(monthly_stats)), monthly_stats['sum'],
                     color=SOLAR_COLORS['production'], alpha=0.7)

        ax.set_xticks(range(len(monthly_stats)))
        ax.set_xticklabels(monthly_stats['month'], rotation=45, ha='right')
        format_solar_axes(ax, 'Month', 'Total Production (kWh)')
        ax.set_title('Monthly Production Totals', fontweight='bold')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=8)

    def _plot_production_distribution(self, ax: plt.Axes, daily_data: pd.DataFrame):
        """Plot production distribution histogram."""
        production = daily_data['production']

        ax.hist(production, bins=30, color=SOLAR_COLORS['production'],
                alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add statistical lines
        mean_prod = production.mean()
        median_prod = production.median()

        ax.axvline(mean_prod, color='red', linestyle='--', alpha=0.8,
                  label=f'Mean: {mean_prod:.1f} kWh')
        ax.axvline(median_prod, color='orange', linestyle='--', alpha=0.8,
                  label=f'Median: {median_prod:.1f} kWh')

        format_solar_axes(ax, 'Daily Production (kWh)', 'Frequency')
        ax.set_title('Production Distribution', fontweight='bold')
        ax.legend()

    def _plot_performance_metrics(self, ax: plt.Axes, daily_data: pd.DataFrame):
        """Plot key performance metrics."""
        production = daily_data['production']

        # Calculate metrics
        metrics = {
            'Total\nProduction': f"{production.sum():.0f} kWh",
            'Average\nDaily': f"{production.mean():.1f} kWh",
            'Best\nDay': f"{production.max():.1f} kWh",
            'Capacity\nFactor': f"{(production.mean() / (10.0 * 24)) * 100:.1f}%"
        }

        # Create text display
        ax.axis('off')
        y_positions = np.linspace(0.8, 0.2, len(metrics))

        for i, (metric, value) in enumerate(metrics.items()):
            ax.text(0.1, y_positions[i], metric, fontsize=10,
                   fontweight='bold', transform=ax.transAxes)
            ax.text(0.6, y_positions[i], value, fontsize=12,
                   color=SOLAR_COLORS['production'], fontweight='bold',
                   transform=ax.transAxes)

        ax.set_title('Performance Metrics', fontweight='bold')

    def _plot_seasonal_patterns(self, ax: plt.Axes, daily_data: pd.DataFrame):
        """Plot seasonal production patterns."""
        daily_data = daily_data.copy()
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data['month'] = daily_data['date'].dt.month
        daily_data['day_of_year'] = daily_data['date'].dt.dayofyear

        # Create monthly boxplot
        monthly_data = [daily_data[daily_data['month'] == month]['production'].values
                       for month in range(1, 13)]

        box_plot = ax.boxplot(monthly_data, patch_artist=True)

        # Style boxes
        for patch in box_plot['boxes']:
            patch.set_facecolor(SOLAR_COLORS['production'])
            patch.set_alpha(0.7)

        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        format_solar_axes(ax, 'Month', 'Daily Production (kWh)')
        ax.set_title('Seasonal Production Patterns', fontweight='bold')

    def _prepare_seasonal_data(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with seasonal indicators."""
        data = daily_data.copy()
        data['date'] = pd.to_datetime(data['date'])

        # Add seasonal indicators
        data['month'] = data['date'].dt.month
        data['season'] = data['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })

        # Calculate day length approximation
        day_of_year = data['date'].dt.dayofyear
        data['daylight_hours'] = 12 + 4 * np.sin(2 * np.pi * (day_of_year - 81) / 365)

        return data

    def _plot_seasonal_comparison(self, ax: plt.Axes, seasonal_data: pd.DataFrame):
        """Plot seasonal production comparison."""
        seasonal_summary = seasonal_data.groupby('season')['production'].agg(['mean', 'std'])

        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        colors = [SOLAR_COLORS['production'], '#FFD700', '#FF8C00', '#4682B4']

        means = [seasonal_summary.loc[season, 'mean'] if season in seasonal_summary.index else 0
                for season in seasons]
        stds = [seasonal_summary.loc[season, 'std'] if season in seasonal_summary.index else 0
               for season in seasons]

        bars = ax.bar(seasons, means, yerr=stds, capsize=5,
                     color=colors, alpha=0.7, edgecolor='black')

        format_solar_axes(ax, 'Season', 'Average Daily Production (kWh)')
        ax.set_title('Seasonal Production Comparison', fontweight='bold')

        # Add value labels
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')

    def _plot_monthly_averages(self, ax: plt.Axes, seasonal_data: pd.DataFrame):
        """Plot monthly average production."""
        monthly_avg = seasonal_data.groupby('month')['production'].mean()

        months = range(1, 13)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        values = [monthly_avg.get(month, 0) for month in months]

        ax.plot(months, values, marker='o', linewidth=2, markersize=6,
               color=SOLAR_COLORS['production'])
        ax.fill_between(months, values, alpha=0.3, color=SOLAR_COLORS['production'])

        ax.set_xticks(months)
        ax.set_xticklabels(month_names, rotation=45)
        format_solar_axes(ax, 'Month', 'Average Production (kWh)')
        ax.set_title('Monthly Production Profile', fontweight='bold')

    def _plot_daylight_correlation(self, ax: plt.Axes, seasonal_data: pd.DataFrame):
        """Plot correlation between daylight hours and production."""
        if 'daylight_hours' in seasonal_data.columns:
            ax.scatter(seasonal_data['daylight_hours'], seasonal_data['production'],
                      alpha=0.6, color=SOLAR_COLORS['production'], s=20)

            # Add trend line
            z = np.polyfit(seasonal_data['daylight_hours'], seasonal_data['production'], 1)
            p = np.poly1d(z)
            ax.plot(seasonal_data['daylight_hours'], p(seasonal_data['daylight_hours']),
                   "r--", alpha=0.8, linewidth=2)

            # Calculate correlation
            corr = seasonal_data['daylight_hours'].corr(seasonal_data['production'])
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                   transform=ax.transAxes, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

            format_solar_axes(ax, 'Daylight Hours', 'Production (kWh)')
            ax.set_title('Daylight vs Production', fontweight='bold')

    def _plot_production_variance(self, ax: plt.Axes, seasonal_data: pd.DataFrame):
        """Plot production variance by season."""
        seasonal_var = seasonal_data.groupby('season')['production'].var()

        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        colors = ['#90EE90', '#FFD700', '#FF8C00', '#87CEEB']

        variances = [seasonal_var.get(season, 0) for season in seasons]

        bars = ax.bar(seasons, variances, color=colors, alpha=0.7, edgecolor='black')

        format_solar_axes(ax, 'Season', 'Production Variance')
        ax.set_title('Production Variability', fontweight='bold')

        for bar, var in zip(bars, variances):
            if var > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{var:.1f}', ha='center', va='bottom', fontsize=8)

    def _plot_efficiency_trends(self, ax: plt.Axes, seasonal_data: pd.DataFrame):
        """Plot efficiency trends over time."""
        # Calculate efficiency as production per daylight hour
        if 'daylight_hours' in seasonal_data.columns:
            seasonal_data['efficiency'] = seasonal_data['production'] / seasonal_data['daylight_hours']

            # Monthly efficiency
            monthly_eff = seasonal_data.groupby('month')['efficiency'].mean()

            months = range(1, 13)
            month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

            values = [monthly_eff.get(month, 0) for month in months]

            ax.plot(months, values, marker='s', linewidth=2, markersize=4,
                   color=SOLAR_COLORS['export'])

            ax.set_xticks(months)
            ax.set_xticklabels(month_names)
            format_solar_axes(ax, 'Month', 'kWh/Daylight Hour')
            ax.set_title('Solar Efficiency Trends', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Efficiency data\nnot available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic')
            ax.set_title('Solar Efficiency Trends', fontweight='bold')

    # Financial analysis helper functions

    def _plot_roi_projection(self, ax: plt.Axes, financial_data: Dict[str, Any]):
        """Plot ROI projection over time."""
        years = range(1, 26)  # 25 year projection
        system_cost = financial_data.get('system_cost', 25000)
        annual_savings = financial_data.get('annual_savings', 2000)

        cumulative_savings = [annual_savings * year for year in years]
        net_benefit = [savings - system_cost for savings in cumulative_savings]

        ax.plot(years, cumulative_savings, label='Cumulative Savings',
               color=SOLAR_COLORS['export'], linewidth=2)
        ax.axhline(y=system_cost, color=SOLAR_COLORS['import'],
                  linestyle='--', label=f'System Cost (${system_cost:,})')
        ax.plot(years, net_benefit, label='Net Benefit',
               color=SOLAR_COLORS['production'], linewidth=2)

        # Find payback period
        try:
            payback_year = next(year for year, benefit in zip(years, net_benefit) if benefit > 0)
            ax.axvline(x=payback_year, color='red', linestyle=':', alpha=0.7,
                      label=f'Payback: {payback_year} years')
        except StopIteration:
            pass

        format_solar_axes(ax, 'Years', 'Dollars ($)')
        ax.set_title('ROI Projection (25 Years)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_savings_breakdown(self, ax: plt.Axes, financial_data: Dict[str, Any]):
        """Plot savings breakdown pie chart."""
        categories = ['Federal Tax Credit', 'Energy Bill Savings', 'SREC Income', 'Other']

        federal_credit = financial_data.get('federal_tax_credit', 7500)
        annual_savings = financial_data.get('annual_savings', 2000)
        srec_income = financial_data.get('srec_annual', 500)
        other_incentives = financial_data.get('other_incentives', 1000)

        values = [federal_credit, annual_savings * 10, srec_income * 10, other_incentives]
        colors = [SOLAR_COLORS['production'], SOLAR_COLORS['export'],
                 SOLAR_COLORS['consumption'], '#FFA500']

        wedges, texts, autotexts = ax.pie(values, labels=categories, colors=colors,
                                         autopct='%1.1f%%', startangle=90)

        ax.set_title('Financial Benefits Breakdown\n(10-Year View)', fontweight='bold')

    def _plot_monthly_financial_impact(self, ax: plt.Axes, daily_data: pd.DataFrame,
                                     financial_data: Dict[str, Any]):
        """Plot monthly financial impact."""
        daily_data = daily_data.copy()
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data['month'] = daily_data['date'].dt.strftime('%b')

        rate_per_kwh = financial_data.get('electricity_rate', 0.12)
        monthly_savings = daily_data.groupby('month')['production'].sum() * rate_per_kwh

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        savings_values = [monthly_savings.get(month, 0) for month in months]

        # Use explicit categorical data to avoid matplotlib warnings
        import numpy as np
        month_positions = np.arange(len(months))
        bars = ax.bar(month_positions, savings_values, color=SOLAR_COLORS['export'], alpha=0.7)
        ax.set_xticks(month_positions)
        ax.set_xticklabels(months)

        format_solar_axes(ax, 'Month', 'Monthly Savings ($)')
        ax.set_title('Monthly Financial Impact', fontweight='bold')

        # Add value labels
        for bar, value in zip(bars, savings_values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'${value:.0f}', ha='center', va='bottom', fontsize=8)

    def _plot_payback_timeline(self, ax: plt.Axes, financial_data: Dict[str, Any]):
        """Plot payback timeline visualization."""
        system_cost = financial_data.get('system_cost', 25000)
        annual_savings = financial_data.get('annual_savings', 2000)
        federal_credit = financial_data.get('federal_tax_credit', 7500)

        # Effective cost after incentives
        effective_cost = system_cost - federal_credit
        payback_years = effective_cost / annual_savings if annual_savings > 0 else 25

        # Create timeline visualization
        years = ['Year 0', f'Year {int(payback_years)}', 'Year 25']
        values = [-effective_cost, 0, annual_savings * (25 - payback_years)]
        colors = [SOLAR_COLORS['import'], 'gray', SOLAR_COLORS['export']]

        # Use explicit categorical data to avoid matplotlib warnings
        import numpy as np
        year_positions = np.arange(len(years))
        bars = ax.bar(year_positions, values, color=colors, alpha=0.7)
        ax.set_xticks(year_positions)
        ax.set_xticklabels(years)

        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        format_solar_axes(ax, 'Timeline', 'Net Financial Position ($)')
        ax.set_title('Payback Timeline', fontweight='bold')

        # Add annotations
        ax.text(0, -effective_cost/2, f'Initial Investment\n${effective_cost:,.0f}',
               ha='center', va='center', fontweight='bold', fontsize=9)

        if len(years) > 2:
            ax.text(2, values[2]/2, f'Total Profit\n${values[2]:,.0f}',
                   ha='center', va='center', fontweight='bold', fontsize=9)

    # Weather correlation helper functions

    def _merge_weather_production_data(self, weather_data: pd.DataFrame,
                                     production_data: pd.DataFrame) -> pd.DataFrame:
        """Merge weather and production data for analysis."""
        # Ensure date columns are datetime
        weather_data = weather_data.copy()
        production_data = production_data.copy()

        if 'date' in weather_data.columns:
            weather_data['date'] = pd.to_datetime(weather_data['date'])
        if 'date' in production_data.columns:
            production_data['date'] = pd.to_datetime(production_data['date'])

        # Merge on date
        merged = pd.merge(production_data, weather_data, on='date', how='inner')
        return merged

    def _plot_temperature_correlation(self, ax: plt.Axes, merged_data: pd.DataFrame):
        """Plot temperature vs production correlation."""
        if 'temperature' in merged_data.columns:
            ax.scatter(merged_data['temperature'], merged_data['production'],
                      alpha=0.6, color=SOLAR_COLORS['production'], s=20)

            # Add trend line
            if len(merged_data) > 1:
                z = np.polyfit(merged_data['temperature'], merged_data['production'], 1)
                p = np.poly1d(z)
                ax.plot(merged_data['temperature'], p(merged_data['temperature']),
                       "r--", alpha=0.8, linewidth=2)

                # Calculate correlation
                corr = merged_data['temperature'].corr(merged_data['production'])
                ax.text(0.05, 0.95, f'R: {corr:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))

            format_solar_axes(ax, 'Temperature (°F)', 'Production (kWh)')
            ax.set_title('Temperature Impact', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Temperature data\nnot available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, style='italic')
            ax.set_title('Temperature Impact', fontweight='bold')

    def _plot_cloud_cover_impact(self, ax: plt.Axes, merged_data: pd.DataFrame):
        """Plot cloud cover impact on production."""
        cloud_cols = [col for col in merged_data.columns if 'cloud' in col.lower()]

        if cloud_cols:
            cloud_col = cloud_cols[0]

            # Create cloud cover bins
            merged_data['cloud_bin'] = pd.cut(merged_data[cloud_col],
                                            bins=[0, 25, 50, 75, 100],
                                            labels=['Clear', 'Partly Cloudy', 'Cloudy', 'Overcast'])

            cloud_impact = merged_data.groupby('cloud_bin', observed=True)['production'].mean()

            bars = ax.bar(cloud_impact.index, cloud_impact.values,
                         color=['#FFD700', '#FFA500', '#FF8C00', '#808080'],
                         alpha=0.7)

            format_solar_axes(ax, 'Cloud Cover', 'Avg Production (kWh)')
            ax.set_title('Cloud Cover Impact', fontweight='bold')

            for bar, value in zip(bars, cloud_impact.values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Cloud cover data\nnot available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, style='italic')
            ax.set_title('Cloud Cover Impact', fontweight='bold')

    def _plot_humidity_effects(self, ax: plt.Axes, merged_data: pd.DataFrame):
        """Plot humidity effects on solar production."""
        if 'humidity' in merged_data.columns:
            # Create humidity bins
            merged_data['humidity_bin'] = pd.cut(merged_data['humidity'],
                                               bins=[0, 40, 60, 80, 100],
                                               labels=['Low', 'Medium', 'High', 'Very High'])

            humidity_impact = merged_data.groupby('humidity_bin', observed=True)['production'].mean()

            bars = ax.bar(humidity_impact.index, humidity_impact.values,
                         color=SOLAR_COLORS['consumption'], alpha=0.7)

            format_solar_axes(ax, 'Humidity Level', 'Avg Production (kWh)')
            ax.set_title('Humidity Effects', fontweight='bold')

            for bar, value in zip(bars, humidity_impact.values):
                if not np.isnan(value):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Humidity data\nnot available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, style='italic')
            ax.set_title('Humidity Effects', fontweight='bold')

    def _plot_weather_pattern_analysis(self, ax: plt.Axes, merged_data: pd.DataFrame):
        """Plot comprehensive weather pattern analysis."""
        # Create weather categories based on multiple factors
        conditions = []
        for _, row in merged_data.iterrows():
            temp = row.get('temperature', 70)
            clouds = row.get('cloud_cover', row.get('clouds', 50))

            if temp > 80 and clouds < 25:
                conditions.append('Hot & Sunny')
            elif temp > 80 and clouds >= 25:
                conditions.append('Hot & Cloudy')
            elif temp <= 80 and clouds < 25:
                conditions.append('Mild & Sunny')
            else:
                conditions.append('Mild & Cloudy')

        merged_data['weather_condition'] = conditions

        # Calculate average production by condition
        condition_impact = merged_data.groupby('weather_condition')['production'].mean()

        bars = ax.bar(condition_impact.index, condition_impact.values,
                     color=SOLAR_COLORS['production'], alpha=0.7)

        format_solar_axes(ax, 'Weather Condition', 'Avg Production (kWh)')
        ax.set_title('Weather Pattern Analysis', fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar, value in zip(bars, condition_impact.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{value:.1f}', ha='center', va='bottom', fontsize=8)

    def _plot_forecast_accuracy(self, ax: plt.Axes, merged_data: pd.DataFrame):
        """Plot forecast accuracy metrics."""
        forecast_cols = [col for col in merged_data.columns if 'forecast' in col.lower()]

        if forecast_cols and len(forecast_cols) > 0:
            forecast_col = forecast_cols[0]
            actual_col = 'production'

            # Calculate accuracy metrics
            mae = np.mean(np.abs(merged_data[forecast_col] - merged_data[actual_col]))
            rmse = np.sqrt(np.mean((merged_data[forecast_col] - merged_data[actual_col])**2))

            # Plot actual vs forecast
            ax.scatter(merged_data[actual_col], merged_data[forecast_col],
                      alpha=0.6, color=SOLAR_COLORS['production'], s=20)

            # Perfect prediction line
            min_val = min(merged_data[actual_col].min(), merged_data[forecast_col].min())
            max_val = max(merged_data[actual_col].max(), merged_data[forecast_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

            format_solar_axes(ax, 'Actual Production (kWh)', 'Forecast (kWh)')
            ax.set_title('Forecast Accuracy', fontweight='bold')

            # Add accuracy metrics
            ax.text(0.05, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}',
                   transform=ax.transAxes, fontweight='bold',
                   bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'Forecast data\nnot available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, style='italic')
            ax.set_title('Forecast Accuracy', fontweight='bold')

    def _plot_weather_correlation(self, ax: plt.Axes, hourly_data: pd.DataFrame):
        """Plot weather correlation for hourly data."""
        # This is a placeholder for hourly weather correlation
        ax.text(0.5, 0.5, 'Hourly Weather\nCorrelation\n(Implementation pending)',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, style='italic')
        ax.set_title('Weather Correlation', fontweight='bold')

    def _plot_hourly_patterns(self, ax: plt.Axes, hourly_data: pd.DataFrame):
        """Plot hourly production patterns."""
        if 'hour' in hourly_data.columns and 'production' in hourly_data.columns:
            hourly_avg = hourly_data.groupby('hour')['production'].mean()

            ax.plot(hourly_avg.index, hourly_avg.values,
                   color=SOLAR_COLORS['production'], linewidth=2, marker='o')
            ax.fill_between(hourly_avg.index, hourly_avg.values,
                           alpha=0.3, color=SOLAR_COLORS['production'])

            format_solar_axes(ax, 'Hour of Day', 'Avg Production (kWh)')
            ax.set_title('Hourly Production Pattern', fontweight='bold')
            ax.set_xlim(0, 23)
            ax.set_xticks(range(0, 24, 4))
        else:
            ax.text(0.5, 0.5, 'Hourly pattern data\nnot available',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, style='italic')
            ax.set_title('Hourly Production Pattern', fontweight='bold')