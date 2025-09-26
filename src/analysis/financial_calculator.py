"""
Financial Analysis Calculator - Standardized solar ROI and financial calculations

Provides comprehensive financial analysis functions for solar energy systems,
including payback calculations, lifetime savings, and ROI analysis.
"""

import pandas as pd
from typing import Dict, Any, Tuple
from datetime import datetime


class SolarFinancialCalculator:
    """
    Comprehensive financial calculator for solar energy systems
    """

    def __init__(self):
        self.annual_degradation = 0.005  # 0.5% annual degradation
        self.system_lifetime = 25  # years

    def calculate_annual_metrics(self, daily_data: pd.DataFrame, days_analyzed: int) -> Dict[str, float]:
        """
        Calculate annualized metrics from daily production data

        Args:
            daily_data: DataFrame with daily production/consumption data
            days_analyzed: Number of days in the dataset

        Returns:
            Dictionary with annualized metrics
        """
        # Calculate totals from the period
        total_production = daily_data['Production (kWh)'].sum()
        total_consumption = daily_data['Consumption (kWh)'].sum()
        total_export = daily_data['Export (kWh)'].sum()
        total_import = daily_data['Import (kWh)'].sum()

        # Annualize the metrics
        annual_multiplier = 365.25 / days_analyzed if days_analyzed > 0 else 0

        return {
            'annual_production_kwh': total_production * annual_multiplier,
            'annual_consumption_kwh': total_consumption * annual_multiplier,
            'annual_export_kwh': total_export * annual_multiplier,
            'annual_import_kwh': total_import * annual_multiplier,
            'daily_average_kwh': total_production / days_analyzed,
            'self_consumption_rate': ((total_production - total_export) / total_production) * 100 if total_production > 0 else 0,
            'grid_independence_rate': (1 - total_import / total_consumption) * 100 if total_consumption > 0 else 0
        }

    def calculate_financial_benefits(self, annual_metrics: Dict[str, float],
                                   electricity_rates: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate annual financial benefits

        Args:
            annual_metrics: Dictionary with annualized production metrics
            electricity_rates: Dictionary with electricity rate information

        Returns:
            Dictionary with financial benefit calculations
        """
        electricity_rate = electricity_rates['annual_cost_per_kwh']
        feed_in_rate = electricity_rates['feed_in_rate_per_kwh']

        # Calculate financial benefits
        annual_export_income = annual_metrics['annual_export_kwh'] * feed_in_rate
        self_consumed_kwh = annual_metrics['annual_production_kwh'] - annual_metrics['annual_export_kwh']
        annual_import_savings = self_consumed_kwh * electricity_rate
        total_annual_savings = annual_export_income + annual_import_savings

        # Calculate what the cost would be without solar
        without_solar_cost = annual_metrics['annual_consumption_kwh'] * electricity_rate
        actual_grid_cost = annual_metrics['annual_import_kwh'] * electricity_rate
        net_electricity_cost = actual_grid_cost - annual_export_income

        return {
            'annual_export_income': annual_export_income,
            'annual_import_savings': annual_import_savings,
            'total_annual_savings': total_annual_savings,
            'without_solar_cost': without_solar_cost,
            'net_electricity_cost': net_electricity_cost,
            'solar_savings_vs_no_solar': without_solar_cost - net_electricity_cost
        }

    def calculate_roi_metrics(self, annual_savings: float, system_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate ROI and payback metrics

        Args:
            annual_savings: Total annual financial savings
            system_config: System configuration with costs and rebates

        Returns:
            Dictionary with ROI metrics
        """
        net_system_cost = system_config['net_system_cost']

        # Basic payback calculation
        simple_payback = net_system_cost / annual_savings if annual_savings > 0 else float('inf')

        # Lifetime analysis with degradation
        lifetime_savings = self._calculate_lifetime_savings(annual_savings)
        net_lifetime_benefit = lifetime_savings - net_system_cost
        roi_percentage = (net_lifetime_benefit / net_system_cost) * 100 if net_system_cost > 0 else 0

        return {
            'simple_payback_years': simple_payback,
            'lifetime_savings': lifetime_savings,
            'net_lifetime_benefit': net_lifetime_benefit,
            'roi_percentage': roi_percentage,
            'net_system_cost': net_system_cost,
            'total_system_cost': system_config['total_cost'],
            'total_rebates': system_config['total_rebates']
        }

    def _calculate_lifetime_savings(self, annual_savings: float) -> float:
        """
        Calculate lifetime savings accounting for system degradation

        Args:
            annual_savings: First year annual savings

        Returns:
            Total lifetime savings
        """
        lifetime_savings = 0
        yearly_savings = annual_savings

        for year in range(1, self.system_lifetime + 1):
            if year > 1:
                yearly_savings *= (1 - self.annual_degradation)
            lifetime_savings += yearly_savings

        return lifetime_savings

    def generate_comprehensive_analysis(self, daily_data: pd.DataFrame,
                                      electricity_rates: Dict[str, Any],
                                      system_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete financial analysis

        Args:
            daily_data: DataFrame with daily production data
            electricity_rates: Dictionary with electricity rates
            system_config: Dictionary with system configuration

        Returns:
            Comprehensive financial analysis results
        """
        days_analyzed = len(daily_data)

        # Calculate all metrics
        annual_metrics = self.calculate_annual_metrics(daily_data, days_analyzed)
        financial_benefits = self.calculate_financial_benefits(annual_metrics, electricity_rates)
        roi_metrics = self.calculate_roi_metrics(financial_benefits['total_annual_savings'], system_config)

        # Capacity factor calculation (if system size provided)
        capacity_factor = None
        if 'system_size_kw' in system_config:
            theoretical_max = system_config['system_size_kw'] * 8760  # kWh per year
            capacity_factor = (annual_metrics['annual_production_kwh'] / theoretical_max) * 100

        return {
            'analysis_period': {
                'days_analyzed': days_analyzed,
                'start_date': daily_data.index.min().strftime('%Y-%m-%d'),
                'end_date': daily_data.index.max().strftime('%Y-%m-%d')
            },
            'annual_metrics': annual_metrics,
            'financial_benefits': financial_benefits,
            'roi_metrics': roi_metrics,
            'capacity_factor': capacity_factor,
            'electricity_rates': electricity_rates
        }

    def print_financial_summary(self, analysis_results: Dict[str, Any], location_name: str = "Unknown") -> None:
        """
        Print comprehensive financial analysis summary

        Args:
            analysis_results: Results from generate_comprehensive_analysis
            location_name: Name of the system location
        """
        annual = analysis_results['annual_metrics']
        financial = analysis_results['financial_benefits']
        roi = analysis_results['roi_metrics']
        rates = analysis_results['electricity_rates']

        print(f"💰 Comprehensive Financial Analysis: {location_name}")
        print("=" * 60)

        # System performance
        print(f"\n⚡ System Performance:")
        print(f"  Annual production: {annual['annual_production_kwh']:,.0f} kWh")
        print(f"  Daily average: {annual['daily_average_kwh']:.1f} kWh")
        print(f"  Self-consumption rate: {annual['self_consumption_rate']:.1f}%")
        print(f"  Grid independence: {annual['grid_independence_rate']:.1f}%")

        if analysis_results['capacity_factor']:
            print(f"  Capacity factor: {analysis_results['capacity_factor']:.1f}%")

        # Financial benefits
        print(f"\n💵 Annual Financial Benefits:")
        print(f"  Export income: ${financial['annual_export_income']:,.0f}")
        print(f"  Import savings: ${financial['annual_import_savings']:,.0f}")
        print(f"  Total solar savings: ${financial['total_annual_savings']:,.0f}")
        print(f"  Cost without solar: ${financial['without_solar_cost']:,.0f}")
        print(f"  Net electricity cost: ${financial['net_electricity_cost']:,.0f}")

        # ROI metrics
        print(f"\n📈 Return on Investment:")
        print(f"  System cost (before rebates): ${roi['total_system_cost']:,.0f}")
        print(f"  Total rebates: ${roi['total_rebates']:,.0f}")
        print(f"  Net investment: ${roi['net_system_cost']:,.0f}")
        print(f"  Simple payback: {roi['simple_payback_years']:.1f} years")
        print(f"  25-year ROI: {roi['roi_percentage']:.0f}%")
        print(f"  Lifetime savings: ${roi['lifetime_savings']:,.0f}")

        # Rate context
        print(f"\n🔌 Electricity Rate Context:")
        print(f"  Purchase rate: {rates['residential_rate']:.2f}¢/kWh")
        print(f"  Feed-in rate: {rates['feed_in_tariff']:.2f}¢/kWh")

        if rates['national_comparison']['is_above_average']:
            print(f"  Rate advantage: {rates['national_comparison']['vs_national_avg']:+.2f}¢ above national average")
        else:
            print(f"  Rate impact: {rates['national_comparison']['vs_national_avg']:+.2f}¢ below national average")


def calculate_quick_roi(daily_data: pd.DataFrame, annual_savings: float, net_system_cost: float) -> Dict[str, float]:
    """
    Quick ROI calculation for simple use cases

    Args:
        daily_data: DataFrame with daily production data
        annual_savings: Total annual financial savings
        net_system_cost: Net system cost after rebates

    Returns:
        Basic ROI metrics
    """
    calculator = SolarFinancialCalculator()

    total_production = daily_data['Production (kWh)'].sum()
    days_analyzed = len(daily_data)
    annual_production = total_production * (365.25 / days_analyzed)

    payback_years = net_system_cost / annual_savings if annual_savings > 0 else float('inf')
    lifetime_savings = calculator._calculate_lifetime_savings(annual_savings)
    roi_percentage = ((lifetime_savings - net_system_cost) / net_system_cost) * 100

    return {
        'annual_production_kwh': annual_production,
        'annual_savings': annual_savings,
        'payback_years': payback_years,
        'lifetime_savings': lifetime_savings,
        'roi_percentage': roi_percentage
    }