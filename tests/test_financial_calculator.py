"""
Tests for financial calculator module
"""

import pytest
import pandas as pd
from datetime import datetime

# Add src to path for imports
import sys
sys.path.append('src')

from analysis.financial_calculator import SolarFinancialCalculator, calculate_quick_roi


class TestSolarFinancialCalculator:
    """Test the SolarFinancialCalculator class"""

    def setup_method(self):
        """Set up test data"""
        self.calculator = SolarFinancialCalculator()

        # Create sample daily data
        dates = pd.date_range('2024-01-01', periods=365, freq='D')
        self.daily_data = pd.DataFrame({
            'Production (kWh)': [25.0] * 365,  # 25 kWh/day average
            'Consumption (kWh)': [30.0] * 365,  # 30 kWh/day average
            'Export (kWh)': [10.0] * 365,      # 10 kWh/day exported
            'Import (kWh)': [15.0] * 365       # 15 kWh/day imported
        }, index=dates)

        # Sample electricity rates
        self.electricity_rates = {
            'annual_cost_per_kwh': 0.12,
            'feed_in_rate_per_kwh': 0.08,
            'residential_rate': 12.0,
            'feed_in_tariff': 8.0,
            'national_comparison': {'is_above_average': False, 'vs_national_avg': -1.0}
        }

        # Sample system config
        self.system_config = {
            'system_size_kw': 10.0,
            'total_cost': 25000,
            'net_system_cost': 17500,
            'total_rebates': 7500
        }

    def test_init(self):
        """Test calculator initialization"""
        calc = SolarFinancialCalculator()
        assert calc.annual_degradation == 0.005
        assert calc.system_lifetime == 25

    def test_calculate_annual_metrics(self):
        """Test annual metrics calculation"""
        metrics = self.calculator.calculate_annual_metrics(self.daily_data, 365)

        assert metrics['annual_production_kwh'] == pytest.approx(9131.25)  # 25 * 365.25
        assert metrics['annual_consumption_kwh'] == pytest.approx(10957.5)  # 30 * 365.25
        assert metrics['annual_export_kwh'] == pytest.approx(3652.5)  # 10 * 365.25
        assert metrics['annual_import_kwh'] == pytest.approx(5478.75)  # 15 * 365.25
        assert metrics['daily_average_kwh'] == pytest.approx(25.0)

        # Self-consumption rate: (production - export) / production * 100
        expected_self_consumption = ((25 - 10) / 25) * 100
        assert metrics['self_consumption_rate'] == pytest.approx(expected_self_consumption)

        # Grid independence: (1 - import / consumption) * 100
        expected_grid_independence = (1 - 15 / 30) * 100
        assert metrics['grid_independence_rate'] == pytest.approx(expected_grid_independence)

    def test_calculate_financial_benefits(self):
        """Test financial benefits calculation"""
        annual_metrics = {
            'annual_production_kwh': 9125.0,
            'annual_export_kwh': 3650.0,
            'annual_import_kwh': 5475.0,
            'annual_consumption_kwh': 10950.0
        }

        benefits = self.calculator.calculate_financial_benefits(annual_metrics, self.electricity_rates)

        # Export income: 3650 * 0.08 = 292
        assert benefits['annual_export_income'] == pytest.approx(292.0)

        # Import savings: (9125 - 3650) * 0.12 = 5475 * 0.12 = 657
        assert benefits['annual_import_savings'] == pytest.approx(657.0)

        # Total savings: 292 + 657 = 949
        assert benefits['total_annual_savings'] == pytest.approx(949.0)

        # Without solar cost: 10950 * 0.12 = 1314
        assert benefits['without_solar_cost'] == pytest.approx(1314.0)

    def test_calculate_roi_metrics(self):
        """Test ROI metrics calculation"""
        annual_savings = 949.0
        roi_metrics = self.calculator.calculate_roi_metrics(annual_savings, self.system_config)

        # Simple payback: 17500 / 949 ≈ 18.4 years
        assert roi_metrics['simple_payback_years'] == pytest.approx(18.4, rel=1e-1)

        assert roi_metrics['net_system_cost'] == 17500
        assert roi_metrics['total_system_cost'] == 25000
        assert roi_metrics['total_rebates'] == 7500

    def test_calculate_lifetime_savings(self):
        """Test lifetime savings calculation with degradation"""
        annual_savings = 1000.0
        lifetime_savings = self.calculator._calculate_lifetime_savings(annual_savings)

        # Should be less than 25 * 1000 due to degradation
        assert lifetime_savings < 25000.0
        assert lifetime_savings > 20000.0  # But still substantial

    def test_generate_comprehensive_analysis(self):
        """Test comprehensive analysis generation"""
        analysis = self.calculator.generate_comprehensive_analysis(
            self.daily_data, self.electricity_rates, self.system_config
        )

        # Check all major sections are present
        assert 'analysis_period' in analysis
        assert 'annual_metrics' in analysis
        assert 'financial_benefits' in analysis
        assert 'roi_metrics' in analysis
        assert 'capacity_factor' in analysis

        # Check analysis period
        assert analysis['analysis_period']['days_analyzed'] == 365

        # Check capacity factor calculation
        # 9131.25 kWh / (10 kW * 8760 hours) * 100
        expected_capacity_factor = (9131.25 / (10.0 * 8760)) * 100
        assert analysis['capacity_factor'] == pytest.approx(expected_capacity_factor)

    def test_print_financial_summary(self, capsys):
        """Test financial summary printing"""
        analysis = self.calculator.generate_comprehensive_analysis(
            self.daily_data, self.electricity_rates, self.system_config
        )

        self.calculator.print_financial_summary(analysis, "Test Location")

        captured = capsys.readouterr()
        assert "Test Location" in captured.out
        assert "System Performance" in captured.out
        assert "Annual Financial Benefits" in captured.out
        assert "Return on Investment" in captured.out

    def test_empty_data(self):
        """Test behavior with empty data"""
        empty_data = pd.DataFrame(columns=['Production (kWh)', 'Consumption (kWh)', 'Export (kWh)', 'Import (kWh)'])

        metrics = self.calculator.calculate_annual_metrics(empty_data, 1)  # Use 1 day to avoid division by zero

        # Should handle empty data gracefully
        assert metrics['annual_production_kwh'] == 0
        assert metrics['self_consumption_rate'] == 0

    def test_zero_division_protection(self):
        """Test protection against zero division"""
        # Data with zero production
        zero_production_data = self.daily_data.copy()
        zero_production_data['Production (kWh)'] = 0

        metrics = self.calculator.calculate_annual_metrics(zero_production_data, 365)

        # Should not raise division by zero error
        assert metrics['self_consumption_rate'] == 0

    def test_system_config_without_size(self):
        """Test analysis without system size in config"""
        config_no_size = self.system_config.copy()
        del config_no_size['system_size_kw']

        analysis = self.calculator.generate_comprehensive_analysis(
            self.daily_data, self.electricity_rates, config_no_size
        )

        # Capacity factor should be None when system size not provided
        assert analysis['capacity_factor'] is None


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_calculate_quick_roi(self):
        """Test quick ROI calculation function"""
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        daily_data = pd.DataFrame({
            'Production (kWh)': [20.0] * 100
        }, index=dates)

        annual_savings = 800.0
        net_system_cost = 15000.0

        result = calculate_quick_roi(daily_data, annual_savings, net_system_cost)

        # Check result structure
        assert 'annual_production_kwh' in result
        assert 'annual_savings' in result
        assert 'payback_years' in result
        assert 'lifetime_savings' in result
        assert 'roi_percentage' in result

        # Check values
        expected_annual_production = 20.0 * 365.25  # Annualized
        assert result['annual_production_kwh'] == pytest.approx(expected_annual_production)
        assert result['annual_savings'] == annual_savings
        assert result['payback_years'] == pytest.approx(15000.0 / 800.0)

    def test_calculate_quick_roi_zero_savings(self):
        """Test quick ROI with zero savings"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        daily_data = pd.DataFrame({
            'Production (kWh)': [20.0] * 100
        }, index=dates)

        result = calculate_quick_roi(daily_data, 0.0, 15000.0)

        # Should handle zero savings gracefully
        assert result['payback_years'] == float('inf')
        assert result['roi_percentage'] < 0  # Negative ROI