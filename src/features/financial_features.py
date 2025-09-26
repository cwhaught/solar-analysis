"""
Financial Feature Engineering - Integrates with existing SolarFinancialCalculator

Creates ML-ready financial features by leveraging the comprehensive financial analysis
capabilities of the existing SolarFinancialCalculator.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from ..analysis.financial_calculator import SolarFinancialCalculator

logger = logging.getLogger(__name__)


class FinancialFeatureEngineer:
    """
    Financial feature engineering that integrates with existing SolarFinancialCalculator.

    Extracts ML-ready features from comprehensive financial analysis results,
    creating efficiency metrics and financial performance indicators.
    """

    def __init__(self):
        """Initialize financial feature engineer with existing calculator"""
        self.calculator = SolarFinancialCalculator()

    def create_financial_ml_features(
        self,
        daily_data: pd.DataFrame,
        electricity_rates: Dict[str, Any],
        system_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Create comprehensive financial features for ML models.

        Leverages existing SolarFinancialCalculator to extract features from
        financial analysis results.

        Args:
            daily_data: Solar production data with datetime index
            electricity_rates: Electricity rate information
            system_config: Solar system configuration

        Returns:
            DataFrame with financial features added
        """
        try:
            logger.info("Creating financial ML features")

            # Use existing comprehensive analysis
            analysis_results = self.calculator.generate_comprehensive_analysis(
                daily_data, electricity_rates, system_config
            )

            # Start with copy of daily data
            ml_data = daily_data.copy()

            # Add financial efficiency features
            ml_data = self._add_efficiency_features(ml_data, analysis_results)
            ml_data = self._add_financial_performance_features(ml_data, analysis_results, electricity_rates)
            ml_data = self._add_economic_indicators(ml_data, analysis_results)
            ml_data = self._add_daily_financial_features(ml_data, electricity_rates)

            logger.info(f"Created {len([c for c in ml_data.columns if 'financial' in c.lower() or 'efficiency' in c.lower()])} financial features")
            return ml_data

        except Exception as e:
            logger.error(f"Error creating financial ML features: {e}")
            # Graceful fallback - return original data
            return daily_data.copy()

    def _add_efficiency_features(
        self,
        data: pd.DataFrame,
        analysis_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Add efficiency features from existing financial analysis.

        Extracts key efficiency metrics that are predictive for ML models.
        """
        try:
            annual_metrics = analysis_results.get('annual_metrics', {})

            # System efficiency features (constant for all rows)
            data['capacity_factor'] = annual_metrics.get('capacity_factor', 0) / 100.0
            data['self_consumption_rate'] = annual_metrics.get('self_consumption_rate', 0) / 100.0
            data['grid_independence'] = annual_metrics.get('grid_independence_rate', 0) / 100.0

            # Daily efficiency variations
            if 'Production (kWh)' in data.columns:
                avg_daily_production = annual_metrics.get('daily_average_kwh', data['Production (kWh)'].mean())

                # Relative daily performance
                data['daily_performance_ratio'] = data['Production (kWh)'] / avg_daily_production
                data['efficiency_normalized'] = data['daily_performance_ratio'] * data['capacity_factor']

            return data

        except Exception as e:
            logger.error(f"Error adding efficiency features: {e}")
            return data

    def _add_financial_performance_features(
        self,
        data: pd.DataFrame,
        analysis_results: Dict[str, Any],
        electricity_rates: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Add financial performance features based on production and rates.
        """
        try:
            financial_benefits = analysis_results.get('financial_benefits', {})
            annual_metrics = analysis_results.get('annual_metrics', {})

            # Rate information
            purchase_rate = electricity_rates.get('residential_rate', 14.0) / 100.0  # Convert cents to dollars
            feed_in_rate = electricity_rates.get('feed_in_tariff', 11.0) / 100.0

            # Daily financial value features
            if all(col in data.columns for col in ['Production (kWh)', 'Consumption (kWh)', 'Export (kWh)', 'Import (kWh)']):
                # Daily savings from production
                data['daily_export_value'] = data['Export (kWh)'] * feed_in_rate
                data['daily_import_savings'] = (data['Consumption (kWh)'] - data['Import (kWh)']) * purchase_rate
                data['daily_total_savings'] = data['daily_export_value'] + data['daily_import_savings']

                # Financial efficiency ratios
                data['financial_efficiency'] = data['daily_total_savings'] / (data['Production (kWh)'] + 0.01)  # Avoid division by zero
                data['savings_rate'] = data['daily_total_savings'] / (data['Consumption (kWh)'] * purchase_rate + 0.01)

            # Annualized financial metrics (constant features)
            data['annual_savings_potential'] = financial_benefits.get('total_annual_savings', 0)
            data['annual_export_income_potential'] = financial_benefits.get('annual_export_income', 0)

            return data

        except Exception as e:
            logger.error(f"Error adding financial performance features: {e}")
            return data

    def _add_economic_indicators(
        self,
        data: pd.DataFrame,
        analysis_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Add economic indicator features from ROI analysis.
        """
        try:
            roi_metrics = analysis_results.get('roi_metrics', {})

            # ROI and payback features (constant for all rows)
            data['simple_payback_years'] = roi_metrics.get('simple_payback_years', 25)
            data['roi_percentage'] = roi_metrics.get('roi_percentage', 0) / 100.0
            data['lifetime_savings'] = roi_metrics.get('lifetime_savings', 0)

            # Economic efficiency indicators
            payback_years = data['simple_payback_years'].iloc[0] if len(data) > 0 else 25
            data['payback_efficiency'] = 1.0 / max(payback_years, 1)  # Inverse of payback (higher is better)

            # Financial health score (composite metric)
            roi_norm = np.clip(data['roi_percentage'] / 2.0, 0, 1)  # Normalize assuming 200% ROI is excellent
            payback_norm = np.clip(1 - (payback_years - 5) / 20, 0, 1)  # 5-25 year payback range
            data['financial_health_score'] = (roi_norm + payback_norm) / 2

            return data

        except Exception as e:
            logger.error(f"Error adding economic indicators: {e}")
            return data

    def _add_daily_financial_features(
        self,
        data: pd.DataFrame,
        electricity_rates: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Add daily financial features that vary with production patterns.
        """
        try:
            if 'Production (kWh)' not in data.columns:
                return data

            purchase_rate = electricity_rates.get('residential_rate', 14.0) / 100.0

            # Rolling financial metrics
            for window in [7, 14, 30]:
                # Rolling average daily savings
                if 'daily_total_savings' in data.columns:
                    data[f'rolling_savings_{window}d'] = data['daily_total_savings'].rolling(
                        window=window, min_periods=1
                    ).mean()

                # Rolling financial efficiency
                if 'financial_efficiency' in data.columns:
                    data[f'rolling_fin_efficiency_{window}d'] = data['financial_efficiency'].rolling(
                        window=window, min_periods=1
                    ).mean()

            # Financial volatility features
            if 'daily_total_savings' in data.columns:
                data['savings_volatility_7d'] = data['daily_total_savings'].rolling(7, min_periods=1).std()
                data['savings_trend_14d'] = data['daily_total_savings'].rolling(14, min_periods=1).apply(
                    lambda x: 1 if len(x) > 1 and x.iloc[-1] > x.iloc[0] else 0, raw=False
                )

            # Daily financial opportunity features
            theoretical_max_daily = data['Production (kWh)'].quantile(0.95)  # 95th percentile as max potential
            data['daily_financial_opportunity'] = (theoretical_max_daily - data['Production (kWh)']) * purchase_rate
            data['opportunity_ratio'] = data['daily_financial_opportunity'] / (
                data['daily_total_savings'].fillna(1) + 0.01
            )

            return data

        except Exception as e:
            logger.error(f"Error adding daily financial features: {e}")
            return data

    def create_financial_summary_features(
        self,
        daily_data: pd.DataFrame,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Create summary financial features for model metadata.

        Returns key financial metrics as a dictionary for model context.
        """
        try:
            roi_metrics = analysis_results.get('roi_metrics', {})
            financial_benefits = analysis_results.get('financial_benefits', {})
            annual_metrics = analysis_results.get('annual_metrics', {})

            summary = {
                'system_roi_percentage': roi_metrics.get('roi_percentage', 0),
                'payback_years': roi_metrics.get('simple_payback_years', 25),
                'annual_production_kwh': annual_metrics.get('annual_production_kwh', 0),
                'annual_savings_dollars': financial_benefits.get('total_annual_savings', 0),
                'capacity_factor': annual_metrics.get('capacity_factor', 0),
                'self_consumption_rate': annual_metrics.get('self_consumption_rate', 0),
                'grid_independence_rate': annual_metrics.get('grid_independence_rate', 0),
                'lifetime_savings': roi_metrics.get('lifetime_savings', 0)
            }

            return summary

        except Exception as e:
            logger.error(f"Error creating financial summary features: {e}")
            return {}

    def get_financial_feature_names(self) -> List[str]:
        """
        Get list of financial feature names created by this engineer.

        Returns:
            List of feature column names
        """
        base_features = [
            'capacity_factor', 'self_consumption_rate', 'grid_independence',
            'daily_performance_ratio', 'efficiency_normalized',
            'daily_export_value', 'daily_import_savings', 'daily_total_savings',
            'financial_efficiency', 'savings_rate',
            'annual_savings_potential', 'annual_export_income_potential',
            'simple_payback_years', 'roi_percentage', 'lifetime_savings',
            'payback_efficiency', 'financial_health_score',
            'daily_financial_opportunity', 'opportunity_ratio'
        ]

        # Rolling features
        rolling_features = [
            f'rolling_savings_{window}d' for window in [7, 14, 30]
        ] + [
            f'rolling_fin_efficiency_{window}d' for window in [7, 14, 30]
        ]

        # Volatility features
        volatility_features = ['savings_volatility_7d', 'savings_trend_14d']

        return base_features + rolling_features + volatility_features