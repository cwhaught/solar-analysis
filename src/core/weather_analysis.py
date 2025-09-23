"""
Weather Analysis - Analyze weather impact on solar production

Provides functions to:
- Correlate weather conditions with solar production
- Calculate weather-based efficiency factors
- Identify optimal/suboptimal weather patterns
- Generate weather-enhanced production forecasts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from .weather_manager import WeatherData, WeatherManager


class WeatherAnalyzer:
    """
    Analyze weather impact on solar production performance
    """

    def __init__(self):
        """Initialize weather analyzer"""
        self.weather_manager = WeatherManager()
        self.scaler = StandardScaler()

    def analyze_weather_correlation(
        self,
        production_data: pd.DataFrame,
        weather_data: List[WeatherData]
    ) -> Dict:
        """
        Analyze correlation between weather conditions and solar production

        Args:
            production_data: DataFrame with production data (timestamp, production)
            weather_data: List of weather data points

        Returns:
            Dictionary with correlation analysis results
        """

        # Convert weather data to DataFrame
        weather_df = self._weather_to_dataframe(weather_data)

        # Merge production and weather data
        merged_data = self._merge_production_weather(production_data, weather_df)

        if merged_data.empty:
            return {'error': 'No overlapping data found between production and weather'}

        # Calculate correlations
        correlations = {}

        # Basic weather correlations
        weather_cols = ['temperature', 'humidity', 'cloud_cover', 'wind_speed', 'pressure']
        for col in weather_cols:
            if col in merged_data.columns:
                corr = merged_data['production'].corr(merged_data[col])
                correlations[col] = {
                    'correlation': corr,
                    'strength': self._interpret_correlation(corr)
                }

        # Solar irradiance correlations (if available)
        solar_cols = ['ghi', 'dni', 'dhi']
        for col in solar_cols:
            if col in merged_data.columns and merged_data[col].notna().any():
                corr = merged_data['production'].corr(merged_data[col])
                correlations[f'solar_{col}'] = {
                    'correlation': corr,
                    'strength': self._interpret_correlation(corr)
                }

        # Weather efficiency analysis
        efficiency_analysis = self._analyze_weather_efficiency(merged_data)

        # Seasonal weather patterns
        seasonal_patterns = self._analyze_seasonal_weather_patterns(merged_data)

        return {
            'correlations': correlations,
            'efficiency_analysis': efficiency_analysis,
            'seasonal_patterns': seasonal_patterns,
            'data_points': len(merged_data),
            'date_range': {
                'start': merged_data.index.min().isoformat(),
                'end': merged_data.index.max().isoformat()
            }
        }

    def calculate_weather_efficiency_factors(
        self,
        production_data: pd.DataFrame,
        weather_data: List[WeatherData]
    ) -> pd.DataFrame:
        """
        Calculate weather-based efficiency factors for solar production

        Args:
            production_data: DataFrame with production data
            weather_data: List of weather data points

        Returns:
            DataFrame with efficiency factors
        """

        weather_df = self._weather_to_dataframe(weather_data)
        merged_data = self._merge_production_weather(production_data, weather_df)

        if merged_data.empty:
            return pd.DataFrame()

        # Calculate baseline expected production (based on GHI if available)
        if 'ghi' in merged_data.columns and merged_data['ghi'].notna().any():
            # Use GHI as baseline for expected production
            baseline_production = merged_data['ghi'] * 0.15  # Rough conversion factor
            merged_data['efficiency_factor'] = merged_data['production'] / baseline_production
            merged_data['efficiency_factor'] = merged_data['efficiency_factor'].clip(0, 2)  # Cap at 200%
        else:
            # Use temperature-based efficiency model
            # Solar panels are most efficient around 25°C
            optimal_temp = 25
            temp_factor = 1 - (np.abs(merged_data['temperature'] - optimal_temp) * 0.004)
            merged_data['efficiency_factor'] = temp_factor.clip(0.5, 1.2)

        # Weather condition factors
        merged_data['cloud_factor'] = 1 - (merged_data['cloud_cover'] / 100) * 0.8
        merged_data['humidity_factor'] = 1 - (merged_data['humidity'] / 100) * 0.1
        merged_data['wind_factor'] = np.minimum(1 + merged_data['wind_speed'] * 0.02, 1.2)  # Wind cooling helps

        # Combined weather efficiency
        merged_data['combined_weather_efficiency'] = (
            merged_data['efficiency_factor'] *
            merged_data['cloud_factor'] *
            merged_data['humidity_factor'] *
            merged_data['wind_factor']
        ).clip(0, 2)

        return merged_data[[
            'production', 'efficiency_factor', 'cloud_factor',
            'humidity_factor', 'wind_factor', 'combined_weather_efficiency'
        ]]

    def predict_weather_impact(
        self,
        forecast_weather: List[WeatherData],
        historical_production: pd.DataFrame,
        historical_weather: List[WeatherData]
    ) -> pd.DataFrame:
        """
        Predict solar production based on weather forecast

        Args:
            forecast_weather: Weather forecast data
            historical_production: Historical production data for training
            historical_weather: Historical weather data for training

        Returns:
            DataFrame with production predictions
        """

        # Train weather-production model
        model = self._train_weather_production_model(historical_production, historical_weather)

        if model is None:
            return pd.DataFrame()

        # Prepare forecast weather data
        forecast_df = self._weather_to_dataframe(forecast_weather)

        # Make predictions
        predictions = self._predict_production(model, forecast_df)

        # Create results DataFrame
        results = pd.DataFrame({
            'timestamp': forecast_df.index,
            'predicted_production': predictions,
            'temperature': forecast_df['temperature'],
            'cloud_cover': forecast_df['cloud_cover'],
            'ghi': forecast_df.get('ghi', np.nan)
        })

        results.set_index('timestamp', inplace=True)

        return results

    def identify_optimal_weather_conditions(
        self,
        production_data: pd.DataFrame,
        weather_data: List[WeatherData]
    ) -> Dict:
        """
        Identify optimal weather conditions for solar production

        Args:
            production_data: Historical production data
            weather_data: Historical weather data

        Returns:
            Dictionary with optimal weather conditions
        """

        weather_df = self._weather_to_dataframe(weather_data)
        merged_data = self._merge_production_weather(production_data, weather_df)

        if merged_data.empty:
            return {}

        # Find top 10% production days
        top_production = merged_data['production'].quantile(0.9)
        optimal_days = merged_data[merged_data['production'] >= top_production]

        # Find bottom 10% production days
        bottom_production = merged_data['production'].quantile(0.1)
        suboptimal_days = merged_data[merged_data['production'] <= bottom_production]

        optimal_conditions = {}
        weather_params = ['temperature', 'humidity', 'cloud_cover', 'wind_speed', 'pressure']

        if 'ghi' in merged_data.columns:
            weather_params.extend(['ghi', 'dni', 'dhi'])

        for param in weather_params:
            if param in optimal_days.columns:
                optimal_conditions[param] = {
                    'optimal_range': {
                        'min': optimal_days[param].quantile(0.25),
                        'max': optimal_days[param].quantile(0.75),
                        'mean': optimal_days[param].mean()
                    },
                    'suboptimal_range': {
                        'min': suboptimal_days[param].quantile(0.25),
                        'max': suboptimal_days[param].quantile(0.75),
                        'mean': suboptimal_days[param].mean()
                    }
                }

        return optimal_conditions

    def generate_weather_report(
        self,
        production_data: pd.DataFrame,
        weather_data: List[WeatherData],
        forecast_weather: Optional[List[WeatherData]] = None
    ) -> Dict:
        """
        Generate comprehensive weather impact report

        Args:
            production_data: Historical production data
            weather_data: Historical weather data
            forecast_weather: Optional forecast data

        Returns:
            Comprehensive weather analysis report
        """

        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_period': {
                'start': production_data.index.min().isoformat(),
                'end': production_data.index.max().isoformat(),
                'days': len(production_data)
            }
        }

        # Correlation analysis
        report['correlation_analysis'] = self.analyze_weather_correlation(production_data, weather_data)

        # Optimal conditions
        report['optimal_conditions'] = self.identify_optimal_weather_conditions(production_data, weather_data)

        # Efficiency factors
        efficiency_data = self.calculate_weather_efficiency_factors(production_data, weather_data)
        if not efficiency_data.empty:
            report['efficiency_summary'] = {
                'average_efficiency': efficiency_data['combined_weather_efficiency'].mean(),
                'best_efficiency_day': efficiency_data['combined_weather_efficiency'].max(),
                'worst_efficiency_day': efficiency_data['combined_weather_efficiency'].min()
            }

        # Forecast (if provided)
        if forecast_weather:
            forecast_results = self.predict_weather_impact(forecast_weather, production_data, weather_data)
            if not forecast_results.empty:
                report['forecast'] = {
                    'next_7_days_total': forecast_results['predicted_production'].sum(),
                    'daily_average': forecast_results['predicted_production'].mean(),
                    'best_day': forecast_results['predicted_production'].idxmax().isoformat(),
                    'worst_day': forecast_results['predicted_production'].idxmin().isoformat()
                }

        return report

    # Helper methods

    def _weather_to_dataframe(self, weather_data: List[WeatherData]) -> pd.DataFrame:
        """Convert weather data list to DataFrame"""
        if not weather_data:
            return pd.DataFrame()

        data = []
        for wd in weather_data:
            row = {
                'timestamp': wd.timestamp,
                'temperature': wd.temperature,
                'humidity': wd.humidity,
                'cloud_cover': wd.cloud_cover,
                'visibility': wd.visibility,
                'wind_speed': wd.wind_speed,
                'wind_direction': wd.wind_direction,
                'pressure': wd.pressure
            }

            # Add solar data if available
            if wd.ghi is not None:
                row['ghi'] = wd.ghi
            if wd.dni is not None:
                row['dni'] = wd.dni
            if wd.dhi is not None:
                row['dhi'] = wd.dhi
            if wd.uv_index is not None:
                row['uv_index'] = wd.uv_index

            data.append(row)

        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        return df

    def _merge_production_weather(self, production_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Merge production and weather data"""
        # Ensure both DataFrames have datetime index
        if not isinstance(production_df.index, pd.DatetimeIndex):
            production_df.index = pd.to_datetime(production_df.index)
        if not isinstance(weather_df.index, pd.DatetimeIndex):
            weather_df.index = pd.to_datetime(weather_df.index)

        # Resample to hourly data for better alignment
        production_hourly = production_df.resample('h').mean()
        weather_hourly = weather_df.resample('h').mean()

        # Merge data
        merged = pd.merge(production_hourly, weather_hourly, left_index=True, right_index=True, how='inner')

        return merged

    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "Very Strong"
        elif abs_corr >= 0.6:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very Weak"

    def _analyze_weather_efficiency(self, merged_data: pd.DataFrame) -> Dict:
        """Analyze weather efficiency patterns"""

        if 'cloud_cover' not in merged_data.columns:
            return {}

        # Efficiency by cloud cover ranges
        cloud_bins = [0, 20, 40, 60, 80, 100]
        cloud_labels = ['Clear (0-20%)', 'Partly Cloudy (20-40%)', 'Cloudy (40-60%)',
                       'Very Cloudy (60-80%)', 'Overcast (80-100%)']

        merged_data['cloud_category'] = pd.cut(merged_data['cloud_cover'], bins=cloud_bins, labels=cloud_labels)

        efficiency_by_clouds = merged_data.groupby('cloud_category', observed=True)['production'].agg(['mean', 'count']).to_dict()

        return {
            'efficiency_by_cloud_cover': efficiency_by_clouds,
            'clear_sky_advantage': (
                merged_data[merged_data['cloud_cover'] <= 20]['production'].mean() /
                merged_data[merged_data['cloud_cover'] >= 80]['production'].mean()
                if merged_data[merged_data['cloud_cover'] >= 80]['production'].mean() > 0 else 1
            )
        }

    def _analyze_seasonal_weather_patterns(self, merged_data: pd.DataFrame) -> Dict:
        """Analyze seasonal weather patterns"""

        merged_data['month'] = merged_data.index.month
        merged_data['season'] = merged_data['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })

        seasonal_stats = {}
        for season in ['Spring', 'Summer', 'Fall', 'Winter']:
            season_data = merged_data[merged_data['season'] == season]
            if not season_data.empty:
                seasonal_stats[season] = {
                    'avg_production': season_data['production'].mean(),
                    'avg_temperature': season_data['temperature'].mean(),
                    'avg_cloud_cover': season_data['cloud_cover'].mean(),
                    'days': len(season_data)
                }

        return seasonal_stats

    def _train_weather_production_model(
        self,
        production_data: pd.DataFrame,
        weather_data: List[WeatherData]
    ) -> Optional[RandomForestRegressor]:
        """Train ML model to predict production from weather"""

        weather_df = self._weather_to_dataframe(weather_data)
        merged_data = self._merge_production_weather(production_data, weather_df)

        if merged_data.empty or len(merged_data) < 50:  # Need minimum data
            return None

        # Prepare features
        feature_cols = ['temperature', 'humidity', 'cloud_cover', 'wind_speed', 'pressure']

        # Add solar irradiance if available
        if 'ghi' in merged_data.columns and merged_data['ghi'].notna().any():
            feature_cols.append('ghi')
        if 'dni' in merged_data.columns and merged_data['dni'].notna().any():
            feature_cols.append('dni')

        # Add time features
        merged_data['hour'] = merged_data.index.hour
        merged_data['day_of_year'] = merged_data.index.dayofyear
        feature_cols.extend(['hour', 'day_of_year'])

        # Clean data
        clean_data = merged_data[feature_cols + ['production']].dropna()

        if len(clean_data) < 50:
            return None

        X = clean_data[feature_cols]
        y = clean_data['production']

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)

        return model

    def _predict_production(self, model: RandomForestRegressor, weather_df: pd.DataFrame) -> np.ndarray:
        """Make production predictions using trained model"""

        # Prepare features (same as training)
        feature_cols = ['temperature', 'humidity', 'cloud_cover', 'wind_speed', 'pressure']

        if 'ghi' in weather_df.columns:
            feature_cols.append('ghi')
        if 'dni' in weather_df.columns:
            feature_cols.append('dni')

        # Add time features
        weather_df['hour'] = weather_df.index.hour
        weather_df['day_of_year'] = weather_df.index.dayofyear
        feature_cols.extend(['hour', 'day_of_year'])

        # Fill missing features with reasonable defaults
        for col in feature_cols:
            if col not in weather_df.columns:
                if col == 'pressure':
                    weather_df[col] = 1013  # Sea level pressure
                elif col == 'ghi':
                    weather_df[col] = 400   # Rough average
                elif col == 'dni':
                    weather_df[col] = 300   # Rough average
                else:
                    weather_df[col] = 0

        X = weather_df[feature_cols].fillna(0)
        predictions = model.predict(X)

        return np.maximum(predictions, 0)  # Ensure non-negative predictions