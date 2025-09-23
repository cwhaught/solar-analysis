"""
Tests for weather integration functionality
"""

import pytest
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.weather_manager import WeatherManager, WeatherData, WeatherProvider
from core.weather_analysis import WeatherAnalyzer


class TestWeatherManager:
    """Test weather manager functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.weather_manager = WeatherManager(cache_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_weather_manager_init(self):
        """Test weather manager initialization"""
        assert self.weather_manager.cache_dir.exists()
        assert self.weather_manager.openweather_api_key is None  # No key in test env
        assert self.weather_manager.nrel_api_key is None

    def test_weather_data_creation(self):
        """Test WeatherData dataclass creation"""
        timestamp = datetime.now()
        weather_data = WeatherData(
            timestamp=timestamp,
            temperature=25.0,
            humidity=60.0,
            cloud_cover=20.0,
            visibility=10.0,
            wind_speed=5.0,
            wind_direction=180.0,
            pressure=1013.0,
            ghi=800.0,
            dni=600.0,
            dhi=200.0
        )

        assert weather_data.timestamp == timestamp
        assert weather_data.temperature == 25.0
        assert weather_data.ghi == 800.0

    def test_weather_provider_enum(self):
        """Test weather provider enumeration"""
        assert WeatherProvider.OPENWEATHER.value == "openweather"
        assert WeatherProvider.OPEN_METEO.value == "open_meteo"
        assert WeatherProvider.NREL.value == "nrel"

    @patch('requests.get')
    def test_open_meteo_forecast_success(self, mock_get):
        """Test successful Open-Meteo forecast request"""
        # Mock API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'hourly': {
                'time': ['2025-01-01T00:00', '2025-01-01T01:00'],
                'temperature_2m': [20.0, 21.0],
                'relative_humidity_2m': [65.0, 60.0],
                'cloud_cover': [30.0, 25.0],
                'visibility': [10000.0, 12000.0],
                'wind_speed_10m': [3.0, 4.0],
                'wind_direction_10m': [180.0, 190.0],
                'surface_pressure': [1013.0, 1014.0],
                'shortwave_radiation': [400.0, 500.0],
                'direct_radiation': [300.0, 400.0],
                'diffuse_radiation': [100.0, 100.0],
                'uv_index': [3.0, 4.0]
            }
        }
        mock_get.return_value = mock_response

        # Test forecast request
        weather_data = self.weather_manager.get_weather_forecast(
            latitude=39.7392,
            longitude=-104.9903,
            days=1,
            provider=WeatherProvider.OPEN_METEO
        )

        assert len(weather_data) == 2
        assert weather_data[0].temperature == 20.0
        assert weather_data[0].ghi == 400.0
        assert weather_data[1].temperature == 21.0

    @patch('requests.get')
    def test_open_meteo_forecast_failure(self, mock_get):
        """Test Open-Meteo forecast request failure"""
        mock_get.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            self.weather_manager.get_weather_forecast(
                latitude=39.7392,
                longitude=-104.9903,
                days=1,
                provider=WeatherProvider.OPEN_METEO
            )

    @patch('requests.get')
    def test_solar_irradiance_forecast(self, mock_get):
        """Test solar irradiance forecast"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'hourly': {
                'time': ['2025-01-01T00:00', '2025-01-01T01:00'],
                'shortwave_radiation': [0.0, 100.0],
                'direct_radiation': [0.0, 80.0],
                'diffuse_radiation': [0.0, 20.0],
                'direct_normal_irradiance': [0.0, 90.0],
                'global_tilted_irradiance': [0.0, 110.0],
                'cloud_cover': [0.0, 10.0],
                'temperature_2m': [15.0, 20.0]
            }
        }
        mock_get.return_value = mock_response

        solar_data = self.weather_manager.get_solar_irradiance_forecast(
            latitude=39.7392,
            longitude=-104.9903,
            days=1
        )

        assert not solar_data.empty
        assert 'ghi' in solar_data.columns
        assert 'dni' in solar_data.columns
        assert 'dhi' in solar_data.columns

    def test_rate_limiting(self):
        """Test API rate limiting"""
        start_time = self.weather_manager.last_request_time.get('test_provider')
        self.weather_manager._rate_limit('test_provider')

        # Should record the request time
        assert 'test_provider' in self.weather_manager.last_request_time

    def test_cache_operations(self):
        """Test weather data caching"""
        # Create test weather data
        weather_data = [
            WeatherData(
                timestamp=datetime.now(),
                temperature=25.0,
                humidity=60.0,
                cloud_cover=20.0,
                visibility=10.0,
                wind_speed=5.0,
                wind_direction=180.0,
                pressure=1013.0
            )
        ]

        # Test save and load
        cache_key = "test_cache_key"
        self.weather_manager._save_to_cache(cache_key, weather_data)
        loaded_data = self.weather_manager._load_from_cache(cache_key)

        assert loaded_data is not None
        assert len(loaded_data) == 1
        assert loaded_data[0].temperature == 25.0

    def test_weather_summary(self):
        """Test weather capabilities summary"""
        summary = self.weather_manager.get_weather_summary(39.7392, -104.9903)

        assert 'location' in summary
        assert 'available_providers' in summary
        assert 'capabilities' in summary
        assert summary['location']['latitude'] == 39.7392
        assert summary['capabilities']['solar_irradiance'] is True

    @patch.dict(os.environ, {'OPENWEATHER_API_KEY': 'test_key'})
    def test_openweather_key_detection(self):
        """Test OpenWeather API key detection"""
        wm = WeatherManager()
        assert wm.openweather_api_key == 'test_key'

        summary = wm.get_weather_summary(39.7392, -104.9903)
        provider_names = [p['name'] for p in summary['available_providers']]
        assert 'OpenWeatherMap' in provider_names


class TestWeatherAnalyzer:
    """Test weather analysis functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.analyzer = WeatherAnalyzer()

    def create_test_production_data(self) -> pd.DataFrame:
        """Create test production data"""
        dates = pd.date_range('2024-01-01', periods=48, freq='h')
        production = np.random.normal(5, 2, 48).clip(0, 20)  # 0-20 kWh

        return pd.DataFrame({
            'production': production
        }, index=dates)

    def create_test_weather_data(self) -> list:
        """Create test weather data"""
        weather_data = []
        base_time = datetime(2024, 1, 1)

        for i in range(48):
            weather_data.append(WeatherData(
                timestamp=base_time + timedelta(hours=i),
                temperature=20 + np.random.normal(0, 5),
                humidity=60 + np.random.normal(0, 20),
                cloud_cover=np.random.uniform(0, 100),
                visibility=10.0,
                wind_speed=np.random.uniform(0, 15),
                wind_direction=np.random.uniform(0, 360),
                pressure=1013 + np.random.normal(0, 10),
                ghi=np.random.uniform(0, 1000),
                dni=np.random.uniform(0, 800),
                dhi=np.random.uniform(0, 300)
            ))

        return weather_data

    def test_weather_to_dataframe(self):
        """Test weather data conversion to DataFrame"""
        weather_data = self.create_test_weather_data()
        df = self.analyzer._weather_to_dataframe(weather_data)

        assert not df.empty
        assert len(df) == 48
        assert 'temperature' in df.columns
        assert 'ghi' in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_merge_production_weather(self):
        """Test merging production and weather data"""
        production_data = self.create_test_production_data()
        weather_data = self.create_test_weather_data()
        weather_df = self.analyzer._weather_to_dataframe(weather_data)

        merged = self.analyzer._merge_production_weather(production_data, weather_df)

        assert not merged.empty
        assert 'production' in merged.columns
        assert 'temperature' in merged.columns
        assert 'ghi' in merged.columns

    def test_correlation_analysis(self):
        """Test weather correlation analysis"""
        production_data = self.create_test_production_data()
        weather_data = self.create_test_weather_data()

        results = self.analyzer.analyze_weather_correlation(production_data, weather_data)

        assert 'correlations' in results
        assert 'data_points' in results
        assert 'date_range' in results
        assert results['data_points'] > 0

        # Check that correlations exist for expected parameters
        correlations = results['correlations']
        assert 'temperature' in correlations
        assert 'cloud_cover' in correlations
        assert 'solar_ghi' in correlations

    def test_efficiency_factors_calculation(self):
        """Test weather efficiency factors calculation"""
        production_data = self.create_test_production_data()
        weather_data = self.create_test_weather_data()

        efficiency_df = self.analyzer.calculate_weather_efficiency_factors(
            production_data, weather_data
        )

        assert not efficiency_df.empty
        assert 'efficiency_factor' in efficiency_df.columns
        assert 'cloud_factor' in efficiency_df.columns
        assert 'combined_weather_efficiency' in efficiency_df.columns

        # Check that efficiency factors are reasonable
        assert efficiency_df['cloud_factor'].min() >= 0
        assert efficiency_df['cloud_factor'].max() <= 1.2

    def test_optimal_conditions_identification(self):
        """Test optimal weather conditions identification"""
        production_data = self.create_test_production_data()
        weather_data = self.create_test_weather_data()

        optimal_conditions = self.analyzer.identify_optimal_weather_conditions(
            production_data, weather_data
        )

        assert 'temperature' in optimal_conditions
        assert 'cloud_cover' in optimal_conditions
        assert 'ghi' in optimal_conditions

        # Check structure of optimal conditions
        temp_conditions = optimal_conditions['temperature']
        assert 'optimal_range' in temp_conditions
        assert 'suboptimal_range' in temp_conditions
        assert 'mean' in temp_conditions['optimal_range']

    def test_weather_impact_prediction(self):
        """Test weather impact prediction"""
        production_data = self.create_test_production_data()
        historical_weather = self.create_test_weather_data()

        # Create forecast weather data
        forecast_weather = []
        base_time = datetime(2024, 1, 3)  # Future dates
        for i in range(24):
            forecast_weather.append(WeatherData(
                timestamp=base_time + timedelta(hours=i),
                temperature=25.0,
                humidity=50.0,
                cloud_cover=20.0,
                visibility=10.0,
                wind_speed=5.0,
                wind_direction=180.0,
                pressure=1013.0,
                ghi=600.0,
                dni=500.0,
                dhi=100.0
            ))

        predictions = self.analyzer.predict_weather_impact(
            forecast_weather, production_data, historical_weather
        )

        if not predictions.empty:  # Model might not train with limited data
            assert 'predicted_production' in predictions.columns
            assert 'temperature' in predictions.columns
            assert all(predictions['predicted_production'] >= 0)  # Non-negative predictions

    def test_weather_report_generation(self):
        """Test comprehensive weather report generation"""
        production_data = self.create_test_production_data()
        weather_data = self.create_test_weather_data()

        report = self.analyzer.generate_weather_report(production_data, weather_data)

        assert 'timestamp' in report
        assert 'analysis_period' in report
        assert 'correlation_analysis' in report
        assert 'optimal_conditions' in report

        # Check analysis period
        assert 'start' in report['analysis_period']
        assert 'end' in report['analysis_period']
        assert 'days' in report['analysis_period']

    def test_correlation_interpretation(self):
        """Test correlation strength interpretation"""
        assert self.analyzer._interpret_correlation(0.9) == "Very Strong"
        assert self.analyzer._interpret_correlation(0.7) == "Strong"
        assert self.analyzer._interpret_correlation(0.5) == "Moderate"
        assert self.analyzer._interpret_correlation(0.3) == "Weak"
        assert self.analyzer._interpret_correlation(0.1) == "Very Weak"
        assert self.analyzer._interpret_correlation(-0.8) == "Very Strong"

    def test_empty_data_handling(self):
        """Test handling of empty or insufficient data"""
        empty_production = pd.DataFrame()
        empty_weather = []

        # Should handle empty data gracefully
        results = self.analyzer.analyze_weather_correlation(empty_production, empty_weather)
        assert 'error' in results

        # Test with minimal data
        minimal_production = pd.DataFrame({
            'production': [5.0]
        }, index=[datetime.now()])

        minimal_weather = [WeatherData(
            timestamp=datetime.now(),
            temperature=25.0,
            humidity=60.0,
            cloud_cover=20.0,
            visibility=10.0,
            wind_speed=5.0,
            wind_direction=180.0,
            pressure=1013.0
        )]

        # Should work with minimal data
        results = self.analyzer.analyze_weather_correlation(minimal_production, minimal_weather)
        assert 'correlations' in results or 'error' in results


class TestWeatherIntegrationEdgeCases:
    """Test edge cases and error conditions"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.weather_manager = WeatherManager(cache_dir=self.temp_dir)
        self.analyzer = WeatherAnalyzer()

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_coordinates(self):
        """Test handling of invalid coordinates"""
        # These should work (Open-Meteo is quite tolerant)
        summary = self.weather_manager.get_weather_summary(999, 999)
        assert 'location' in summary

    def test_extreme_forecast_days(self):
        """Test handling of extreme forecast day requests"""
        # Test very large number of days (should be capped)
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                'hourly': {
                    'time': ['2025-01-01T00:00'],
                    'temperature_2m': [20.0],
                    'relative_humidity_2m': [60.0],
                    'cloud_cover': [30.0],
                    'visibility': [10000.0],
                    'wind_speed_10m': [5.0],
                    'wind_direction_10m': [180.0],
                    'surface_pressure': [1013.0],
                    'shortwave_radiation': [400.0],
                    'direct_radiation': [300.0],
                    'diffuse_radiation': [100.0],
                    'uv_index': [3.0]
                }
            }
            mock_get.return_value = mock_response

            # Should cap at maximum allowed days
            weather_data = self.weather_manager.get_weather_forecast(
                latitude=39.7392,
                longitude=-104.9903,
                days=30,  # More than Open-Meteo's 16-day limit
                provider=WeatherProvider.OPEN_METEO
            )

            # Should still work (API will handle the limit)
            assert isinstance(weather_data, list)

    def test_missing_solar_data(self):
        """Test handling of weather data without solar irradiance"""
        weather_data = [WeatherData(
            timestamp=datetime.now(),
            temperature=25.0,
            humidity=60.0,
            cloud_cover=20.0,
            visibility=10.0,
            wind_speed=5.0,
            wind_direction=180.0,
            pressure=1013.0
            # No solar data (ghi, dni, dhi are None)
        )]

        # Should handle missing solar data gracefully
        df = self.analyzer._weather_to_dataframe(weather_data)
        assert 'temperature' in df.columns
        # Solar columns might not be present or might be NaN

    def test_cache_corruption_handling(self):
        """Test handling of corrupted cache files"""
        # Create corrupted cache file
        cache_file = self.weather_manager.cache_dir / "corrupted_cache.json"
        with open(cache_file, 'w') as f:
            f.write("invalid json content {")

        # Should handle corrupted cache gracefully
        result = self.weather_manager._load_from_cache("corrupted_cache")
        assert result is None

    def test_network_timeout_simulation(self):
        """Test handling of network timeouts"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Network timeout")

            # Should raise exception for network errors
            with pytest.raises(Exception):
                self.weather_manager.get_weather_forecast(
                    latitude=39.7392,
                    longitude=-104.9903,
                    days=1,
                    provider=WeatherProvider.OPEN_METEO
                )

    def test_malformed_api_response(self):
        """Test handling of malformed API responses"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                'invalid': 'response structure'
            }
            mock_get.return_value = mock_response

            # Should handle malformed responses
            with pytest.raises(KeyError):
                self.weather_manager.get_weather_forecast(
                    latitude=39.7392,
                    longitude=-104.9903,
                    days=1,
                    provider=WeatherProvider.OPEN_METEO
                )