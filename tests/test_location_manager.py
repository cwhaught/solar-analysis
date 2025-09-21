"""
Tests for LocationManager class
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

# Import the module to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.location_manager import LocationManager


class TestLocationManager(unittest.TestCase):
    """Test LocationManager functionality"""

    def setUp(self):
        """Setup for each test"""
        # Test location: Denver, CO
        self.denver = LocationManager(39.7392, -104.9903, 'America/Denver', 'Denver, CO')

        # Test location: Miami, FL (subtropical)
        self.miami = LocationManager(25.7617, -80.1918, 'America/New_York', 'Miami, FL')

        # Test location: Anchorage, AK (extreme latitude)
        self.anchorage = LocationManager(61.2181, -149.9003, 'America/Anchorage', 'Anchorage, AK')

    def test_init_valid_coordinates(self):
        """Test initialization with valid coordinates"""
        loc = LocationManager(40.7128, -74.0060, 'America/New_York', 'New York, NY')

        self.assertEqual(loc.latitude, 40.7128)
        self.assertEqual(loc.longitude, -74.0060)
        self.assertEqual(loc.timezone_str, 'America/New_York')
        self.assertEqual(loc.location_name, 'New York, NY')

    def test_init_invalid_coordinates(self):
        """Test initialization with invalid coordinates"""
        # Invalid latitude
        with self.assertRaises(ValueError):
            LocationManager(95.0, -74.0060)  # > 90

        with self.assertRaises(ValueError):
            LocationManager(-95.0, -74.0060)  # < -90

        # Invalid longitude
        with self.assertRaises(ValueError):
            LocationManager(40.7128, 185.0)  # > 180

        with self.assertRaises(ValueError):
            LocationManager(40.7128, -185.0)  # < -180

    def test_from_city_valid(self):
        """Test creating LocationManager from valid city name"""
        denver = LocationManager.from_city('denver')

        self.assertEqual(denver.latitude, 39.7392)
        self.assertEqual(denver.longitude, -104.9903)
        self.assertEqual(denver.location_name, 'Denver, CO')
        self.assertEqual(denver.timezone_str, 'America/Denver')

    def test_from_city_invalid(self):
        """Test creating LocationManager from invalid city name"""
        with self.assertRaises(ValueError):
            LocationManager.from_city('nonexistent_city')

    def test_solar_declination(self):
        """Test solar declination calculation"""
        # Test known values
        summer_solstice = self.denver.get_solar_declination(172)  # ~June 21
        winter_solstice = self.denver.get_solar_declination(356)  # ~Dec 22

        # Summer solstice should be close to +23.45 degrees
        self.assertAlmostEqual(summer_solstice, 23.45, delta=1.0)

        # Winter solstice should be close to -23.45 degrees
        self.assertAlmostEqual(winter_solstice, -23.45, delta=1.0)

        # Equinoxes should be close to 0
        spring_equinox = self.denver.get_solar_declination(80)  # ~March 21
        self.assertAlmostEqual(spring_equinox, 0, delta=2.0)

    def test_sunrise_sunset(self):
        """Test sunrise and sunset calculations"""
        # Test summer solstice in Denver
        summer_date = datetime(2024, 6, 21)
        sunrise, sunset = self.denver.get_sunrise_sunset(summer_date)

        # Denver should have long summer days
        daylight_hours = sunset - sunrise
        self.assertGreater(daylight_hours, 14)  # Longer than 14 hours
        self.assertLess(daylight_hours, 16)     # But less than 16 hours

        # Test winter solstice in Denver
        winter_date = datetime(2024, 12, 21)
        sunrise, sunset = self.denver.get_sunrise_sunset(winter_date)

        # Denver should have shorter winter days
        winter_daylight = sunset - sunrise
        self.assertLess(winter_daylight, 10)    # Shorter than 10 hours
        self.assertGreater(winter_daylight, 8)  # But longer than 8 hours

    def test_solar_elevation(self):
        """Test solar elevation calculation"""
        # Test noon in summer - should be high
        summer_noon = datetime(2024, 6, 21, 12, 0)  # Summer solstice at noon
        elevation = self.denver.get_solar_elevation(summer_noon)

        # Should be positive and reasonably high
        self.assertGreater(elevation, 70)  # High summer sun
        self.assertLess(elevation, 90)     # But not directly overhead

        # Test midnight - should be 0
        midnight = datetime(2024, 6, 21, 0, 0)
        elevation = self.denver.get_solar_elevation(midnight)
        self.assertEqual(elevation, 0)  # Sun below horizon

    def test_theoretical_solar_irradiance(self):
        """Test theoretical solar irradiance calculation"""
        # Test clear summer noon
        summer_noon = datetime(2024, 6, 21, 12, 0)
        irradiance = self.denver.get_theoretical_solar_irradiance(summer_noon)

        # Should be high but realistic
        self.assertGreater(irradiance, 500)   # Decent irradiance
        self.assertLess(irradiance, 1400)     # But not more than solar constant

        # Test midnight - should be 0
        midnight = datetime(2024, 6, 21, 0, 0)
        irradiance = self.denver.get_theoretical_solar_irradiance(midnight)
        self.assertEqual(irradiance, 0)

    def test_seasonal_adjustment_factor(self):
        """Test seasonal adjustment factor"""
        # Summer should have higher factor
        summer_date = datetime(2024, 6, 21)
        summer_factor = self.denver.get_seasonal_adjustment_factor(summer_date)

        # Winter should have lower factor
        winter_date = datetime(2024, 12, 21)
        winter_factor = self.denver.get_seasonal_adjustment_factor(winter_date)

        self.assertGreater(summer_factor, winter_factor)

        # Both should be positive and reasonable
        self.assertGreater(summer_factor, 0.2)
        self.assertGreater(winter_factor, 0.2)
        self.assertLess(summer_factor, 2.0)
        self.assertLess(winter_factor, 2.0)

    def test_weather_adjustment_factor(self):
        """Test weather adjustment factor"""
        test_date = datetime(2024, 6, 21)

        # Test with randomness
        factor1 = self.denver.get_weather_adjustment_factor(test_date, base_randomness=True)

        # Test without randomness (should be deterministic)
        factor2 = self.denver.get_weather_adjustment_factor(test_date, base_randomness=False)
        factor3 = self.denver.get_weather_adjustment_factor(test_date, base_randomness=False)

        # Non-random should be identical
        self.assertEqual(factor2, factor3)

        # All should be between 0 and 1
        for factor in [factor1, factor2, factor3]:
            self.assertGreaterEqual(factor, 0)
            self.assertLessEqual(factor, 1)

    def test_calculate_location_solar_profile(self):
        """Test complete solar profile calculation"""
        test_date = datetime(2024, 6, 21)  # Summer solstice
        profile = self.denver.calculate_location_solar_profile(test_date)

        # Check all expected keys are present
        expected_keys = ['sunrise_hour', 'sunset_hour', 'daylight_hours',
                        'seasonal_factor', 'weather_factor', 'peak_irradiance',
                        'daily_solar_factor']

        for key in expected_keys:
            self.assertIn(key, profile)

        # Check reasonable values
        self.assertGreater(profile['daylight_hours'], 14)  # Long summer day
        self.assertGreater(profile['peak_irradiance'], 0)
        self.assertEqual(profile['daylight_hours'],
                        profile['sunset_hour'] - profile['sunrise_hour'])

    def test_enhance_solar_data(self):
        """Test solar data enhancement"""
        # Create sample solar data
        dates = pd.date_range('2024-06-01', periods=48, freq='H')  # 2 days hourly
        sample_data = pd.DataFrame({
            'Production (kWh)': np.random.uniform(0, 5, len(dates)),
            'Consumption (kWh)': np.random.uniform(1, 3, len(dates))
        }, index=dates)

        enhanced = self.denver.enhance_solar_data(sample_data)

        # Check new columns were added
        expected_new_cols = ['solar_elevation', 'theoretical_irradiance',
                           'seasonal_factor', 'weather_factor', 'sunrise_hour',
                           'sunset_hour', 'daylight_hours']

        for col in expected_new_cols:
            self.assertIn(col, enhanced.columns)

        # Check original data is preserved
        self.assertTrue(enhanced['Production (kWh)'].equals(sample_data['Production (kWh)']))

        # Check metadata
        self.assertEqual(enhanced.attrs['location_name'], 'Denver, CO')
        self.assertTrue(enhanced.attrs['enhanced_with_location'])

    def test_get_location_summary(self):
        """Test location summary generation"""
        summary = self.denver.get_location_summary()

        # Check all expected keys
        expected_keys = ['location_name', 'latitude', 'longitude', 'timezone',
                        'climate_type', 'summer_daylight_hours', 'winter_daylight_hours',
                        'seasonal_variation', 'typical_cloud_factor', 'atmospheric_clarity']

        for key in expected_keys:
            self.assertIn(key, summary)

        # Check values are reasonable
        self.assertEqual(summary['location_name'], 'Denver, CO')
        self.assertEqual(summary['latitude'], 39.7392)
        self.assertIn(summary['climate_type'], ['Tropical', 'Subtropical', 'Temperate',
                                               'Subarctic/Subantarctic', 'Arctic/Antarctic'])

    def test_climate_classification(self):
        """Test climate classification for different latitudes"""
        # Miami should be subtropical
        miami_summary = self.miami.get_location_summary()
        self.assertEqual(miami_summary['climate_type'], 'Subtropical')

        # Denver should be temperate
        denver_summary = self.denver.get_location_summary()
        self.assertEqual(denver_summary['climate_type'], 'Temperate')

        # Anchorage should be subarctic
        anchorage_summary = self.anchorage.get_location_summary()
        self.assertEqual(anchorage_summary['climate_type'], 'Subarctic/Subantarctic')

    def test_extreme_latitudes(self):
        """Test handling of extreme latitudes"""
        # Test Arctic location
        arctic = LocationManager(70.0, -150.0, location_name='Arctic Test')

        # Test that it doesn't crash
        winter_date = datetime(2024, 12, 21)
        summer_date = datetime(2024, 6, 21)

        winter_sunrise, winter_sunset = arctic.get_sunrise_sunset(winter_date)
        summer_sunrise, summer_sunset = arctic.get_sunrise_sunset(summer_date)

        # Arctic should have extreme day/night cycles
        winter_daylight = winter_sunset - winter_sunrise
        summer_daylight = summer_sunset - summer_sunrise

        # Should handle polar night/day gracefully
        self.assertGreaterEqual(winter_daylight, 0)
        self.assertLessEqual(summer_daylight, 24)

    def test_string_representations(self):
        """Test string representations"""
        str_repr = str(self.denver)
        self.assertIn('Denver, CO', str_repr)
        self.assertIn('39.739', str_repr)

        repr_str = repr(self.denver)
        self.assertIn('LocationManager', repr_str)
        self.assertIn('latitude=39.7392', repr_str)
        self.assertIn('longitude=-104.9903', repr_str)


if __name__ == '__main__':
    unittest.main()