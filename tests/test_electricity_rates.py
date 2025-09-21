"""
Tests for electricity rates functionality
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, Mock

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.electricity_rates import ElectricityRatesManager, get_electricity_rates_for_location


class TestElectricityRatesManager:
    """Test electricity rates manager functionality"""

    def test_init(self):
        """Test ElectricityRatesManager initialization"""
        manager = ElectricityRatesManager()
        assert manager.nrel_api_key is None
        assert manager._state_rates is not None
        assert "by_name" in manager._state_rates
        assert "by_abbreviation" in manager._state_rates
        assert "national_average" in manager._state_rates

    def test_init_with_api_key(self):
        """Test initialization with NREL API key"""
        api_key = "test_api_key"
        manager = ElectricityRatesManager(api_key)
        assert manager.nrel_api_key == api_key

    def test_state_rates_data_structure(self):
        """Test that state rates data has expected structure"""
        manager = ElectricityRatesManager()

        # Check national average
        assert manager._state_rates["national_average"] == 16.22

        # Check a few known states
        assert "colorado" in manager._state_rates["by_name"]
        assert manager._state_rates["by_name"]["colorado"]["rate"] == 14.30
        assert manager._state_rates["by_name"]["colorado"]["abbreviation"] == "CO"

        # Check abbreviation lookup
        assert "CO" in manager._state_rates["by_abbreviation"]
        assert manager._state_rates["by_abbreviation"]["CO"]["rate"] == 14.30

    def test_get_state_from_coordinates_known_location(self):
        """Test coordinate to state mapping for known locations"""
        manager = ElectricityRatesManager()

        # Test Denver coordinates
        state = manager._get_state_from_coordinates(39.7392, -104.9903)
        assert state == "CO"

        # Test San Francisco coordinates
        state = manager._get_state_from_coordinates(37.7749, -122.4194)
        assert state == "CA"

    def test_get_state_from_coordinates_unknown_location(self):
        """Test coordinate to state mapping for unknown locations"""
        manager = ElectricityRatesManager()

        # Test coordinates in the middle of the ocean
        state = manager._get_state_from_coordinates(0.0, 0.0)
        assert state is None

    def test_get_rates_by_coordinates_known_state(self):
        """Test getting rates by coordinates for known state"""
        manager = ElectricityRatesManager()

        # Test Denver coordinates
        rates = manager.get_rates_by_coordinates(39.7392, -104.9903)

        assert rates["residential_rate"] == 14.30
        assert rates["source"] == "state_average"
        assert rates["state"] == "CO"
        assert rates["confidence"] == "medium"

    def test_get_rates_by_coordinates_unknown_location(self):
        """Test getting rates by coordinates for unknown location"""
        manager = ElectricityRatesManager()

        # Test coordinates that don't match any state
        rates = manager.get_rates_by_coordinates(0.0, 0.0)

        assert rates["residential_rate"] == 16.22  # National average
        assert rates["source"] == "national_average"
        assert rates["state"] is None
        assert rates["confidence"] == "low"

    def test_get_rates_by_state_name(self):
        """Test getting rates by state name"""
        manager = ElectricityRatesManager()

        # Test valid state name
        rates = manager.get_rates_by_state("Colorado")
        assert rates["residential_rate"] == 14.30
        assert rates["source"] == "state_average"
        assert rates["state"] == "CO"
        assert rates["confidence"] == "medium"

    def test_get_rates_by_state_abbreviation(self):
        """Test getting rates by state abbreviation"""
        manager = ElectricityRatesManager()

        # Test valid state abbreviation
        rates = manager.get_rates_by_state("CA")
        assert rates["residential_rate"] == 30.62
        assert rates["source"] == "state_average"
        assert rates["state"] == "CA"
        assert rates["confidence"] == "medium"

    def test_get_rates_by_state_invalid(self):
        """Test getting rates for invalid state"""
        manager = ElectricityRatesManager()

        # Test invalid state name
        rates = manager.get_rates_by_state("InvalidState")
        assert rates["residential_rate"] == 16.22  # National average
        assert rates["source"] == "national_average"
        assert rates["state"] is None
        assert rates["confidence"] == "low"

    def test_get_feed_in_tariff_estimate(self):
        """Test feed-in tariff estimation"""
        manager = ElectricityRatesManager()

        # Test with a known rate
        residential_rate = 20.0  # 20 cents/kWh
        feed_in_rate = manager.get_feed_in_tariff_estimate(residential_rate)

        assert feed_in_rate == 16.0  # 80% of residential rate
        assert isinstance(feed_in_rate, float)

    def test_get_rate_summary(self):
        """Test comprehensive rate summary"""
        manager = ElectricityRatesManager()

        # Test for Denver coordinates
        summary = manager.get_rate_summary(39.7392, -104.9903)

        # Check required fields
        required_fields = [
            "residential_rate", "feed_in_tariff", "annual_cost_per_kwh",
            "feed_in_rate_per_kwh", "source", "state", "confidence",
            "national_comparison"
        ]

        for field in required_fields:
            assert field in summary

        # Check conversions
        assert summary["annual_cost_per_kwh"] == summary["residential_rate"] / 100
        assert summary["feed_in_rate_per_kwh"] == summary["feed_in_tariff"] / 100

        # Check national comparison
        assert "vs_national_avg" in summary["national_comparison"]
        assert "is_above_average" in summary["national_comparison"]

    @patch('requests.get')
    def test_query_nrel_api_success(self, mock_get):
        """Test successful NREL API query"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "outputs": [{
                "utility_name": "Test Utility",
                "residential": 0.15,  # $0.15/kWh
                "commercial": 0.12,
                "industrial": 0.10
            }]
        }
        mock_get.return_value = mock_response

        manager = ElectricityRatesManager("test_api_key")
        result = manager._query_nrel_api(39.7392, -104.9903)

        assert result is not None
        assert result["utility_name"] == "Test Utility"
        assert result["residential"] == 0.15

    @patch('requests.get')
    def test_query_nrel_api_failure(self, mock_get):
        """Test NREL API query failure"""
        # Mock API failure
        mock_get.side_effect = Exception("API Error")

        manager = ElectricityRatesManager("test_api_key")
        result = manager._query_nrel_api(39.7392, -104.9903)

        assert result is None

    def test_query_nrel_api_no_key(self):
        """Test NREL API query without API key"""
        manager = ElectricityRatesManager()
        result = manager._query_nrel_api(39.7392, -104.9903)

        assert result is None

    @patch('requests.get')
    def test_get_rates_by_coordinates_with_nrel_api(self, mock_get):
        """Test getting rates with NREL API enhancement"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "outputs": [{
                "utility_name": "Test Utility",
                "residential": 0.18,  # $0.18/kWh = 18 cents/kWh
                "commercial": 0.15,
                "industrial": 0.12
            }]
        }
        mock_get.return_value = mock_response

        manager = ElectricityRatesManager("test_api_key")
        rates = manager.get_rates_by_coordinates(39.7392, -104.9903)

        # Should use NREL API data instead of state average
        assert rates["residential_rate"] == 18.0  # Converted from $/kWh to cents/kWh
        assert rates["source"] == "nrel_api"
        assert rates["utility_name"] == "Test Utility"
        assert rates["confidence"] == "high"


def test_get_electricity_rates_for_location():
    """Test convenience function"""
    rates = get_electricity_rates_for_location(39.7392, -104.9903)

    # Should return a summary with all required fields
    required_fields = [
        "residential_rate", "feed_in_tariff", "annual_cost_per_kwh",
        "feed_in_rate_per_kwh", "source", "confidence", "national_comparison"
    ]

    for field in required_fields:
        assert field in rates

    # Should be Colorado rates
    assert rates["residential_rate"] == 14.30
    assert rates["source"] == "state_average"


class TestElectricityRatesEdgeCases:
    """Test edge cases and error conditions"""

    def test_extreme_coordinates(self):
        """Test extreme coordinate values"""
        manager = ElectricityRatesManager()

        # Test with extreme latitudes
        rates = manager.get_rates_by_coordinates(90.0, 0.0)  # North Pole
        assert rates["source"] == "national_average"

        rates = manager.get_rates_by_coordinates(-90.0, 0.0)  # South Pole
        assert rates["source"] == "national_average"

    def test_feed_in_tariff_calculation(self):
        """Test feed-in tariff calculation edge cases"""
        manager = ElectricityRatesManager()

        # Test with zero rate
        assert manager.get_feed_in_tariff_estimate(0.0) == 0.0

        # Test with very high rate
        high_rate = 100.0
        feed_in = manager.get_feed_in_tariff_estimate(high_rate)
        assert feed_in == 80.0  # 80% of 100

    def test_state_name_variations(self):
        """Test various state name formats"""
        manager = ElectricityRatesManager()

        # Test with spaces
        rates = manager.get_rates_by_state("North Dakota")
        assert rates["residential_rate"] == 11.70

        # Test case insensitive
        rates = manager.get_rates_by_state("COLORADO")
        assert rates["residential_rate"] == 14.30

        # Test lowercase
        rates = manager.get_rates_by_state("california")
        assert rates["residential_rate"] == 30.62