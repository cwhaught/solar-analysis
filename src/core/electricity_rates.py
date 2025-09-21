"""
Electricity Rates Module - Location-based utility rate lookup

Provides functionality to lookup electricity rates based on geographic location
using multiple data sources including state averages and utility databases.
"""

import requests
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ElectricityRatesManager:
    """
    Manager for looking up electricity rates by location
    """

    def __init__(self, nrel_api_key: Optional[str] = None):
        """
        Initialize the electricity rates manager

        Args:
            nrel_api_key: Optional NREL API key for more detailed rate data
        """
        self.nrel_api_key = nrel_api_key
        self._state_rates = self._load_state_rates_data()

    def _load_state_rates_data(self) -> Dict[str, Dict]:
        """Load 2025 state electricity rates data"""
        # 2025 residential electricity rates by state (cents per kWh)
        state_rates = {
            "idaho": {"rate": 11.69, "abbreviation": "ID"},
            "north_dakota": {"rate": 11.70, "abbreviation": "ND"},
            "louisiana": {"rate": 11.80, "abbreviation": "LA"},
            "utah": {"rate": 11.81, "abbreviation": "UT"},
            "nebraska": {"rate": 11.90, "abbreviation": "NE"},
            "nevada": {"rate": 11.42, "abbreviation": "NV"},
            "washington": {"rate": 12.05, "abbreviation": "WA"},
            "oregon": {"rate": 12.25, "abbreviation": "OR"},
            "wyoming": {"rate": 12.30, "abbreviation": "WY"},
            "south_dakota": {"rate": 12.45, "abbreviation": "SD"},
            "oklahoma": {"rate": 12.60, "abbreviation": "OK"},
            "montana": {"rate": 12.75, "abbreviation": "MT"},
            "kansas": {"rate": 13.10, "abbreviation": "KS"},
            "iowa": {"rate": 13.25, "abbreviation": "IA"},
            "missouri": {"rate": 13.40, "abbreviation": "MO"},
            "tennessee": {"rate": 13.55, "abbreviation": "TN"},
            "kentucky": {"rate": 13.70, "abbreviation": "KY"},
            "west_virginia": {"rate": 13.85, "abbreviation": "WV"},
            "arkansas": {"rate": 14.00, "abbreviation": "AR"},
            "north_carolina": {"rate": 14.15, "abbreviation": "NC"},
            "colorado": {"rate": 14.30, "abbreviation": "CO"},
            "texas": {"rate": 14.45, "abbreviation": "TX"},
            "alabama": {"rate": 14.60, "abbreviation": "AL"},
            "minnesota": {"rate": 14.75, "abbreviation": "MN"},
            "mississippi": {"rate": 14.90, "abbreviation": "MS"},
            "indiana": {"rate": 15.05, "abbreviation": "IN"},
            "georgia": {"rate": 15.20, "abbreviation": "GA"},
            "wisconsin": {"rate": 15.35, "abbreviation": "WI"},
            "south_carolina": {"rate": 15.50, "abbreviation": "SC"},
            "ohio": {"rate": 15.65, "abbreviation": "OH"},
            "arizona": {"rate": 15.80, "abbreviation": "AZ"},
            "virginia": {"rate": 15.95, "abbreviation": "VA"},
            "illinois": {"rate": 16.10, "abbreviation": "IL"},
            "florida": {"rate": 16.25, "abbreviation": "FL"},
            "michigan": {"rate": 16.40, "abbreviation": "MI"},
            "new_mexico": {"rate": 16.55, "abbreviation": "NM"},
            "delaware": {"rate": 16.70, "abbreviation": "DE"},
            "pennsylvania": {"rate": 16.85, "abbreviation": "PA"},
            "maryland": {"rate": 17.00, "abbreviation": "MD"},
            "maine": {"rate": 17.15, "abbreviation": "ME"},
            "alaska": {"rate": 24.50, "abbreviation": "AK"},
            "vermont": {"rate": 18.25, "abbreviation": "VT"},
            "new_hampshire": {"rate": 22.15, "abbreviation": "NH"},
            "new_jersey": {"rate": 17.80, "abbreviation": "NJ"},
            "new_york": {"rate": 25.30, "abbreviation": "NY"},
            "massachusetts": {"rate": 27.40, "abbreviation": "MA"},
            "rhode_island": {"rate": 27.84, "abbreviation": "RI"},
            "connecticut": {"rate": 30.24, "abbreviation": "CT"},
            "california": {"rate": 30.62, "abbreviation": "CA"},
            "hawaii": {"rate": 42.49, "abbreviation": "HI"},
        }

        # Add reverse lookup by abbreviation
        by_abbrev = {}
        for state, info in state_rates.items():
            by_abbrev[info["abbreviation"]] = {
                "rate": info["rate"],
                "state_name": state
            }

        return {
            "by_name": state_rates,
            "by_abbreviation": by_abbrev,
            "national_average": 16.22
        }

    def _get_state_from_coordinates(self, latitude: float, longitude: float) -> Optional[str]:
        """
        Get state abbreviation from coordinates using a simple coordinate-to-state mapping

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees

        Returns:
            State abbreviation or None if not found
        """
        # Simple coordinate-based state lookup for major areas
        # This is a simplified approach - a complete solution would use a geographic database
        coordinate_to_state = {
            # Western states
            (37.7749, -122.4194): "CA",  # San Francisco
            (34.0522, -118.2437): "CA",  # Los Angeles
            (47.6062, -122.3321): "WA",  # Seattle
            (45.5152, -122.6784): "OR",  # Portland
            (39.7392, -104.9903): "CO",  # Denver
            (40.7589, -111.8883): "UT",  # Salt Lake City
            (33.4484, -112.0740): "AZ",  # Phoenix
            (36.1699, -115.1398): "NV",  # Las Vegas

            # Texas
            (29.7604, -95.3698): "TX",  # Houston
            (32.7767, -96.7970): "TX",  # Dallas
            (30.2672, -97.7431): "TX",  # Austin

            # Central states
            (41.8781, -87.6298): "IL",  # Chicago
            (39.0458, -76.6413): "MD",  # Baltimore
            (38.9072, -77.0369): "VA",  # Washington DC area

            # East coast
            (40.7128, -74.0060): "NY",  # New York
            (42.3601, -71.0589): "MA",  # Boston
            (39.9526, -75.1652): "PA",  # Philadelphia
            (25.7617, -80.1918): "FL",  # Miami
            (33.7490, -84.3880): "GA",  # Atlanta
            (35.2271, -80.8431): "NC",  # Charlotte
        }

        # Find closest match within reasonable distance
        min_distance = float('inf')
        closest_state = None

        for (lat, lon), state in coordinate_to_state.items():
            distance = ((latitude - lat) ** 2 + (longitude - lon) ** 2) ** 0.5
            if distance < min_distance and distance < 5.0:  # Within ~5 degrees
                min_distance = distance
                closest_state = state

        return closest_state

    def get_rates_by_coordinates(
        self, latitude: float, longitude: float
    ) -> Dict[str, float]:
        """
        Get electricity rates for a specific location by coordinates

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees

        Returns:
            Dictionary with rate information
        """
        result = {
            "residential_rate": self._state_rates["national_average"],
            "source": "national_average",
            "state": None,
            "confidence": "low"
        }

        # Try to determine state from coordinates
        state_abbrev = self._get_state_from_coordinates(latitude, longitude)

        if state_abbrev and state_abbrev in self._state_rates["by_abbreviation"]:
            state_data = self._state_rates["by_abbreviation"][state_abbrev]
            result.update({
                "residential_rate": state_data["rate"],
                "source": "state_average",
                "state": state_abbrev,
                "confidence": "medium"
            })

        # Try NREL API if available
        if self.nrel_api_key:
            try:
                nrel_data = self._query_nrel_api(latitude, longitude)
                if nrel_data and nrel_data.get("residential"):
                    # Convert from $/kWh to cents/kWh
                    nrel_rate = nrel_data["residential"] * 100
                    result.update({
                        "residential_rate": nrel_rate,
                        "source": "nrel_api",
                        "utility_name": nrel_data.get("utility_name"),
                        "confidence": "high"
                    })
            except Exception as e:
                logger.warning(f"NREL API query failed: {e}")

        return result

    def _query_nrel_api(self, latitude: float, longitude: float) -> Optional[Dict]:
        """
        Query NREL API for utility rate data

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees

        Returns:
            API response data or None if failed
        """
        if not self.nrel_api_key:
            return None

        url = "https://developer.nrel.gov/api/utility_rates/v3.json"
        params = {
            "api_key": self.nrel_api_key,
            "lat": latitude,
            "lon": longitude,
            "limit": 1
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("outputs") and len(data["outputs"]) > 0:
                return data["outputs"][0]

        except Exception as e:
            logger.error(f"NREL API error: {e}")

        return None

    def get_rates_by_state(self, state: str) -> Dict[str, float]:
        """
        Get electricity rates for a specific state

        Args:
            state: State name or abbreviation

        Returns:
            Dictionary with rate information
        """
        state_key = state.lower().replace(" ", "_")

        # Try by state name first
        if state_key in self._state_rates["by_name"]:
            return {
                "residential_rate": self._state_rates["by_name"][state_key]["rate"],
                "source": "state_average",
                "state": self._state_rates["by_name"][state_key]["abbreviation"],
                "confidence": "medium"
            }

        # Try by abbreviation
        state_upper = state.upper()
        if state_upper in self._state_rates["by_abbreviation"]:
            return {
                "residential_rate": self._state_rates["by_abbreviation"][state_upper]["rate"],
                "source": "state_average",
                "state": state_upper,
                "confidence": "medium"
            }

        # Default to national average
        return {
            "residential_rate": self._state_rates["national_average"],
            "source": "national_average",
            "state": None,
            "confidence": "low"
        }

    def get_feed_in_tariff_estimate(self, residential_rate: float) -> float:
        """
        Estimate feed-in tariff (net metering rate) based on residential rate

        Args:
            residential_rate: Residential electricity rate in cents/kWh

        Returns:
            Estimated feed-in tariff rate in cents/kWh
        """
        # Feed-in tariffs are typically 70-90% of retail rates in most states
        # Some states have full net metering (100%), others have lower rates
        return residential_rate * 0.80  # Conservative 80% estimate

    def get_rate_summary(self, latitude: float, longitude: float) -> Dict:
        """
        Get comprehensive rate summary for a location

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees

        Returns:
            Comprehensive rate information
        """
        rates = self.get_rates_by_coordinates(latitude, longitude)
        residential_rate = rates["residential_rate"]
        feed_in_rate = self.get_feed_in_tariff_estimate(residential_rate)

        return {
            **rates,
            "feed_in_tariff": feed_in_rate,
            "annual_cost_per_kwh": residential_rate / 100,  # Convert to $/kWh
            "feed_in_rate_per_kwh": feed_in_rate / 100,  # Convert to $/kWh
            "national_comparison": {
                "vs_national_avg": residential_rate - self._state_rates["national_average"],
                "is_above_average": residential_rate > self._state_rates["national_average"]
            }
        }


def get_electricity_rates_for_location(latitude: float, longitude: float, nrel_api_key: Optional[str] = None) -> Dict:
    """
    Convenience function to get electricity rates for a location

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        nrel_api_key: Optional NREL API key

    Returns:
        Rate information dictionary
    """
    manager = ElectricityRatesManager(nrel_api_key)
    return manager.get_rate_summary(latitude, longitude)