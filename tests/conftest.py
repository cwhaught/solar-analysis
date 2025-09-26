"""
Pytest configuration and shared fixtures
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_solar_data():
    """Generate sample solar production data for testing"""
    dates = pd.date_range(
        "2023-01-01", periods=96, freq="15min"
    )  # 1 day of 15-min data

    # Simulate realistic solar production pattern
    hours = dates.hour + dates.minute / 60
    production = []

    for hour in hours:
        if 6 <= hour <= 18:  # Daylight hours
            # Peak around noon, zero at night
            peak_factor = 1 - abs(hour - 12) / 6
            base_production = max(0, peak_factor * 5000)  # Max 5kW
            # Add some randomness
            production.append(
                base_production * (0.8 + 0.4 * abs(hash(hour) % 100) / 100)
            )
        else:
            production.append(0)

    consumption = [
        1000 + 500 * abs(hash(d) % 100) / 100 for d in dates
    ]  # 1-1.5 kW baseline

    df = pd.DataFrame(
        {
            "Date/Time": dates,
            "Production (Wh)": production,
            "Consumption (Wh)": consumption,
            "Export (Wh)": [max(0, p - c) for p, c in zip(production, consumption)],
            "Import (Wh)": [max(0, c - p) for p, c in zip(production, consumption)],
        }
    )

    return df


@pytest.fixture
def temp_csv_file(sample_solar_data):
    """Create temporary CSV file with sample data"""
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "test_solar_data.csv")
    sample_solar_data.to_csv(csv_path, index=False)

    yield csv_path

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_enphase_response():
    """Mock Enphase API response data"""
    return {
        "system_id": 12345,
        "current_power": 3250,
        "energy_today": 25000,
        "energy_lifetime": 50000000,
        "modules": 24,
        "last_report_at": 1640995200,
        "operational_at": 1577836800,
    }


@pytest.fixture
def sample_api_energy_data():
    """Sample energy lifetime API response"""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "production": [
                20000 + 5000 * abs(hash(d) % 100) / 100 for d in dates
            ],  # 20-25 kWh per day
            "consumption": [
                25000 + 3000 * abs(hash(d) % 100) / 100 for d in dates
            ],  # 25-28 kWh per day
        }
    )


@pytest.fixture
def oauth_test_config():
    """Test configuration for OAuth setup"""
    return {
        "client_id": "test_client_id_12345",
        "client_secret": "test_client_secret_abcdef",
        "redirect_uri": "https://localhost:8080/callback",
        "api_key": "test_api_key_xyz789",
        "system_id": "67890",
    }


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_daily_data():
    """Generate extended sample daily solar data for feature engineering tests"""
    dates = pd.date_range("2023-01-01", periods=90, freq="D")  # 3 months for rolling features

    # Generate realistic production pattern
    day_of_year = dates.dayofyear
    seasonal_factor = 1 + 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365)  # Peak in summer
    base_production = 30 * seasonal_factor + np.random.normal(0, 5, len(dates))
    base_production = np.maximum(base_production, 0)  # No negative production

    consumption = 35 + np.random.normal(0, 8, len(dates))
    consumption = np.maximum(consumption, 5)  # Minimum consumption

    export = np.maximum(base_production - consumption, 0)
    import_energy = np.maximum(consumption - base_production, 0)

    df = pd.DataFrame({
        'Production (kWh)': base_production,
        'Consumption (kWh)': consumption,
        'Export (kWh)': export,
        'Import (kWh)': import_energy
    }, index=dates)

    return df


@pytest.fixture
def sample_weather_data():
    """Generate sample weather data for feature engineering tests"""
    dates = pd.date_range("2023-01-01", periods=90, freq="D")

    # Import numpy for fixtures
    import numpy as np

    df = pd.DataFrame({
        'sunshine_hours': np.random.uniform(4, 12, len(dates)),
        'cloudcover_mean_pct': np.random.uniform(10, 90, len(dates)),
        'temp_mean_c': np.random.uniform(15, 30, len(dates)),
        'temp_max_c': np.random.uniform(20, 35, len(dates)),
        'temp_min_c': np.random.uniform(10, 25, len(dates)),
        'precipitation_mm': np.random.exponential(2, len(dates)),
        'solar_radiation_mj': np.random.uniform(10, 30, len(dates))
    }, index=dates)

    return df


@pytest.fixture
def mock_location_manager():
    """Mock location manager for feature engineering tests"""
    from unittest.mock import Mock

    mock = Mock()
    mock.latitude = 35.663
    mock.longitude = -78.844
    mock.location_name = "Holly Springs, NC"
    mock.timezone_str = "America/New_York"

    # Mock methods with realistic returns
    mock.get_sunrise_sunset.return_value = (6.0, 18.0)  # 12 hours daylight
    mock.get_solar_elevation.return_value = 45.0
    mock.get_theoretical_solar_irradiance.return_value = 1000.0
    mock.get_location_summary.return_value = {
        'climate_type': 'Temperate',
        'seasonal_variation': 0.5,
        'winter_daylight_hours': 9.6,
        'summer_daylight_hours': 14.4
    }

    return mock
