"""
Location Loader - Load location configuration from environment variables

Provides utility functions to load location settings from .env files
and create LocationManager instances automatically.
"""

import os
from pathlib import Path
from typing import Optional
from .location_manager import LocationManager
from .electricity_rates import ElectricityRatesManager


def load_location_from_env(env_path: Optional[str] = None) -> Optional[LocationManager]:
    """
    Load location configuration from environment variables

    Args:
        env_path: Path to .env file (optional, searches for .env if not provided)

    Returns:
        LocationManager instance if location is configured, None otherwise
    """

    # Load environment variables from .env file if it exists
    if env_path is None:
        env_path = Path(".env")
    else:
        env_path = Path(env_path)

    if env_path.exists():
        _load_env_file(env_path)

    # Try to create location from environment variables

    # Option 1: City name
    city_name = os.environ.get("SOLAR_LOCATION_CITY")
    if city_name and city_name.strip() and city_name != "your_city_here":
        try:
            location = LocationManager.from_city(city_name.strip())
            return location
        except ValueError:
            print(f"⚠️ Unknown city '{city_name}' in SOLAR_LOCATION_CITY")
            print("   Available cities: new_york, los_angeles, chicago, denver, miami,")
            print(
                "                     seattle, phoenix, atlanta, london, berlin, tokyo, sydney"
            )

    # Option 2: Custom coordinates
    lat_str = os.environ.get("SOLAR_LOCATION_LATITUDE")
    lon_str = os.environ.get("SOLAR_LOCATION_LONGITUDE")

    if lat_str and lon_str and lat_str.strip() and lon_str.strip():
        try:
            latitude = float(lat_str.strip())
            longitude = float(lon_str.strip())

            # Optional parameters
            location_name = os.environ.get(
                "SOLAR_LOCATION_NAME", f"{latitude:.3f}, {longitude:.3f}"
            )
            timezone_str = os.environ.get("SOLAR_LOCATION_TIMEZONE")

            # Clean up values
            if (
                location_name
                and location_name.strip()
                and location_name != "your_location_here"
            ):
                location_name = location_name.strip()
            else:
                location_name = f"{latitude:.3f}, {longitude:.3f}"

            if (
                timezone_str
                and timezone_str.strip()
                and timezone_str != "your_timezone_here"
            ):
                timezone_str = timezone_str.strip()
            else:
                timezone_str = None

            location = LocationManager(
                latitude=latitude,
                longitude=longitude,
                timezone_str=timezone_str,
                location_name=location_name,
            )
            return location

        except (ValueError, TypeError) as e:
            print(f"⚠️ Invalid coordinates in environment variables: {e}")
            print(f"   SOLAR_LOCATION_LATITUDE: {lat_str}")
            print(f"   SOLAR_LOCATION_LONGITUDE: {lon_str}")

    # No valid location configuration found
    return None


def _load_env_file(env_path: Path) -> None:
    """Load environment variables from .env file"""
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Only set if not already in environment (env vars take precedence)
                    if key not in os.environ:
                        os.environ[key] = value
    except Exception as e:
        print(f"⚠️ Warning: Could not load .env file {env_path}: {e}")


def get_location_summary_from_env(env_path: Optional[str] = None) -> dict:
    """
    Get a summary of location configuration from environment

    Args:
        env_path: Path to .env file (optional)

    Returns:
        Dictionary with location configuration status
    """

    location = load_location_from_env(env_path)

    if location:
        summary = location.get_location_summary()
        summary["configured"] = True
        summary["source"] = "environment"
        return summary
    else:
        return {
            "configured": False,
            "source": "none",
            "message": "No location configured in environment variables",
        }


def create_location_with_fallback(
    preferred_city: Optional[str] = None,
    fallback_city: str = "denver",
    env_path: Optional[str] = None,
) -> LocationManager:
    """
    Create location with fallback strategy

    Priority order:
    1. Environment variables (.env file)
    2. Preferred city parameter
    3. Fallback city

    Args:
        preferred_city: Preferred city name if env not configured
        fallback_city: Final fallback city (default: denver)
        env_path: Path to .env file (optional)

    Returns:
        LocationManager instance (always succeeds)
    """

    # Try environment first
    location = load_location_from_env(env_path)
    if location:
        return location

    # Try preferred city
    if preferred_city:
        try:
            return LocationManager.from_city(preferred_city)
        except ValueError:
            print(f"⚠️ Unknown preferred city '{preferred_city}', using fallback")

    # Use fallback
    try:
        return LocationManager.from_city(fallback_city)
    except ValueError:
        # Final fallback to Denver coordinates
        print(f"⚠️ Unknown fallback city '{fallback_city}', using Denver coordinates")
        return LocationManager(
            39.7392, -104.9903, "America/Denver", "Denver, CO (fallback)"
        )


# Convenience function for notebooks
def create_notebook_location() -> LocationManager:
    """
    Convenience function for notebooks to create location with smart defaults

    Returns:
        LocationManager instance from .env or Denver fallback
    """
    return create_location_with_fallback(
        env_path="../.env",  # Look for .env in parent directory (project root)
        fallback_city="denver",
    )


def get_location_electricity_rates(location: LocationManager, nrel_api_key: Optional[str] = None) -> dict:
    """
    Get electricity rates for a location

    Args:
        location: LocationManager instance
        nrel_api_key: Optional NREL API key for more accurate data

    Returns:
        Dictionary with electricity rate information
    """
    rates_manager = ElectricityRatesManager(nrel_api_key)
    return rates_manager.get_rate_summary(location.latitude, location.longitude)


def create_notebook_location_with_rates() -> tuple[LocationManager, dict]:
    """
    Convenience function for notebooks to create location with electricity rates

    Returns:
        Tuple of (LocationManager instance, electricity rates dict)
    """
    location = create_notebook_location()

    # Try to get NREL API key from environment
    nrel_api_key = os.environ.get('NREL_API_KEY')
    rates = get_location_electricity_rates(location, nrel_api_key)

    return location, rates


def load_system_financial_config(env_path: Optional[str] = None) -> dict:
    """
    Load solar system financial configuration from environment variables

    Args:
        env_path: Path to .env file (optional, searches for .env if not provided)

    Returns:
        Dictionary with system financial configuration
    """

    # Load environment variables from .env file if it exists
    if env_path is None:
        env_path = Path(".env")
    else:
        env_path = Path(env_path)

    if env_path.exists():
        _load_env_file(env_path)

    # Load system specifications with defaults
    system_size_kw = float(os.environ.get('SOLAR_SYSTEM_SIZE_KW', '10.0'))
    cost_per_kw = float(os.environ.get('SOLAR_SYSTEM_COST_PER_KW', '2500'))
    total_cost = float(os.environ.get('SOLAR_SYSTEM_TOTAL_COST', str(system_size_kw * cost_per_kw)))

    # Load rebates and incentives
    federal_tax_credit_percent = float(os.environ.get('SOLAR_FEDERAL_TAX_CREDIT_PERCENT', '30'))
    state_rebate = float(os.environ.get('SOLAR_STATE_REBATE', '0'))
    utility_rebate = float(os.environ.get('SOLAR_UTILITY_REBATE', '0'))
    other_rebates = float(os.environ.get('SOLAR_OTHER_REBATES', '0'))

    # Calculate derived values
    federal_tax_credit = total_cost * (federal_tax_credit_percent / 100)
    total_rebates = federal_tax_credit + state_rebate + utility_rebate + other_rebates
    net_system_cost = total_cost - total_rebates

    return {
        'system_size_kw': system_size_kw,
        'cost_per_kw': cost_per_kw,
        'total_cost': total_cost,
        'federal_tax_credit_percent': federal_tax_credit_percent,
        'federal_tax_credit': federal_tax_credit,
        'state_rebate': state_rebate,
        'utility_rebate': utility_rebate,
        'other_rebates': other_rebates,
        'total_rebates': total_rebates,
        'net_system_cost': net_system_cost,
        'source': 'environment' if env_path.exists() else 'defaults'
    }


def get_complete_financial_config(env_path: Optional[str] = None) -> tuple[LocationManager, dict, dict]:
    """
    Get complete financial configuration including location, rates, and system costs

    Args:
        env_path: Path to .env file (optional)

    Returns:
        Tuple of (LocationManager, electricity_rates_dict, system_financial_dict)
    """
    location = create_notebook_location() if env_path is None else create_location_with_fallback(env_path=env_path)

    # Try to get NREL API key from environment
    nrel_api_key = os.environ.get('NREL_API_KEY')
    rates = get_location_electricity_rates(location, nrel_api_key)

    system_config = load_system_financial_config(env_path)

    return location, rates, system_config
