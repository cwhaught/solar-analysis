"""
Location Loader - Load location configuration from environment variables

Provides utility functions to load location settings from .env files
and create LocationManager instances automatically.
"""

import os
from pathlib import Path
from typing import Optional
from .location_manager import LocationManager


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
    city_name = os.environ.get('SOLAR_LOCATION_CITY')
    if city_name and city_name.strip() and city_name != 'your_city_here':
        try:
            location = LocationManager.from_city(city_name.strip())
            return location
        except ValueError:
            print(f"⚠️ Unknown city '{city_name}' in SOLAR_LOCATION_CITY")
            print("   Available cities: new_york, los_angeles, chicago, denver, miami,")
            print("                     seattle, phoenix, atlanta, london, berlin, tokyo, sydney")

    # Option 2: Custom coordinates
    lat_str = os.environ.get('SOLAR_LOCATION_LATITUDE')
    lon_str = os.environ.get('SOLAR_LOCATION_LONGITUDE')

    if lat_str and lon_str and lat_str.strip() and lon_str.strip():
        try:
            latitude = float(lat_str.strip())
            longitude = float(lon_str.strip())

            # Optional parameters
            location_name = os.environ.get('SOLAR_LOCATION_NAME', f"{latitude:.3f}, {longitude:.3f}")
            timezone_str = os.environ.get('SOLAR_LOCATION_TIMEZONE')

            # Clean up values
            if location_name and location_name.strip() and location_name != 'your_location_here':
                location_name = location_name.strip()
            else:
                location_name = f"{latitude:.3f}, {longitude:.3f}"

            if timezone_str and timezone_str.strip() and timezone_str != 'your_timezone_here':
                timezone_str = timezone_str.strip()
            else:
                timezone_str = None

            location = LocationManager(
                latitude=latitude,
                longitude=longitude,
                timezone_str=timezone_str,
                location_name=location_name
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
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
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
        summary['configured'] = True
        summary['source'] = 'environment'
        return summary
    else:
        return {
            'configured': False,
            'source': 'none',
            'message': 'No location configured in environment variables'
        }


def create_location_with_fallback(
    preferred_city: Optional[str] = None,
    fallback_city: str = 'denver',
    env_path: Optional[str] = None
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
        return LocationManager(39.7392, -104.9903, 'America/Denver', 'Denver, CO (fallback)')


# Convenience function for notebooks
def create_notebook_location() -> LocationManager:
    """
    Convenience function for notebooks to create location with smart defaults

    Returns:
        LocationManager instance from .env or Denver fallback
    """
    return create_location_with_fallback(
        env_path="../.env",  # Look for .env in parent directory (project root)
        fallback_city='denver'
    )