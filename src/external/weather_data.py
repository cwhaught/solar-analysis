import requests
import pandas as pd
from pathlib import Path
import json


def get_weather_data(lat, lon, start_date, end_date):
    """
    Get historical weather data using Open-Meteo API (free, no key required)

    Args:
        lat, lon: Your location coordinates
        start_date, end_date: Date range as strings 'YYYY-MM-DD'

    Returns:
        pandas.DataFrame: Weather data with solar-relevant metrics
    """

    base_url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "sunshine_duration",
            "precipitation_sum",
            "windspeed_10m_max",
            "shortwave_radiation_sum",
            "cloudcover_mean",
        ],
        "timezone": "auto",
    }

    print(f"Fetching weather data for coordinates ({lat}, {lon})")
    print(f"Date range: {start_date} to {end_date}")

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()

        # Convert to DataFrame
        weather_df = pd.DataFrame(data["daily"])
        weather_df["date"] = pd.to_datetime(weather_df["time"])
        weather_df.set_index("date", inplace=True)
        weather_df.drop("time", axis=1, inplace=True)

        # Rename columns for clarity
        weather_df.columns = [
            "temp_max_c",
            "temp_min_c",
            "temp_mean_c",
            "sunshine_duration_s",
            "precipitation_mm",
            "windspeed_max_kmh",
            "solar_radiation_mj",
            "cloudcover_mean_pct",
        ]

        # Convert sunshine duration to hours
        weather_df["sunshine_hours"] = weather_df["sunshine_duration_s"] / 3600

        # Create weather quality score for solar production
        weather_df["solar_weather_score"] = (
            (weather_df["sunshine_hours"] / 12) * 0.4  # Sunshine weight
            + ((100 - weather_df["cloudcover_mean_pct"]) / 100)
            * 0.4  # Clear sky weight
            + (
                weather_df["solar_radiation_mj"]
                / weather_df["solar_radiation_mj"].max()
            )
            * 0.2
            # Solar radiation weight
        )

        print(f"Successfully fetched {len(weather_df)} days of weather data")
        return weather_df

    else:
        print(f"Error fetching weather data: {response.status_code}")
        print(f"Response: {response.text}")
        return None


def merge_weather_solar_data(solar_daily, weather_df):
    """Merge solar production data with weather data"""

    # Ensure both have datetime indices
    if not isinstance(solar_daily.index, pd.DatetimeIndex):
        solar_daily.index = pd.to_datetime(solar_daily.index)
    if not isinstance(weather_df.index, pd.DatetimeIndex):
        weather_df.index = pd.to_datetime(weather_df.index)

    # Merge on date index
    merged_data = solar_daily.join(weather_df, how="inner")

    print(f"Merged dataset has {len(merged_data)} days")
    print(f"Weather data coverage: {len(merged_data) / len(solar_daily) * 100:.1f}%")

    return merged_data


def save_weather_data(weather_df, filename="weather_data.csv"):
    """Save weather data to avoid re-fetching"""
    project_root = Path(__file__).parent.parent
    file_path = project_root / "data" / "processed" / filename

    # Create processed directory if it doesn't exist
    file_path.parent.mkdir(exist_ok=True)

    weather_df.to_csv(file_path)
    print(f"Weather data saved to {file_path}")


def load_weather_data(filename="weather_data.csv"):
    """Load previously saved weather data"""
    project_root = Path(__file__).parent.parent
    file_path = project_root / "data" / "processed" / filename

    if file_path.exists():
        weather_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"Loaded weather data from {file_path}")
        return weather_df
    else:
        print(f"No saved weather data found at {file_path}")
        return None
