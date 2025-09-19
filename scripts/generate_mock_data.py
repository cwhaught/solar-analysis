#!/usr/bin/env python3
"""
Generate mock solar energy data for demonstration purposes

Creates realistic 15-minute interval solar production and consumption data
based on patterns observed in real Enphase system data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import random

def generate_mock_solar_data(start_date="2024-01-01", num_days=90):
    """
    Generate realistic mock solar energy data

    Args:
        start_date: Start date for data generation
        num_days: Number of days to generate

    Returns:
        DataFrame with mock solar data
    """

    # Create 15-minute intervals
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=num_days)

    # Generate timestamp range (15-minute intervals)
    timestamps = pd.date_range(start=start_dt, end=end_dt, freq='15min')[:-1]  # Exclude last point

    data = []

    for ts in timestamps:
        # Calculate solar production based on time of day and season
        production = calculate_solar_production(ts)

        # Calculate realistic consumption
        consumption = calculate_consumption(ts)

        # Calculate grid import/export based on production vs consumption
        if production > consumption:
            # Surplus: export to grid
            exported = production - consumption
            imported = 0
        else:
            # Deficit: import from grid
            exported = 0
            imported = consumption - production

        data.append({
            'Date/Time': ts.strftime('%m/%d/%Y %H:%M'),
            'Energy Produced (Wh)': max(0, int(production)),
            'Energy Consumed (Wh)': max(0, int(consumption)),
            'Exported to Grid (Wh)': max(0, int(exported)),
            'Imported from Grid (Wh)': max(0, int(imported))
        })

    return pd.DataFrame(data)

def calculate_solar_production(timestamp):
    """Calculate realistic solar production for given timestamp"""

    # Base production parameters (for a ~10kW system)
    max_production = 1950  # Peak 15-min production in Wh (roughly 7.8kW)

    # Get time components
    hour = timestamp.hour
    minute = timestamp.minute
    day_of_year = timestamp.timetuple().tm_yday

    # No production at night (before 6 AM or after 7 PM)
    if hour < 6 or hour >= 19:
        return 0

    # Seasonal adjustment (higher in summer, lower in winter)
    # Peak around day 172 (summer solstice ~June 21)
    seasonal_factor = 0.7 + 0.3 * math.cos(2 * math.pi * (day_of_year - 172) / 365)

    # Daily solar curve (bell curve peaking around noon)
    time_decimal = hour + minute / 60.0
    peak_time = 12.5  # Peak around 12:30 PM
    curve_width = 4.5  # How wide the production curve is

    # Gaussian-like curve for daily production
    time_factor = math.exp(-((time_decimal - peak_time) ** 2) / (2 * curve_width ** 2))

    # Weather variability (random clouds, etc.)
    weather_factor = random.uniform(0.6, 1.0)  # 60-100% of clear sky
    if random.random() < 0.15:  # 15% chance of very cloudy
        weather_factor *= random.uniform(0.1, 0.4)

    # Calculate production
    production = max_production * seasonal_factor * time_factor * weather_factor

    # Add some random noise
    production *= random.uniform(0.95, 1.05)

    return max(0, production)

def calculate_consumption(timestamp):
    """Calculate realistic household energy consumption"""

    # Base consumption patterns
    hour = timestamp.hour
    day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
    day_of_year = timestamp.timetuple().tm_yday

    # Base consumption (varies by time of day)
    if 0 <= hour < 6:  # Late night - low consumption
        base_consumption = random.uniform(80, 150)
    elif 6 <= hour < 9:  # Morning - medium consumption
        base_consumption = random.uniform(200, 450)
    elif 9 <= hour < 17:  # Daytime - variable consumption
        base_consumption = random.uniform(150, 400)
    elif 17 <= hour < 22:  # Evening - high consumption
        base_consumption = random.uniform(300, 800)
    else:  # 22-24 - medium consumption
        base_consumption = random.uniform(150, 300)

    # Weekend adjustment (higher consumption during day)
    if day_of_week >= 5:  # Weekend
        if 9 <= hour < 17:
            base_consumption *= random.uniform(1.2, 1.8)

    # Seasonal adjustment (heating/cooling)
    # Higher consumption in winter and summer
    seasonal_factor = 1.0 + 0.4 * abs(math.cos(2 * math.pi * day_of_year / 365))

    # Occasional high-consumption events (appliances, etc.)
    if random.random() < 0.05:  # 5% chance
        base_consumption *= random.uniform(2.0, 4.0)

    consumption = base_consumption * seasonal_factor

    # Add random variation
    consumption *= random.uniform(0.8, 1.2)

    return max(50, consumption)  # Minimum base load

def main():
    """Generate and save mock data"""
    print("🔄 Generating mock solar energy data...")

    # Generate 3 months of data (good for demo)
    mock_data = generate_mock_solar_data(start_date="2024-01-01", num_days=90)

    print(f"✅ Generated {len(mock_data):,} records ({len(mock_data)/96:.0f} days)")
    print(f"📅 Date range: {mock_data.iloc[0]['Date/Time']} to {mock_data.iloc[-1]['Date/Time']}")

    # Display sample statistics
    production_kwh = mock_data['Energy Produced (Wh)'].sum() / 1000
    consumption_kwh = mock_data['Energy Consumed (Wh)'].sum() / 1000
    exported_kwh = mock_data['Exported to Grid (Wh)'].sum() / 1000

    print(f"\n📊 Summary Statistics:")
    print(f"  Total production: {production_kwh:.1f} kWh")
    print(f"  Total consumption: {consumption_kwh:.1f} kWh")
    print(f"  Total exported: {exported_kwh:.1f} kWh")
    print(f"  Net energy balance: {production_kwh - consumption_kwh:.1f} kWh")
    print(f"  Daily avg production: {production_kwh / 90:.1f} kWh/day")
    print(f"  Daily avg consumption: {consumption_kwh / 90:.1f} kWh/day")

    # Save to CSV
    output_path = "data/raw/mock_solar_data.csv"
    mock_data.to_csv(output_path, index=False)
    print(f"\n💾 Saved mock data to: {output_path}")

    # Display first few rows
    print(f"\n📋 Sample data (first 5 rows):")
    print(mock_data.head().to_string(index=False))

    print(f"\n🎯 Mock data is ready for analysis!")
    print(f"   To use: Update notebooks to point to 'mock_solar_data.csv'")

if __name__ == "__main__":
    main()