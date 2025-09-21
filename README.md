# Solar Panel Energy Analysis

AI-powered analysis of Enphase solar panel data for production forecasting and optimization.

## Features
- 🌍 **Location-aware modeling** - Solar calculations based on your geographic location
- 🤖 **Machine learning forecasting** - Predict solar production using advanced ML models
- 📊 **Seasonal pattern analysis** - Understand how location affects seasonal variations
- ⚡ **Energy optimization** - Optimize consumption patterns for maximum solar benefit
- 🚨 **Anomaly detection** - Monitor system health and identify performance issues
- ☀️ **Solar geometry** - Accurate sunrise/sunset times and solar elevation calculations
- 💰 **Location-based electricity rates** - Automatic electricity rate lookup for accurate financial analysis

## Quick Start (No Solar Data Required!)
🚀 **Try immediately with location-specific mock data:**
1. Clone repository
2. Install UV: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or see [UV docs](https://docs.astral.sh/uv/))
3. Install dependencies: `uv sync`
4. Generate location-specific data: `uv run python scripts/generate_mock_data.py denver` (or your city)
5. Run demo: `uv run jupyter lab notebooks/00_quick_start_demo.ipynb`

🌍 **Location options:**
- **Predefined cities**: `denver`, `phoenix`, `miami`, `seattle`, `new_york`, `los_angeles`, `chicago`, `atlanta`, `london`, `berlin`, `tokyo`, `sydney`
- **Custom coordinates**: `uv run python scripts/generate_mock_data.py 37.7749 -122.4194 "San Francisco"`
- **Generic data**: `uv run python scripts/generate_mock_data.py` (no location)

## Full Setup (For Real Solar Data)
1. Clone repository
2. Install UV: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or see [UV docs](https://docs.astral.sh/uv/))
3. Install dependencies: `uv sync`
4. Activate environment: `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
5. Add your Enphase data to `data/raw/`
6. **Optional**: Configure API (copy `.env.template` → `.env` and add credentials)
7. Run analysis: `uv run jupyter lab` or work in notebooks/

## API Setup (Optional)
For live API data integration, configure your Enphase credentials:

**Setup Steps:**
1. **Copy template**: `cp .env.template .env`
2. **Get credentials**: Register at [Enphase Developer Portal](https://developer.enphase.com/)
3. **Create application** to get Client ID and Secret
4. **Run OAuth setup**: `uv run python src/setup/oauth_setup.py`
5. **Fill in `.env`** with your actual credentials
6. **Verify**: The notebooks will automatically detect and use these credentials

**Your `.env` file should look like:**
```bash
# Enphase API Credentials - DO NOT COMMIT TO GIT
ENPHASE_CLIENT_ID=your_actual_client_id
ENPHASE_CLIENT_SECRET=your_actual_client_secret
ENPHASE_API_KEY=your_actual_api_key
ENPHASE_ACCESS_TOKEN=your_actual_access_token
ENPHASE_REFRESH_TOKEN=your_actual_refresh_token
ENPHASE_SYSTEM_ID=your_actual_system_id

# Location Configuration (Optional)
SOLAR_LOCATION_CITY=denver  # or your city
# OR use custom coordinates:
# SOLAR_LOCATION_LATITUDE=39.7392
# SOLAR_LOCATION_LONGITUDE=-104.9903

# Electricity Rates (Optional - for enhanced financial analysis)
# Get free API key from https://developer.nrel.gov/signup/
NREL_API_KEY=your_nrel_api_key_here
```

**Without API credentials:** All notebooks work perfectly with CSV data only (demo mode).

## Electricity Rates & Financial Analysis
💰 **Automatic location-based electricity rate lookup for accurate financial projections:**

**Features:**
- 🌍 **Automatic rate detection** - Uses your configured location to find local electricity rates
- 📊 **Multiple data sources** - NREL API for utility-specific rates + state averages fallback
- 💡 **Feed-in tariff estimation** - Calculates solar export compensation (typically 80% of retail rate)
- 📈 **National comparisons** - Shows how your local rates compare to national average
- 🎯 **High accuracy** - Uses 2025 state-level data with optional NREL API enhancement

**Rate Sources (in priority order):**
1. **NREL API** (highest accuracy) - Utility-specific rates from coordinates
2. **State averages** (medium accuracy) - 2025 state-level residential rates
3. **National average** (fallback) - 16.22¢/kWh national residential average

**Configuration:**
```bash
# Basic location (uses state average rates)
SOLAR_LOCATION_CITY=denver

# Enhanced rates (utility-specific via NREL API)
NREL_API_KEY=your_free_api_key_from_nrel_gov
```

**Usage in notebooks:**
All financial analysis automatically uses location-specific rates - no hardcoded values!

## Development
- Install with dev dependencies: `uv sync --extra dev`
- Run tests: `uv run pytest`
- Format code: `uv run black .`

## Data Structure
**Hybrid Data Approach:**
- **CSV Data**: 15-minute interval historical data (2+ years) for comprehensive analysis
- **API Data**: Live daily updates for recent production data
- **Mock Data**: Realistic simulated data for testing and demos
- **Intelligent Merging**: Automatically combines CSV + API for complete dataset

**Data Types:**
- Production, consumption, and grid import/export tracking
- 15-minute granularity for detailed analysis
- Daily aggregations for ML model training
- Mock data generator creates realistic seasonal and daily patterns

**Available Datasets:**
- 🌍 **Location-specific mock data**: `mock_solar_data_[city].csv` with accurate solar geometry
- 🏠 **Generic mock data**: `mock_solar_data.csv` for general testing
- 📊 **Your Enphase CSV exports**: Real historical data from your system
- 🔄 **Live API integration**: Fresh data from Enphase API

**Location Features:**
- Accurate sunrise/sunset times for your latitude
- Solar elevation angles and theoretical irradiance calculations
- Climate-specific weather patterns and seasonal variations
- Location-based electricity rates for accurate financial modeling
- Enhanced ML model accuracy with location-aware features