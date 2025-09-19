# Solar Panel Energy Analysis

AI-powered analysis of Enphase solar panel data for production forecasting and optimization.

## Features
- Solar production forecasting using machine learning
- Seasonal pattern analysis
- Energy consumption optimization
- Anomaly detection for system health

## Quick Start (No Solar Data Required!)
🚀 **Try immediately with mock data:**
1. Clone repository
2. Install UV: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or see [UV docs](https://docs.astral.sh/uv/))
3. Install dependencies: `uv sync`
4. Generate mock data: `uv run python scripts/generate_mock_data.py`
5. Run demo: `uv run jupyter lab notebooks/00_quick_start_demo.ipynb`

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
```

**Without API credentials:** All notebooks work perfectly with CSV data only (demo mode).

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
- `mock_solar_data.csv`: 3 months of simulated 10kW residential system data
- Your own Enphase CSV exports
- Live API data integration