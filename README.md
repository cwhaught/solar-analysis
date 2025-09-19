# Solar Panel Energy Analysis

AI-powered analysis of Enphase solar panel data for production forecasting and optimization.

## Features
- Solar production forecasting using machine learning
- Seasonal pattern analysis
- Energy consumption optimization
- Anomaly detection for system health

## Setup
1. Clone repository
2. Install UV: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or see [UV docs](https://docs.astral.sh/uv/))
3. Install dependencies: `uv sync`
4. Activate environment: `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
5. Add your Enphase data to `data/raw/`
6. **Optional**: Configure Enphase API for live data (see [API Setup](#api-setup))
7. Run analysis: `uv run jupyter lab` or work in notebooks/

## API Setup (Optional)
For live API data integration, create a `.env` file in the project root:

```bash
# Enphase API Credentials - DO NOT COMMIT TO GIT
ENPHASE_CLIENT_ID=your_client_id
ENPHASE_CLIENT_SECRET=your_client_secret
ENPHASE_API_KEY=your_api_key
ENPHASE_ACCESS_TOKEN=your_access_token
ENPHASE_REFRESH_TOKEN=your_refresh_token
ENPHASE_SYSTEM_ID=your_system_id
```

**Getting Credentials:**
1. Register at [Enphase Developer Portal](https://developer.enphase.com/)
2. Create an application to get Client ID and Secret
3. Use the OAuth setup utility: `uv run python src/setup/oauth_setup.py`
4. The notebooks will automatically detect and use these credentials

**Without API credentials:** All notebooks work perfectly with CSV data only (demo mode).

## Development
- Install with dev dependencies: `uv sync --extra dev`
- Run tests: `uv run pytest`
- Format code: `uv run black .`

## Data Structure
**Hybrid Data Approach:**
- **CSV Data**: 15-minute interval historical data (2+ years) for comprehensive analysis
- **API Data**: Live daily updates for recent production data
- **Intelligent Merging**: Automatically combines CSV + API for complete dataset

**Data Types:**
- Production, consumption, and grid import/export tracking
- 15-minute granularity for detailed analysis
- Daily aggregations for ML model training