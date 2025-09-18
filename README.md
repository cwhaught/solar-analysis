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
6. Run analysis: `uv run jupyter lab` or work in notebooks/

## Development
- Install with dev dependencies: `uv sync --extra dev`
- Run tests: `uv run pytest`
- Format code: `uv run black .`

## Data Structure
- 15-minute interval data from Enphase systems
- Production, consumption, and grid import/export tracking
- 2+ years of historical data for ML training