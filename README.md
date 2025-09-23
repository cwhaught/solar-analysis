# Solar Panel Energy Analysis

🌟 **AI-powered solar analysis that works instantly - no setup required!**

Get immediate insights into solar production patterns, financial returns, and optimization opportunities using realistic location-specific data or your own Enphase system data.

## ⚡ Try It Now (60 seconds)

```bash
# Clone and run demo instantly
git clone <https://github.com/cwhaught/solar-analysis.git | git@github.com:cwhaught/solar-analysis.git>
cd solar-analysis
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install UV package manager
uv sync                                           # Install dependencies
uv run python scripts/generate_mock_data.py denver  # Generate realistic data
uv run jupyter lab notebooks/00_quick_start_demo.ipynb  # Launch analysis!
```

**🌍 Available locations:** `denver`, `phoenix`, `miami`, `seattle`, `new_york`, `los_angeles`, `chicago`, `atlanta`, `london`, `berlin`, `tokyo`, `sydney`

**🎯 Custom coordinates:** `uv run python scripts/generate_mock_data.py 37.7749 -122.4194 "San Francisco"`

## 🎯 What You Get

- 🤖 **ML-powered forecasting** - Predict solar production with advanced models
- 🌍 **Location-aware analysis** - Solar geometry, weather patterns, seasonal variations
- 💰 **Smart financial modeling** - ROI, payback periods, location-specific electricity rates
- ⚡ **Energy optimization** - Maximize self-consumption and grid export revenue
- 🚨 **Performance monitoring** - Anomaly detection and system health analysis
- 📊 **Interactive dashboards** - Jupyter notebooks with rich visualizations

## 🏠 Have Real Solar Data?

<details>
<summary><strong>📤 Setup for Enphase System Owners</strong></summary>

### CSV Data (Recommended)
1. Download your Enphase production data as CSV from [Enphase Enlighten](https://enlighten.enphaseenergy.com/)
2. Place CSV files in `data/raw/` directory
3. Run notebooks - they automatically detect and use your real data!

### Live API Integration (Optional)
For real-time data updates:

```bash
# Copy configuration template
cp .env.template .env

# Get API credentials from Enphase Developer Portal
https://developer-v4.enphase.com/docs/quickstart.html

# Run OAuth setup wizard
uv run python src/setup/oauth_setup.py

# Edit .env with your credentials and run notebooks
```

**Your `.env` file:**
```bash
# Enphase API Credentials
ENPHASE_CLIENT_ID=your_client_id
ENPHASE_CLIENT_SECRET=your_client_secret
ENPHASE_SYSTEM_ID=your_system_id

# Location (for electricity rates and solar modeling)
SOLAR_LOCATION_CITY=denver

# System Financial Configuration
SOLAR_SYSTEM_SIZE_KW=10.0
SOLAR_SYSTEM_TOTAL_COST=25000
SOLAR_FEDERAL_TAX_CREDIT_PERCENT=30
```

</details>

## ⚙️ Configuration Options

<details>
<summary><strong>🏠 System Financial Configuration</strong></summary>

Configure your solar system specifications for accurate financial modeling:

```bash
# System specifications
SOLAR_SYSTEM_SIZE_KW=10.0              # System size in kilowatts
SOLAR_SYSTEM_COST_PER_KW=2500          # Cost per kW installed
SOLAR_SYSTEM_TOTAL_COST=25000          # Total system cost (overrides per-kW calc)

# Rebates and incentives
SOLAR_FEDERAL_TAX_CREDIT_PERCENT=30    # Federal tax credit (%)
SOLAR_STATE_REBATE=5000                # State rebates ($)
SOLAR_UTILITY_REBATE=2000              # Utility incentives ($)
SOLAR_OTHER_REBATES=1000               # Other rebates ($)
```

**Automatic calculations:**
- Federal tax credit = Total cost × Tax credit percentage
- Net system cost = Total cost - All rebates and credits
- Payback period = Net cost ÷ Annual electricity savings

</details>

<details>
<summary><strong>💰 Electricity Rates & Location</strong></summary>

Get accurate financial projections with location-specific electricity rates:

```bash
# Location configuration (choose one)
SOLAR_LOCATION_CITY=denver             # Use predefined city
# OR custom coordinates:
# SOLAR_LOCATION_LATITUDE=39.7392
# SOLAR_LOCATION_LONGITUDE=-104.9903

# Enhanced rate accuracy (optional)
NREL_API_KEY=your_free_api_key_here    # Get from https://developer.nrel.gov/signup/
```

**Rate sources (in priority order):**
1. **NREL API** - Utility-specific rates (most accurate)
2. **State averages** - 2025 state-level residential rates
3. **National average** - 16.22¢/kWh fallback

**Features:**
- Automatic rate detection based on your location
- Feed-in tariff estimation for solar exports
- National comparisons and benchmarking

</details>

## 📊 Data Sources

The system supports multiple data sources and automatically combines them:

- **🌍 Location-specific mock data** - Realistic simulated data with accurate solar geometry
- **📤 Your Enphase CSV exports** - Historical production data from your system
- **🔄 Live Enphase API** - Real-time updates and recent production data
- **📈 Hybrid approach** - Automatically merges CSV + API for complete datasets

**Data features:**
- 15-minute interval granularity for detailed analysis
- Production, consumption, and grid import/export tracking
- Climate-specific weather patterns and seasonal variations
- Enhanced ML model accuracy with location-aware features

## 🛠️ Development

```bash
# Development setup
uv sync --extra dev
uv run pytest                    # Run test suite
uv run black .                   # Format code
```

**Project structure:**
- `notebooks/` - Interactive analysis notebooks
- `src/core/` - Core analysis modules (location, rates, ML models)
- `src/api/` - Enphase API integration
- `data/` - Data storage (CSV files, generated data)
- `tests/` - Comprehensive test suite

## 🎨 Example Outputs

The notebooks generate rich visualizations including:
- Daily/monthly/seasonal production patterns
- Financial projections and ROI analysis
- ML model performance and forecasting accuracy
- Location-specific solar geometry and irradiance
- Performance benchmarking and anomaly detection

---

**🚀 Ready to analyze your solar data? Start with the 60-second demo above!**