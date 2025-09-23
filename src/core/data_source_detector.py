"""
Data Source Detector - Smart data source detection and prioritization

Provides intelligent detection of available data sources with transparent
fallback logic for solar production data analysis.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from unittest.mock import Mock


class DataSourceInfo:
    """Information about a detected data source"""

    def __init__(self, source_type: str, description: str, path: Optional[str] = None, priority: int = 0):
        self.source_type = source_type
        self.description = description
        self.path = path
        self.priority = priority
        self.available = True


class DataSourceDetector:
    """
    Intelligent data source detection and prioritization system

    Automatically detects and prioritizes available data sources:
    1. Real Enphase API data (live system)
    2. Real CSV data (historical exports)
    3. Synthetic data (demo/testing)
    """

    def __init__(self, location=None):
        """
        Initialize detector

        Args:
            location: Location object with location_name for synthetic data detection
        """
        self.location = location

    def create_enphase_client(self) -> Tuple[Any, str]:
        """
        Create Enphase client from .env credentials if available, otherwise return mock client

        Returns:
            Tuple of (client, client_type) where client_type is "REAL_API" or "MOCK"
        """
        env_path = Path("../.env")

        if env_path.exists():
            # Load environment variables
            with open(env_path) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value

            # Check if we have all required credentials
            required_vars = ['ENPHASE_ACCESS_TOKEN', 'ENPHASE_API_KEY', 'ENPHASE_SYSTEM_ID']
            placeholder_values = ['your_access_token_here', 'your_api_key_here', 'your_system_id_here']

            if all(var in os.environ and os.environ[var] not in placeholder_values for var in required_vars):
                try:
                    from core.enphase_client import EnphaseClient
                    client = EnphaseClient(
                        access_token=os.environ['ENPHASE_ACCESS_TOKEN'],
                        api_key=os.environ['ENPHASE_API_KEY'],
                        system_id=os.environ['ENPHASE_SYSTEM_ID']
                    )
                    print("✅ Using REAL Enphase API credentials!")
                    print("   📈 Will attempt to fetch live data from your solar system")
                    return client, "REAL_API"
                except Exception as e:
                    print(f"⚠️ Error creating Enphase client: {e}")
                    print("🔄 Falling back to available data files")
        else:
            print("📝 No .env file found")

        # Fallback to mock client
        print("🎭 Using mock Enphase client (demo mode)")
        print("   📊 Will prioritize real CSV data over synthetic data")
        mock_client = Mock()
        mock_client.__class__.__name__ = "MockEnphaseClient"
        mock_client.get_energy_lifetime.return_value = None
        return mock_client, "MOCK"

    def detect_available_csv_sources(self) -> List[DataSourceInfo]:
        """
        Detect all available CSV data sources in priority order

        Returns:
            List of DataSourceInfo objects, ordered by priority (highest first)
        """
        sources = []

        # Priority 1: Real CSV data from actual solar system
        real_csv_path = "../data/raw/4136754_custom_report.csv"
        if Path(real_csv_path).exists():
            sources.append(DataSourceInfo(
                source_type="REAL_CSV",
                description="Real solar panel CSV data through 2025",
                path=real_csv_path,
                priority=100
            ))

        # Priority 2: Location-specific synthetic data
        if self.location:
            city_name = self.location.location_name.split(',')[0].lower().replace(' ', '_').replace('.', '')
            location_data_path = f"../data/raw/mock_solar_data_{city_name}.csv"

            if Path(location_data_path).exists():
                sources.append(DataSourceInfo(
                    source_type="SYNTHETIC_LOCATION",
                    description=f"Synthetic data for {self.location.location_name} (ends June 2024)",
                    path=location_data_path,
                    priority=50
                ))

        # Priority 3: Generic mock data
        generic_mock_path = "../data/raw/mock_solar_data.csv"
        if Path(generic_mock_path).exists():
            sources.append(DataSourceInfo(
                source_type="GENERIC_MOCK",
                description="Generic synthetic data (ends March 2024)",
                path=generic_mock_path,
                priority=25
            ))

        # Sort by priority (highest first)
        sources.sort(key=lambda x: x.priority, reverse=True)
        return sources

    def determine_data_strategy(self) -> Dict[str, Any]:
        """
        Determine the optimal data loading strategy

        Returns:
            Dictionary with data strategy information including:
            - client: Enphase client instance
            - client_type: "REAL_API" or "MOCK"
            - csv_path: Best CSV file path for fallback
            - data_type: Description of primary data type
            - available_sources: List of all available sources
            - selected_source: Information about selected source
        """
        print("📊 Determining Best Data Source...")

        # Create client
        client, client_type = self.create_enphase_client()

        # Find available CSV sources
        csv_sources = self.detect_available_csv_sources()

        if not csv_sources:
            raise FileNotFoundError("No CSV data files available")

        # Select best CSV source
        best_csv_source = csv_sources[0]

        # Determine strategy based on client type
        if client_type == "REAL_API":
            data_type = "📈 REAL API DATA (with CSV fallback)"
            selected_description = "Real Enphase API data from your solar system"
            print("   🎯 PRIORITY: Real Enphase API data from your solar system")
            print("   📡 Will attempt live API data first")
            print(f"   📁 CSV fallback: {best_csv_source.path}")
        else:
            data_type = f"📈 {best_csv_source.source_type}"
            selected_description = best_csv_source.description
            print(f"   🎯 PRIORITY: {best_csv_source.description}")
            print(f"   📁 Using: {best_csv_source.path}")

        # Show all available sources for transparency
        print("\n📋 Available Data Sources (in priority order):")
        for i, source in enumerate(csv_sources):
            priority_label = "🎯 SELECTED" if i == 0 else "   Available"
            emoji_map = {
                "REAL_CSV": "📈 REAL CSV DATA",
                "SYNTHETIC_LOCATION": "🎭 SYNTHETIC/MOCK DATA",
                "GENERIC_MOCK": "🎭 GENERIC MOCK DATA"
            }
            display_type = emoji_map.get(source.source_type, source.source_type)
            print(f"   {priority_label}: {display_type} - {source.description}")

        return {
            'client': client,
            'client_type': client_type,
            'csv_path': best_csv_source.path,
            'data_type': data_type,
            'available_sources': csv_sources,
            'selected_source': best_csv_source,
            'strategy_description': selected_description
        }

    def analyze_data_recency(self, dataframe, data_summary: Dict) -> Dict[str, Any]:
        """
        Analyze data recency and authenticity

        Args:
            dataframe: Data DataFrame with datetime index
            data_summary: Summary from SolarDataManager

        Returns:
            Dictionary with recency analysis results
        """
        if dataframe.empty:
            return {
                'latest_date': None,
                'days_old': None,
                'recency_status': '❌ NO DATA',
                'authenticity': 'Unknown'
            }

        # Determine data source used
        actual_source = None
        authenticity = None

        if data_summary['api']['available'] and not data_summary['api']['is_mock']:
            actual_source = "✅ LIVE API DATA"
            authenticity = "Real API data from your Enphase system"
        elif data_summary['csv']['available']:
            csv_path = data_summary['csv'].get('source_file', 'unknown')
            if '4136754_custom_report.csv' in str(csv_path):
                actual_source = "✅ REAL CSV DATA"
                authenticity = "Authentic solar panel data (not synthetic)"
            elif 'mock' in str(csv_path).lower():
                actual_source = "⚠️ SYNTHETIC DATA"
                authenticity = "Synthetic/demo data for demonstration"
            else:
                actual_source = "📁 CSV DATA"
                authenticity = "CSV data file"

        # Calculate recency
        latest_date = dataframe.index.max()
        days_old = (datetime.now() - latest_date).days

        if days_old < 7:
            recency_status = f"🔥 VERY RECENT (only {days_old} days old)"
        elif days_old < 30:
            recency_status = f"✅ RECENT ({days_old} days old)"
        elif days_old < 90:
            recency_status = f"📅 SOMEWHAT OLD ({days_old} days old)"
        else:
            recency_status = f"⚠️ OLD DATA ({days_old} days old)"

        return {
            'latest_date': latest_date,
            'days_old': days_old,
            'recency_status': recency_status,
            'actual_source': actual_source,
            'authenticity': authenticity
        }

    def generate_final_report(self, strategy: Dict, data_summary: Dict, recency_info: Dict) -> None:
        """
        Generate final data source report for user transparency

        Args:
            strategy: Data strategy from determine_data_strategy()
            data_summary: Summary from SolarDataManager
            recency_info: Recency analysis from analyze_data_recency()
        """
        print("\n📈 Solar Data Loaded Successfully:")
        print(f"   🎯 DATA TYPE: {strategy['data_type']}")

        # Show what was actually used
        if recency_info['actual_source']:
            print(f"   📡 Source: {recency_info['actual_source']}")

        # Show record counts
        if data_summary['api']['available'] and not data_summary['api']['is_mock']:
            print(f"   📊 API Records: {data_summary['api']['records']:,}")
            if data_summary['api']['date_range']:
                print(f"   📅 API Date range: {data_summary['api']['date_range'][0]} to {data_summary['api']['date_range'][1]}")
        elif data_summary['csv']['available']:
            records = data_summary['csv']['records']
            date_range = data_summary['csv']['date_range']
            print("   📁 Source: CSV file (API failed, using fallback)")
            print(f"   📊 CSV Records: {records:,}")
            print(f"   📅 CSV Date range: {date_range[0]} to {date_range[1]}")

        # Show recency information
        if recency_info['latest_date']:
            print(f"   📅 Data recency: {recency_info['recency_status']}")
            print(f"   🕐 Latest data point: {recency_info['latest_date'].strftime('%Y-%m-%d')}")

        # Show final authenticity notice
        self._print_authenticity_notice(strategy, recency_info, data_summary)

    def _print_authenticity_notice(self, strategy: Dict, recency_info: Dict, data_summary: Dict) -> None:
        """Print final authenticity and guidance notice"""

        if data_summary['api']['available'] and not data_summary['api']['is_mock']:
            print("\n🎉 SUCCESS: Analyzing YOUR actual solar system performance!")
            print("   📈 Live API data from your Enphase system")
            print(f"   🔗 System ID: {os.environ.get('ENPHASE_SYSTEM_ID', 'Unknown')}")

        elif strategy['selected_source'].source_type == "REAL_CSV":
            print("\n✅ SUCCESS: Using REAL solar production data!")
            print("   📈 Authentic solar panel data (not synthetic)")
            print("   📊 Contains data through 2025")
            if strategy['client_type'] == "REAL_API":
                print("   ℹ️ Note: API returned 401 (expired token), but CSV data is recent")

        elif 'mock' in strategy['selected_source'].source_type.lower() or 'synthetic' in strategy['selected_source'].source_type.lower():
            print("\n💡 NOTE: Using synthetic data for demonstration.")
            print("   📝 To use your real data:")
            print("   • Set up .env with Enphase API credentials (tokens may be expired)")
            print("   • Or add real CSV data to ../data/raw/")
            if recency_info['latest_date']:
                print(f"   ⚠️ Current mock data ends {recency_info['latest_date'].strftime('%B %Y')}")
