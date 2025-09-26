"""
Notebook Utilities - Common initialization patterns for Jupyter notebooks

Provides standardized functions for setting up data managers, locations, and
other common notebook initialization tasks to eliminate code duplication.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Add src to path for imports
if '../src' not in sys.path:
    sys.path.append('../src')

from core.data_manager import SolarDataManager
from core.data_source_detector import DataSourceDetector
from core.location_loader import create_notebook_location


class NotebookEnvironment:
    """
    Manages common notebook setup and initialization
    """

    def __init__(self):
        self.location = None
        self.detector = None
        self.strategy = None
        self.data_manager = None
        self.data_summary = None
        self.recency_info = None

    def setup_complete_environment(self, cache_dir: str = "../data/processed") -> Dict[str, Any]:
        """
        Complete notebook setup with location, data source detection, and data manager

        Args:
            cache_dir: Directory for caching processed data

        Returns:
            Dictionary containing all initialized components
        """
        # Initialize location
        self.location = create_notebook_location()

        # Initialize smart data source detector
        self.detector = DataSourceDetector(location=self.location)

        # Determine optimal data loading strategy
        self.strategy = self.detector.determine_data_strategy()

        # Initialize data manager with detected strategy
        self.data_manager = SolarDataManager(
            csv_path=self.strategy['csv_path'],
            enphase_client=self.strategy['client'],
            cache_dir=cache_dir
        )

        print(f"✅ Notebook environment initialized for {self.location.location_name}")

        return {
            'location': self.location,
            'detector': self.detector,
            'strategy': self.strategy,
            'data_manager': self.data_manager
        }

    def load_and_analyze_data(self) -> Dict[str, Any]:
        """
        Load data and perform comprehensive analysis

        Returns:
            Dictionary containing loaded data and analysis results
        """
        if not self.data_manager:
            raise RuntimeError("Must call setup_complete_environment() first")

        # Load data
        csv_data = self.data_manager.load_csv_data()
        daily_data = self.data_manager.get_daily_production()

        # Get comprehensive data summary and recency analysis
        self.data_summary = self.data_manager.get_data_summary()
        self.recency_info = self.detector.analyze_data_recency(csv_data, self.data_summary)

        # Generate comprehensive data source report
        self.detector.generate_final_report(self.strategy, self.data_summary, self.recency_info)

        return {
            'csv_data': csv_data,
            'daily_data': daily_data,
            'data_summary': self.data_summary,
            'recency_info': self.recency_info
        }

    def get_location_context(self) -> Dict[str, Any]:
        """
        Get location-specific context information

        Returns:
            Dictionary with location summary and context
        """
        if not self.location:
            raise RuntimeError("Must call setup_complete_environment() first")

        summary = self.location.get_location_summary()

        print(f"\n🌍 Location Context:")
        print(f"  System location: {self.location.location_name}")
        print(f"  Coordinates: {self.location.latitude:.3f}°N, {self.location.longitude:.3f}°W")
        print(f"  Climate type: {summary['climate_type']}")
        print(f"  Timezone: {self.location.timezone_str or 'Not specified'}")

        return summary


def quick_setup() -> Tuple[SolarDataManager, DataSourceDetector, Any, Dict]:
    """
    Quick setup for simple notebooks - one-line initialization

    Returns:
        Tuple of (data_manager, detector, location, strategy)
    """
    env = NotebookEnvironment()
    components = env.setup_complete_environment()

    return (
        components['data_manager'],
        components['detector'],
        components['location'],
        components['strategy']
    )


def load_with_analysis() -> Dict[str, Any]:
    """
    Complete setup with data loading and analysis

    Returns:
        Dictionary containing all components and loaded data
    """
    env = NotebookEnvironment()
    components = env.setup_complete_environment()
    data_results = env.load_and_analyze_data()
    location_context = env.get_location_context()

    # Combine all results
    return {
        **components,
        **data_results,
        'location_context': location_context
    }


def print_notebook_header(title: str, description: str = "") -> None:
    """
    Print standardized notebook header

    Args:
        title: Main title for the notebook
        description: Optional description text
    """
    print(f"📊 {title}")
    print("=" * (len(title) + 4))

    if description:
        print(f"\n{description}")
        print()