"""
Core business logic modules

This package contains the core functionality for the solar energy analysis system:
- EnphaseClient: API client for Enphase Energy systems
- SolarDataManager: Unified data management for CSV and API sources
- EnphaseOAuthSetup: OAuth authentication setup and credential management
"""

from .enphase_client import EnphaseClient
from .data_manager import SolarDataManager
from .oauth_setup import EnphaseOAuthSetup

__all__ = ["EnphaseClient", "SolarDataManager", "EnphaseOAuthSetup"]
