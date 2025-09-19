import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
import os
from typing import Optional, Dict, Any


class EnphaseAPI:
    """
    Enphase Energy API client for accessing solar production data
    """

    def __init__(self, api_key: str = None, user_id: str = None):
        """
        Initialize Enphase API client

        Args:
            api_key: Enphase API key
            user_id: Enphase user ID
        """
        self.api_key = api_key or os.getenv('ENPHASE_API_KEY')
        self.user_id = user_id or os.getenv('ENPHASE_USER_ID')
        self.base_url = "https://api.enphaseenergy.com/api/v2"
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })

    def get_systems(self) -> Optional[Dict]:
        """Get list of systems for the user"""
        if not self.api_key:
            print("API key required for this method")
            return None

        url = f"{self.base_url}/systems"
        response = self.session.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting systems: {response.status_code}")
            print(response.text)
            return None

    def get_production_data(self, system_id: str, start_date: str, end_date: str,
                            granularity: str = 'day') -> Optional[pd.DataFrame]:
        """
        Get production data for a system

        Args:
            system_id: Enphase system ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            granularity: 'day', 'week', or '15mins'

        Returns:
            DataFrame with production data
        """
        if not self.api_key:
            print("API key required for this method")
            return None

        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        url = f"{self.base_url}/systems/{system_id}/stats"
        params = {
            'start_at': start_ts,
            'end_at': end_ts,
            'granularity': granularity
        }

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            return self._parse_production_data(data, granularity)
        else:
            print(f"Error getting production data: {response.status_code}")
            print(response.text)
            return None

    def get_consumption_data(self, system_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get consumption data if available"""
        if not self.api_key:
            print("API key required for this method")
            return None

        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        url = f"{self.base_url}/systems/{system_id}/consumption_stats"
        params = {
            'start_at': start_ts,
            'end_at': end_ts
        }

        response = self.session.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            return self._parse_consumption_data(data)
        else:
            print(f"Error getting consumption data: {response.status_code}")
            return None

    def _parse_production_data(self, data: Dict, granularity: str) -> pd.DataFrame:
        """Parse production data from API response"""
        intervals = data.get('intervals', [])

        records = []
        for interval in intervals:
            record = {
                'datetime': pd.to_datetime(interval['end_at'], unit='s'),
                'production_wh': interval.get('powr', 0),
                'devices_reporting': interval.get('devices_reporting', 0)
            }
            records.append(record)

        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)

        return df

    def _parse_consumption_data(self, data: Dict) -> pd.DataFrame:
        """Parse consumption data from API response"""
        intervals = data.get('intervals', [])

        records = []
        for interval in intervals:
            record = {
                'datetime': pd.to_datetime(interval['end_at'], unit='s'),
                'consumption_wh': interval.get('enwh', 0)
            }
            records.append(record)

        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)

        return df


class EnphaseLocalAPI:
    """
    Access Enphase data through local Envoy gateway
    """

    def __init__(self, envoy_ip: str = "envoy.local"):
        """
        Initialize local Envoy connection

        Args:
            envoy_ip: IP address or hostname of Envoy gateway
        """
        self.envoy_ip = envoy_ip
        self.base_url = f"http://{envoy_ip}"

    def get_current_production(self) -> Optional[Dict]:
        """Get current production data from Envoy"""
        try:
            url = f"{self.base_url}/api/v1/production"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error accessing Envoy: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Cannot connect to Envoy at {self.envoy_ip}: {e}")
            return None

    def test_connection(self) -> bool:
        """Test if Envoy is accessible"""
        try:
            response = requests.get(f"{self.base_url}/info", timeout=5)
            return response.status_code == 200
        except:
            return False


def setup_enphase_credentials():
    """
    Interactive setup for Enphase API credentials
    """
    print("=== ENPHASE API SETUP ===\n")

    print("To use the Enphase API, you need:")
    print("1. Developer account at https://developer.enphase.com/")
    print("2. API key from your developer dashboard")
    print("3. Your system ID from Enlighten portal\n")

    print("Alternatively, you can:")
    print("- Try local Envoy access (if you have an Envoy gateway)")
    print("- Continue using CSV data export method\n")

    choice = input("Choose option (1=API setup, 2=Local Envoy, 3=Skip): ")

    if choice == "1":
        api_key = input("Enter your Enphase API key: ").strip()
        system_id = input("Enter your system ID: ").strip()

        # Save to environment file
        env_file = Path(".env")
        with open(env_file, "w") as f:
            f.write(f"ENPHASE_API_KEY={api_key}\n")
            f.write(f"ENPHASE_SYSTEM_ID={system_id}\n")

        print(f"Credentials saved to {env_file}")
        return "api"

    elif choice == "2":
        envoy_ip = input("Enter Envoy IP (or press Enter for 'envoy.local'): ").strip()
        if not envoy_ip:
            envoy_ip = "envoy.local"

        local_api = EnphaseLocalAPI(envoy_ip)
        if local_api.test_connection():
            print(f"✅ Successfully connected to Envoy at {envoy_ip}")
            return "local"
        else:
            print(f"❌ Cannot connect to Envoy at {envoy_ip}")
            return None
    else:
        print("Skipping API setup - continuing with CSV workflow")
        return None


def get_live_data(method: str = "auto") -> Optional[pd.DataFrame]:
    """
    Get live data using the best available method

    Args:
        method: "api", "local", or "auto"
    """
    if method == "auto":
        # Try local first, then API
        local_api = EnphaseLocalAPI()
        if local_api.test_connection():
            method = "local"
        elif os.getenv('ENPHASE_API_KEY'):
            method = "api"
        else:
            print("No Enphase connection available")
            return None

    if method == "local":
        local_api = EnphaseLocalAPI()
        data = local_api.get_current_production()
        if data:
            # Convert to DataFrame format compatible with your existing code
            current_time = datetime.now()
            df = pd.DataFrame([{
                'datetime': current_time,
                'production_w': data.get('wattsNow', 0),
                'total_production_mwh': data.get('wattHoursLifetime', 0) / 1000000
            }])
            df.set_index('datetime', inplace=True)
            return df

    elif method == "api":
        api = EnphaseAPI()
        systems = api.get_systems()
        if systems and len(systems.get('systems', [])) > 0:
            system_id = systems['systems'][0]['system_id']
            today = datetime.now().strftime('%Y-%m-%d')
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

            return api.get_production_data(system_id, yesterday, today, '15mins')

    return None