"""
Enphase Energy API Client

A comprehensive client for interacting with the Enphase Energy API v4,
providing OAuth 2.0 authentication and access to solar production data.

Usage:
    from enphase_client import EnphaseClient

    client = EnphaseClient(access_token, api_key, system_id)
    status = client.get_current_status()
    daily_data = client.get_energy_lifetime()
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from typing import Optional, Dict, List, Union


class EnphaseClient:
    """
    Client for Enphase Energy API v4

    Provides methods for accessing solar production data, system status,
    and historical energy production with proper rate limiting and error handling.
    """

    def __init__(self, access_token: str, api_key: str, system_id: str):
        """
        Initialize Enphase API client

        Args:
            access_token: OAuth 2.0 access token
            api_key: Enphase API key
            system_id: Enphase system ID
        """
        self.access_token = access_token
        self.api_key = api_key
        self.system_id = system_id
        self.base_url = "https://api.enphaseenergy.com/api/v4"
        self.headers = {
            'Authorization': f'Bearer {access_token}',
            'key': api_key
        }
        self.rate_limit_delay = 2  # seconds between requests

    def get_current_status(self) -> Optional[Dict]:
        """
        Get current system status and summary

        Returns:
            Dict with current power, energy today, lifetime energy, etc.
            None if request fails
        """
        summary_url = f"{self.base_url}/systems/{self.system_id}/summary"
        response = requests.get(summary_url, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print("Rate limit exceeded. Please wait before making more requests.")
            return None
        else:
            print(f"Error getting system status: {response.status_code}")
            return None

    def get_energy_lifetime(self, start_date: Optional[Union[str, datetime]] = None,
                          end_date: Optional[Union[str, datetime]] = None,
                          production: str = 'default') -> pd.DataFrame:
        """
        Get daily energy production time series over system lifetime

        This endpoint provides complete daily historical data without the 7-day
        limitation of other endpoints.

        Args:
            start_date: Start date (defaults to system start if None)
            end_date: End date (defaults to current date if None)
            production: 'default' (merged), 'all', or specific source

        Returns:
            DataFrame with daily energy production time series
        """
        url = f"{self.base_url}/systems/{self.system_id}/energy_lifetime"
        params = {}

        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            params['start_date'] = start_date.strftime('%Y-%m-%d')

        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            params['end_date'] = end_date.strftime('%Y-%m-%d')

        if production != 'default':
            params['production'] = production

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            data = response.json()
            return self._parse_energy_lifetime(data)

        elif response.status_code == 429:
            print("Rate limit exceeded. Please wait before making more requests.")
            return pd.DataFrame()
        else:
            print(f"Error getting lifetime energy: {response.status_code}")
            if response.status_code == 422:
                error_detail = response.json().get('details', 'Unknown error')
                print(f"Details: {error_detail}")
            return pd.DataFrame()

    def _parse_energy_lifetime(self, data: Dict) -> pd.DataFrame:
        """Parse energy_lifetime API response into DataFrame"""
        # The API returns 'production' not 'energy_lifetime'
        if 'production' not in data:
            return pd.DataFrame()

        energy_data = data['production']
        start_date_str = data.get('start_date')

        if not start_date_str or not energy_data:
            return pd.DataFrame()

        # Create date range
        dates = pd.date_range(
            start=start_date_str,
            periods=len(energy_data),
            freq='D'
        )

        df = pd.DataFrame({
            'date': dates,
            'daily_energy_wh': energy_data,
            'daily_energy_kwh': [wh/1000 if wh > 0 else 0 for wh in energy_data]
        })
        df.set_index('date', inplace=True)

        # Add metadata
        if 'meter_start_date' in data:
            df.attrs['meter_start_date'] = data['meter_start_date']

        return df

    def get_daily_production(self, start_date: Union[str, datetime],
                           end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get daily production totals by aggregating 15-minute intervals

        Note: This method has a 7-day limitation per request.
        For longer historical data, use get_energy_lifetime() instead.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with daily production totals
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Check 7-day limit
        if (end_date - start_date).days > 7:
            print("Warning: Daily production endpoint limited to 7 days. Use get_energy_lifetime() for longer ranges.")
            return pd.DataFrame()

        # Check 2-year limit
        two_years_ago = datetime.now() - timedelta(days=730)
        if start_date < two_years_ago:
            print(f"Warning: Start date {start_date.strftime('%Y-%m-%d')} is beyond 2-year API limit")
            start_date = two_years_ago

        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        url = f"{self.base_url}/systems/{self.system_id}/telemetry/production_meter"
        params = {
            'start_at': start_ts,
            'end_at': end_ts,
            'granularity': 'day'  # Returns 15-min intervals
        }

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            data = response.json()
            intervals = data.get('intervals', [])

            if not intervals:
                return pd.DataFrame()

            # Convert intervals to DataFrame and aggregate to daily
            records = []
            for interval in intervals:
                dt = pd.to_datetime(interval['end_at'], unit='s')
                record = {
                    'datetime': dt,
                    'production_wh': interval.get('wh_del', 0),
                    'devices_reporting': interval.get('devices_reporting', 0)
                }
                records.append(record)

            df = pd.DataFrame(records)
            df.set_index('datetime', inplace=True)
            df['production_kwh'] = df['production_wh'] / 1000

            # Aggregate to daily totals
            daily_df = df.groupby(df.index.date).agg({
                'production_wh': 'sum',
                'production_kwh': 'sum',
                'devices_reporting': 'mean'
            })
            daily_df.index = pd.to_datetime(daily_df.index)
            daily_df.index.name = 'date'

            return daily_df

        elif response.status_code == 422:
            error_detail = response.json().get('details', 'Unknown error')
            print(f"API Error 422: {error_detail}")
            return pd.DataFrame()
        elif response.status_code == 429:
            print("Rate limit exceeded. Please wait before making more requests.")
            return pd.DataFrame()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return pd.DataFrame()

    def get_fifteen_minute_intervals(self, target_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get 15-minute intervals for a specific day

        Args:
            target_date: Date to get intervals for

        Returns:
            DataFrame with 15-minute production intervals
        """
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d')

        # Set to start and end of day
        start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = target_date.replace(hour=23, minute=59, second=59, microsecond=0)

        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        url = f"{self.base_url}/systems/{self.system_id}/telemetry/production_meter"
        params = {
            'start_at': start_ts,
            'end_at': end_ts,
            'granularity': 'day'  # Returns 15-min intervals for the day
        }

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            data = response.json()
            intervals = data.get('intervals', [])

            if not intervals:
                return pd.DataFrame()

            records = []
            for interval in intervals:
                dt = pd.to_datetime(interval['end_at'], unit='s')
                record = {
                    'datetime': dt,
                    'production_wh': interval.get('wh_del', 0),
                    'production_kwh': interval.get('wh_del', 0) / 1000,
                    'devices_reporting': interval.get('devices_reporting', 0)
                }
                records.append(record)

            df = pd.DataFrame(records)
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)

            return df

        elif response.status_code == 429:
            print("Rate limit exceeded. Please wait before making more requests.")
            return pd.DataFrame()
        else:
            print(f"Error getting 15-minute data: {response.status_code}")
            return pd.DataFrame()

    def get_recent_summary(self, days_back: int = 7) -> pd.DataFrame:
        """
        Get daily production summary for recent days with rate limiting

        Args:
            days_back: Number of days to retrieve

        Returns:
            DataFrame with recent daily production summary
        """
        daily_data = []

        for i in range(days_back):
            date = datetime.now() - timedelta(days=i+1)  # Skip today (incomplete)

            # Add delay to respect rate limits
            if i > 0:
                time.sleep(self.rate_limit_delay)

            try:
                # Use energy_lifetime for better historical data access
                day_data = self.get_energy_lifetime(date, date)

                if not day_data.empty:
                    total_kwh = day_data['daily_energy_kwh'].iloc[0]
                else:
                    total_kwh = 0.0

                daily_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'production_kwh': total_kwh,
                    'has_data': total_kwh > 0
                })

            except Exception as e:
                print(f"Error getting data for {date.strftime('%Y-%m-%d')}: {e}")
                daily_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'production_kwh': 0.0,
                    'has_data': False
                })

        return pd.DataFrame(daily_data)

    def check_api_health(self) -> Dict:
        """
        Check if API is accessible and within rate limits

        Returns:
            Dict with API health status and basic system info
        """
        try:
            status = self.get_current_status()
            if status is not None:
                return {
                    'api_accessible': True,
                    'current_power': status.get('current_power', 0),
                    'energy_today': status.get('energy_today', 0) / 1000,
                    'energy_lifetime': status.get('energy_lifetime', 0) / 1000000,
                    'last_report': status.get('last_report_at')
                }
            else:
                return {'api_accessible': False, 'error': 'Could not retrieve status'}
        except Exception as e:
            return {'api_accessible': False, 'error': str(e)}

    def get_historical_data(self, start_date: Optional[Union[str, datetime]] = None,
                          end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Get comprehensive historical data using the best available method

        This method automatically chooses the most appropriate endpoint based on
        the date range requested.

        Args:
            start_date: Start date (defaults to system start)
            end_date: End date (defaults to current date)

        Returns:
            DataFrame with historical daily production data
        """
        # For comprehensive historical data, use energy_lifetime endpoint
        return self.get_energy_lifetime(start_date, end_date)

    def export_to_csv(self, filename: str, start_date: Optional[Union[str, datetime]] = None,
                     end_date: Optional[Union[str, datetime]] = None) -> bool:
        """
        Export historical data to CSV file

        Args:
            filename: Output CSV filename
            start_date: Start date for export
            end_date: End date for export

        Returns:
            True if export successful, False otherwise
        """
        try:
            data = self.get_historical_data(start_date, end_date)

            if not data.empty:
                data.to_csv(filename)
                print(f"Exported {len(data)} days of data to {filename}")
                return True
            else:
                print("No data available for export")
                return False

        except Exception as e:
            print(f"Error exporting data: {e}")
            return False