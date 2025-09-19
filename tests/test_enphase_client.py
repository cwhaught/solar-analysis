"""
Tests for EnphaseClient class
"""

import pytest
import requests
from unittest.mock import Mock, patch
from datetime import datetime
import pandas as pd

from src.enphase_client import EnphaseClient


class TestEnphaseClient:
    """Test cases for EnphaseClient"""

    def setup_method(self):
        """Set up test fixtures"""
        self.access_token = "test_access_token"
        self.api_key = "test_api_key"
        self.system_id = "test_system_id"
        self.client = EnphaseClient(self.access_token, self.api_key, self.system_id)

    def test_init(self):
        """Test client initialization"""
        assert self.client.access_token == self.access_token
        assert self.client.api_key == self.api_key
        assert self.client.system_id == self.system_id
        assert self.client.base_url == "https://api.enphaseenergy.com/api/v4"
        assert self.client.headers['Authorization'] == f'Bearer {self.access_token}'
        assert self.client.headers['key'] == self.api_key

    @patch('src.enphase_client.requests.get')
    def test_get_current_status_success(self, mock_get):
        """Test successful current status retrieval"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "current_power": 1500,
            "energy_today": 25000,
            "energy_lifetime": 50000000
        }
        mock_get.return_value = mock_response

        result = self.client.get_current_status()

        assert result is not None
        assert result["current_power"] == 1500
        assert result["energy_today"] == 25000
        mock_get.assert_called_once()

    @patch('src.enphase_client.requests.get')
    def test_get_current_status_rate_limit(self, mock_get):
        """Test rate limit handling"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        result = self.client.get_current_status()

        assert result is None
        mock_get.assert_called_once()

    @patch('src.enphase_client.requests.get')
    def test_get_current_status_error(self, mock_get):
        """Test error handling"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = self.client.get_current_status()

        assert result is None
        mock_get.assert_called_once()

    @patch('src.enphase_client.requests.get')
    def test_get_energy_lifetime_success(self, mock_get):
        """Test successful energy lifetime retrieval"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "start_date": "2022-01-01",
            "production": [100, 200, 300],
            "meta": {"status": "normal"}
        }
        mock_get.return_value = mock_response

        result = self.client.get_energy_lifetime()

        assert isinstance(result, pd.DataFrame)
        mock_get.assert_called_once()

    def test_date_parameter_handling(self):
        """Test date parameter conversion"""
        # Test string date
        start_date = "2023-01-01"
        end_date = "2023-12-31"

        with patch('src.enphase_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"production": []}
            mock_get.return_value = mock_response

            self.client.get_energy_lifetime(start_date=start_date, end_date=end_date)

            call_args = mock_get.call_args
            params = call_args[1]['params']
            assert params['start_date'] == '2023-01-01'
            assert params['end_date'] == '2023-12-31'

    def test_datetime_parameter_handling(self):
        """Test datetime object parameter conversion"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        with patch('src.enphase_client.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"production": []}
            mock_get.return_value = mock_response

            self.client.get_energy_lifetime(start_date=start_date, end_date=end_date)

            call_args = mock_get.call_args
            params = call_args[1]['params']
            assert params['start_date'] == '2023-01-01'
            assert params['end_date'] == '2023-12-31'