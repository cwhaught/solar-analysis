"""
Tests for OAuth setup functionality
"""

import pytest
from unittest.mock import Mock, patch, mock_open
import tempfile
import os
from pathlib import Path

from src.core.oauth_setup import EnphaseOAuthSetup


class TestEnphaseOAuthSetup:
    """Test cases for EnphaseOAuthSetup"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.env_file = os.path.join(self.temp_dir, ".env")
        self.oauth_setup = EnphaseOAuthSetup(self.env_file)

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test OAuth setup initialization"""
        assert self.oauth_setup.env_file == Path(self.env_file)
        assert "api.enphaseenergy.com" in self.oauth_setup.redirect_uri
        assert "api.enphaseenergy.com" in self.oauth_setup.auth_base_url
        assert "api.enphaseenergy.com" in self.oauth_setup.api_base_url

    def test_save_credentials(self):
        """Test credential saving"""
        credentials = {
            "ENPHASE_CLIENT_ID": "test_client_id",
            "ENPHASE_CLIENT_SECRET": "test_client_secret",
            "ENPHASE_API_KEY": "test_api_key",
        }

        result = self.oauth_setup.save_credentials(credentials)

        assert result is True
        assert Path(self.env_file).exists()

        # Check file content
        with open(self.env_file, "r") as f:
            content = f.read()
            assert "ENPHASE_CLIENT_ID=test_client_id" in content
            assert "ENPHASE_CLIENT_SECRET=test_client_secret" in content
            assert "ENPHASE_API_KEY=test_api_key" in content

    def test_load_credentials_empty_file(self):
        """Test loading credentials when no file exists"""
        credentials = self.oauth_setup.load_credentials()

        expected_keys = [
            "ENPHASE_CLIENT_ID",
            "ENPHASE_CLIENT_SECRET",
            "ENPHASE_API_KEY",
            "ENPHASE_ACCESS_TOKEN",
            "ENPHASE_REFRESH_TOKEN",
            "ENPHASE_SYSTEM_ID",
        ]

        for key in expected_keys:
            assert key in credentials
            assert credentials[key] is None

    def test_load_credentials_existing_file(self):
        """Test loading credentials from existing file"""
        # Create .env file with test data
        env_content = """# Test env file
ENPHASE_CLIENT_ID=test_client_123
ENPHASE_API_KEY=test_api_456
ENPHASE_SYSTEM_ID=789
"""
        with open(self.env_file, "w") as f:
            f.write(env_content)

        credentials = self.oauth_setup.load_credentials()

        assert credentials["ENPHASE_CLIENT_ID"] == "test_client_123"
        assert credentials["ENPHASE_API_KEY"] == "test_api_456"
        assert credentials["ENPHASE_SYSTEM_ID"] == "789"
        assert credentials["ENPHASE_CLIENT_SECRET"] is None  # Not in file

    def test_get_authorization_url(self):
        """Test authorization URL generation"""
        client_id = "test_client_id"
        url = self.oauth_setup.get_authorization_url(client_id)

        assert url.startswith(self.oauth_setup.auth_base_url)
        assert "response_type=code" in url
        assert f"client_id={client_id}" in url
        assert "redirect_uri=" in url
        assert "scope=read" in url

    @patch("src.core.oauth_setup.requests.post")
    def test_exchange_code_for_tokens_success(self, mock_post):
        """Test successful token exchange"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_post.return_value = mock_response

        result = self.oauth_setup.exchange_code_for_tokens(
            "test_code", "client_id", "client_secret"
        )

        assert result is not None
        assert result["access_token"] == "test_access_token"
        assert result["refresh_token"] == "test_refresh_token"

    @patch("src.core.oauth_setup.requests.post")
    def test_exchange_code_for_tokens_failure(self, mock_post):
        """Test failed token exchange"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Invalid authorization code"
        mock_post.return_value = mock_response

        result = self.oauth_setup.exchange_code_for_tokens(
            "invalid_code", "client_id", "client_secret"
        )

        assert result is None

    def test_check_security_setup_no_files(self):
        """Test security check with no files"""
        checks = self.oauth_setup.check_security_setup()

        assert checks["env_exists"] is False
        assert checks["gitignore_exists"] is False
        assert checks["env_in_gitignore"] is False
        assert checks["secure_permissions"] is False

    def test_check_security_setup_with_files(self):
        """Test security check with proper setup"""
        # Create .env file
        with open(self.env_file, "w") as f:
            f.write("ENPHASE_CLIENT_ID=test")

        # Create .gitignore file
        gitignore_path = os.path.join(self.temp_dir, ".gitignore")
        with open(gitignore_path, "w") as f:
            f.write(".env\n*.log\n")

        checks = self.oauth_setup.check_security_setup()

        assert checks["env_exists"] is True
        assert checks["gitignore_exists"] is True
        assert checks["env_in_gitignore"] is True

    def test_save_credentials_handles_empty_values(self):
        """Test that empty values are not saved"""
        credentials = {
            "ENPHASE_CLIENT_ID": "test_client_id",
            "ENPHASE_CLIENT_SECRET": "",  # Empty value
            "ENPHASE_API_KEY": None,  # None value
        }

        result = self.oauth_setup.save_credentials(credentials)

        assert result is True

        with open(self.env_file, "r") as f:
            content = f.read()
            assert "ENPHASE_CLIENT_ID=test_client_id" in content
            assert "ENPHASE_CLIENT_SECRET=" not in content
            assert "ENPHASE_API_KEY=" not in content
