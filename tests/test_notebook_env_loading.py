"""
Tests for .env credential loading functionality in notebooks
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import pandas as pd


class TestNotebookEnvLoading(unittest.TestCase):
    """Test the .env credential loading functionality used in notebooks"""

    def setUp(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.env_file = Path(self.temp_dir) / ".env"

        # Sample environment content
        self.env_content = """# Enphase API Credentials
ENPHASE_CLIENT_ID=test_client_123
ENPHASE_CLIENT_SECRET=test_secret_456
ENPHASE_API_KEY=test_api_789
ENPHASE_ACCESS_TOKEN=test_token_abc
ENPHASE_REFRESH_TOKEN=test_refresh_def
ENPHASE_SYSTEM_ID=test_system_123
"""

    def tearDown(self):
        """Cleanup after each test"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_enphase_client_function(self, env_path):
        """Recreate the create_enphase_client function from notebooks"""
        def create_enphase_client():
            """Create Enphase client from .env credentials if available, otherwise return mock client"""
            if env_path.exists():
                # Load environment variables
                with open(env_path) as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value

                # Check if we have all required credentials
                required_vars = ['ENPHASE_ACCESS_TOKEN', 'ENPHASE_API_KEY', 'ENPHASE_SYSTEM_ID']
                if all(var in os.environ for var in required_vars):
                    try:
                        # Mock the EnphaseClient import for testing
                        class MockRealEnphaseClient:
                            def __init__(self, access_token, api_key, system_id):
                                self.access_token = access_token
                                self.api_key = api_key
                                self.system_id = system_id
                                self.is_real_client = True

                            def get_energy_lifetime(self, start_date=None, end_date=None):
                                return pd.DataFrame({'daily_energy_kwh': [10, 20, 30]})

                        client = MockRealEnphaseClient(
                            access_token=os.environ['ENPHASE_ACCESS_TOKEN'],
                            api_key=os.environ['ENPHASE_API_KEY'],
                            system_id=os.environ['ENPHASE_SYSTEM_ID']
                        )
                        return client
                    except Exception:
                        pass

            # Fallback to mock client
            class MockEnphaseClient:
                def __init__(self):
                    self.is_real_client = False

                def get_energy_lifetime(self, start_date=None, end_date=None):
                    return pd.DataFrame()  # Return empty for demo

            return MockEnphaseClient()

        return create_enphase_client

    def test_env_file_exists_with_valid_credentials(self):
        """Test that real client is created when .env exists with valid credentials"""
        # Create .env file with valid credentials
        with open(self.env_file, 'w') as f:
            f.write(self.env_content)

        create_client = self.create_enphase_client_function(self.env_file)
        client = create_client()

        # Should create real client
        self.assertTrue(hasattr(client, 'is_real_client'))
        self.assertTrue(client.is_real_client)
        self.assertEqual(client.access_token, 'test_token_abc')
        self.assertEqual(client.api_key, 'test_api_789')
        self.assertEqual(client.system_id, 'test_system_123')

    def test_env_file_missing(self):
        """Test that mock client is created when .env file doesn't exist"""
        # Don't create .env file
        create_client = self.create_enphase_client_function(self.env_file)
        client = create_client()

        # Should create mock client
        self.assertTrue(hasattr(client, 'is_real_client'))
        self.assertFalse(client.is_real_client)

    def test_env_file_missing_required_credentials(self):
        """Test that mock client is created when .env exists but missing required credentials"""
        # Clear any existing environment variables first
        for var in ['ENPHASE_ACCESS_TOKEN', 'ENPHASE_API_KEY', 'ENPHASE_SYSTEM_ID']:
            if var in os.environ:
                del os.environ[var]

        # Create .env file with incomplete credentials
        incomplete_content = """# Enphase API Credentials
ENPHASE_CLIENT_ID=test_client_123
ENPHASE_CLIENT_SECRET=test_secret_456
# Missing ACCESS_TOKEN, API_KEY and SYSTEM_ID
"""
        with open(self.env_file, 'w') as f:
            f.write(incomplete_content)

        create_client = self.create_enphase_client_function(self.env_file)
        client = create_client()

        # Should fall back to mock client
        self.assertTrue(hasattr(client, 'is_real_client'))
        self.assertFalse(client.is_real_client)

    def test_env_file_with_comments_and_empty_lines(self):
        """Test that .env parsing handles comments and empty lines correctly"""
        content_with_comments = """# Enphase API Credentials - DO NOT COMMIT
# This is a test file

ENPHASE_CLIENT_ID=test_client_123
# Another comment
ENPHASE_CLIENT_SECRET=test_secret_456

ENPHASE_API_KEY=test_api_789
ENPHASE_ACCESS_TOKEN=test_token_abc
ENPHASE_REFRESH_TOKEN=test_refresh_def
ENPHASE_SYSTEM_ID=test_system_123

# End of file
"""
        with open(self.env_file, 'w') as f:
            f.write(content_with_comments)

        create_client = self.create_enphase_client_function(self.env_file)
        client = create_client()

        # Should create real client despite comments
        self.assertTrue(hasattr(client, 'is_real_client'))
        self.assertTrue(client.is_real_client)
        self.assertEqual(client.access_token, 'test_token_abc')

    def test_environment_variable_loading(self):
        """Test that environment variables are correctly loaded from .env file"""
        with open(self.env_file, 'w') as f:
            f.write(self.env_content)

        # Clear any existing env vars
        for var in ['ENPHASE_CLIENT_ID', 'ENPHASE_API_KEY', 'ENPHASE_ACCESS_TOKEN', 'ENPHASE_SYSTEM_ID']:
            if var in os.environ:
                del os.environ[var]

        create_client = self.create_enphase_client_function(self.env_file)
        client = create_client()

        # Check that environment variables were loaded
        self.assertEqual(os.environ.get('ENPHASE_CLIENT_ID'), 'test_client_123')
        self.assertEqual(os.environ.get('ENPHASE_API_KEY'), 'test_api_789')
        self.assertEqual(os.environ.get('ENPHASE_ACCESS_TOKEN'), 'test_token_abc')
        self.assertEqual(os.environ.get('ENPHASE_SYSTEM_ID'), 'test_system_123')

    def test_mock_client_functionality(self):
        """Test that mock client returns expected empty data"""
        create_client = self.create_enphase_client_function(self.env_file)  # No file created
        mock_client = create_client()

        # Test mock client returns empty DataFrame
        result = mock_client.get_energy_lifetime()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_real_client_functionality(self):
        """Test that real client returns expected data structure"""
        with open(self.env_file, 'w') as f:
            f.write(self.env_content)

        create_client = self.create_enphase_client_function(self.env_file)
        real_client = create_client()

        # Test real client returns data
        result = real_client.get_energy_lifetime()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertIn('daily_energy_kwh', result.columns)


if __name__ == '__main__':
    unittest.main()