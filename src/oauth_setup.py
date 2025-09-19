"""
Enphase OAuth 2.0 Setup Module

Handles secure OAuth authentication flow for Enphase Energy API.
Stores credentials securely in .env file with proper security practices.
"""

import requests
import base64
import webbrowser
import os
from pathlib import Path
from typing import Dict, Optional, Tuple


class EnphaseOAuthSetup:
    """Handles Enphase OAuth 2.0 setup and credential management"""

    def __init__(self, env_file_path: str = ".env"):
        """
        Initialize OAuth setup

        Args:
            env_file_path: Path to .env file for storing credentials
        """
        self.env_file = Path(env_file_path)
        self.redirect_uri = "https://api.enphaseenergy.com/oauth/redirect_uri"
        self.auth_base_url = "https://api.enphaseenergy.com/oauth"
        self.api_base_url = "https://api.enphaseenergy.com/api/v4"

    def check_security_setup(self) -> Dict[str, bool]:
        """Check if security best practices are in place"""
        gitignore_file = self.env_file.parent / ".gitignore"

        checks = {
            'env_exists': self.env_file.exists(),
            'gitignore_exists': gitignore_file.exists(),
            'env_in_gitignore': False,
            'secure_permissions': False
        }

        if gitignore_file.exists():
            with open(gitignore_file, 'r') as f:
                gitignore_content = f.read()
                checks['env_in_gitignore'] = '.env' in gitignore_content

        if self.env_file.exists():
            try:
                stat = self.env_file.stat()
                # Check if file has secure permissions (owner read/write only)
                checks['secure_permissions'] = oct(stat.st_mode)[-3:] == '600'
            except:
                pass

        return checks

    def save_credentials(self, credentials: Dict[str, str]) -> bool:
        """
        Save credentials securely to .env file

        Args:
            credentials: Dictionary of credential key-value pairs

        Returns:
            True if saved successfully
        """
        try:
            # Build .env content
            env_content = "# Enphase API Credentials - DO NOT COMMIT TO GIT\n"

            for key, value in credentials.items():
                if value:  # Only save non-empty values
                    env_content += f"{key}={value}\n"

            # Write to file
            with open(self.env_file, 'w') as f:
                f.write(env_content)

            # Set secure permissions (Unix/Mac only)
            try:
                os.chmod(self.env_file, 0o600)
            except (OSError, NotImplementedError):
                # Windows or permission issue
                pass

            return True

        except Exception as e:
            print(f"Error saving credentials: {e}")
            return False

    def load_credentials(self) -> Dict[str, Optional[str]]:
        """Load credentials from .env file"""
        credentials = {
            'ENPHASE_CLIENT_ID': None,
            'ENPHASE_CLIENT_SECRET': None,
            'ENPHASE_API_KEY': None,
            'ENPHASE_ACCESS_TOKEN': None,
            'ENPHASE_REFRESH_TOKEN': None,
            'ENPHASE_SYSTEM_ID': None
        }

        if not self.env_file.exists():
            return credentials

        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        if key in credentials:
                            credentials[key] = value
        except Exception as e:
            print(f"Error loading credentials: {e}")

        return credentials

    def collect_initial_credentials(self) -> Tuple[str, str, str]:
        """
        Collect initial credentials from user input

        Returns:
            Tuple of (client_id, client_secret, api_key)
        """
        print("Enter your Enphase developer credentials:")
        print("(Get these from https://developer.enphase.com/)")

        client_id = input("Client ID: ").strip()
        client_secret = input("Client Secret: ").strip()
        api_key = input("API Key: ").strip()

        return client_id, client_secret, api_key

    def get_authorization_url(self, client_id: str) -> str:
        """
        Generate OAuth authorization URL

        Args:
            client_id: OAuth client ID

        Returns:
            Authorization URL for user to visit
        """
        return (
            f"{self.auth_base_url}/authorize?"
            f"response_type=code&"
            f"client_id={client_id}&"
            f"redirect_uri={self.redirect_uri}&"
            f"scope=read"
        )

    def open_authorization_url(self, auth_url: str) -> None:
        """Open authorization URL in browser"""
        print(f"Authorization URL: {auth_url}")

        try:
            webbrowser.open(auth_url)
            print("Browser opened automatically")
        except:
            print("Copy the URL above into your browser manually")

    def exchange_code_for_tokens(self, authorization_code: str, client_id: str,
                                 client_secret: str) -> Optional[Dict[str, str]]:
        """
        Exchange authorization code for access tokens

        Args:
            authorization_code: Code from OAuth callback
            client_id: OAuth client ID
            client_secret: OAuth client secret

        Returns:
            Dictionary with token information or None if failed
        """
        # Create Basic Auth header
        auth_string = f"{client_id}:{client_secret}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')

        headers = {
            'Authorization': f'Basic {auth_b64}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        data = {
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri,
            'code': authorization_code
        }

        try:
            response = requests.post(f"{self.auth_base_url}/token",
                                     data=data, headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Token exchange failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except Exception as e:
            print(f"Error during token exchange: {e}")
            return None

    def test_api_access(self, access_token: str, api_key: str) -> Optional[Dict]:
        """
        Test API access and get system information

        Args:
            access_token: OAuth access token
            api_key: Enphase API key

        Returns:
            Systems data or None if failed
        """
        headers = {
            'Authorization': f'Bearer {access_token}',
            'key': api_key
        }

        try:
            response = requests.get(f"{self.api_base_url}/systems", headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"API test failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except Exception as e:
            print(f"Error testing API: {e}")
            return None

    def complete_setup(self) -> bool:
        """
        Complete OAuth setup process interactively

        Returns:
            True if setup completed successfully
        """
        print("=== ENPHASE OAUTH SETUP ===\n")

        # Check security setup
        security_checks = self.check_security_setup()

        if not security_checks['env_in_gitignore']:
            print("WARNING: .env should be added to .gitignore to prevent credential exposure!")

        # Step 1: Collect initial credentials
        client_id, client_secret, api_key = self.collect_initial_credentials()

        if not all([client_id, client_secret, api_key]):
            print("ERROR: All credentials are required")
            return False

        # Step 2: Get authorization
        auth_url = self.get_authorization_url(client_id)
        print(f"\nStep 1: Authorize the application")
        self.open_authorization_url(auth_url)

        auth_code = input("\nEnter authorization code: ").strip()
        if not auth_code:
            print("ERROR: Authorization code required")
            return False

        # Step 3: Exchange for tokens
        print("Step 2: Exchanging code for tokens...")
        tokens = self.exchange_code_for_tokens(auth_code, client_id, client_secret)

        if not tokens:
            return False

        # Step 4: Test API access
        print("Step 3: Testing API access...")
        systems_data = self.test_api_access(tokens['access_token'], api_key)

        if not systems_data:
            return False

        # Step 5: Save everything
        systems_list = systems_data.get('systems', [])
        system_id = systems_list[0]['system_id'] if systems_list else None

        credentials = {
            'ENPHASE_CLIENT_ID': client_id,
            'ENPHASE_CLIENT_SECRET': client_secret,
            'ENPHASE_API_KEY': api_key,
            'ENPHASE_ACCESS_TOKEN': tokens['access_token'],
            'ENPHASE_REFRESH_TOKEN': tokens.get('refresh_token'),
            'ENPHASE_SYSTEM_ID': system_id
        }

        if self.save_credentials(credentials):
            print(f"\nSUCCESS! OAuth setup complete")
            print(f"Found {len(systems_list)} system(s)")

            for i, system in enumerate(systems_list):
                print(f"  System {i + 1}: {system.get('system_id')} ({system.get('size_w', 'Unknown')}W)")

            print(f"\nCredentials saved securely to {self.env_file}")
            return True
        else:
            print("ERROR: Failed to save credentials")
            return False


def setup_oauth(env_file_path: str = ".env") -> bool:
    """
    Convenience function to run complete OAuth setup

    Args:
        env_file_path: Path to .env file

    Returns:
        True if setup completed successfully
    """
    oauth_setup = EnphaseOAuthSetup(env_file_path)
    return oauth_setup.complete_setup()


def verify_setup(env_file_path: str = ".env") -> bool:
    """
    Verify OAuth setup is complete and working

    Args:
        env_file_path: Path to .env file

    Returns:
        True if setup is valid
    """
    oauth_setup = EnphaseOAuthSetup(env_file_path)
    credentials = oauth_setup.load_credentials()

    required_creds = ['ENPHASE_ACCESS_TOKEN', 'ENPHASE_API_KEY']

    if not all(credentials.get(key) for key in required_creds):
        print("Missing required credentials")
        return False

    # Test API access
    systems_data = oauth_setup.test_api_access(
        credentials['ENPHASE_ACCESS_TOKEN'],
        credentials['ENPHASE_API_KEY']
    )

    return systems_data is not None