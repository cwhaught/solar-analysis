#!/usr/bin/env python3
"""
Command-line OAuth setup for Enphase API

Run this script to generate fresh access and refresh tokens.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from core.oauth_setup import setup_oauth

if __name__ == "__main__":
    print("🔐 Enphase OAuth Setup Utility")
    print("=" * 40)
    print()
    print("This will guide you through setting up OAuth credentials for Enphase API.")
    print("You'll need:")
    print("  1. Client ID from https://developer.enphase.com/")
    print("  2. Client Secret from https://developer.enphase.com/")
    print("  3. API Key from https://developer.enphase.com/")
    print()

    # Check if .env already exists
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  Found existing .env file")
        overwrite = input("Do you want to update your credentials? (y/N): ").strip().lower()
        if overwrite not in ['y', 'yes']:
            print("OAuth setup cancelled.")
            sys.exit(0)

    try:
        success = setup_oauth('.env')
        if success:
            print("\n✅ OAuth setup completed successfully!")
            print("\nYour credentials have been saved to .env")
            print("You can now use the Enphase API in your notebooks.")
        else:
            print("\n❌ OAuth setup failed.")
            print("Please check your credentials and try again.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nOAuth setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during OAuth setup: {e}")
        sys.exit(1)