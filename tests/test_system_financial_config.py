"""
Tests for system financial configuration functionality
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.location_loader import load_system_financial_config, get_complete_financial_config


class TestSystemFinancialConfig:
    """Test system financial configuration loading"""

    def setup_method(self):
        """Clean environment before each test"""
        # Store original environment
        self.original_env = os.environ.copy()
        # Remove any solar-related environment variables
        for key in list(os.environ.keys()):
            if key.startswith('SOLAR_'):
                del os.environ[key]

    def teardown_method(self):
        """Restore environment after each test"""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_load_system_financial_config_defaults(self):
        """Test loading system config with default values"""
        # Test with no .env file
        config = load_system_financial_config('/nonexistent/.env')

        # Check default values
        assert config['system_size_kw'] == 10.0
        assert config['cost_per_kw'] == 2500.0
        assert config['total_cost'] == 25000.0  # 10 * 2500
        assert config['federal_tax_credit_percent'] == 30.0
        assert config['federal_tax_credit'] == 7500.0  # 30% of 25000
        assert config['state_rebate'] == 0.0
        assert config['utility_rebate'] == 0.0
        assert config['other_rebates'] == 0.0
        assert config['total_rebates'] == 7500.0  # Only federal tax credit
        assert config['net_system_cost'] == 17500.0  # 25000 - 7500
        assert config['source'] == 'defaults'

    def test_load_system_financial_config_from_env(self):
        """Test loading system config from .env file"""
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("""
# Solar System Configuration
SOLAR_SYSTEM_SIZE_KW=12.5
SOLAR_SYSTEM_COST_PER_KW=3000
SOLAR_SYSTEM_TOTAL_COST=37500
SOLAR_FEDERAL_TAX_CREDIT_PERCENT=30
SOLAR_STATE_REBATE=5000
SOLAR_UTILITY_REBATE=2000
SOLAR_OTHER_REBATES=1000
""")
            env_path = f.name

        try:
            config = load_system_financial_config(env_path)

            # Check configured values
            assert config['system_size_kw'] == 12.5
            assert config['cost_per_kw'] == 3000.0
            assert config['total_cost'] == 37500.0
            assert config['federal_tax_credit_percent'] == 30.0
            assert config['federal_tax_credit'] == 11250.0  # 30% of 37500
            assert config['state_rebate'] == 5000.0
            assert config['utility_rebate'] == 2000.0
            assert config['other_rebates'] == 1000.0
            assert config['total_rebates'] == 19250.0  # 11250 + 5000 + 2000 + 1000
            assert config['net_system_cost'] == 18250.0  # 37500 - 19250
            assert config['source'] == 'environment'

        finally:
            os.unlink(env_path)

    def test_load_system_financial_config_partial_env(self):
        """Test loading with partial configuration in .env"""
        # Create temporary .env file with only some values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("""
# Partial configuration
SOLAR_SYSTEM_SIZE_KW=8.0
SOLAR_STATE_REBATE=3000
""")
            env_path = f.name

        try:
            config = load_system_financial_config(env_path)

            # Check mix of configured and default values
            assert config['system_size_kw'] == 8.0  # From env
            assert config['cost_per_kw'] == 2500.0  # Default
            assert config['total_cost'] == 20000.0  # 8.0 * 2500 (calculated)
            assert config['federal_tax_credit'] == 6000.0  # 30% of 20000
            assert config['state_rebate'] == 3000.0  # From env
            assert config['utility_rebate'] == 0.0  # Default
            assert config['total_rebates'] == 9000.0  # 6000 + 3000
            assert config['net_system_cost'] == 11000.0  # 20000 - 9000

        finally:
            os.unlink(env_path)

    def test_load_system_financial_config_custom_total_cost(self):
        """Test when total cost is explicitly set vs calculated"""
        # Create temporary .env file with explicit total cost
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("""
SOLAR_SYSTEM_SIZE_KW=10.0
SOLAR_SYSTEM_COST_PER_KW=2500
SOLAR_SYSTEM_TOTAL_COST=30000
""")
            env_path = f.name

        try:
            config = load_system_financial_config(env_path)

            # Total cost should be explicit value, not calculated
            assert config['system_size_kw'] == 10.0
            assert config['cost_per_kw'] == 2500.0
            assert config['total_cost'] == 30000.0  # Explicit, not 10 * 2500
            assert config['federal_tax_credit'] == 9000.0  # 30% of 30000

        finally:
            os.unlink(env_path)

    def test_federal_tax_credit_calculation(self):
        """Test federal tax credit calculation with different percentages"""
        # Test with different tax credit percentage
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("""
SOLAR_SYSTEM_TOTAL_COST=20000
SOLAR_FEDERAL_TAX_CREDIT_PERCENT=26
""")
            env_path = f.name

        try:
            config = load_system_financial_config(env_path)

            assert config['total_cost'] == 20000.0
            assert config['federal_tax_credit_percent'] == 26.0
            assert config['federal_tax_credit'] == 5200.0  # 26% of 20000
            assert config['net_system_cost'] == 14800.0  # 20000 - 5200

        finally:
            os.unlink(env_path)

    def test_zero_rebates(self):
        """Test configuration with no rebates"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("""
SOLAR_SYSTEM_TOTAL_COST=25000
SOLAR_FEDERAL_TAX_CREDIT_PERCENT=0
""")
            env_path = f.name

        try:
            config = load_system_financial_config(env_path)

            assert config['federal_tax_credit'] == 0.0
            assert config['total_rebates'] == 0.0
            assert config['net_system_cost'] == 25000.0  # No rebates

        finally:
            os.unlink(env_path)

    def test_env_file_parsing_with_comments(self):
        """Test .env file parsing handles comments and empty lines"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("""
# This is a comment
SOLAR_SYSTEM_SIZE_KW=15.0

# Another comment
SOLAR_STATE_REBATE=4000
# SOLAR_UTILITY_REBATE=1000  # This is commented out
""")
            env_path = f.name

        try:
            config = load_system_financial_config(env_path)

            assert config['system_size_kw'] == 15.0
            assert config['state_rebate'] == 4000.0
            assert config['utility_rebate'] == 0.0  # Should be default since commented out

        finally:
            os.unlink(env_path)

    @patch.dict(os.environ, {
        'SOLAR_SYSTEM_SIZE_KW': '20.0',
        'SOLAR_STATE_REBATE': '6000'
    })
    def test_environment_variables_priority(self):
        """Test that environment variables take priority over .env file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("""
SOLAR_SYSTEM_SIZE_KW=10.0
SOLAR_STATE_REBATE=3000
SOLAR_UTILITY_REBATE=2000
""")
            env_path = f.name

        try:
            config = load_system_financial_config(env_path)

            # Environment variables should take precedence
            assert config['system_size_kw'] == 20.0  # From env var, not file
            assert config['state_rebate'] == 6000.0  # From env var, not file
            assert config['utility_rebate'] == 2000.0  # From file (no env var)

        finally:
            os.unlink(env_path)


class TestCompleteFinancialConfig:
    """Test complete financial configuration integration"""

    def setup_method(self):
        """Clean environment before each test"""
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith('SOLAR_'):
                del os.environ[key]

    def teardown_method(self):
        """Restore environment after each test"""
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch('core.location_loader.create_location_with_fallback')
    @patch('core.location_loader.get_location_electricity_rates')
    def test_get_complete_financial_config(self, mock_get_rates, mock_create_location):
        """Test getting complete financial configuration"""
        # Mock location
        mock_location = type('MockLocation', (), {
            'latitude': 39.7392,
            'longitude': -104.9903,
            'location_name': 'Denver, CO'
        })()
        mock_create_location.return_value = mock_location

        # Mock electricity rates
        mock_rates = {
            'residential_rate': 14.30,
            'annual_cost_per_kwh': 0.143,
            'feed_in_rate_per_kwh': 0.114,
            'source': 'state_average'
        }
        mock_get_rates.return_value = mock_rates

        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("""
SOLAR_SYSTEM_SIZE_KW=12.0
SOLAR_SYSTEM_TOTAL_COST=30000
SOLAR_STATE_REBATE=5000
""")
            env_path = f.name

        try:
            location, rates, system_config = get_complete_financial_config(env_path)

            # Check that all components are returned
            assert location.location_name == 'Denver, CO'
            assert rates['residential_rate'] == 14.30
            assert system_config['system_size_kw'] == 12.0
            assert system_config['total_cost'] == 30000.0
            assert system_config['state_rebate'] == 5000.0

        finally:
            os.unlink(env_path)

    def test_invalid_numeric_values(self):
        """Test handling of invalid numeric values in .env file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("""
SOLAR_SYSTEM_SIZE_KW=invalid_number
SOLAR_STATE_REBATE=5000
""")
            env_path = f.name

        try:
            # Should raise ValueError for invalid float conversion
            with pytest.raises(ValueError, match="could not convert string to float"):
                load_system_financial_config(env_path)

        finally:
            os.unlink(env_path)


class TestSystemFinancialConfigEdgeCases:
    """Test edge cases and error conditions"""

    def setup_method(self):
        """Clean environment before each test"""
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith('SOLAR_'):
                del os.environ[key]

    def teardown_method(self):
        """Restore environment after each test"""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_extremely_high_values(self):
        """Test with extremely high system cost values"""
        config = load_system_financial_config()

        # Override with high values
        import os
        old_env = os.environ.copy()
        try:
            os.environ['SOLAR_SYSTEM_TOTAL_COST'] = '1000000'  # $1M system
            os.environ['SOLAR_FEDERAL_TAX_CREDIT_PERCENT'] = '30'

            config = load_system_financial_config()

            assert config['total_cost'] == 1000000.0
            assert config['federal_tax_credit'] == 300000.0
            assert config['net_system_cost'] == 700000.0

        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_zero_system_cost(self):
        """Test with zero system cost"""
        import os
        old_env = os.environ.copy()
        try:
            os.environ['SOLAR_SYSTEM_TOTAL_COST'] = '0'

            config = load_system_financial_config()

            assert config['total_cost'] == 0.0
            assert config['federal_tax_credit'] == 0.0
            assert config['net_system_cost'] == 0.0

        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_rebates_exceed_system_cost(self):
        """Test when total rebates exceed system cost"""
        import os
        old_env = os.environ.copy()
        try:
            os.environ['SOLAR_SYSTEM_TOTAL_COST'] = '10000'
            os.environ['SOLAR_FEDERAL_TAX_CREDIT_PERCENT'] = '30'  # $3000
            os.environ['SOLAR_STATE_REBATE'] = '5000'
            os.environ['SOLAR_UTILITY_REBATE'] = '5000'  # Total rebates = $13000 > $10000 cost

            config = load_system_financial_config()

            assert config['total_cost'] == 10000.0
            assert config['total_rebates'] == 13000.0
            assert config['net_system_cost'] == -3000.0  # Negative cost (profit!)

        finally:
            os.environ.clear()
            os.environ.update(old_env)