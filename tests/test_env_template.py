"""
Tests for .env.template file and related functionality
"""

import unittest
import tempfile
import shutil
from pathlib import Path


class TestEnvTemplate(unittest.TestCase):
    """Test .env.template file functionality"""

    def setUp(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.template_path = Path(self.temp_dir) / ".env.template"
        self.env_path = Path(self.temp_dir) / ".env"

    def tearDown(self):
        """Cleanup after each test"""
        shutil.rmtree(self.temp_dir)

    def test_env_template_exists(self):
        """Test that .env.template exists in project root"""
        template_path = Path(".env.template")
        self.assertTrue(template_path.exists(), ".env.template file should exist in project root")

    def test_env_template_format(self):
        """Test that .env.template has correct format"""
        template_path = Path(".env.template")

        with open(template_path, 'r') as f:
            content = f.read()

        # Check for required environment variables
        required_vars = [
            'ENPHASE_CLIENT_ID',
            'ENPHASE_CLIENT_SECRET',
            'ENPHASE_API_KEY',
            'ENPHASE_ACCESS_TOKEN',
            'ENPHASE_REFRESH_TOKEN',
            'ENPHASE_SYSTEM_ID'
        ]

        for var in required_vars:
            self.assertIn(var, content, f"{var} should be in .env.template")

        # Check that template values are placeholders
        self.assertIn('your_client_id_here', content)
        self.assertIn('your_system_id_here', content)

        # Check for helpful comments
        self.assertIn('# Enphase API Credentials', content)
        self.assertIn('DO NOT COMMIT TO GIT', content)

    def test_env_template_copy_instructions(self):
        """Test that copying .env.template works as expected"""
        # Copy the real template to temp location
        real_template = Path(".env.template")
        shutil.copy2(real_template, self.template_path)

        # Test copying template to .env
        shutil.copy2(self.template_path, self.env_path)

        # Verify .env was created
        self.assertTrue(self.env_path.exists())

        # Verify content is identical
        with open(self.template_path, 'r') as template_file:
            template_content = template_file.read()

        with open(self.env_path, 'r') as env_file:
            env_content = env_file.read()

        self.assertEqual(template_content, env_content)

    def test_template_has_instructions(self):
        """Test that template includes helpful instructions"""
        template_path = Path(".env.template")

        with open(template_path, 'r') as f:
            content = f.read()

        # Check for step-by-step instructions
        instruction_keywords = [
            'Copy this file',
            'developer.enphase.com',
            'oauth_setup.py',
            'Replace the values',
            'Developer Portal'
        ]

        for keyword in instruction_keywords:
            self.assertIn(keyword, content, f"Template should include instruction about '{keyword}'")

    def test_template_variable_grouping(self):
        """Test that template groups variables logically"""
        template_path = Path(".env.template")

        with open(template_path, 'r') as f:
            lines = f.readlines()

        content = ''.join(lines)

        # Check that related variables are grouped with comments
        self.assertIn('Developer Portal', content)
        self.assertIn('OAuth setup', content)
        self.assertIn('Enlighten account', content)


if __name__ == '__main__':
    unittest.main()