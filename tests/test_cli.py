"""
Tests for the CLI module of MicroPyzzotMet.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
from micropyzzotmet.cli import main


class TestCLI:
    """Test suite for command-line interface."""

    def test_main_requires_config_argument(self):
        """Test that main() requires a config file argument."""
        with patch.object(sys, 'argv', ['micropyzzotmet']):
            with pytest.raises(SystemExit):
                # argparse calls sys.exit(2) for missing arguments
                main()
    
    @patch('micropyzzotmet.main_micromet.run_micropezzomet')
    def test_main_accepts_config_file(self, mock_run):
        """Test that main accepts a config file argument."""
        test_config = "test_config.json"
        
        with patch.object(sys, 'argv', ['micropyzzotmet', test_config]):
            main()
            
        mock_run.assert_called_once_with(test_config)
    
    @patch('micropyzzotmet.main_micromet.run_micropezzomet')
    def test_main_handles_exception(self, mock_run):
        """Test exception handling in main."""
        mock_run.side_effect = Exception("Test error")
        
        with patch.object(sys, 'argv', ['micropyzzotmet', 'config.json']):
            with pytest.raises(Exception):
                main()
