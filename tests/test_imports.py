"""
Tests for package imports and basic structure.
"""

import pytest


class TestPackageImports:
    """Test that package modules can be imported."""

    def test_import_micropyzzotmet(self):
        """Test importing the main package."""
        import micropyzzotmet
        assert hasattr(micropyzzotmet, '__version__')

    def test_import_cli(self):
        """Test importing the CLI module."""
        from micropyzzotmet import cli
        assert hasattr(cli, 'main')

    def test_import_utils(self):
        """Test importing the utils module."""
        from micropyzzotmet import utils
        assert hasattr(utils, 'build_earthdatahub_url')
        assert hasattr(utils, 'get_earthdatahub_credentials')

    def test_import_main_micromet(self):
        """Test importing the main_micromet module."""
        from micropyzzotmet import main_micromet
        assert hasattr(main_micromet, 'run_micropezzomet')

    def test_import_get_era5_land(self):
        """Test importing the get_era5_land module."""
        from micropyzzotmet import get_era5_land
        # Module should exist even if not all functions are used
        assert get_era5_land is not None

    def test_import_downscaling_variables(self):
        """Test importing the downscaling_variables module."""
        from micropyzzotmet import downscaling_variables
        # Module should exist
        assert downscaling_variables is not None
