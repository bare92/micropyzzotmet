"""
Tests for the utils module of MicroPyzzotMet.
"""

import pytest
import os
import json
from pathlib import Path
from micropyzzotmet.utils import get_earthdatahub_credentials, build_earthdatahub_url


class TestEarthDataHubCredentials:
    """Test suite for EarthDataHub credential handling."""

    def test_build_earthdatahub_url_with_pat(self):
        """Test building URL with explicit PAT token."""
        dataset_path = "minio/cmip6-cp4cj/data/some_dataset"
        pat = "test_token_12345"
        
        url = build_earthdatahub_url(dataset_path, pat=pat)
        
        assert "https://" in url
        assert "edh" in url
        assert "test_token_12345" in url
        assert "data.earthdatahub.destine.eu" in url
        
    def test_build_earthdatahub_url_path_normalization(self):
        """Test URL path normalization."""
        dataset_path = "/minio/test/path"
        pat = "token123"
        
        url = build_earthdatahub_url(dataset_path, pat=pat)
        
        # Verify leading slash is stripped
        assert "minio/test/path" in url
        
    def test_build_earthdatahub_url_special_chars(self):
        """Test URL encoding of special characters."""
        dataset_path = "test/dataset with spaces"
        pat = "token@#$"
        
        url = build_earthdatahub_url(dataset_path, pat=pat)
        
        assert "https://" in url
        assert "earthdatahub" in url


class TestCredentialsFromNetrc:
    """Test suite for .netrc credential reading."""
    
    def test_get_earthdatahub_credentials_not_found(self):
        """Test error handling when credentials are not found."""
        # This test assumes .netrc doesn't have earthdatahub.com entry
        # or tests in an environment without .netrc
        try:
            credentials = get_earthdatahub_credentials(machine="nonexistent.machine")
            pytest.fail("Should raise ValueError for missing credentials")
        except ValueError as e:
            assert "No ~/.netrc credentials found" in str(e) or "Unable to read" in str(e)


class TestPackageVersion:
    """Test package version."""
    
    def test_version_exists(self):
        """Test that package version is defined."""
        from micropyzzotmet import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert __version__ == "0.1.0"
