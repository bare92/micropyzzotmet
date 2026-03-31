"""
Pytest configuration and fixtures for MicroPyzzotMet tests.
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample configuration file for testing."""
    import json
    config = {
        "DEM_file": str(temp_dir / "dem.tif"),
        "working_directory": str(temp_dir / "work"),
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
        "spatial_resolution": 100,
        "variables": ["temperature", "precipitation"],
    }
    
    config_file = temp_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f)
    
    return config_file
