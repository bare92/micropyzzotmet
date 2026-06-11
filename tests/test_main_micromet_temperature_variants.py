"""Tests for temperature variant wrappers in main_micromet."""

from unittest.mock import patch

from micropyzzotmet.main_micromet import run_temperature_min, run_temperature_max


def test_run_temperature_min_uses_t_min_mapping():
    """Wrapper should call downscale_Temperature with t_min settings."""
    with patch("micropyzzotmet.main_micromet.downscale_Temperature") as mock_downscale:
        run_temperature_min(
            curr_climate_file="input.nc",
            dem_path="dem.tif",
            working_directory="/tmp/work",
            variables_to_downscale={},
            custom_lapse_rates={},
            dem_nodata=-9999,
            time_chunk=24,
        )

    kwargs = mock_downscale.call_args.kwargs
    assert kwargs["source_temperature_var"] == "t_min"
    assert kwargs["output_temperature_var"] == "t_min"
    assert kwargs["output_file_prefix"] == "temperature_min_downscaled"


def test_run_temperature_max_uses_t_max_mapping():
    """Wrapper should call downscale_Temperature with t_max settings."""
    with patch("micropyzzotmet.main_micromet.downscale_Temperature") as mock_downscale:
        run_temperature_max(
            curr_climate_file="input.nc",
            dem_path="dem.tif",
            working_directory="/tmp/work",
            variables_to_downscale={},
            custom_lapse_rates={},
            dem_nodata=-9999,
            time_chunk=24,
        )

    kwargs = mock_downscale.call_args.kwargs
    assert kwargs["source_temperature_var"] == "t_max"
    assert kwargs["output_temperature_var"] == "t_max"
    assert kwargs["output_file_prefix"] == "temperature_max_downscaled"
