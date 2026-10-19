"""
Tests for ERA5-Land aggregation helpers.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from micropyzzotmet.get_era5_land import (
    _is_transient_http_error,
    aggregate_to_daily,
    process_month,
    subset_era5_domain,
)


def test_aggregate_to_daily_adds_temperature_extrema():
    """Daily aggregation should include t_min and t_max derived from t2m."""
    times = pd.date_range("2024-01-01", periods=48, freq="h")

    # Day 1: 270..293 K, Day 2: 250..273 K
    t2m_values = np.array(list(range(270, 294)) + list(range(250, 274)), dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "t2m": ("valid_time", t2m_values),
            "u10": ("valid_time", np.ones(48, dtype=np.float32)),
        },
        coords={"valid_time": times},
    )

    ds["t2m"].attrs["units"] = "K"

    daily = aggregate_to_daily(ds)

    assert "t2m" in daily
    assert "t_min" in daily
    assert "t_max" in daily

    np.testing.assert_allclose(daily["t_min"].values, np.array([270.0, 250.0], dtype=np.float32))
    np.testing.assert_allclose(daily["t_max"].values, np.array([293.0, 273.0], dtype=np.float32))
    np.testing.assert_allclose(daily["t2m"].values, np.array([281.5, 261.5], dtype=np.float32))

    assert daily["t_min"].attrs["units"] == "K"
    assert daily["t_max"].attrs["units"] == "K"


def test_subset_era5_domain_handles_prime_meridian_crossing():
    """Longitude subsetting should keep cells on both sides of 0 degrees."""
    ds = xr.Dataset(
        coords={
            "latitude": np.array([15.0, 14.5, 14.0], dtype=np.float32),
            "longitude": np.array([353.0, 354.0, 355.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        },
    )

    subset = subset_era5_domain(ds, lat_min=14.0, lat_max=15.0, lon_min=-6.0, lon_max=2.0)

    np.testing.assert_allclose(
        subset.longitude.values,
        np.array([354.0, 355.0, 0.0, 1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        subset.latitude.values,
        np.array([15.0, 14.5, 14.0], dtype=np.float32),
    )


def test_subset_era5_domain_handles_standard_longitude_window():
    """Longitude subsetting should still work for non-wrapping extents."""
    ds = xr.Dataset(
        coords={
            "latitude": np.array([15.0, 14.5, 14.0], dtype=np.float32),
            "longitude": np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32),
        },
    )

    subset = subset_era5_domain(ds, lat_min=14.0, lat_max=15.0, lon_min=10.5, lon_max=12.5)

    np.testing.assert_allclose(
        subset.longitude.values,
        np.array([11.0, 12.0], dtype=np.float32),
    )


def test_is_transient_http_error_detects_http_502():
    """HTTP 502 responses should be treated as retryable transient failures."""

    class FakeHttpError(Exception):
        pass

    exc = FakeHttpError("502, message='Bad Gateway'")

    assert _is_transient_http_error(exc) is True


def test_process_month_raises_serializable_runtime_error_for_final_failure(tmp_path):
    """Final worker failure should be rethrown as a plain picklable RuntimeError."""

    class FakeHttpError(Exception):
        pass

    class FailingSurface:
        def load(self):
            raise FakeHttpError("502, message='Bad Gateway'")

    class FailingMonth:
        def __getitem__(self, _surface_vars):
            return FailingSurface()

    class FailingDataset:
        def sel(self, **_kwargs):
            return FailingMonth()

    with pytest.raises(RuntimeError, match="ERA download/read failed") as exc_info:
        process_month(
            FailingDataset(),
            pd.Timestamp("2024-01-01"),
            str(tmp_path),
            ["t2m"],
            max_retries=1,
        )

    assert "502" in str(exc_info.value)
    pickle.dumps(exc_info.value)
