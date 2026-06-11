"""
Tests for ERA5-Land aggregation helpers.
"""

import numpy as np
import pandas as pd
import xarray as xr

from micropyzzotmet.get_era5_land import aggregate_to_daily


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
