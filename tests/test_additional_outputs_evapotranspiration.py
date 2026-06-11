"""Tests for evapotranspiration utilities in additional_outputs."""

import numpy as np

from micropyzzotmet.additional_outputs import (
    compute_extraterrestrial_radiation,
    compute_reference_evapotranspiration,
)


def test_compute_extraterrestrial_radiation_reasonable_range():
    """Ra should be positive and within expected climatological range."""
    ra = compute_extraterrestrial_radiation(latitude_deg=45.0, day_of_year=172)
    assert ra > 30.0
    assert ra < 50.0


def test_hargreaves_samani_et0_positive():
    """Standard HS ET0 should be positive under warm, non-zero diurnal range conditions."""
    et0 = compute_reference_evapotranspiration(
        t_min_c=10.0,
        t_max_c=25.0,
        latitude_deg=45.0,
        day_of_year=172,
        method="hargreaves_samani",
    )
    assert et0 > 0.0


def test_unsupported_method_raises_value_error():
    """Only HS method should be accepted."""
    try:
        compute_reference_evapotranspiration(
            t_min_c=10.0,
            t_max_c=25.0,
            latitude_deg=45.0,
            day_of_year=172,
            method="modified_hargreaves_samani",
        )
        raise AssertionError("Expected ValueError for unsupported method")
    except ValueError as exc:
        assert "Only HS" in str(exc)


def test_aliases_return_same_result_for_hs():
    """Method aliases for HS should produce equivalent results."""
    kwargs = dict(t_min_c=12.0, t_max_c=26.0, latitude_deg=42.0, day_of_year=200)
    et_hs = compute_reference_evapotranspiration(method="hs", **kwargs)
    et_full = compute_reference_evapotranspiration(method="hargreaves_samani", **kwargs)
    np.testing.assert_allclose(et_hs, et_full)
