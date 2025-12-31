"""
Global test configuration for QuantLab-TA.

This module defines strict accuracy requirements and common test utilities
to ensure all indicators match TA-Lib output within acceptable tolerances.
"""

# Strict global tolerance for TA-Lib comparison
# As requested, we use 1e-6 as the threshold for largest error
GLOBAL_RTOL = 1e-6  # Relative tolerance
GLOBAL_ATOL = 1e-10  # Absolute tolerance for near-zero values

# Parameter ranges for comprehensive testing
PERIOD_PARAMS = [2, 3, 5, 10, 14, 20, 50, 100, 200]
FAST_SLOW_PARAMS = [(3, 10), (5, 15), (12, 26), (10, 30), (20, 50)]
FACTOR_PARAMS = [1.0, 2.0, 2.5, 3.0, 5.0]


# Edge case data generators
def generate_edge_cases():
    """Generate common edge case test data."""
    import numpy as np

    return {
        "all_nan": np.full(100, np.nan),
        "single_value": np.full(100, 42.0),
        "constant": np.full(100, 100.0),
        "linear_increasing": np.arange(100, dtype=float),
        "linear_decreasing": np.arange(100, 0, -1, dtype=float),
        "single_spike": np.concatenate([np.ones(50), [1000.0], np.ones(49)]),
        "alternating": np.array([100.0, 50.0] * 50),
        "mostly_nan": np.concatenate([np.full(90, np.nan), np.arange(10, dtype=float)]),
        "short_data": np.array([1.0, 2.0, 3.0]),
    }


def assert_indicator_match(
    result,
    expected,
    indicator_name,
    rtol=None,
    atol=None,
    allow_nans_differ=False,
):
    """
    Assert that indicator results match expected values within tolerance.

    Parameters
    ----------
    result : array-like
        Calculated indicator values
    expected : array-like
        Expected values (typically from TA-Lib)
    indicator_name : str
        Name of the indicator for error messages
    rtol : float, optional
        Relative tolerance (defaults to GLOBAL_RTOL)
    atol : float, optional
        Absolute tolerance (defaults to GLOBAL_ATOL)
    allow_nans_differ : bool, default False
        If True, allows NaN patterns to differ (use with caution)
    """
    import numpy as np
    from numpy.testing import assert_allclose

    if rtol is None:
        rtol = GLOBAL_RTOL
    if atol is None:
        atol = GLOBAL_ATOL

    # Convert to numpy arrays
    result = np.asarray(result)
    expected = np.asarray(expected)

    # Check shapes match
    if result.shape != expected.shape:
        raise AssertionError(
            f"{indicator_name}: Shape mismatch - "
            f"result: {result.shape}, expected: {expected.shape}",
        )

    # Check NaN patterns match (unless explicitly allowed to differ)
    if not allow_nans_differ:
        result_nan_mask = np.isnan(result)
        expected_nan_mask = np.isnan(expected)
        if not np.array_equal(result_nan_mask, expected_nan_mask):
            raise AssertionError(
                f"{indicator_name}: NaN pattern mismatch\\n"
                f"Result NaNs at: {np.where(result_nan_mask)[0]}\\n"
                f"Expected NaNs at: {np.where(expected_nan_mask)[0]}",
            )

    # Compare non-NaN values
    valid_mask = ~(np.isnan(result) | np.isnan(expected))
    if np.any(valid_mask):
        try:
            assert_allclose(
                result[valid_mask],
                expected[valid_mask],
                rtol=rtol,
                atol=atol,
                err_msg=f"{indicator_name} accuracy test failed",
            )
        except AssertionError as e:
            # Enhanced error message with actual deviations
            abs_diff = np.abs(result[valid_mask] - expected[valid_mask])
            rel_diff = abs_diff / np.abs(expected[valid_mask])
            max_abs_diff = np.max(abs_diff)
            max_rel_diff = np.max(rel_diff)
            max_abs_idx = np.argmax(abs_diff)

            raise AssertionError(
                f"{indicator_name} accuracy test failed:\\n"
                f"Max absolute difference: {max_abs_diff:.2e} at index {max_abs_idx}\\n"
                f"Max relative difference: {max_rel_diff:.2e} ({max_rel_diff * 100:.4f}%)\\n"
                f"Required rtol: {rtol}, atol: {atol}\\n"
                f"Original error: {e!s}",
            ) from e
