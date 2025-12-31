# mypy: disable-error-code="no-any-return,arg-type,dict-item"
"""Fractional differencing for stationarity with memory preservation.

This module implements the Fixed-Width Window Fractional Differencing (FFD) method,
which transforms non-stationary time series to achieve stationarity while preserving
as much memory of the original series as possible.

Based on advances in financial machine learning, particularly the work on
fractional differencing for feature engineering.
"""

from functools import lru_cache

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit
from statsmodels.tsa.stattools import adfuller

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _compute_ffd_weights_nb(d: float, threshold: float, max_length: int) -> npt.NDArray[np.float64]:
    """Compute FFD weights using Numba for performance.

    Parameters
    ----------
    d : float
        Fractional differencing parameter
    threshold : float
        Minimum weight magnitude to include
    max_length : int
        Maximum number of weights to compute

    Returns
    -------
    npt.NDArray
        Array of weights
    """
    # Start with first weight = 1
    weights = np.zeros(max_length)
    weights[0] = 1.0

    # Compute subsequent weights
    for k in range(1, max_length):
        weight = -weights[k - 1] * (d - k + 1) / k

        if abs(weight) < threshold:
            # Truncate when weight becomes negligible
            return weights[:k]

        weights[k] = weight

    return weights


@lru_cache(maxsize=128)
def get_ffd_weights(
    d: float,
    threshold: float = 1e-5,
    max_length: int = 10000,
) -> npt.NDArray[np.float64]:
    """Get fractional differencing weights with caching.

    The weights follow the binomial close:
    w_k = prod_{i=0}^{k-1} (d-i) / (k!)

    Parameters
    ----------
    d : float
        Fractional differencing parameter (0 <= d <= 2)
    threshold : float, default 1e-5
        Minimum weight magnitude to include
    max_length : int, default 10000
        Maximum number of weights to compute

    Returns
    -------
    npt.NDArray
        Array of weights

    Raises
    ------
    InvalidParameterError
        If d is outside valid range
    """
    if not 0 <= d <= 2:
        raise InvalidParameterError(f"d must be between 0 and 2, got {d}")

    if d == 0:
        return np.array([1.0])

    weights = _compute_ffd_weights_nb(d, threshold, max_length)
    # Trim zeros from the end
    last_nonzero = np.nonzero(weights)[0]
    if len(last_nonzero) > 0:
        return weights[: last_nonzero[-1] + 1]
    return np.array([1.0])


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _apply_ffd_weights_nb(
    close: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Apply FFD weights to close using Numba.

    Parameters
    ----------
    close : npt.NDArray
        Input close
    weights : npt.NDArray
        FFD weights

    Returns
    -------
    npt.NDArray
        Fractionally differenced close
    """
    n = len(close)
    m = len(weights)
    result = np.full(n, np.nan)

    # Apply convolution - use min(i+1, m) weights for each position
    for i in range(n):
        # Determine how many weights to use (at most i+1 or m)
        n_weights_to_use = min(i + 1, m)
        window_start = i - n_weights_to_use + 1

        # Check if we have enough non-NaN close
        window = close[window_start : i + 1]

        # Skip if any NaN in window
        has_nan = False
        for val in window:
            if np.isnan(val):
                has_nan = True
                break

        if not has_nan:
            # Apply weights (need to reverse for convolution)
            weighted_sum = 0.0
            for j in range(n_weights_to_use):
                weighted_sum += close[i - j] * weights[j]
            result[i] = weighted_sum

    return result


@feature(
    name="ffdiff",
    category="ml",
    description="Fractional differencing for stationarity with memory preservation (FFD)",
    lookback="variable",
    # Note: Output is stationary but value_range is data-dependent (based on input volatility)
    # so we don't set normalized=True to avoid normalization warnings
    formula="w_k = prod_{i=0}^{k-1} (d-i) / k! ; result_t = sum_{k=0}^K w_k * x_{t-k}",
    input_type="close",
    parameters={"d": 0.5, "threshold": 1e-5},
    tags=["stationarity", "preprocessing", "memory", "ffd", "machine-learning"],
)
def ffdiff(
    close: pl.Series | pl.Expr | str,
    d: float,
    threshold: float = 1e-5,
) -> pl.Expr | pl.Series:
    """Apply fractional differencing to a series.

    This implements the Fixed-Width Window Fractional Differencing (FFD) method,
    which achieves stationarity while preserving maximum memory of the original series.

    Parameters
    ----------
    close : pl.Series, pl.Expr, or str
        Input close to difference. If str, interpreted as column name
    d : float
        Fractional differencing parameter (0 <= d <= 2)
        - d=0: returns original series
        - d=1: equivalent to first differencing
        - 0<d<1: partial differencing for memory preservation
    threshold : float, default 1e-5
        Minimum weight magnitude to include in calculation

    Returns
    -------
    Union[pl.Expr, pl.Series]
        Fractionally differenced close as Polars expression or Series

    Examples
    --------
    >>> import polars as pl
    >>> import ml4t_features as qf
    >>>
    >>> # Apply to a column
    >>> df = pl.DataFrame({"price": [100, 102, 101, 103, 105]})
    >>> df.with_columns(price_ffd=qf.features.fdiff.ffdiff("price", d=0.5))
    """
    # Get weights
    weights = get_ffd_weights(d, threshold)

    # Convert to expression if needed
    if isinstance(close, str):
        expr = pl.col(close)
    elif isinstance(close, pl.Series):
        expr = pl.lit(close)
    else:
        expr = close

    # Apply weights using map_batches for Numba acceleration
    def apply_ffd(s: pl.Series) -> pl.Series:
        values_np = s.to_numpy()
        result_np = _apply_ffd_weights_nb(values_np, weights)
        return pl.Series(result_np)

    # If input was a Series, apply directly and return Series
    if isinstance(close, pl.Series):
        return apply_ffd(close)
    # For expressions or column names, return expression
    return expr.map_batches(apply_ffd, return_dtype=pl.Float64)


def find_optimal_d(
    close: pl.Series | pl.Expr,
    d_range: tuple[float, float] = (0.0, 1.0),
    step: float = 0.01,
    adf_pvalue_threshold: float = 0.05,
) -> dict[str, float]:
    """Find minimum d that achieves stationarity.

    Uses grid search to find the minimum fractional differencing parameter
    that makes the series stationary according to the ADF test.

    Parameters
    ----------
    close : pl.Series or pl.Expr
        Input close to test
    d_range : tuple, default (0.0, 1.0)
        Range of d close to search
    step : float, default 0.01
        Step size for grid search
    adf_pvalue_threshold : float, default 0.05
        P-value threshold for stationarity

    Returns
    -------
    dict
        Dictionary containing:
        - optimal_d: the minimum d achieving stationarity
        - adf_pvalue: p-value at optimal d
        - correlation: correlation with original series

    Examples
    --------
    >>> result = find_optimal_d(df["price"])
    >>> print(f"Optimal d: {result['optimal_d']:.3f}")
    """
    # Convert to numpy for analysis
    if isinstance(close, pl.Expr):
        raise ValueError("find_optimal_d requires a Series, not an Expr")

    values_np = close.to_numpy()

    # Remove NaN close for analysis
    clean_values = values_np[~np.isnan(values_np)]

    # Test if already stationary
    adf_result = adfuller(clean_values, autolag="AIC")
    if adf_result[1] < adf_pvalue_threshold:
        return {"optimal_d": 0.0, "adf_pvalue": adf_result[1], "correlation": 1.0}

    # Grid search
    d_values = np.arange(d_range[0], d_range[1] + step, step)

    for d in d_values:
        if d == 0:
            continue

        # Apply fractional differencing
        ffd_result = ffdiff(close, d=d)
        # Since we pass a Series, ffdiff will return a Series
        assert isinstance(ffd_result, pl.Series)  # type narrowing for mypy
        ffd_values = ffd_result.to_numpy()

        # Clean NaN close
        clean_ffd = ffd_values[~np.isnan(ffd_values)]

        if len(clean_ffd) < 50:  # Need enough data for ADF test
            continue

        # Test stationarity
        adf_result = adfuller(clean_ffd, autolag="AIC")

        if adf_result[1] < adf_pvalue_threshold:
            # Calculate correlation with original
            # Align the series properly
            n_ffd = len(clean_ffd)
            correlation = np.corrcoef(clean_values[-n_ffd:], clean_ffd)[0, 1]

            return {
                "optimal_d": d,
                "adf_pvalue": adf_result[1],
                "correlation": correlation,
            }

    # If no d achieves stationarity, return the last tested
    return {"optimal_d": d_range[1], "adf_pvalue": adf_result[1], "correlation": 0.0}


def fdiff_diagnostics(
    close: pl.Series | pl.Expr,
    d: float,
    threshold: float = 1e-5,
) -> dict[str, float | int]:
    """Get detailed diagnostics for fractional differencing.

    Parameters
    ----------
    close : pl.Series or pl.Expr
        Input close
    d : float
        Fractional differencing parameter
    threshold : float, default 1e-5
        Weight threshold

    Returns
    -------
    dict
        Dictionary containing:
        - d: the differencing parameter used
        - adf_statistic: ADF test statistic
        - adf_pvalue: ADF test p-value
        - correlation: correlation with original series
        - n_weights: number of weights used
        - weight_sum: sum of absolute weights
    """
    # Convert to numpy
    if isinstance(close, pl.Expr):
        raise ValueError("fdiff_diagnostics requires a Series, not an Expr")

    values_np = close.to_numpy()
    clean_values = values_np[~np.isnan(values_np)]

    # Apply fractional differencing
    ffd_result = ffdiff(close, d=d, threshold=threshold)
    # Since we pass a Series, ffdiff will return a Series
    assert isinstance(ffd_result, pl.Series)  # type narrowing for mypy
    ffd_values = ffd_result.to_numpy()
    clean_ffd = ffd_values[~np.isnan(ffd_values)]

    # Get weights info
    weights = get_ffd_weights(d, threshold)

    # Calculate diagnostics
    if len(clean_ffd) >= 50:
        adf_result = adfuller(clean_ffd, autolag="AIC")
        adf_stat = adf_result[0]
        adf_pvalue = adf_result[1]

        # Calculate correlation
        n_ffd = len(clean_ffd)
        correlation = np.corrcoef(clean_values[-n_ffd:], clean_ffd)[0, 1]
    else:
        adf_stat = np.nan
        adf_pvalue = np.nan
        correlation = np.nan

    return {
        "d": d,
        "adf_statistic": adf_stat,
        "adf_pvalue": adf_pvalue,
        "correlation": correlation,
        "n_weights": len(weights),
        "weight_sum": np.sum(np.abs(weights)),
    }


__all__ = [
    "fdiff_diagnostics",
    "ffdiff",
    "find_optimal_d",
    "get_ffd_weights",
]
