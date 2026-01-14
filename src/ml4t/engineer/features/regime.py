# mypy: disable-error-code="no-any-return"
"""Market regime identification features.

Exports:
    choppiness_index(high, low, close, period=14) -> Expr
        Identifies trending vs choppy/range-bound markets (0-100).

    variance_ratio(close, period=20, holding_period=5) -> Expr
        Lo-MacKinlay variance ratio for random walk testing.

    fractal_efficiency(close, period=10) -> Expr
        Price path efficiency (0-1). Higher = more trending.

    hurst_exponent(close, window=100, max_lag=20) -> Expr
        Long memory/persistence measure. H>0.5 = trending, H<0.5 = mean reverting.

    trend_intensity_index(close, period=30) -> Expr
        Directional strength based on positive vs negative closes.

    market_regime_classifier(data, ...) -> DataFrame
        Multi-indicator regime classification.

This module provides features to identify market regimes (trending vs range-bound)
which is critical for ML models to adapt their predictions based on market state.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_list_length,
    validate_positive,
    validate_threshold,
    validate_window,
)


@feature(
    name="choppiness_index",
    category="regime",
    description="Choppiness Index - identifies trending vs choppy/range-bound markets",
    lookback="period",
    normalized=True,
    value_range=(0.0, 100.0),
    formula="CI = 100 * LOG10(SUM(ATR, n) / (MAX(HIGH, n) - MIN(LOW, n))) / LOG10(n)",
    input_type="OHLC",
    parameters={"period": 14},
    tags=["regime", "trend-strength", "choppiness"],
)
def choppiness_index(
    high: pl.Expr | str,
    low: pl.Expr | str,
    close: pl.Expr | str,
    period: int = 14,
) -> pl.Expr:
    """Calculate Choppiness Index to identify trending vs choppy markets.

    The Choppiness Index determines if the market is trending or consolidating.
    Values range from 0 to 100:
    - Below 38.2: Strong trending market
    - 38.2 to 61.8: Transitional
    - Above 61.8: Choppy/range-bound market

    Parameters
    ----------
    high : pl.Expr | str
        High price column
    low : pl.Expr | str
        Low price column
    close : pl.Expr | str
        Close price column
    period : int, default 14
        Lookback period

    Returns
    -------
    pl.Expr
        Choppiness Index close

    Raises
    ------
    ValueError
        If period is not positive
    TypeError
        If period is not an integer

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     "high": [101, 102, 103, 102, 101],
    ...     "low": [99, 100, 101, 100, 99],
    ...     "close": [100, 101, 102, 101, 100]
    ... })
    >>> df.with_columns(choppiness_index("high", "low", "close", 3).alias("chop"))
    """
    # Validate inputs
    validate_window(period, min_window=2, name="period")

    # Convert strings to expressions
    high = pl.col(high) if isinstance(high, str) else high
    low = pl.col(low) if isinstance(low, str) else low
    close = pl.col(close) if isinstance(close, str) else close

    # Calculate True Range components
    true_range = pl.max_horizontal(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
    )

    # Sum of true ranges over period
    atr_sum = true_range.rolling_sum(period)

    # Highest high and lowest low over period
    highest_high = high.rolling_max(period)
    lowest_low = low.rolling_min(period)

    # Choppiness Index formula
    # CI = 100 * LOG10(SUM(ATR, n) / (MAX(HIGH, n) - MIN(LOW, n))) / LOG10(n)
    choppiness = 100 * (atr_sum / (highest_high - lowest_low)).log10() / np.log10(period)

    return choppiness.cast(pl.Float64)


def variance_ratio(
    close: pl.Expr | str,
    periods: list[int] | None = None,
    base_period: int = 1,
    window: int | None = None,
) -> dict[str, pl.Expr]:
    """Calculate Variance Ratio test statistics for market efficiency.

    The variance ratio tests the random walk hypothesis. Values significantly
    different from 1 indicate either trending (>1) or mean-reverting (<1) behavior.

    Parameters
    ----------
    close : pl.Expr | str
        Close price column
    periods : list[int], optional
        Periods to calculate variance ratios for (default: [2, 4, 8, 16])
    base_period : int, default 1
        Base period for comparison
    window : int, optional
        Rolling window for variance calculation. If None, uses global variance (legacy behavior)

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary of variance ratio expressions for each period

    Raises
    ------
    ValueError
        If periods is empty, contains non-positive close, or base_period is not positive
    TypeError
        If periods contains non-integers or base_period is not an integer

    Examples
    --------
    >>> df = pl.DataFrame({"close": [100, 101, 102, 101, 100, 99, 98, 99, 100]})
    >>> vr = variance_ratio("close", periods=[2, 4])
    >>> df.with_columns([vr[f"vr_{p}"].alias(f"vr_{p}") for p in [2, 4]])
    """
    if periods is None:
        periods = [2, 4, 8, 16]

    # Validate inputs
    validate_list_length(periods, min_length=1, name="periods")
    for i, period in enumerate(periods):
        if not isinstance(period, int):
            raise TypeError(
                f"periods[{i}] must be an integer, got {type(period).__name__}",
            )
        if period <= 0:
            raise ValueError(f"periods[{i}] must be positive, got {period}")

    validate_window(base_period, min_window=1, name="base_period")

    close = pl.col(close) if isinstance(close, str) else close

    # Calculate returns
    base_returns = close.pct_change(base_period)

    # Use rolling variance if window is specified, otherwise global (for backwards compatibility)
    if window is not None:
        validate_window(window, min_window=max(periods) + 1, name="window")
        base_var = base_returns.rolling_var(window_size=window)
    else:
        base_var = base_returns.var()

    results = {}
    for period in periods:
        # Calculate k-period returns
        k_returns = close.pct_change(period)

        # Use rolling variance if window is specified
        k_var = k_returns.rolling_var(window_size=window) if window is not None else k_returns.var()

        # Variance ratio = Var(k-period returns) / (k * Var(1-period returns))
        vr = k_var / (period * base_var)
        results[f"vr_{period}"] = vr

    return results


@feature(
    name="fractal_efficiency",
    category="regime",
    description="Fractal Efficiency Ratio - measures how efficiently price moves",
    lookback="period",
    normalized=True,
    value_range=(0.0, 1.0),
    formula="FE = (|close[t] - close[t-n]|) / sum(|close[i] - close[i-1]|)",
    input_type="close",
    parameters={"window": 20},
    tags=["regime", "efficiency", "trend-quality"],
)
def fractal_efficiency(close: pl.Expr | str, period: int = 10) -> pl.Expr:
    """Calculate Fractal Efficiency Ratio (Efficiency Ratio).

    Measures the "straightness" of price movement. Values near 1 indicate
    efficient trending movement, while close near 0 indicate choppy action.

    Parameters
    ----------
    close : pl.Expr | str
        Close price column
    period : int, default 10
        Lookback period

    Returns
    -------
    pl.Expr
        Fractal efficiency close (0 to 1)

    Raises
    ------
    ValueError
        If period is not positive
    TypeError
        If period is not an integer
    """
    # Validate inputs
    validate_window(period, min_window=2, name="period")

    close = pl.col(close) if isinstance(close, str) else close

    # Net change over period
    net_change = (close - close.shift(period)).abs()

    # Sum of absolute period-to-period changes
    period_changes = close.diff().abs().rolling_sum(period)

    # Efficiency = Net Change / Sum of Individual Changes
    efficiency = net_change / period_changes

    return efficiency


@jit(nopython=True, cache=True)  # type: ignore[misc]
def hurst_exponent_nb(
    close: npt.NDArray[np.float64],
    min_lag: int = 2,
    max_lag: int = 100,
) -> float:
    """Calculate Hurst Exponent using R/S analysis (Numba optimized).

    H > 0.5: Trending (persistent)
    H = 0.5: Random walk
    H < 0.5: Mean reverting (anti-persistent)
    """
    if len(close) < max_lag:
        return float(np.nan)

    # Calculate returns
    returns = np.diff(np.log(close))

    # R/S analysis
    # Use np.arange instead of range for Numba compatibility
    lags = np.arange(min_lag, min(max_lag, len(returns) // 2))
    rs_values = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        # Divide series into non-overlapping blocks
        n_blocks = len(returns) // lag

        rs_block = np.zeros(n_blocks)
        for j in range(n_blocks):
            block = returns[j * lag : (j + 1) * lag]

            # Skip blocks with insufficient data
            if len(block) < 2:
                continue

            # Mean-adjusted series
            mean_adj = block - np.mean(block)

            # Cumulative sum
            cum_sum = np.cumsum(mean_adj)

            # Range
            R = float(np.max(cum_sum) - np.min(cum_sum))

            # Standard deviation (ddof=1 for sample std)
            S = np.sqrt(np.var(block) * len(block) / (len(block) - 1))

            if S > 0:
                rs_block[j] = R / S
            else:
                rs_block[j] = 0

        # Only compute mean if there are non-zero values
        valid_rs = rs_block[rs_block > 0]
        if len(valid_rs) > 0:
            rs_values[i] = np.mean(valid_rs)
        else:
            rs_values[i] = 0

    # Linear regression of log(R/S) on log(lag)
    valid_mask = rs_values > 0
    if np.sum(valid_mask) < 2:
        return float(np.nan)

    log_lags = np.log(lags[valid_mask])
    log_rs = np.log(rs_values[valid_mask])

    # Fit line: log(R/S) = log(c) + H * log(lag)
    # Use tuple instead of list for Numba compatibility
    A = np.vstack((log_lags, np.ones(len(log_lags)))).T
    # Use rcond=-1 instead of None for Numba compatibility (uses machine precision)
    hurst, _ = np.linalg.lstsq(A, log_rs, rcond=-1.0)[0]

    return float(hurst)


@feature(
    name="hurst_exponent",
    category="regime",
    description="Hurst Exponent - measures long-term memory and trend persistence",
    lookback="period",
    normalized=True,
    value_range=(0.0, 1.0),
    formula="H from R/S analysis: log(R/S) = log(c) + H * log(lag)",
    input_type="close",
    parameters={"period": 100, "min_lag": 2, "max_lag": None},
    tags=["regime", "persistence", "memory", "r-s-analysis"],
)
def hurst_exponent(
    close: pl.Expr | str,
    period: int = 100,
    min_lag: int = 2,
    max_lag: int | None = None,
) -> pl.Expr:
    """Calculate rolling Hurst Exponent.

    The Hurst Exponent measures the long-term memory of a time close:
    - H > 0.5: Trending behavior (persistent)
    - H = 0.5: Random walk
    - H < 0.5: Mean-reverting behavior (anti-persistent)

    Parameters
    ----------
    close : pl.Expr | str
        Close price column
    period : int, default 100
        Rolling window size
    min_lag : int, default 2
        Minimum lag for R/S analysis
    max_lag : int, optional
        Maximum lag for R/S analysis (defaults to period//2)

    Returns
    -------
    pl.Expr
        Hurst exponent close

    Raises
    ------
    ValueError
        If period, min_lag, or max_lag are not positive, or if min_lag >= max_lag
    TypeError
        If period, min_lag, or max_lag are not integers
    """
    # Validate inputs
    validate_window(
        period,
        min_window=10,
        name="period",
    )  # Need reasonable window for R/S analysis
    validate_window(min_lag, min_window=2, name="min_lag")

    if max_lag is not None:
        validate_window(max_lag, min_window=min_lag + 1, name="max_lag")
        if min_lag >= max_lag:
            raise ValueError(
                f"min_lag ({min_lag}) must be less than max_lag ({max_lag})",
            )

    close = pl.col(close) if isinstance(close, str) else close

    if max_lag is None:
        max_lag = period // 2

    # Apply rolling window with custom function
    # Wrap in try/except for numerical stability with edge cases
    def safe_hurst(x: pl.Series) -> float:
        try:
            arr = x.to_numpy() if hasattr(x, "to_numpy") else x
            return float(hurst_exponent_nb(arr, min_lag, max_lag))
        except (ZeroDivisionError, ValueError, RuntimeError):
            return float(np.nan)

    return close.rolling_map(
        safe_hurst,
        window_size=period,
        weights=None,
        min_samples=period,
        center=False,
    )


@feature(
    name="trend_intensity_index",
    category="regime",
    description="Trend Intensity Index - measures trend strength",
    lookback="period",
    normalized=True,
    value_range=(0.0, 100.0),
    formula="TII = 100 * (closes_above_MA / period)",
    input_type="close",
    parameters={"period": 60},
    tags=["regime", "trend-strength"],
)
def trend_intensity_index(close: pl.Expr | str, period: int = 30) -> pl.Expr:
    """Calculate Trend Intensity Index.

    Measures the strength of the current trend by comparing price movements
    in the trend direction vs total price movement.

    Parameters
    ----------
    close : pl.Expr | str
        Close price column
    period : int, default 30
        Lookback period

    Returns
    -------
    pl.Expr
        Trend intensity close (0 to 100)

    Raises
    ------
    ValueError
        If period is not positive
    TypeError
        If period is not an integer
    """
    # Validate inputs
    validate_window(period, min_window=2, name="period")

    close = pl.col(close) if isinstance(close, str) else close

    # Calculate period-to-period changes
    changes = close.diff()

    # Separate positive and negative changes
    pos_changes = pl.when(changes > 0).then(changes).otherwise(0)
    neg_changes = pl.when(changes < 0).then(changes.abs()).otherwise(0)

    # Sum over period
    sum_pos = pos_changes.rolling_sum(period)
    sum_neg = neg_changes.rolling_sum(period)

    # TII = 100 * sum(positive moves) / sum(all moves)
    # For downtrend: TII = 100 * sum(negative moves) / sum(all moves)
    tii_up = 100 * sum_pos / (sum_pos + sum_neg)
    tii_down = 100 * sum_neg / (sum_pos + sum_neg)

    # Return the dominant trend intensity
    return pl.when(sum_pos > sum_neg).then(tii_up).otherwise(-tii_down)


def market_regime_classifier(
    high: pl.Expr | str,
    low: pl.Expr | str,
    close: pl.Expr | str,
    volume: pl.Expr | str,
    adx_threshold: float = 25.0,
    chop_threshold_high: float = 61.8,
    chop_threshold_low: float = 38.2,
) -> pl.Expr:
    """Classify market regime using multiple indicators.

    Combines ADX, Choppiness Index, and volume patterns to classify:
    - 1: Strong Trend
    - 0: Transitional
    - -1: Range-bound/Choppy

    Parameters
    ----------
    high, low, close, volume : pl.Expr | str
        Price and volume columns
    adx_threshold : float, default 25.0
        ADX threshold for trending market
    chop_threshold_high : float, default 61.8
        Upper threshold for choppy market
    chop_threshold_low : float, default 38.2
        Lower threshold for trending market

    Returns
    -------
    pl.Expr
        Market regime classification (-1, 0, 1)

    Raises
    ------
    ValueError
        If thresholds are invalid or threshold ordering is incorrect
    TypeError
        If thresholds are not numeric
    """
    # Validate inputs
    validate_positive(adx_threshold, name="adx_threshold")
    validate_threshold(chop_threshold_high, 0.0, 100.0, name="chop_threshold_high")
    validate_threshold(chop_threshold_low, 0.0, 100.0, name="chop_threshold_low")

    if chop_threshold_low >= chop_threshold_high:
        raise ValueError(
            f"chop_threshold_low ({chop_threshold_low}) must be less than chop_threshold_high ({chop_threshold_high})",
        )

    from ml4t.engineer.features.momentum import adx

    # Calculate indicators
    adx_val = adx(high, low, close, 14)
    chop = choppiness_index(high, low, close, 14)

    # Volume trend (increasing volume in trends)
    volume = pl.col(volume) if isinstance(volume, str) else volume
    vol_sma_short = volume.rolling_mean(5)
    vol_sma_long = volume.rolling_mean(20)
    vol_trend = vol_sma_short > vol_sma_long

    # Classify regime
    regime = (
        pl.when((adx_val > adx_threshold) & (chop < chop_threshold_low) & vol_trend)
        .then(1)  # Strong trend
        .when((adx_val < adx_threshold) & (chop > chop_threshold_high))
        .then(-1)  # Range-bound
        .otherwise(0)  # Transitional
    )

    return regime
