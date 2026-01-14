"""Risk management and measurement features for quantitative finance.

Exports:
    value_at_risk(returns, confidence=0.95, window=252, method="historical") -> Expr
        VaR calculation: historical, parametric, or cornish_fisher methods.

    conditional_value_at_risk(returns, confidence=0.95, window=252) -> Expr
        CVaR/Expected Shortfall (average loss beyond VaR threshold).

    maximum_drawdown(data, price_col="close", window=None) -> Expr
        Maximum peak-to-trough drawdown.

    downside_deviation(returns, target=0.0, window=252) -> Expr
        Standard deviation of returns below target.

    tail_ratio(returns, window=252) -> Expr
        Ratio of upper to lower tail quantiles.

    higher_moments(returns, window=252) -> dict[str, Expr]
        Skewness, kurtosis, and Jarque-Bera test statistics.

    risk_adjusted_returns(returns, benchmark, ...) -> dict[str, Expr]
        Sharpe, Sortino, Calmar, and information ratios.

    ulcer_index(prices, window=14) -> Expr
        Ulcer Performance Index (drawdown-based volatility).

    information_ratio(returns, benchmark, window=252) -> Expr
        Excess return per unit of tracking error.

This module provides comprehensive risk metrics including Value at Risk (VaR),
Conditional VaR (CVaR/Expected Shortfall), maximum drawdown analysis, and
higher moment calculations for financial risk management.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit
from scipy import stats

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_threshold,
    validate_window,
)


def value_at_risk(
    returns: pl.Expr | str,
    confidence_level: float = 0.95,
    window: int = 252,
    method: str = "historical",
) -> pl.Expr:
    """Calculate Value at Risk (VaR) using various methods.

    VaR estimates the maximum loss that won't be exceeded with a given
    confidence level over a specific time period.

    Parameters
    ----------
    returns : pl.Expr | str
        Return series
    confidence_level : float, default 0.95
        Confidence level (0.95 = 95% confidence)
    window : int, default 252
        Rolling window size (252 = 1 year daily data)
    method : str, default "historical"
        VaR calculation method: "historical", "parametric", "cornish_fisher"

    Returns
    -------
    pl.Expr
        Value at Risk (negative close indicate losses)

    Raises
    ------
    ValueError
        If method is invalid or confidence_level out of range

    Notes
    -----
    VaR at α confidence level answers: "What is the worst loss I can expect
    with (1-α)% probability over the time horizon?"
    """
    # Validate inputs
    validate_threshold(confidence_level, 0.0, 1.0, "confidence_level")
    validate_window(window, min_window=20)

    if method not in ["historical", "parametric", "cornish_fisher"]:
        raise ValueError(
            f"method must be 'historical', 'parametric', or 'cornish_fisher', got {method}",
        )

    returns = pl.col(returns) if isinstance(returns, str) else returns
    alpha = 1 - confidence_level

    if method == "historical":
        # Historical VaR: empirical quantile
        return returns.rolling_quantile(
            alpha,
            window_size=window,
            min_samples=window // 2,
        )

    if method == "parametric":
        # Parametric VaR: assume normal distribution
        mean = returns.rolling_mean(window, min_samples=window // 2)
        std = returns.rolling_std(window, min_samples=window // 2)
        z_score = pl.lit(stats.norm.ppf(alpha))
        return mean + z_score * std

    # cornish_fisher
    # Cornish-Fisher VaR: adjust for skewness and kurtosis
    mean = returns.rolling_mean(window, min_samples=window // 2)
    std = returns.rolling_std(window, min_samples=window // 2)

    # Calculate rolling skewness and kurtosis
    skew = returns.rolling_skew(window)
    kurt = returns.rolling_kurtosis(window, fisher=False)

    # Cornish-Fisher expansion
    z = pl.lit(stats.norm.ppf(alpha))
    cf_adjustment = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * (kurt - 3) / 24
        - (2 * z**3 - 5 * z) * (skew**2) / 36
    )

    return mean + cf_adjustment * std


def conditional_value_at_risk(
    returns: pl.Expr | str,
    confidence_level: float = 0.95,
    window: int = 252,
    method: str = "historical",
) -> pl.Expr:
    """Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

    CVaR measures the expected loss given that the loss exceeds the VaR threshold.
    It provides a more complete picture of tail risk than VaR alone.

    Parameters
    ----------
    returns : pl.Expr | str
        Return series
    confidence_level : float, default 0.95
        Confidence level (0.95 = 95% confidence)
    window : int, default 252
        Rolling window size
    method : str, default "historical"
        CVaR calculation method: "historical" or "parametric"

    Returns
    -------
    pl.Expr
        Conditional Value at Risk

    Notes
    -----
    CVaR answers: "If things go bad (beyond VaR), what is the expected loss?"
    Also known as Expected Shortfall (ES) or Average Value at Risk (AVaR).
    """
    # Validate inputs
    validate_threshold(confidence_level, 0.0, 1.0, "confidence_level")
    validate_window(window, min_window=20)

    if method not in ["historical", "parametric"]:
        raise ValueError(f"method must be 'historical' or 'parametric', got {method}")

    returns = pl.col(returns) if isinstance(returns, str) else returns

    # First calculate VaR (not used in historical method)
    # var = value_at_risk(returns, confidence_level, window, method)

    if method == "historical":
        # Historical CVaR: average of returns worse than VaR
        def calculate_cvar(close: pl.Series) -> float:
            """Calculate CVaR for a window of returns."""
            arr = close.to_numpy()
            arr = arr[~np.isnan(arr)]

            if len(arr) < window // 2:
                return float(np.nan)

            var_threshold = np.percentile(arr, (1 - confidence_level) * 100)
            tail_losses = arr[arr <= var_threshold]

            if len(tail_losses) > 0:
                return float(np.mean(tail_losses))
            return float(var_threshold)

        return returns.rolling_map(
            calculate_cvar,
            window_size=window,
            min_samples=window // 2,
        )

    # parametric
    # Parametric CVaR: analytical formula for normal distribution
    mean = returns.rolling_mean(window, min_samples=window // 2)
    std = returns.rolling_std(window, min_samples=window // 2)
    alpha = 1 - confidence_level

    # For normal distribution: ES = μ - σ * φ(z_α) / α
    z_alpha = stats.norm.ppf(alpha)
    pdf_z = stats.norm.pdf(z_alpha)

    return mean - std * pl.lit(pdf_z / alpha)


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _calculate_drawdowns_nb(
    close: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate drawdown series, duration, and recovery time using Numba.

    Parameters
    ----------
    close : npt.NDArray
        Price series

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (drawdown_series, drawdown_duration, time_to_recovery)
    """
    n = len(close)
    drawdowns = np.zeros(n)
    durations = np.zeros(n)
    recovery_times = np.zeros(n)

    running_max = close[0]
    drawdown_start = 0
    in_drawdown = False

    for i in range(n):
        if close[i] > running_max:
            running_max = close[i]

            # Recovery occurred
            if in_drawdown:
                # Fill recovery time for the drawdown period
                for j in range(drawdown_start, i):
                    recovery_times[j] = i - drawdown_start
                in_drawdown = False

        # Calculate current drawdown
        drawdowns[i] = (close[i] - running_max) / running_max

        # Track drawdown duration
        if drawdowns[i] < 0:
            if not in_drawdown:
                drawdown_start = i
                in_drawdown = True
            durations[i] = i - drawdown_start + 1
        else:
            durations[i] = 0

    # Handle case where series ends in drawdown
    if in_drawdown:
        for j in range(drawdown_start, n):
            recovery_times[j] = np.nan  # Not recovered

    return drawdowns, durations, recovery_times


@feature(
    name="maximum_drawdown",
    category="risk",
    description="Maximum Drawdown - largest peak-to-trough decline",
    lookback="window",  # Uses window parameter for rolling calculation
    normalized=False,
    formula="MDD = max((peak - trough) / peak)",
    input_type="close",
    parameters={"window": 252},
    tags=["risk", "drawdown"],
)
def maximum_drawdown(
    close: pl.Expr | str,
    window: int | None = None,
) -> dict[str, pl.Expr]:
    """Calculate maximum drawdown and related statistics.

    Maximum drawdown measures the largest peak-to-trough decline in value.
    It's a key risk metric for understanding worst-case scenarios.

    Parameters
    ----------
    close : pl.Expr | str
        Price series (not returns)
    window : int, optional
        Rolling window size. If None, calculates expanding maximum drawdown

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary containing:
        - max_drawdown: Maximum drawdown (negative percentage)
        - max_duration: Longest drawdown duration (periods)
        - current_drawdown: Current drawdown from peak
        - time_underwater: Periods in drawdown

    Notes
    -----
    Drawdown = (Current Value - Running Maximum) / Running Maximum
    """
    if window is not None:
        validate_window(window, min_window=2)

    close = pl.col(close) if isinstance(close, str) else close

    if window is None:
        # Expanding window - use cumulative maximum
        running_max = close.cum_max()
        drawdown = (close - running_max) / running_max

        # For expanding window, return simple metrics
        return {
            "max_drawdown": drawdown.cum_min(),
            "max_duration": pl.lit(None),  # Not easily computed in expanding mode
            "current_drawdown": drawdown,
            "time_underwater": (drawdown < 0).cast(pl.Int32).cum_sum()
            / pl.int_range(1, pl.len() + 1),
        }
    # Rolling window calculations
    running_max = close.rolling_max(window, min_samples=window // 2)
    drawdown = (close - running_max) / running_max

    # Calculate rolling maximum drawdown
    max_drawdown = drawdown.rolling_min(window, min_samples=window // 2)

    # Calculate drawdown duration using a custom function
    def calculate_dd_duration(s: pl.Series) -> float:
        """Calculate maximum drawdown duration in a window."""
        close = s.to_numpy()
        mask = ~np.isnan(close)
        clean_values = close[mask]

        if len(clean_values) < 2:
            return float(np.nan)

        # Simple duration calculation
        max_duration = 0
        current_duration = 0

        for val in clean_values:
            if val < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return float(max_duration)

    # Apply duration calculation
    max_duration = drawdown.rolling_map(
        calculate_dd_duration,
        window_size=window,
        min_samples=window // 2,
    )

    # Time underwater (percentage of time in drawdown)
    time_underwater = (drawdown < 0).cast(pl.Float64).rolling_mean(window, min_samples=window // 2)

    return {
        "max_drawdown": max_drawdown,
        "max_duration": max_duration,
        "current_drawdown": drawdown,
        "time_underwater": time_underwater,
    }


@feature(
    name="downside_deviation",
    category="risk",
    description="Downside Deviation (Semi-Deviation) - volatility of negative returns",
    lookback="window",
    normalized=True,
    value_range=(0.0, 2.0),  # Typical range for daily return volatility
    formula="DD = sqrt(mean((min(r - target, 0))^2))",
    input_type="returns",
    parameters={"window": 252},
    tags=["risk", "downside", "semi-deviation"],
)
def downside_deviation(
    returns: pl.Expr | str,
    target_return: float = 0.0,
    window: int = 252,
) -> pl.Expr:
    """Calculate downside deviation (semi-deviation).

    Downside deviation measures volatility of returns below a target threshold,
    focusing only on "bad" volatility.

    Parameters
    ----------
    returns : pl.Expr | str
        Return series
    target_return : float, default 0.0
        Minimum acceptable return (MAR)
    window : int, default 252
        Rolling window size

    Returns
    -------
    pl.Expr
        Downside deviation

    Notes
    -----
    DD = sqrt(E[min(R - MAR, 0)²])
    Used in Sortino Ratio calculation.
    """
    validate_window(window, min_window=2)

    returns = pl.col(returns) if isinstance(returns, str) else returns

    # Calculate downside returns (only negative deviations from target)
    downside_returns = (returns - target_return).clip(upper_bound=0.0)

    # Calculate root mean square of downside returns
    return (downside_returns**2).rolling_mean(window, min_samples=window // 2).sqrt()


@feature(
    name="tail_ratio",
    category="risk",
    description="Tail Ratio - ratio of positive to negative tail events",
    lookback="window",
    normalized=True,
    value_range=(0.0, 10.0),  # Ratio range, typically 0.5-2.0, but wider for safety
    formula="TR = abs(95th percentile) / abs(5th percentile)",
    input_type="returns",
    parameters={"window": 252},
    tags=["risk", "tails", "extremes"],
)
def tail_ratio(
    returns: pl.Expr | str,
    confidence_level: float = 0.95,
    window: int = 252,
) -> pl.Expr:
    """Calculate tail ratio comparing positive to negative tails.

    Tail ratio measures the ratio of gains in the right tail to losses
    in the left tail, indicating asymmetry of extreme returns.

    Parameters
    ----------
    returns : pl.Expr | str
        Return series
    confidence_level : float, default 0.95
        Confidence level for tail definition
    window : int, default 252
        Rolling window size

    Returns
    -------
    pl.Expr
        Tail ratio (higher is better)

    Notes
    -----
    Tail Ratio = |95th percentile| / |5th percentile|
    Values > 1 indicate positive skew in extreme returns.
    """
    validate_threshold(confidence_level, 0.5, 1.0, "confidence_level")
    validate_window(window, min_window=20)

    returns = pl.col(returns) if isinstance(returns, str) else returns

    # Calculate percentiles
    upper_percentile = returns.rolling_quantile(
        confidence_level,
        window_size=window,
        min_samples=window // 2,
    )
    lower_percentile = returns.rolling_quantile(
        1 - confidence_level,
        window_size=window,
        min_samples=window // 2,
    )

    # Tail ratio = |upper tail| / |lower tail|
    return upper_percentile.abs() / (lower_percentile.abs() + 1e-10)


@feature(
    name="higher_moments",
    category="risk",
    description="Higher Moments - skewness and kurtosis of returns",
    lookback="window",
    normalized=False,  # Returns dict with multiple metrics
    formula="Skew = E[(r - mean)^3] / std^3, Kurt = E[(r - mean)^4] / std^4",
    input_type="returns",
    parameters={"window": 252},
    tags=["risk", "skewness", "kurtosis", "moments"],
)
def higher_moments(
    returns: pl.Expr | str,
    window: int = 252,
) -> dict[str, pl.Expr]:
    """Calculate higher statistical moments for risk analysis.

    Higher moments provide insight into the shape of return distributions
    beyond mean and variance.

    Parameters
    ----------
    returns : pl.Expr | str
        Return series
    window : int, default 252
        Rolling window size

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary containing:
        - skewness: Third moment (asymmetry)
        - kurtosis: Fourth moment (fat tails)
        - hyperskewness: Fifth moment
        - hyperkurtosis: Sixth moment

    Notes
    -----
    - Skewness < 0: Left tail is longer (crash risk)
    - Kurtosis > 3: Fatter tails than normal distribution
    - Higher moments help identify non-normal return distributions
    """
    validate_window(window, min_window=4)

    returns = pl.col(returns) if isinstance(returns, str) else returns

    # Standardized returns for moment calculation
    mean = returns.rolling_mean(window, min_samples=window // 2)
    std = returns.rolling_std(window, min_samples=window // 2)
    standardized = (returns - mean) / (std + 1e-10)

    return {
        "skewness": returns.rolling_skew(window),
        "kurtosis": returns.rolling_kurtosis(window, fisher=True),  # Excess kurtosis
        "hyperskewness": (standardized**5).rolling_mean(
            window,
            min_samples=window // 2,
        ),
        "hyperkurtosis": (standardized**6).rolling_mean(window, min_samples=window // 2)
        - 15,  # Excess
    }


@feature(
    name="risk_adjusted_returns",
    category="risk",
    description="Risk-Adjusted Return Metrics - Sharpe, Sortino, Calmar ratios",
    lookback="window",
    normalized=False,  # Returns dict with multiple ratios
    formula="Sharpe = (mean_return - rf) / std_return",
    input_type="returns",
    parameters={"window": 252},
    tags=["risk", "sharpe", "sortino", "risk-adjusted"],
)
def risk_adjusted_returns(
    returns: pl.Expr | str,
    risk_free_rate: float = 0.0,
    window: int = 252,
    close: pl.Expr | str | None = None,
) -> dict[str, pl.Expr]:
    """Calculate various risk-adjusted return metrics.

    Risk-adjusted returns help compare investments with different risk profiles.

    Parameters
    ----------
    returns : pl.Expr | str
        Return series
    risk_free_rate : float, default 0.0
        Risk-free rate (annualized)
    window : int, default 252
        Rolling window size
    close : pl.Expr | str, optional
        Price series for more accurate Calmar ratio calculation.
        If None, close are approximated from returns using cumulative product.

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary containing:
        - sharpe_ratio: Excess return per unit of total risk
        - sortino_ratio: Excess return per unit of downside risk
        - calmar_ratio: Return per unit of max drawdown risk
        - omega_ratio: Probability-weighted ratio of gains vs losses

    Notes
    -----
    All ratios are annualized assuming 252 trading days.

    The Calmar ratio calculation is more accurate when actual price data is provided
    via the `close` parameter, as it avoids potential numerical precision issues
    from approximating close using cumulative returns.

    Examples
    --------
    >>> # Using only returns (backward compatible)
    >>> metrics = risk_adjusted_returns("returns")

    >>> # Using actual close for better accuracy
    >>> metrics = risk_adjusted_returns("returns", close="close")
    """
    validate_window(window, min_window=20)

    returns = pl.col(returns) if isinstance(returns, str) else returns

    # Convert risk-free rate to period rate
    period_rf = risk_free_rate / 252

    # Calculate components
    mean_return = returns.rolling_mean(window, min_samples=window // 2)
    excess_return = mean_return - period_rf
    volatility = returns.rolling_std(window, min_samples=window // 2)
    downside_dev = downside_deviation(returns, period_rf, window)

    # For Calmar ratio, we need close
    if close is not None:
        # Use actual price series for more accurate calculation
        price_series = pl.col(close) if isinstance(close, str) else close
        dd_stats = maximum_drawdown(price_series, window)
    else:
        # Fall back to approximation using cumulative returns
        # Note: This starts from 1.0 and may have precision issues over time
        cum_returns = (1 + returns).cum_prod()
        dd_stats = maximum_drawdown(cum_returns, window)

    max_dd = dd_stats["max_drawdown"].abs() + 1e-10

    # Omega ratio calculation
    def calculate_omega(close: pl.Series) -> float:
        """Calculate Omega ratio for a window."""
        arr = close.to_numpy()
        arr = arr[~np.isnan(arr)]

        if len(arr) < window // 2:
            return float(np.nan)

        gains = arr[arr > period_rf] - period_rf
        losses = period_rf - arr[arr <= period_rf]

        if len(losses) > 0 and np.sum(losses) > 0:
            return float(np.sum(gains) / np.sum(losses))
        return float(np.inf) if len(gains) > 0 else 1.0

    omega = returns.rolling_map(
        calculate_omega,
        window_size=window,
        min_samples=window // 2,
    )

    return {
        "sharpe_ratio": (excess_return / (volatility + 1e-10)) * np.sqrt(252),
        "sortino_ratio": (excess_return / (downside_dev + 1e-10)) * np.sqrt(252),
        "calmar_ratio": (mean_return * 252) / max_dd,
        "omega_ratio": omega,
    }


@feature(
    name="ulcer_index",
    category="risk",
    description="Ulcer Index - drawdown volatility measure",
    lookback="window",
    normalized=False,
    formula="UI = sqrt(mean(drawdown^2))",
    input_type="close",
    parameters={"window": 14},
    tags=["risk", "drawdown", "volatility"],
)
def ulcer_index(
    close: pl.Expr | str,
    window: int = 252,
) -> pl.Expr:
    """Calculate Ulcer Index measuring downside volatility.

    The Ulcer Index measures the depth and duration of drawdowns,
    providing a volatility measure focused on downside risk.

    Parameters
    ----------
    close : pl.Expr | str
        Price series (not returns)
    window : int, default 252
        Rolling window size

    Returns
    -------
    pl.Expr
        Ulcer Index (lower is better)

    Notes
    -----
    UI = sqrt(mean(drawdown²))
    Developed by Peter Martin and Byron McCann.
    """
    validate_window(window, min_window=2)

    close = pl.col(close) if isinstance(close, str) else close

    # Calculate rolling maximum
    rolling_max = close.rolling_max(window, min_samples=window // 2)

    # Calculate percentage drawdown
    drawdown_pct = ((close - rolling_max) / rolling_max) * 100

    # Ulcer Index = RMS of drawdowns
    return (drawdown_pct**2).rolling_mean(window, min_samples=window // 2).sqrt()


def information_ratio(
    returns: pl.Expr | str,
    benchmark_returns: pl.Expr | str,
    window: int = 252,
) -> pl.Expr:
    """Calculate Information Ratio (active return per unit of tracking error).

    Information Ratio measures the consistency of active returns relative
    to a benchmark.

    Parameters
    ----------
    returns : pl.Expr | str
        Portfolio return series
    benchmark_returns : pl.Expr | str
        Benchmark return series
    window : int, default 252
        Rolling window size

    Returns
    -------
    pl.Expr
        Information Ratio (annualized)

    Notes
    -----
    IR = (R_p - R_b) / TE
    where TE is tracking error (std of active returns)
    """
    validate_window(window, min_window=20)

    returns = pl.col(returns) if isinstance(returns, str) else returns
    benchmark_returns = (
        pl.col(benchmark_returns) if isinstance(benchmark_returns, str) else benchmark_returns
    )

    # Calculate active returns
    active_returns = returns - benchmark_returns

    # Calculate components
    mean_active = active_returns.rolling_mean(window, min_samples=window // 2)
    tracking_error = active_returns.rolling_std(window, min_samples=window // 2)

    # Annualized IR
    result: pl.Expr = (mean_active / (tracking_error + 1e-10)) * np.sqrt(252)
    return result.cast(pl.Float64)


__all__ = [
    "conditional_value_at_risk",
    "downside_deviation",
    "higher_moments",
    "information_ratio",
    "maximum_drawdown",
    "risk_adjusted_returns",
    "tail_ratio",
    "ulcer_index",
    "value_at_risk",
]
