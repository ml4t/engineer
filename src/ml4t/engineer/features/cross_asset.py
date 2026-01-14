"""Cross-asset relationship features for multi-asset ML models.

Exports:
    rolling_correlation(series1, series2, window=20) -> Expr
        Rolling correlation between two series.

    beta_to_market(returns, market_returns, window=60) -> Expr
        Rolling beta to market index.

    correlation_regime_indicator(series1, series2, ...) -> Expr
        High/low correlation regime detection.

    lead_lag_correlation(series1, series2, max_lag=5, window=60) -> Expr
        Cross-correlation at various lags.

    multi_asset_dispersion(returns_list, window=20) -> Expr
        Cross-sectional return dispersion.

    correlation_matrix_features(returns_df, window=60) -> DataFrame
        Eigenvalue-based correlation features.

    relative_strength_index_spread(series1, series2, period=14) -> Expr
        RSI spread between assets.

    volatility_ratio(series1, series2, window=20) -> Expr
        Relative volatility between assets.

    transfer_entropy(series1, series2, window=20, lag=1, bins=5) -> Expr
        Information flow between time series.

    co_integration_score(series1, series2, window=252) -> Expr
        Rolling cointegration test score.

    cross_asset_momentum(returns_df, lookback=20, n_top=3) -> DataFrame
        Cross-sectional momentum features.

This module provides features that capture relationships between different assets,
useful for portfolio models and cross-market signal detection.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.validation import (
    validate_lag,
    validate_list_length,
    validate_threshold,
    validate_window,
)


def rolling_correlation(
    series1: pl.Expr | str,
    series2: pl.Expr | str,
    window: int = 20,
    min_periods: int | None = None,
) -> pl.Expr:
    """Calculate rolling correlation between two series.

    Measures the linear relationship between two assets over time.

    Parameters
    ----------
    series1, series2 : pl.Expr | str
        Two series to correlate
    window : int, default 20
        Rolling window size
    min_periods : int, optional
        Minimum periods required (default: window // 2)

    Returns
    -------
    pl.Expr
        Rolling correlation coefficient (-1 to 1)

    Raises
    ------
    ValueError
        If window is not positive
    TypeError
        If window is not an integer
    """
    # Validate inputs
    validate_window(window, min_window=2)
    if min_periods is not None:
        validate_window(min_periods, min_window=1, name="min_periods")
        if min_periods > window:
            raise ValueError(
                f"min_periods ({min_periods}) cannot exceed window ({window})",
            )

    series1 = pl.col(series1) if isinstance(series1, str) else series1
    series2 = pl.col(series2) if isinstance(series2, str) else series2

    if min_periods is None:
        min_periods = window // 2

    # Standardize series
    mean1 = series1.rolling_mean(window, min_samples=min_periods)
    mean2 = series2.rolling_mean(window, min_samples=min_periods)
    std1 = series1.rolling_std(window, min_samples=min_periods)
    std2 = series2.rolling_std(window, min_samples=min_periods)

    # Calculate covariance
    cov = ((series1 - mean1) * (series2 - mean2)).rolling_mean(
        window,
        min_samples=min_periods,
    )

    # Correlation = covariance / (std1 * std2)
    corr = cov / (std1 * std2 + 1e-10)

    return corr.clip(-1, 1)  # Ensure valid correlation range


def beta_to_market(
    asset_returns: pl.Expr | str,
    market_returns: pl.Expr | str,
    window: int = 60,
    min_periods: int | None = None,
) -> pl.Expr:
    """Calculate rolling beta of asset to market.

    Beta measures systematic risk - how much the asset moves relative
    to the market. Beta > 1 means more volatile than market.

    Parameters
    ----------
    asset_returns : pl.Expr | str
        Asset returns column
    market_returns : pl.Expr | str
        Market returns column
    window : int, default 60
        Rolling window size
    min_periods : int, optional
        Minimum periods required

    Returns
    -------
    pl.Expr
        Rolling beta close

    Raises
    ------
    ValueError
        If window is not positive
    TypeError
        If window is not an integer
    """
    # Validate inputs
    validate_window(window, min_window=2)
    if min_periods is not None:
        validate_window(min_periods, min_window=1, name="min_periods")
        if min_periods > window:
            raise ValueError(
                f"min_periods ({min_periods}) cannot exceed window ({window})",
            )

    asset_returns = pl.col(asset_returns) if isinstance(asset_returns, str) else asset_returns
    market_returns = pl.col(market_returns) if isinstance(market_returns, str) else market_returns

    if min_periods is None:
        min_periods = window // 2

    # Calculate covariance and market variance
    mean_asset = asset_returns.rolling_mean(window, min_samples=min_periods)
    mean_market = market_returns.rolling_mean(window, min_samples=min_periods)

    covariance = ((asset_returns - mean_asset) * (market_returns - mean_market)).rolling_mean(
        window,
        min_samples=min_periods,
    )
    market_variance = market_returns.rolling_var(window, min_samples=min_periods)

    # Beta = Cov(asset, market) / Var(market)
    beta = covariance / (market_variance + 1e-10)

    return beta


def correlation_regime_indicator(
    correlation: pl.Expr | str,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7,
    lookback: int = 20,
) -> dict[str, pl.Expr]:
    """Identify correlation regimes.

    Classifies correlation into regimes useful for portfolio allocation
    and risk management.

    Parameters
    ----------
    correlation : pl.Expr | str
        Correlation series
    low_threshold : float, default 0.3
        Threshold for low correlation
    high_threshold : float, default 0.7
        Threshold for high correlation
    lookback : int, default 20
        Lookback for regime persistence

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary with regime indicators

    Raises
    ------
    ValueError
        If thresholds are invalid or lookback is not positive
    """
    # Validate inputs
    validate_threshold(low_threshold, 0.0, 1.0, "low_threshold")
    validate_threshold(high_threshold, 0.0, 1.0, "high_threshold")
    validate_window(lookback, min_window=1, name="lookback")

    if low_threshold >= high_threshold:
        raise ValueError(
            f"low_threshold ({low_threshold}) must be less than high_threshold ({high_threshold})",
        )

    correlation = pl.col(correlation) if isinstance(correlation, str) else correlation

    # Classify correlation regimes
    low_corr = (correlation.abs() < low_threshold).cast(pl.Int32)
    mid_corr = ((correlation.abs() >= low_threshold) & (correlation.abs() < high_threshold)).cast(
        pl.Int32,
    )
    high_corr = (correlation.abs() >= high_threshold).cast(pl.Int32)

    # Calculate regime persistence
    return {
        "corr_regime_low": low_corr.rolling_mean(lookback),
        "corr_regime_mid": mid_corr.rolling_mean(lookback),
        "corr_regime_high": high_corr.rolling_mean(lookback),
        "corr_trend": correlation.diff(lookback),  # Correlation trend
        "corr_stability": correlation.rolling_std(lookback),  # Correlation stability
    }


def lead_lag_correlation(
    series1: pl.Expr | str,
    series2: pl.Expr | str,
    max_lag: int = 10,
    window: int = 20,
) -> dict[str, pl.Expr]:
    """Calculate lead-lag correlations between series.

    Identifies which series leads or lags the other, useful for
    predictive relationships.

    Parameters
    ----------
    series1, series2 : pl.Expr | str
        Series to analyze
    max_lag : int, default 10
        Maximum lag to test
    window : int, default 20
        Window for correlation calculation

    Returns
    -------
    dict[str, pl.Expr]
        Correlations at different lags

    Raises
    ------
    ValueError
        If max_lag or window are not positive
    """
    # Validate inputs
    validate_lag(max_lag, name="max_lag")
    validate_window(window, min_window=2)

    series1 = pl.col(series1) if isinstance(series1, str) else series1
    series2 = pl.col(series2) if isinstance(series2, str) else series2

    correlations = {}

    # Test different lags
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # series1 leads series2
            shifted_series1 = series1.shift(-lag)
            corr = rolling_correlation(shifted_series1, series2, window)
            correlations[f"lead_{-lag}"] = corr
        elif lag > 0:
            # series2 leads series1
            shifted_series2 = series2.shift(lag)
            corr = rolling_correlation(series1, shifted_series2, window)
            correlations[f"lag_{lag}"] = corr
        else:
            # Contemporaneous
            correlations["lag_0"] = rolling_correlation(series1, series2, window)

    return correlations


def multi_asset_dispersion(
    returns_list: list[pl.Expr | str],
    window: int = 20,
    method: str = "std",
) -> pl.Expr:
    """Calculate dispersion across multiple assets.

    Measures how much individual assets deviate from the average,
    useful for identifying market stress or opportunities.

    Parameters
    ----------
    returns_list : List[pl.Expr | str]
        List of return series
    window : int, default 20
        Rolling window
    method : str, default "std"
        Dispersion method: "std", "mad" (mean absolute deviation)

    Returns
    -------
    pl.Expr
        Cross-sectional dispersion

    Raises
    ------
    ValueError
        If returns_list is empty, window is invalid, or method is unknown
    """
    # Validate inputs
    validate_list_length(returns_list, min_length=2, name="returns_list")
    validate_window(window, min_window=1)
    if method not in ["std", "mad"]:
        raise ValueError(f"Unknown method: {method}. Supported methods: ['std', 'mad']")

    # Convert strings to expressions
    returns_list = [pl.col(r) if isinstance(r, str) else r for r in returns_list]

    # Stack returns horizontally
    returns_matrix = pl.concat_list(returns_list)

    if method == "std":
        # Cross-sectional standard deviation
        dispersion = returns_matrix.list.eval(pl.element().std()).list.first()
    elif method == "mad":
        # Mean absolute deviation
        dispersion = returns_matrix.list.eval(
            (pl.element() - pl.element().mean()).abs().mean(),
        ).list.first()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply rolling window
    return dispersion.rolling_mean(window)


def correlation_matrix_features(
    returns_list: list[pl.Expr | str],
    window: int = 20,
    min_periods: int | None = None,
) -> dict[str, pl.Expr]:
    """Extract features from correlation matrix.

    Calculates summary statistics from the correlation matrix
    that capture market-wide relationships.

    Parameters
    ----------
    returns_list : List[pl.Expr | str]
        List of return series for different assets
    window : int, default 20
        Rolling window
    min_periods : int, optional
        Minimum periods required

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary of correlation matrix features
    """
    if min_periods is None:
        min_periods = window // 2

    # Convert strings to expressions
    returns_list = [pl.col(r) if isinstance(r, str) else r for r in returns_list]
    n_assets = len(returns_list)

    # Calculate all pairwise correlations
    correlations = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr = rolling_correlation(
                returns_list[i],
                returns_list[j],
                window,
                min_periods,
            )
            correlations.append(corr)

    # Stack correlations
    corr_matrix = pl.concat_list(correlations)

    # Extract features
    features = {
        # Average correlation (market integration)
        "avg_correlation": corr_matrix.list.eval(pl.element().mean()).list.first(),
        # Maximum correlation (highest dependency)
        "max_correlation": corr_matrix.list.eval(pl.element().max()).list.first(),
        # Minimum correlation (best diversification)
        "min_correlation": corr_matrix.list.eval(pl.element().min()).list.first(),
        # Correlation dispersion (heterogeneity)
        "corr_dispersion": corr_matrix.list.eval(pl.element().std()).list.first(),
    }

    return features


def relative_strength_index_spread(
    rsi1: pl.Expr | str,
    rsi2: pl.Expr | str,
    smooth_period: int = 5,
) -> pl.Expr:
    """Calculate RSI spread between two assets.

    Identifies relative momentum differences, useful for
    pair trading or relative value strategies.

    Parameters
    ----------
    rsi1, rsi2 : pl.Expr | str
        RSI close for two assets
    smooth_period : int, default 5
        Smoothing period for spread

    Returns
    -------
    pl.Expr
        Smoothed RSI spread
    """
    rsi1 = pl.col(rsi1) if isinstance(rsi1, str) else rsi1
    rsi2 = pl.col(rsi2) if isinstance(rsi2, str) else rsi2

    # Calculate spread
    spread = rsi1 - rsi2

    # Smooth to reduce noise
    return spread.rolling_mean(smooth_period)


def volatility_ratio(
    vol1: pl.Expr | str,
    vol2: pl.Expr | str,
    log_ratio: bool = True,
) -> pl.Expr:
    """Calculate volatility ratio between two assets.

    Measures relative volatility, useful for volatility arbitrage
    or risk allocation.

    Parameters
    ----------
    vol1, vol2 : pl.Expr | str
        Volatility series for two assets
    log_ratio : bool, default True
        Whether to return log ratio (more stable)

    Returns
    -------
    pl.Expr
        Volatility ratio
    """
    vol1 = pl.col(vol1) if isinstance(vol1, str) else vol1
    vol2 = pl.col(vol2) if isinstance(vol2, str) else vol2

    if log_ratio:
        # Log ratio is more stable and symmetric
        return (vol1 / (vol2 + 1e-10)).log()
    return vol1 / (vol2 + 1e-10)


@jit(nopython=True, cache=True)  # type: ignore[misc]
def transfer_entropy_nb(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    lag: int = 1,
    bins: int = 10,
) -> float:
    """Calculate transfer entropy from X to Y (Numba optimized).

    Measures information flow from X to Y.
    """
    n = len(x) - lag
    if n < 10:  # Need minimum data
        return float(np.nan)

    # Discretize data
    x_min: float = float(np.min(x))
    x_max: float = float(np.max(x))
    y_min: float = float(np.min(y))
    y_max: float = float(np.max(y))

    if x_min == x_max or y_min == y_max:
        return 0.0

    # Create bins
    x_bins = np.linspace(x_min, x_max, bins + 1)
    y_bins = np.linspace(y_min, y_max, bins + 1)

    # Discretize
    x_disc = np.searchsorted(x_bins[1:-1], x[:-lag])
    y_disc = np.searchsorted(y_bins[1:-1], y[:-lag])
    y_future_disc = np.searchsorted(y_bins[1:-1], y[lag:])

    # Calculate joint and marginal probabilities
    # This is a simplified implementation
    te = 0.0
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                # Count occurrences
                count_ijk = 0
                count_jk = 0
                count_j = 0

                for idx in range(n):
                    if int(y_disc[idx]) == j:
                        count_j += 1
                        if int(y_future_disc[idx]) == k:
                            count_jk += 1
                            if int(x_disc[idx]) == i:
                                count_ijk += 1

                # Calculate probabilities
                if count_ijk > 0 and count_jk > 0 and count_j > 0:
                    p_ijk = count_ijk / n
                    p_k_given_ij = count_ijk / count_j
                    p_k_given_j = count_jk / count_j

                    if p_k_given_ij > 0 and p_k_given_j > 0:
                        te += p_ijk * np.log2(p_k_given_ij / p_k_given_j)

    return te


def transfer_entropy(
    series1: pl.Expr | str,
    series2: pl.Expr | str,
    lag: int = 1,
    window: int = 100,
    bins: int = 10,
) -> dict[str, pl.Expr]:
    """Calculate transfer entropy between series.

    This function is currently not implemented due to technical complexity
    with Polars rolling operations on custom functions. The previous
    implementation was returning incorrect zero close.

    The transfer entropy calculation requires complex rolling operations
    that are not efficiently supported in the current Polars API.

    A working Numba implementation exists (transfer_entropy_nb) for
    single-calculation use cases.

    Parameters
    ----------
    series1, series2 : pl.Expr | str
        Series to analyze
    lag : int, default 1
        Time lag for causality detection
    window : int, default 100
        Rolling window size
    bins : int, default 10
        Number of bins for discretization

    Returns
    -------
    dict[str, pl.Expr]
        Would return transfer entropy in both directions

    Raises
    ------
    NotImplementedError
        This function is not yet implemented for production use
    """
    raise NotImplementedError(
        "Transfer entropy calculation is not yet implemented for Polars expressions. "
        "This feature was returning incorrect zero close and has been disabled. "
        "For single calculations, use transfer_entropy_nb() directly on NumPy arrays. "
        "See: https://github.com/ml4t/ml4t-features/issues/transfer-entropy-implementation"
    )


def co_integration_score(
    price1: pl.Expr | str,
    price2: pl.Expr | str,
    window: int = 60,
) -> pl.Expr:
    """Calculate simple co-integration score.

    A simplified measure of how well two price series move together
    in the long term, useful for pair trading.

    Parameters
    ----------
    price1, price2 : pl.Expr | str
        Price series (not returns)
    window : int, default 60
        Rolling window

    Returns
    -------
    pl.Expr
        Co-integration score (lower = more co-integrated)
    """
    price1 = pl.col(price1) if isinstance(price1, str) else price1
    price2 = pl.col(price2) if isinstance(price2, str) else price2

    # Calculate spread
    # Simple approach: normalized price difference
    norm_price1 = price1 / price1.shift(window)
    norm_price2 = price2 / price2.shift(window)

    spread = norm_price1 - norm_price2

    # Co-integration score = standard deviation of spread
    # Lower close indicate better co-integration
    return spread.rolling_std(window)


def cross_asset_momentum(
    returns_list: list[pl.Expr | str],
    lookback: int = 20,
    method: str = "rank",
) -> dict[str, pl.Expr]:
    """Calculate cross-asset momentum features.

    Identifies which assets are outperforming/underperforming
    on a relative basis.

    Parameters
    ----------
    returns_list : List[pl.Expr | str]
        List of return series
    lookback : int, default 20
        Lookback period for momentum
    method : str, default "rank"
        Method: "rank" or "zscore"

    Returns
    -------
    dict[str, pl.Expr]
        Cross-asset momentum features
    """
    # Convert strings to expressions
    returns_list = [pl.col(r) if isinstance(r, str) else r for r in returns_list]

    # Calculate cumulative returns over lookback
    cum_returns = []
    for r in returns_list:
        if isinstance(r, str):
            cum_returns.append(pl.col(r).rolling_sum(lookback))
        else:
            cum_returns.append(r.rolling_sum(lookback))

    # Stack returns
    returns_matrix = pl.concat_list(cum_returns)

    features = {}

    if method == "rank":
        # Percentile rank across assets
        features["momentum_rank"] = returns_matrix.list.eval(
            pl.element().rank() / pl.element().len(),
        ).list.first()

    elif method == "zscore":
        # Z-score across assets
        features["momentum_zscore"] = returns_matrix.list.eval(
            (pl.element() - pl.element().mean()) / (pl.element().std() + 1e-10),
        ).list.first()

    # Momentum dispersion
    features["momentum_dispersion"] = returns_matrix.list.eval(
        pl.element().std(),
    ).list.first()

    return features
