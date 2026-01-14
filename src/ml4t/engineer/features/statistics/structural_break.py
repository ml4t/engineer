# mypy: disable-error-code="misc,no-any-return"
"""
Structural Break Detection Features.

Exports:
    coefficient_of_variation(data, window=20) -> Expr
        Rolling CV (std/mean). Key "gate" feature for regime detection.

    variance_ratio(data, window=20, holding_period=5) -> Expr
        Lo-MacKinlay variance ratio test for random walk.

    rolling_kl_divergence(data, ref_window=252, test_window=20) -> Expr
        KL divergence between reference and test distributions.

    rolling_wasserstein(data, ref_window=252, test_window=20) -> Expr
        Wasserstein distance for distribution shift detection.

    rolling_drift(data, ref_window=252, test_window=20) -> Expr
        Combined drift score from multiple metrics.

    rolling_cv_zscore(data, window=20, lookback=252) -> Expr
        Z-score of CV relative to historical distribution.

Implements statistical features for detecting regime changes and structural breaks
in financial time series. Based on 2025 ADIA Lab Structural Break Challenge insights
and modern machine learning approaches.

Key Insights from ADIA Lab 2025:
- Winning teams used feature engineering + tree ensembles (XGBoost, RF)
- Coefficient of Variation (CV) was a key "gate" feature between regimes
- Distributional divergence metrics outperformed traditional tests
- Multi-scale features capture different types of structural changes
- "Clarity beats complexity" - compact feature sets + disciplined validation

References:
    ADIA Lab Structural Break Challenge (2025) - Competition results
    Lo & MacKinlay (1988). Stock Market Prices Do Not Follow Random Walks
    Wasserstein Distance for Distribution Comparison
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import validate_window
from ml4t.engineer.logging import logged_feature

__all__ = [
    "coefficient_of_variation",
    "variance_ratio",
    "rolling_kl_divergence",
    "rolling_wasserstein",
    "rolling_drift",
    "rolling_cv_zscore",
]


# =============================================================================
# Coefficient of Variation (CV)
# =============================================================================


@jit(nopython=True, cache=True)
def _cv_nb(values: npt.NDArray[np.float64]) -> float:
    """Coefficient of Variation: std/mean (normalized dispersion)."""
    valid = values[~np.isnan(values)]
    if len(valid) < 2:
        return np.nan

    mean = np.mean(valid)
    if abs(mean) < 1e-10:
        return np.nan

    std = np.std(valid)
    return std / abs(mean)


@logged_feature("coefficient_of_variation", warn_threshold_ms=100.0, log_data_quality=True)
@feature(
    name="coefficient_of_variation",
    category="statistics",
    description="Rolling Coefficient of Variation - key regime detection metric",
    lookback="window",
    normalized=True,
    value_range=(0.0, 10.0),
    formula="CV = std(x) / |mean(x)|",
    ta_lib_compatible=False,
)
def coefficient_of_variation(
    feature: pl.Expr | str,
    window: int = 50,
) -> pl.Expr:
    """Rolling Coefficient of Variation (CV).

    CV measures relative variability - the ratio of standard deviation
    to absolute mean. Critical feature for structural break detection
    as identified in ADIA Lab 2025 competition.

    High CV indicates high relative volatility / regime uncertainty.
    Low CV indicates stable regime with consistent mean.

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to calculate CV for (typically returns or prices)
    window : int, default 50
        Rolling window size

    Returns
    -------
    pl.Expr
        Rolling CV values (dimensionless ratio)

    Notes
    -----
    CV is particularly useful because:
    - Scale-invariant: works across different price levels
    - Regime-sensitive: spikes during transitions
    - Acts as "gate" between high/low volatility regimes

    References
    ----------
    .. [1] ADIA Lab Structural Break Challenge (2025)
    """
    validate_window(window, min_window=2, name="window")
    feature_expr = pl.col(feature) if isinstance(feature, str) else feature

    return feature_expr.rolling_map(
        lambda x: _cv_nb(x.to_numpy().astype(np.float64)),
        window_size=window,
        weights=None,
        min_samples=window // 2,
        center=False,
    )


# =============================================================================
# CV Z-Score (Normalized CV for comparison across assets)
# =============================================================================


@jit(nopython=True, cache=True)
def _cv_zscore_nb(
    values: npt.NDArray[np.float64],
    lookback: int,
) -> float:
    """CV Z-score: how many standard deviations current CV is from historical mean."""
    n = len(values)
    if n < lookback + 2:
        return np.nan

    # Calculate current CV
    current_window = values[n - lookback :]
    current_cv = _cv_nb(current_window)

    if np.isnan(current_cv):
        return np.nan

    # Calculate historical CV distribution
    cv_values = np.empty(n - lookback + 1, dtype=np.float64)
    for i in range(n - lookback + 1):
        cv_values[i] = _cv_nb(values[i : i + lookback])

    valid_cvs = cv_values[~np.isnan(cv_values)]
    if len(valid_cvs) < 10:
        return np.nan

    cv_mean = np.mean(valid_cvs)
    cv_std = np.std(valid_cvs)

    if cv_std < 1e-10:
        return 0.0

    return (current_cv - cv_mean) / cv_std


@logged_feature("rolling_cv_zscore", warn_threshold_ms=200.0, log_data_quality=True)
@feature(
    name="rolling_cv_zscore",
    category="statistics",
    description="CV Z-score - normalized CV relative to historical distribution",
    lookback="window",
    normalized=True,
    value_range=(-10.0, 10.0),
    formula="z = (CV - mean(CV_hist)) / std(CV_hist)",
    ta_lib_compatible=False,
)
def rolling_cv_zscore(
    feature: pl.Expr | str,
    window: int = 50,
    lookback_multiplier: int = 5,
) -> pl.Expr:
    """Rolling CV Z-score for normalized regime detection.

    Z-score normalizes CV relative to its historical distribution,
    making it comparable across different assets and time periods.

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to calculate CV Z-score for
    window : int, default 50
        Window for CV calculation
    lookback_multiplier : int, default 5
        Total lookback is window * lookback_multiplier

    Returns
    -------
    pl.Expr
        CV Z-score (typically -3 to +3, extremes indicate regime change)

    Notes
    -----
    Z-score > 2: Unusually high volatility (potential regime change)
    Z-score < -2: Unusually low volatility (potential compression)
    """
    validate_window(window, min_window=2, name="window")
    feature_expr = pl.col(feature) if isinstance(feature, str) else feature
    total_lookback = window * lookback_multiplier

    return feature_expr.rolling_map(
        lambda x: _cv_zscore_nb(x.to_numpy().astype(np.float64), window),
        window_size=total_lookback,
        weights=None,
        min_samples=total_lookback // 2,
        center=False,
    )


# =============================================================================
# Variance Ratio Test
# =============================================================================


@jit(nopython=True, cache=True)
def _variance_ratio_nb(
    values: npt.NDArray[np.float64],
    q: int,
) -> float:
    """Variance Ratio: Var(q-period returns) / (q * Var(1-period returns)).

    Under random walk hypothesis, VR should equal 1.
    VR > 1 suggests momentum / trending
    VR < 1 suggests mean reversion
    """
    valid = values[~np.isnan(values)]
    n = len(valid)

    if n < q + 2:
        return np.nan

    # 1-period returns (differences for general features, or assume returns input)
    returns_1 = valid[1:] - valid[:-1]

    # q-period returns
    returns_q = valid[q:] - valid[:-q]

    var_1 = np.var(returns_1)
    var_q = np.var(returns_q)

    if var_1 < 1e-15:
        return np.nan

    return var_q / (q * var_1)


@logged_feature("variance_ratio", warn_threshold_ms=100.0, log_data_quality=True)
@feature(
    name="variance_ratio",
    category="statistics",
    description="Variance Ratio test statistic - random walk vs mean reversion",
    lookback="window",
    normalized=True,
    value_range=(0.0, 5.0),
    formula="VR(q) = Var(r_q) / (q * Var(r_1))",
    ta_lib_compatible=False,
)
def variance_ratio(
    feature: pl.Expr | str,
    window: int = 100,
    q: int = 5,
) -> pl.Expr:
    """Rolling Variance Ratio test statistic.

    Tests whether variance of q-period changes equals q times variance
    of 1-period changes (random walk hypothesis).

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to test (prices or cumulative returns)
    window : int, default 100
        Rolling window size
    q : int, default 5
        Aggregation period for comparison

    Returns
    -------
    pl.Expr
        Variance ratio (1 = random walk, >1 = trending, <1 = mean-reverting)

    Notes
    -----
    Useful for detecting:
    - Regime changes (VR shifts from >1 to <1 or vice versa)
    - Market efficiency changes
    - Structural breaks in return autocorrelation

    References
    ----------
    .. [1] Lo & MacKinlay (1988). Stock Market Prices Do Not Follow Random Walks
    """
    validate_window(window, min_window=q * 2, name="window")
    validate_window(q, min_window=2, name="q")
    feature_expr = pl.col(feature) if isinstance(feature, str) else feature

    # Cast to Float64 to ensure consistent return type from rolling_map
    return feature_expr.cast(pl.Float64).rolling_map(
        lambda x: _variance_ratio_nb(x.to_numpy().astype(np.float64), q),
        window_size=window,
        weights=None,
        min_samples=window // 2,
        center=False,
    )


# =============================================================================
# KL Divergence (Distributional Divergence)
# =============================================================================


@jit(nopython=True, cache=True)
def _kl_divergence_nb(
    values: npt.NDArray[np.float64],
    _split_point: int,  # Reserved for future use (custom split points)
    n_bins: int,
) -> float:
    """KL Divergence between first half and second half distributions.

    KL(P||Q) = sum(P(x) * log(P(x) / Q(x)))

    Where P is the first half distribution and Q is the second half.
    """
    valid = values[~np.isnan(values)]
    n = len(valid)

    if n < 4:
        return np.nan

    # Split into two halves
    half = n // 2
    first_half = valid[:half]
    second_half = valid[half:]

    if len(first_half) < 2 or len(second_half) < 2:
        return np.nan

    # Create histogram bins based on full range
    all_min = min(np.min(first_half), np.min(second_half))
    all_max = max(np.max(first_half), np.max(second_half))

    if all_max - all_min < 1e-10:
        return 0.0  # Identical distributions

    bins = np.linspace(all_min, all_max, n_bins + 1)

    # Compute histograms (probability distributions)
    hist_p, _ = np.histogram(first_half, bins)
    hist_q, _ = np.histogram(second_half, bins)

    # Convert to probabilities with small epsilon to avoid log(0)
    eps = 1e-10
    p = hist_p / len(first_half) + eps
    q = hist_q / len(second_half) + eps

    # Normalize to sum to 1 after adding epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)

    # KL divergence
    kl = float(np.sum(p * np.log(p / q)))

    return kl


@logged_feature("rolling_kl_divergence", warn_threshold_ms=200.0, log_data_quality=True)
@feature(
    name="rolling_kl_divergence",
    category="statistics",
    description="Rolling KL divergence - measures distribution change within window",
    lookback="window",
    normalized=True,
    value_range=(0.0, 10.0),
    formula="KL(P||Q) = sum(P * log(P/Q))",
    ta_lib_compatible=False,
)
def rolling_kl_divergence(
    feature: pl.Expr | str,
    window: int = 100,
    n_bins: int = 20,
) -> pl.Expr:
    """Rolling KL Divergence between window halves.

    Measures how much the distribution has changed within the window
    by comparing first half to second half.

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to measure divergence
    window : int, default 100
        Rolling window size (split in half for comparison)
    n_bins : int, default 20
        Number of histogram bins for distribution estimation

    Returns
    -------
    pl.Expr
        KL divergence (0 = identical distributions, higher = more different)

    Notes
    -----
    High KL divergence indicates:
    - Distribution has shifted significantly within window
    - Potential structural break or regime change
    - Non-stationarity in the time series

    References
    ----------
    .. [1] ADIA Lab Structural Break Challenge (2025)
    """
    validate_window(window, min_window=10, name="window")
    validate_window(n_bins, min_window=5, name="n_bins")
    feature_expr = pl.col(feature) if isinstance(feature, str) else feature

    return feature_expr.rolling_map(
        lambda x: _kl_divergence_nb(x.to_numpy().astype(np.float64), len(x) // 2, n_bins),
        window_size=window,
        weights=None,
        min_samples=window // 2,
        center=False,
    )


# =============================================================================
# Wasserstein Distance (Earth Mover's Distance)
# =============================================================================


@jit(nopython=True, cache=True)
def _wasserstein_1d_nb(
    values: npt.NDArray[np.float64],
) -> float:
    """1D Wasserstein distance between first half and second half.

    For 1D distributions, Wasserstein distance is the integral of the
    absolute difference between CDFs.
    """
    valid = values[~np.isnan(values)]
    n = len(valid)

    if n < 4:
        return np.nan

    # Split into two halves
    half = n // 2
    first_half = np.sort(valid[:half])
    second_half = np.sort(valid[half:])

    n1 = len(first_half)
    n2 = len(second_half)

    if n1 < 2 or n2 < 2:
        return np.nan

    # Simple Wasserstein for equal-sized samples: mean of sorted differences
    # For unequal sizes, use interpolation
    if n1 == n2:
        return float(np.mean(np.abs(first_half - second_half)))

    # Interpolate to common size
    common_size = min(n1, n2)
    indices_1 = np.linspace(0, n1 - 1, common_size).astype(np.int64)
    indices_2 = np.linspace(0, n2 - 1, common_size).astype(np.int64)

    vals_1 = np.empty(common_size, dtype=np.float64)
    vals_2 = np.empty(common_size, dtype=np.float64)

    for i in range(common_size):
        vals_1[i] = first_half[indices_1[i]]
        vals_2[i] = second_half[indices_2[i]]

    return float(np.mean(np.abs(vals_1 - vals_2)))


@logged_feature("rolling_wasserstein", warn_threshold_ms=200.0, log_data_quality=True)
@feature(
    name="rolling_wasserstein",
    category="statistics",
    description="Rolling Wasserstein distance - optimal transport between distributions",
    lookback="window",
    normalized=False,  # Scale depends on input scale
    formula="W_1(P, Q) = integral |F_P(x) - F_Q(x)| dx",
    ta_lib_compatible=False,
)
def rolling_wasserstein(
    feature: pl.Expr | str,
    window: int = 100,
) -> pl.Expr:
    """Rolling Wasserstein Distance (Earth Mover's Distance).

    Measures the minimum "work" needed to transform one distribution
    into another. More robust than KL divergence for structural breaks.

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to measure distance
    window : int, default 100
        Rolling window size (split in half for comparison)

    Returns
    -------
    pl.Expr
        Wasserstein distance (same units as input)

    Notes
    -----
    Advantages over KL divergence:
    - Well-defined even when distributions have different support
    - Provides meaningful distance metric (not just divergence)
    - More robust to outliers and sparse bins

    High Wasserstein distance indicates significant distribution shift.

    References
    ----------
    .. [1] ADIA Lab Structural Break Challenge (2025)
    """
    validate_window(window, min_window=10, name="window")
    feature_expr = pl.col(feature) if isinstance(feature, str) else feature

    return feature_expr.rolling_map(
        lambda x: _wasserstein_1d_nb(x.to_numpy().astype(np.float64)),
        window_size=window,
        weights=None,
        min_samples=window // 2,
        center=False,
    )


# =============================================================================
# Drift Detection
# =============================================================================


@jit(nopython=True, cache=True)
def _drift_nb(
    values: npt.NDArray[np.float64],
) -> float:
    """Drift: difference between second half mean and first half mean."""
    valid = values[~np.isnan(values)]
    n = len(valid)

    if n < 4:
        return np.nan

    half = n // 2
    first_mean = np.mean(valid[:half])
    second_mean = np.mean(valid[half:])

    return second_mean - first_mean


@jit(nopython=True, cache=True)
def _drift_zscore_nb(
    values: npt.NDArray[np.float64],
) -> float:
    """Drift Z-score: drift normalized by pooled standard error."""
    valid = values[~np.isnan(values)]
    n = len(valid)

    if n < 4:
        return np.nan

    half = n // 2
    first_half = valid[:half]
    second_half = valid[half:]

    first_mean = np.mean(first_half)
    second_mean = np.mean(second_half)
    drift = second_mean - first_mean

    # Pooled standard error
    first_var = np.var(first_half)
    second_var = np.var(second_half)

    pooled_se = np.sqrt(first_var / len(first_half) + second_var / len(second_half))

    if pooled_se < 1e-15:
        return 0.0 if abs(drift) < 1e-15 else np.sign(drift) * 100.0

    return drift / pooled_se


@logged_feature("rolling_drift", warn_threshold_ms=100.0, log_data_quality=True)
@feature(
    name="rolling_drift",
    category="statistics",
    description="Rolling drift Z-score - mean shift detection within window",
    lookback="window",
    normalized=True,
    value_range=(-10.0, 10.0),
    formula="z = (mean_2 - mean_1) / SE_pooled",
    ta_lib_compatible=False,
)
def rolling_drift(
    feature: pl.Expr | str,
    window: int = 100,
    normalize: bool = True,
) -> pl.Expr:
    """Rolling drift detection between window halves.

    Measures the shift in mean between first half and second half
    of the window, optionally normalized by standard error.

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to detect drift in
    window : int, default 100
        Rolling window size (split in half for comparison)
    normalize : bool, default True
        If True, return Z-score; if False, return raw drift

    Returns
    -------
    pl.Expr
        Drift Z-score (or raw drift if normalize=False)

    Notes
    -----
    |Z| > 2: Significant mean shift (potential structural break)
    Positive: Mean increased (upward drift)
    Negative: Mean decreased (downward drift)

    References
    ----------
    .. [1] ADIA Lab Structural Break Challenge (2025)
    """
    validate_window(window, min_window=10, name="window")
    feature_expr = pl.col(feature) if isinstance(feature, str) else feature

    fn = _drift_zscore_nb if normalize else _drift_nb

    return feature_expr.rolling_map(
        lambda x: fn(x.to_numpy().astype(np.float64)),
        window_size=window,
        weights=None,
        min_samples=window // 2,
        center=False,
    )
