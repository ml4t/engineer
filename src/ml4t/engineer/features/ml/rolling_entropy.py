# mypy: disable-error-code="misc,no-any-return"
"""
Entropy Features for Financial Time Series.

Implements entropy estimators from AFML Chapter 18 for measuring market complexity
and predictability. Updated with 2025 best practices.

Key Concepts:
- Shannon entropy: Information content based on histogram probabilities
- Kontoyiannis (LZ) entropy: Compression-based complexity measure
- Plug-in entropy: Word/pattern frequency-based estimator

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning. Ch. 18.
    Kontoyiannis et al. (1998). "Nonparametric Entropy Estimation"
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
    "encode_binary",
    "encode_quantile",
    "encode_sigma",
    "rolling_entropy",
    "rolling_entropy_lz",
    "rolling_entropy_plugin",
]


# =============================================================================
# Encoding Schemes (AFML Ch. 18.2)
# =============================================================================


@jit(nopython=True, cache=True)
def _encode_binary_nb(
    returns: npt.NDArray[np.float64],
) -> npt.NDArray[np.int32]:
    """Binary encoding: 1 for positive, 0 for negative/zero."""
    n = len(returns)
    result = np.empty(n, dtype=np.int32)
    for i in range(n):
        if np.isnan(returns[i]):
            result[i] = -1  # Use -1 for NaN
        elif returns[i] > 0:
            result[i] = 1
        else:
            result[i] = 0
    return result


@jit(nopython=True, cache=True)
def _encode_quantile_nb(
    values: npt.NDArray[np.float64],
    n_bins: int,
) -> npt.NDArray[np.int32]:
    """Quantile encoding: assign to bins with equal probability mass."""
    n = len(values)
    result = np.empty(n, dtype=np.int32)

    # Get valid values for quantile calculation
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]

    if len(valid_values) == 0:
        result[:] = -1
        return result

    # Calculate quantile boundaries
    quantiles = np.empty(n_bins + 1, dtype=np.float64)
    for i in range(n_bins + 1):
        p = i / n_bins
        quantiles[i] = np.percentile(valid_values, p * 100)

    # Assign bins
    for i in range(n):
        if np.isnan(values[i]):
            result[i] = -1
        else:
            # Find bin
            bin_idx = n_bins - 1  # Default to last bin
            for j in range(n_bins):
                if values[i] <= quantiles[j + 1]:
                    bin_idx = j
                    break
            result[i] = bin_idx

    return result


@jit(nopython=True, cache=True)
def _encode_sigma_nb(
    values: npt.NDArray[np.float64],
    n_bins: int,
) -> npt.NDArray[np.int32]:
    """Sigma encoding: fixed-width bins based on standard deviations.

    Bins are centered at mean, each bin spans (sigma_width) standard deviations.
    Example: n_bins=5, sigma_width=1.0 creates bins at [-2σ, -1σ, 0, +1σ, +2σ]
    """
    n = len(values)
    result = np.empty(n, dtype=np.int32)

    # Get valid values for mean/std calculation
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]

    if len(valid_values) == 0:
        result[:] = -1
        return result

    mean = np.mean(valid_values)
    std = np.std(valid_values)

    if std < 1e-10:
        # All values same, assign to middle bin
        result[:] = n_bins // 2
        for i in range(n):
            if np.isnan(values[i]):
                result[i] = -1
        return result

    # Calculate bin boundaries (symmetric around mean)
    half_bins = n_bins // 2
    sigma_per_bin = 4.0 / n_bins  # Total range of 4 sigma centered at mean

    for i in range(n):
        if np.isnan(values[i]):
            result[i] = -1
        else:
            # Calculate z-score
            z = (values[i] - mean) / std
            # Map to bin (clamp to valid range)
            bin_idx = int((z / sigma_per_bin) + half_bins)
            bin_idx = max(0, min(n_bins - 1, bin_idx))
            result[i] = bin_idx

    return result


def encode_binary(
    feature: pl.Expr | str,
) -> pl.Expr:
    """Binary encoding of returns (sign-based).

    Encodes positive returns as 1, non-positive as 0.
    Useful for direction-based entropy analysis.

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to encode (typically returns)

    Returns
    -------
    pl.Expr
        Binary encoded values (0 or 1)

    References
    ----------
    .. [1] AFML Chapter 18.2 - Encoding Schemes
    """
    feature_expr = pl.col(feature) if isinstance(feature, str) else feature
    return feature_expr.map_batches(
        lambda x: pl.Series(_encode_binary_nb(x.to_numpy().astype(np.float64))),
        return_dtype=pl.Int32,
    )


def encode_quantile(
    feature: pl.Expr | str,
    n_bins: int = 10,
) -> pl.Expr:
    """Quantile encoding - equal probability mass per bin.

    Maps values to bins such that each bin has approximately
    equal number of observations (uniform probability).

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to encode
    n_bins : int, default 10
        Number of quantile bins

    Returns
    -------
    pl.Expr
        Quantile bin indices (0 to n_bins-1)

    References
    ----------
    .. [1] AFML Chapter 18.2 - Encoding Schemes
    """
    validate_window(n_bins, min_window=2, name="n_bins")
    feature_expr = pl.col(feature) if isinstance(feature, str) else feature
    return feature_expr.map_batches(
        lambda x: pl.Series(_encode_quantile_nb(x.to_numpy().astype(np.float64), n_bins)),
        return_dtype=pl.Int32,
    )


def encode_sigma(
    feature: pl.Expr | str,
    n_bins: int = 10,
) -> pl.Expr:
    """Sigma encoding - fixed-width bins based on standard deviations.

    Creates bins of fixed width in terms of standard deviations,
    centered at the mean. Useful when distribution shape matters.

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to encode
    n_bins : int, default 10
        Number of sigma bins

    Returns
    -------
    pl.Expr
        Sigma bin indices (0 to n_bins-1)

    References
    ----------
    .. [1] AFML Chapter 18.2 - Encoding Schemes
    """
    validate_window(n_bins, min_window=2, name="n_bins")
    feature_expr = pl.col(feature) if isinstance(feature, str) else feature
    return feature_expr.map_batches(
        lambda x: pl.Series(_encode_sigma_nb(x.to_numpy().astype(np.float64), n_bins)),
        return_dtype=pl.Int32,
    )


# =============================================================================
# Shannon Entropy (Histogram-based)
# =============================================================================


@jit(nopython=True, cache=True)
def _shannon_entropy_nb(
    values: npt.NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Calculate Shannon entropy of a distribution."""
    if len(values) == 0:
        return np.nan

    # Filter out NaN values
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return np.nan

    # Discretize values
    min_val: float = float(np.min(valid_values))
    max_val: float = float(np.max(valid_values))

    # Check if all values are the same
    if min_val == max_val or (max_val - min_val) < 1e-10:
        return 0.0

    bins = np.linspace(min_val, max_val, n_bins + 1)
    hist, _ = np.histogram(valid_values, bins)

    # Calculate probabilities
    probs = hist / len(valid_values)
    probs = probs[probs > 0]  # Remove zeros

    # Shannon entropy
    entropy = float(-np.sum(probs * np.log2(probs)))

    return entropy


# =============================================================================
# Kontoyiannis (LZ) Entropy Estimator (AFML Ch. 18.3.2)
# =============================================================================


@jit(nopython=True, cache=True)
def _lz_match_length(
    sequence: npt.NDArray[np.int32],
    i: int,
    window_size: int,
) -> int:
    """Find longest match length for position i looking back in window.

    Returns L_i^n + 1 where L_i^n is the length of longest match.
    """
    n = len(sequence)
    if i == 0:
        return 1

    # Look back in the window for matches
    max_match = 0
    start = max(0, i - window_size)

    for j in range(start, i):
        # Count match length starting from j
        match_len = 0
        while (i + match_len < n) and (j + match_len < i):
            if sequence[i + match_len] == sequence[j + match_len]:
                match_len += 1
            else:
                break

        if match_len > max_match:
            max_match = match_len

    return max_match + 1


@jit(nopython=True, cache=True)
def _kontoyiannis_entropy_nb(
    encoded_sequence: npt.NDArray[np.int32],
    window_size: int = 100,
) -> float:
    """Kontoyiannis (Lempel-Ziv) entropy estimator.

    Estimates entropy based on compression - measures how predictable
    the sequence is by looking for repeated patterns.

    Formula: H_hat = [1/k * sum(L_i^n / log2(n))]^(-1)

    where L_i^n is the match length + 1 for position i in window n.

    Parameters
    ----------
    encoded_sequence : array
        Discretized sequence (integer symbols)
    window_size : int
        Size of look-back window for pattern matching

    Returns
    -------
    float
        Estimated entropy in bits

    References
    ----------
    .. [1] Kontoyiannis et al. (1998). Nonparametric Entropy Estimation
    .. [2] AFML Chapter 18.3.2
    """
    # Filter out invalid entries (-1 represents NaN)
    valid_mask = encoded_sequence >= 0
    valid_seq = encoded_sequence[valid_mask]
    k = len(valid_seq)

    if k < 3:
        return np.nan

    # Sum of normalized match lengths
    sum_ratio = 0.0
    log2_n = np.log2(float(k))

    if log2_n < 1e-10:
        return np.nan

    for i in range(1, k):
        match_len = _lz_match_length(valid_seq, i, min(i, window_size))
        sum_ratio += match_len / log2_n

    # Kontoyiannis estimator
    if sum_ratio < 1e-10:
        return np.nan

    entropy = float(k - 1) / sum_ratio
    return entropy


# =============================================================================
# Plug-in (ML) Entropy Estimator (AFML Ch. 18.3.1)
# =============================================================================


@jit(nopython=True, cache=True)
def _plugin_entropy_nb(
    encoded_sequence: npt.NDArray[np.int32],
    word_length: int = 1,
) -> float:
    """Plug-in (Maximum Likelihood) entropy estimator.

    Estimates entropy from empirical word frequencies.
    More efficient than LZ for short sequences.

    Parameters
    ----------
    encoded_sequence : array
        Discretized sequence (integer symbols)
    word_length : int
        Length of words/patterns to count

    Returns
    -------
    float
        Estimated entropy in bits per symbol

    References
    ----------
    .. [1] AFML Chapter 18.3.1
    """
    # Filter out invalid entries
    valid_mask = encoded_sequence >= 0
    valid_seq = encoded_sequence[valid_mask]
    n = len(valid_seq)

    if n < word_length:
        return np.nan

    # For word_length=1, just count symbol frequencies
    if word_length == 1:
        # Count unique symbols and their frequencies
        max_symbol = np.max(valid_seq)
        counts = np.zeros(max_symbol + 1, dtype=np.float64)

        for i in range(n):
            counts[valid_seq[i]] += 1

        # Calculate entropy
        total = float(n)
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * np.log2(p)

        return entropy

    # For longer words, use a different approach
    # Count (word_length)-grams
    n_words = n - word_length + 1
    if n_words <= 0:
        return np.nan

    # Create word signatures (simple hash)
    max_symbol = np.max(valid_seq) + 1
    word_hashes = np.empty(n_words, dtype=np.int64)

    for i in range(n_words):
        word_hash = np.int64(0)
        for j in range(word_length):
            word_hash = word_hash * max_symbol + valid_seq[i + j]
        word_hashes[i] = word_hash

    # Count unique words
    sorted_hashes = np.sort(word_hashes)
    unique_count = 1
    for i in range(1, len(sorted_hashes)):
        if sorted_hashes[i] != sorted_hashes[i - 1]:
            unique_count += 1

    # Count frequencies for each unique word
    word_counts = np.zeros(unique_count, dtype=np.float64)
    sorted_hashes = np.sort(word_hashes)

    current_idx = 0
    current_count = 1

    for i in range(1, len(sorted_hashes)):
        if sorted_hashes[i] == sorted_hashes[i - 1]:
            current_count += 1
        else:
            word_counts[current_idx] = float(current_count)
            current_idx += 1
            current_count = 1
    word_counts[current_idx] = float(current_count)

    # Calculate entropy
    total = float(n_words)
    entropy = 0.0
    for c in word_counts:
        if c > 0:
            p = c / total
            entropy -= p * np.log2(p)

    # Normalize by word length to get bits per symbol
    return entropy / word_length


# =============================================================================
# Rolling Entropy Features
# =============================================================================


@logged_feature("rolling_entropy", warn_threshold_ms=300.0, log_data_quality=True)
@feature(
    name="rolling_entropy",
    category="ml",
    description="Rolling Shannon entropy - information content over window",
    lookback="window",
    normalized=True,
    value_range=(0.0, 10.0),
    formula="H = -sum(p_i * log2(p_i))",
    ta_lib_compatible=False,
)
def rolling_entropy(
    feature: pl.Expr | str,
    window: int = 50,
    n_bins: int = 10,
) -> pl.Expr:
    """Rolling Shannon entropy as a measure of uncertainty.

    High entropy indicates high uncertainty/randomness in the distribution.
    Low entropy suggests more predictability.

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to calculate entropy for
    window : int, default 50
        Rolling window size
    n_bins : int, default 10
        Number of bins for discretization

    Returns
    -------
    pl.Expr
        Rolling entropy (bits)

    References
    ----------
    .. [1] AFML Chapter 18 - Entropy Features
    """
    validate_window(window, min_window=2, name="window")
    validate_window(n_bins, min_window=2, name="n_bins")

    feature_expr = pl.col(feature) if isinstance(feature, str) else feature

    return feature_expr.rolling_map(
        lambda x: _shannon_entropy_nb(
            x.to_numpy().astype(np.float64) if hasattr(x, "to_numpy") else x,
            n_bins,
        ),
        window_size=window,
        weights=None,
        min_samples=window // 2,
        center=False,
    )


@logged_feature("rolling_entropy_lz", warn_threshold_ms=500.0, log_data_quality=True)
@feature(
    name="rolling_entropy_lz",
    category="ml",
    description="Rolling Kontoyiannis (LZ) entropy - compression-based complexity",
    lookback="window",
    normalized=True,
    value_range=(0.0, 10.0),
    formula="H = [1/k * sum(L_i / log2(n))]^(-1)",
    ta_lib_compatible=False,
)
def rolling_entropy_lz(
    feature: pl.Expr | str,
    window: int = 100,
    encoding: str = "quantile",
    n_bins: int = 10,
) -> pl.Expr:
    """Rolling Kontoyiannis (Lempel-Ziv) entropy estimator.

    Measures complexity via compression - highly repetitive sequences
    have low entropy, complex/random sequences have high entropy.

    This estimator is particularly useful for:
    - Detecting regime changes (complexity shifts)
    - Measuring market efficiency
    - Identifying structural breaks

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to calculate entropy for
    window : int, default 100
        Rolling window size (larger windows give better estimates)
    encoding : str, default "quantile"
        Encoding scheme: "binary", "quantile", or "sigma"
    n_bins : int, default 10
        Number of bins for quantile/sigma encoding

    Returns
    -------
    pl.Expr
        Rolling LZ entropy (bits)

    References
    ----------
    .. [1] Kontoyiannis et al. (1998). Nonparametric Entropy Estimation
    .. [2] AFML Chapter 18.3.2
    """
    validate_window(window, min_window=10, name="window")
    validate_window(n_bins, min_window=2, name="n_bins")

    if encoding not in ("binary", "quantile", "sigma"):
        raise ValueError(f"encoding must be 'binary', 'quantile', or 'sigma', got {encoding}")

    feature_expr = pl.col(feature) if isinstance(feature, str) else feature

    def compute_lz_entropy(x: pl.Series) -> float:
        arr = x.to_numpy().astype(np.float64)

        # Encode based on scheme
        if encoding == "binary":
            encoded = _encode_binary_nb(arr)
        elif encoding == "quantile":
            encoded = _encode_quantile_nb(arr, n_bins)
        else:  # sigma
            encoded = _encode_sigma_nb(arr, n_bins)

        return _kontoyiannis_entropy_nb(encoded, window_size=len(arr) // 2)

    return feature_expr.rolling_map(
        compute_lz_entropy,
        window_size=window,
        weights=None,
        min_samples=window // 2,
        center=False,
    )


@logged_feature("rolling_entropy_plugin", warn_threshold_ms=300.0, log_data_quality=True)
@feature(
    name="rolling_entropy_plugin",
    category="ml",
    description="Rolling plug-in (ML) entropy - word frequency estimator",
    lookback="window",
    normalized=True,
    value_range=(0.0, 10.0),
    formula="H = -sum(f_w/n * log2(f_w/n))",
    ta_lib_compatible=False,
)
def rolling_entropy_plugin(
    feature: pl.Expr | str,
    window: int = 50,
    encoding: str = "quantile",
    n_bins: int = 10,
    word_length: int = 1,
) -> pl.Expr:
    """Rolling plug-in (Maximum Likelihood) entropy estimator.

    Estimates entropy from empirical symbol/word frequencies.
    Fast and effective for shorter windows.

    Parameters
    ----------
    feature : pl.Expr | str
        Feature to calculate entropy for
    window : int, default 50
        Rolling window size
    encoding : str, default "quantile"
        Encoding scheme: "binary", "quantile", or "sigma"
    n_bins : int, default 10
        Number of bins for quantile/sigma encoding
    word_length : int, default 1
        Length of words to count (1 = individual symbols)

    Returns
    -------
    pl.Expr
        Rolling plug-in entropy (bits per symbol)

    References
    ----------
    .. [1] AFML Chapter 18.3.1
    """
    validate_window(window, min_window=2, name="window")
    validate_window(n_bins, min_window=2, name="n_bins")
    validate_window(word_length, min_window=1, name="word_length")

    if encoding not in ("binary", "quantile", "sigma"):
        raise ValueError(f"encoding must be 'binary', 'quantile', or 'sigma', got {encoding}")

    feature_expr = pl.col(feature) if isinstance(feature, str) else feature

    def compute_plugin_entropy(x: pl.Series) -> float:
        arr = x.to_numpy().astype(np.float64)

        # Encode based on scheme
        if encoding == "binary":
            encoded = _encode_binary_nb(arr)
        elif encoding == "quantile":
            encoded = _encode_quantile_nb(arr, n_bins)
        else:  # sigma
            encoded = _encode_sigma_nb(arr, n_bins)

        return _plugin_entropy_nb(encoded, word_length)

    return feature_expr.rolling_map(
        compute_plugin_entropy,
        window_size=window,
        weights=None,
        min_samples=window // 2,
        center=False,
    )
