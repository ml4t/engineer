# mypy: disable-error-code="no-any-return,arg-type,call-arg,return-value,assignment"
"""Core labeling functionality.

Implements the generalized triple-barrier labeling method for financial machine learning.
The triple-barrier method labels price movements based on three barriers:
1. Upper barrier (profit target)
2. Lower barrier (stop loss)
3. Time barrier (maximum holding period)

This implementation supports:
- Fixed and dynamic barriers
- Trailing stop losses
- Path-dependent features
- Asymmetric barriers
- Side-aware labeling

Exports:
    triple_barrier_labels(data, upper_barrier, lower_barrier, max_holding, ...) -> DataFrame
        Main API for triple-barrier labeling. Returns labels (+1/-1/0), returns, touch times.

    fixed_time_horizon_labels(data, horizon=1, method="returns") -> DataFrame
        Simple forward return labels (returns, log_returns, or binary).

    trend_scanning_labels(data, min_window=5, max_window=50) -> DataFrame
        De Prado's t-statistic method. Optimal window selection per observation.

    sequential_bootstrap(data, n_samples) -> DataFrame
        Sequential bootstrap sampling for overlapping labels.

    calculate_label_uniqueness(data, label_col) -> DataFrame
        Uniqueness score per label for sample weighting.

    calculate_sample_weights(data, label_col) -> DataFrame
        Sample weights based on label uniqueness.

    build_concurrency(data, timestamps, end_times) -> DataFrame
        Build concurrency matrix for overlapping events.

Internal:
    _calculate_barrier_prices() - Side-aware barrier price calculation
    _process_single_event() - Process one event through barriers
    _apply_triple_barrier_nb() - Numba-compiled triple barrier loop
"""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit
from numpy.random import Generator, default_rng

from ml4t.engineer.core.exceptions import DataValidationError
from ml4t.engineer.labeling.barriers import BarrierConfig


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _calculate_barrier_prices(
    event_price: float,
    upper_barrier: float,
    lower_barrier: float,
    side: int,
) -> tuple[float, float]:
    """Calculate actual barrier price levels based on position side.

    Both barriers are expected as POSITIVE distances from the entry price.
    The position side determines the direction of the barriers.

    Parameters
    ----------
    event_price : float
        Entry price for the event
    upper_barrier : float
        Upper barrier distance (positive percentage for profit target)
    lower_barrier : float
        Lower barrier distance (positive percentage for stop loss)
    side : int
        Position side: 1 (long), -1 (short), 0 (symmetric)

    Returns
    -------
    tuple[float, float]
        (upper_price, lower_price) actual barrier price levels

    Examples
    --------
    For a long position with entry at 100:
    - upper_barrier=0.02 (2% profit) -> upper_price = 102.0
    - lower_barrier=0.01 (1% loss) -> lower_price = 99.0

    For a short position with entry at 100:
    - upper_barrier=0.02 (2% profit) -> upper_price = 98.0 (profit is downward)
    - lower_barrier=0.01 (1% loss) -> lower_price = 101.0 (stop is upward)
    """
    # Convert barriers to absolute values to ensure consistency
    upper_barrier = abs(upper_barrier)
    lower_barrier = abs(lower_barrier)

    if side == 1:  # Long position
        upper_price = event_price * (1 + upper_barrier)  # Profit target above entry
        lower_price = event_price * (1 - lower_barrier)  # Stop loss below entry
    elif side == -1:  # Short position
        upper_price = event_price * (1 - upper_barrier)  # Profit target below entry
        lower_price = event_price * (1 + lower_barrier)  # Stop loss above entry
    else:  # No side, symmetric (assume long-like behavior)
        upper_price = event_price * (1 + upper_barrier)
        lower_price = event_price * (1 - lower_barrier)

    return upper_price, lower_price


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _initialize_trailing_stop(
    event_price: float,
    trailing_stop: float,
    side: int,
) -> float:
    """Initialize trailing stop price.

    Parameters
    ----------
    event_price : float
        Entry price for the event
    trailing_stop : float
        Trailing stop percentage (positive value)
    side : int
        Position side: 1 (long), -1 (short), 0 (symmetric)

    Returns
    -------
    float
        Initial trailing stop price level
    """
    if trailing_stop > 0:
        if side == 1:
            return event_price * (1 - trailing_stop)
        return event_price * (1 + trailing_stop)
    return float(-np.inf) if side == 1 else float(np.inf)


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _update_trailing_stop(
    current_price: float,
    trailing_stop_price: float,
    trailing_stop: float,
    side: int,
) -> float:
    """Update trailing stop price based on current price movement.

    Parameters
    ----------
    current_price : float
        Current market price
    trailing_stop_price : float
        Current trailing stop price level
    trailing_stop : float
        Trailing stop percentage
    side : int
        Position side: 1 (long), -1 (short), 0 (symmetric)

    Returns
    -------
    float
        Updated trailing stop price level
    """
    if trailing_stop > 0:
        if side == 1:
            # For long, trail up with price
            new_stop = current_price * (1 - trailing_stop)
            return max(trailing_stop_price, new_stop)
        # For short, trail down with price
        new_stop = current_price * (1 + trailing_stop)
        return min(trailing_stop_price, new_stop)
    return trailing_stop_price


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _check_barrier_touch(
    high_price: float,
    low_price: float,
    upper_price: float,
    lower_price: float,
    trailing_stop_price: float,
    side: int,
) -> int:
    """Check if any barrier has been touched using OHLC prices.

    For realistic barrier checking:
    - LONG positions: TP triggers on high >= target, SL triggers on low <= stop
    - SHORT positions: TP triggers on low <= target, SL triggers on high >= stop

    Parameters
    ----------
    high_price : float
        High price of the bar (used to check TP for LONG, SL for SHORT)
    low_price : float
        Low price of the bar (used to check SL for LONG, TP for SHORT)
    upper_price : float
        Upper barrier price level (profit target)
    lower_price : float
        Lower barrier price level (stop loss)
    trailing_stop_price : float
        Trailing stop price level
    side : int
        Position side: 1 (long), -1 (short), 0 (symmetric)

    Returns
    -------
    int
        Barrier touched: 1 (upper/profit), -1 (lower/stop), 0 (none)
    """
    # Check upper barrier (profit target)
    # LONG: high can reach TP, SHORT: low can reach TP (which is below entry)
    upper_hit = (
        (side == 1 and high_price >= upper_price)
        or (side == -1 and low_price <= upper_price)
        or (side == 0 and high_price >= upper_price)
    )

    # Check lower barrier (stop loss)
    # LONG: low can hit SL, SHORT: high can hit SL (which is above entry)
    lower_hit = (
        (side == 1 and (low_price <= lower_price or low_price <= trailing_stop_price))
        or (side == -1 and (high_price >= lower_price or high_price >= trailing_stop_price))
        or (side == 0 and low_price <= lower_price)
    )

    if upper_hit:
        return 1
    if lower_hit:
        return -1
    return 0


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _calculate_label_return(event_price: float, label_price: float, side: int) -> float:
    """Calculate return from event to label based on position side.

    Parameters
    ----------
    event_price : float
        Entry price
    label_price : float
        Exit price
    side : int
        Position side: 1 (long), -1 (short), 0 (symmetric)

    Returns
    -------
    float
        Return from entry to exit
    """
    # Defensive check: prices should never be zero in financial data
    if event_price == 0:
        return 0.0

    if side == -1:  # Short position
        return (event_price - label_price) / event_price
    # Long or symmetric
    return (label_price - event_price) / event_price


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _process_single_event(
    closes: npt.NDArray[np.float64],
    highs: npt.NDArray[np.float64],
    lows: npt.NDArray[np.float64],
    event_idx: int,
    upper_barrier: float,
    lower_barrier: float,
    max_period: int,
    side: int,
    trailing_stop: float,
    n_prices: int,
) -> tuple[int, int, float, float, int]:
    """Process a single event through the triple barrier method.

    Parameters
    ----------
    closes : npt.NDArray
        Array of close prices (used for entry and return calculation)
    highs : npt.NDArray
        Array of high prices (used for barrier checking)
    lows : npt.NDArray
        Array of low prices (used for barrier checking)
    event_idx : int
        Index of the event in the price array
    upper_barrier : float
        Upper barrier percentage
    lower_barrier : float
        Lower barrier percentage
    max_period : int
        Maximum holding period
    side : int
        Position side
    trailing_stop : float
        Trailing stop percentage
    n_prices : int
        Total number of prices

    Returns
    -------
    tuple[int, int, float, float, int]
        (label, label_index, label_price, label_return, bar_duration)
    """
    if event_idx >= n_prices:
        return 0, event_idx, closes[min(event_idx, n_prices - 1)], 0.0, 0

    event_price = closes[event_idx]

    # Calculate barrier prices
    upper_price, lower_price = _calculate_barrier_prices(
        event_price,
        upper_barrier,
        lower_barrier,
        side,
    )

    # Initialize trailing stop
    trailing_stop_price = _initialize_trailing_stop(event_price, trailing_stop, side)

    # Scan forward to find first barrier touch
    end_idx = min(event_idx + max_period, n_prices)

    for j in range(event_idx + 1, end_idx):
        high_price = highs[j]
        low_price = lows[j]
        close_price = closes[j]

        # Update trailing stop using high for LONG (trails up), low for SHORT (trails down)
        # Use high for LONG to maximize trailing benefit, close is also acceptable
        trailing_update_price = high_price if side == 1 else low_price
        trailing_stop_price = _update_trailing_stop(
            trailing_update_price,
            trailing_stop_price,
            trailing_stop,
            side,
        )

        # Check barriers using high/low for more realistic detection
        barrier_touched = _check_barrier_touch(
            high_price,
            low_price,
            upper_price,
            lower_price,
            trailing_stop_price,
            side,
        )

        if barrier_touched != 0:
            # Barrier was touched - use close for return calculation
            label_return = _calculate_label_return(event_price, close_price, side)
            bar_duration = j - event_idx
            return barrier_touched, j, close_price, label_return, bar_duration

    # If no barrier hit, time barrier touched
    final_idx = end_idx - 1
    final_price = closes[final_idx] if final_idx < n_prices else event_price
    label_return = _calculate_label_return(event_price, final_price, side)
    bar_duration = final_idx - event_idx

    return 0, final_idx, final_price, label_return, bar_duration


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _apply_triple_barrier_nb(
    closes: npt.NDArray[np.float64],
    highs: npt.NDArray[np.float64],
    lows: npt.NDArray[np.float64],
    event_times: npt.NDArray[np.float64],
    upper_barriers: npt.NDArray[np.float64],
    lower_barriers: npt.NDArray[np.float64],
    max_periods: npt.NDArray[np.float64],
    sides: npt.NDArray[np.float64],
    trailing_stops: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Apply triple-barrier labeling using Numba for performance - refactored version.

    This refactored version breaks down the monolithic triple barrier calculation into
    smaller, testable components while maintaining exact numerical compatibility.

    Now uses OHLC prices for more realistic barrier detection:
    - LONG: TP checked against high, SL checked against low
    - SHORT: TP checked against low, SL checked against high

    Returns
    -------
    tuple of arrays
        (labels, label_indices, label_prices, label_returns, bar_durations)
    """
    n_events = len(event_times)
    n_prices = len(closes)

    # Output arrays
    labels = np.zeros(n_events, dtype=np.int32)
    label_indices = np.zeros(n_events, dtype=np.int64)
    label_prices = np.zeros(n_events, dtype=np.float64)
    label_returns = np.zeros(n_events, dtype=np.float64)
    bar_durations = np.zeros(n_events, dtype=np.int64)

    for i in range(n_events):
        event_idx = event_times[i]
        upper = upper_barriers[i]
        lower = lower_barriers[i]
        max_period = max_periods[i]
        side = sides[i]
        trailing_stop = trailing_stops[i]

        # Process single event using helper function
        label, label_idx, label_price, label_return, bar_duration = _process_single_event(
            closes,
            highs,
            lows,
            event_idx,
            upper,
            lower,
            max_period,
            side,
            trailing_stop,
            n_prices,
        )

        labels[i] = label
        label_indices[i] = label_idx
        label_prices[i] = label_price
        label_returns[i] = label_return
        bar_durations[i] = bar_duration

    return labels, label_indices, label_prices, label_returns, bar_durations


# ============================================================================
# UNIQUENESS AND CONCURRENCY CALCULATIONS (De Prado AFML Ch. 4)
# ============================================================================


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _build_concurrency_nb(
    n_bars: int,
    starts: npt.NDArray[np.int64],
    ends: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """
    Numba-compiled efficient difference-array sweep for concurrency calculation.
    Internal implementation - use build_concurrency() for public API.
    """
    diff = np.zeros(n_bars + 1, dtype=np.int64)
    for s, e in zip(starts, ends):
        if s < 0 or s >= n_bars:
            continue
        e = min(max(e, s), n_bars - 1)
        diff[s] += 1
        if e + 1 <= n_bars - 1:
            diff[e + 1] -= 1
    return np.cumsum(diff[:-1])


def build_concurrency(
    event_indices: npt.NDArray[np.float64],
    label_indices: npt.NDArray[np.float64],
    n_bars: int | None = None,
) -> npt.NDArray[np.float64]:
    """
    Calculate per-bar concurrency (how many labels are active at each time).

    This function computes c[t] = number of labels active at time t using
    an efficient O(n) difference-array algorithm.

    Parameters
    ----------
    event_indices : array
        Start indices of labels (when positions were entered)
    label_indices : array
        End indices of labels (when barriers were hit)
    n_bars : int, optional
        Total number of bars. If None, uses max(label_indices) + 1

    Returns
    -------
    array
        Concurrency at each timestamp (length = n_bars)

    Notes
    -----
    Concurrency is used to calculate label uniqueness. High concurrency
    at time t means many labels overlap there, indicating redundancy.

    References
    ----------
    .. [1] López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 4: Sample Weights.

    Examples
    --------
    >>> concurrency = build_concurrency(event_indices, label_indices, len(prices))
    >>> # concurrency[t] = number of active labels at time t
    >>> max_overlap = concurrency.max()  # Maximum label overlap
    """
    if n_bars is None:
        n_bars = int(np.max(label_indices)) + 1

    return _build_concurrency_nb(n_bars, event_indices, label_indices)


# Keep old name as private alias for backward compatibility within the module
def _build_concurrency(
    n_bars: int,
    starts: npt.NDArray[np.int64],
    ends: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Legacy private function - use build_concurrency() instead."""
    return _build_concurrency_nb(n_bars, starts, ends)


def calculate_label_uniqueness(
    event_indices: npt.NDArray[np.intp],
    label_indices: npt.NDArray[np.intp],
    n_bars: int | None = None,
) -> npt.NDArray[np.float64]:
    """
    Calculate average uniqueness for each label based on overlapping periods.

    Uniqueness measures how "independent" a label is from others. Labels that
    overlap with many others have low uniqueness (redundant information), while
    labels that are relatively isolated have high uniqueness.

    Parameters
    ----------
    event_indices : array
        Start indices of labels (when positions were entered)
    label_indices : array
        End indices of labels (when barriers were hit)
    n_bars : int, optional
        Total number of bars. If None, uses max(label_indices) + 1

    Returns
    -------
    array
        Average uniqueness score for each label (between 0 and 1)

    Notes
    -----
    From López de Prado's AFML:
    u_i = (1/T_i) * Σ(1/c_t) for t in [start_i, end_i]

    Where:
    - T_i is the length of label i's active period
    - c_t is the concurrency at time t (number of active labels)
    - Higher uniqueness means more independent information

    References
    ----------
    .. [1] López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 4: Sample Weights.
    """
    # Input validation
    if len(event_indices) != len(label_indices):
        raise ValueError(
            f"event_indices and label_indices must have same length, "
            f"got {len(event_indices)} and {len(label_indices)}"
        )

    if len(event_indices) == 0:
        return np.array([])

    if np.any(event_indices < 0) or np.any(label_indices < 0):
        raise ValueError("Indices must be non-negative")

    if n_bars is None:
        n_bars = int(np.max(label_indices)) + 1

    # Build concurrency array
    concurrency = _build_concurrency(
        n_bars,
        event_indices.astype(np.int64),
        label_indices.astype(np.int64),
    )

    # Calculate uniqueness for each label
    n_labels = len(event_indices)
    uniqueness = np.zeros(n_labels, dtype=np.float64)

    for i in range(n_labels):
        start = int(event_indices[i])
        end = int(label_indices[i])

        if start < n_bars and start <= end:
            # Ensure we don't go out of bounds
            start = max(0, start)
            end = min(end, n_bars - 1)

            # Average of 1/c_t over the label's active period
            c_slice = concurrency[start : end + 1]
            # Avoid division by zero (though concurrency should always be >= 1)
            uniqueness[i] = np.mean(1.0 / np.maximum(c_slice, 1.0))
        else:
            uniqueness[i] = 1.0  # Default for invalid ranges

    return uniqueness


def calculate_sample_weights(
    uniqueness: npt.NDArray[np.float64],
    returns: npt.NDArray[np.float64],
    weight_scheme: Literal[
        "returns_uniqueness", "uniqueness_only", "returns_only", "equal"
    ] = "returns_uniqueness",
) -> npt.NDArray[np.float64]:
    """
    Calculate sample weights combining statistical uniqueness and economic significance.

    Parameters
    ----------
    uniqueness : array
        Average uniqueness scores from calculate_label_uniqueness
    returns : array
        Label returns (from entry to exit)
    weight_scheme : str
        Weighting scheme to use:
        - "returns_uniqueness": u_i * |r_i| (De Prado's recommendation)
        - "uniqueness_only": u_i only (statistical correction)
        - "returns_only": |r_i| only (economic significance)
        - "equal": uniform weights

    Returns
    -------
    array
        Sample weights for training (normalized to sum to len(weights))

    Notes
    -----
    De Prado recommends "returns_uniqueness" to balance:
    - Statistical independence (uniqueness)
    - Economic importance (return magnitude)

    This prevents overweighting "boring" full-horizon labels while
    preserving the importance of profitable trades.

    References
    ----------
    .. [1] López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 4: Sample Weights.
    """
    # Input validation
    if len(uniqueness) != len(returns):
        raise ValueError(
            f"uniqueness and returns must have same length, "
            f"got {len(uniqueness)} and {len(returns)}"
        )

    if len(uniqueness) == 0:
        return np.array([])

    if weight_scheme == "returns_uniqueness":
        # De Prado's formula: combine uniqueness with economic significance
        weights = uniqueness * np.abs(returns)
    elif weight_scheme == "uniqueness_only":
        weights = uniqueness
    elif weight_scheme == "returns_only":
        weights = np.abs(returns)
    else:  # "equal"
        weights = np.ones_like(uniqueness)

    # Normalize weights to sum to len(weights) for compatibility with ML libraries
    total = np.sum(weights)
    weights = weights * len(weights) / total if total > 0 else np.ones_like(uniqueness)

    return weights


def _expected_uniqueness_for_candidate(
    starts: npt.NDArray[np.int64],
    ends: npt.NDArray[np.int64],
    concurrency: npt.NDArray[np.int64],
    cand_idx: int,
) -> float:
    """
    Calculate expected uniqueness if we add candidate to the sample.
    u_j = mean_t(1 / (c_t + 1)) for t in [s_j, e_j]
    """
    s = int(starts[cand_idx])
    e = int(ends[cand_idx])
    if e < s:
        return 0.0

    c_slice = concurrency[s : e + 1]
    # +1 because we're calculating marginal uniqueness (if we add this label)
    return float(np.mean(1.0 / (c_slice + 1.0)))


def sequential_bootstrap(
    starts: npt.NDArray[np.int64],
    ends: npt.NDArray[np.int64],
    n_bars: int | None = None,
    n_draws: int | None = None,
    with_replacement: bool = True,
    random_state: int | Generator | None = None,
) -> npt.NDArray[np.int64]:
    """
    Sequential bootstrap that favors events with high marginal uniqueness.

    This method creates a bootstrapped sample that minimizes redundancy by
    probabilistically selecting labels based on how unique they would be
    given the already-selected labels.

    Parameters
    ----------
    starts : array
        Start indices of labels (event_indices)
    ends : array
        End indices of labels (label_indices)
    n_bars : int, optional
        Total number of bars. If None, uses max(ends) + 1
    n_draws : int, optional
        Number of selections to make. Defaults to len(starts)
    with_replacement : bool, default True
        If False, each event can be selected at most once
    random_state : int or Generator, optional
        RNG seed or Generator for reproducibility

    Returns
    -------
    array
        Indices of selected events in the order drawn (length = n_draws)

    Notes
    -----
    From López de Prado's AFML Chapter 4:
    - At each step, pick the event that maximizes expected average uniqueness
    - Probability of selection is proportional to marginal uniqueness
    - Creates less redundant training sets compared to random sampling

    References
    ----------
    .. [1] López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 4: Sample Weights.

    Examples
    --------
    >>> # After triple barrier labeling
    >>> order = sequential_bootstrap(event_indices, label_indices, len(prices))
    >>> # Use order to select training samples
    >>> X_train = X[order]
    >>> y_train = y[order]
    >>> weights_train = sample_weights[order]
    """
    # Input validation
    if len(starts) != len(ends):
        raise ValueError(
            f"starts and ends must have same length, got {len(starts)} and {len(ends)}"
        )

    if len(starts) == 0:
        return np.array([], dtype=np.int64)

    if np.any(starts < 0) or np.any(ends < 0):
        raise ValueError("Indices must be non-negative")

    m = len(starts)
    if n_bars is None:
        n_bars = int(np.max(ends)) + 1
    if n_draws is None:
        n_draws = m

    if n_draws <= 0:
        raise ValueError(f"n_draws must be positive, got {n_draws}")

    rng: Generator = (
        default_rng(random_state) if not isinstance(random_state, Generator) else random_state
    )

    # Start with empty concurrency
    concurrency = np.zeros(n_bars, dtype=np.int64)
    available = np.ones(m, dtype=bool)  # track availability if sampling w/o replacement
    order = np.empty(n_draws, dtype=np.int64)

    for k in range(n_draws):
        # Compute marginal expected uniqueness for all available candidates
        u = np.zeros(m, dtype=np.float64)
        for j in range(m):
            if with_replacement or available[j]:
                u[j] = _expected_uniqueness_for_candidate(starts, ends, concurrency, j)
            else:
                u[j] = 0.0

        total = float(u.sum())
        if total <= 0.0:
            # Fallback to uniform over available items
            probs = np.where(available | with_replacement, 1.0, 0.0)
            prob_sum = probs.sum()
            if prob_sum == 0:
                # No valid candidates remaining
                raise ValueError(
                    f"Cannot draw {n_draws} samples without replacement from {m} candidates. "
                    f"Either reduce n_draws or set with_replacement=True."
                )
            probs = probs / prob_sum
        else:
            probs = u / total

        # Draw next index
        j = int(rng.choice(m, p=probs))
        order[k] = j

        # Update concurrency with the chosen interval
        s, e = int(starts[j]), int(ends[j])
        if s <= e and s < n_bars:
            s_clamped = max(0, min(s, n_bars - 1))
            e_clamped = max(0, min(e, n_bars - 1))
            concurrency[s_clamped : e_clamped + 1] += 1

        if not with_replacement:
            available[j] = False

    return order


# =============================================================================
# Triple Barrier Labeling Helper Functions
# =============================================================================


def _prepare_barrier_arrays(
    data: pl.DataFrame,
    config: BarrierConfig,
    event_indices: npt.NDArray[np.intp],
) -> tuple[
    npt.NDArray[np.float64],  # upper_barriers
    npt.NDArray[np.float64],  # lower_barriers
    npt.NDArray[np.int64],  # max_periods
    npt.NDArray[np.int32],  # sides
    npt.NDArray[np.float64],  # trailing_stops
]:
    """Prepare barrier arrays from config.

    Handles both fixed values and dynamic column-based barriers.

    Parameters
    ----------
    data : pl.DataFrame
        Input data
    config : BarrierConfig
        Barrier configuration
    event_indices : npt.NDArray
        Indices of events to label

    Returns
    -------
    tuple
        (upper_barriers, lower_barriers, max_periods, sides, trailing_stops)

    Raises
    ------
    DataValidationError
        If a column specified in config is not found
    """
    n_events = len(event_indices)

    # Upper barriers
    if config.upper_barrier is None:
        upper_barriers = np.full(n_events, np.inf)
    elif isinstance(config.upper_barrier, (int, float)):
        upper_barriers = np.full(n_events, float(config.upper_barrier))
    else:
        if config.upper_barrier not in data.columns:
            raise DataValidationError(
                f"Upper barrier column '{config.upper_barrier}' not found",
            )
        upper_barriers = data[config.upper_barrier].to_numpy()[event_indices]

    # Lower barriers
    if config.lower_barrier is None:
        lower_barriers = np.full(n_events, -np.inf)
    elif isinstance(config.lower_barrier, (int, float)):
        lower_barriers = np.full(n_events, float(config.lower_barrier))
    else:
        if config.lower_barrier not in data.columns:
            raise DataValidationError(
                f"Lower barrier column '{config.lower_barrier}' not found",
            )
        lower_barriers = data[config.lower_barrier].to_numpy()[event_indices]

    # Max periods
    if isinstance(config.max_holding_period, int):
        max_periods = np.full(n_events, config.max_holding_period, dtype=np.int64)
    else:
        if config.max_holding_period not in data.columns:
            raise DataValidationError(
                f"Max holding period column '{config.max_holding_period}' not found",
            )
        max_periods = data[config.max_holding_period].to_numpy()[event_indices].astype(np.int64)

    # Sides
    if config.side is None:
        sides = np.zeros(n_events, dtype=np.int32)
    elif isinstance(config.side, int):
        sides = np.full(n_events, config.side, dtype=np.int32)
    else:
        if config.side not in data.columns:
            raise DataValidationError(f"Side column '{config.side}' not found")
        sides = data[config.side].to_numpy()[event_indices].astype(np.int32)

    # Trailing stops
    if config.trailing_stop is False or config.trailing_stop is None:
        trailing_stops = np.zeros(n_events)
    elif config.trailing_stop is True:
        if config.lower_barrier is not None and isinstance(
            config.lower_barrier,
            (int, float),
        ):
            trailing_stops = np.full(n_events, abs(float(config.lower_barrier)))
        else:
            trailing_stops = np.full(n_events, 0.01)  # Default 1%
    elif isinstance(config.trailing_stop, (int, float)):
        trailing_stops = np.full(n_events, float(config.trailing_stop))
    else:
        if config.trailing_stop not in data.columns:
            raise DataValidationError(
                f"Trailing stop column '{config.trailing_stop}' not found",
            )
        trailing_stops = data[config.trailing_stop].to_numpy()[event_indices]

    return upper_barriers, lower_barriers, max_periods, sides, trailing_stops


def _determine_barrier_hits(
    labels: npt.NDArray[np.int32],
) -> npt.NDArray[np.object_]:
    """Determine which barrier was hit for each label.

    Parameters
    ----------
    labels : npt.NDArray
        Label values (-1, 0, 1)

    Returns
    -------
    npt.NDArray
        Array of strings ("upper", "lower", "time")
    """
    n_events = len(labels)
    barrier_hit = np.empty(n_events, dtype=object)

    for i in range(n_events):
        if labels[i] == 1:
            barrier_hit[i] = "upper"
        elif labels[i] == -1:
            barrier_hit[i] = "lower"
        else:
            barrier_hit[i] = "time"

    return barrier_hit


def _calculate_time_durations(
    data: pl.DataFrame,
    event_indices: npt.NDArray[np.intp],
    label_indices: npt.NDArray[np.int64],
    timestamp_col: str | None,
) -> tuple[list[Any], npt.NDArray[Any]]:
    """Calculate label times and durations.

    Parameters
    ----------
    data : pl.DataFrame
        Input data
    event_indices : npt.NDArray
        Event indices
    label_indices : npt.NDArray
        Label indices (when barrier was hit)
    timestamp_col : str | None
        Name of timestamp column, or None

    Returns
    -------
    tuple
        (label_times, time_durations)
    """
    n_events = len(event_indices)

    if timestamp_col is not None:
        timestamps = data[timestamp_col].to_list()
        label_times = [timestamps[idx] if idx < len(timestamps) else None for idx in label_indices]

        timestamps_array = data[timestamp_col].to_numpy()
        entry_times = timestamps_array[event_indices]
        exit_times = timestamps_array[label_indices]
        time_durations = exit_times - entry_times
    else:
        label_times = list(label_indices)
        time_durations = np.array([None] * n_events, dtype=object)

    return label_times, time_durations


def _build_labeling_result(
    data: pl.DataFrame,
    event_indices: npt.NDArray[np.intp],
    labels: npt.NDArray[np.int32],
    label_times: list[Any],
    label_prices: npt.NDArray[np.float64],
    label_returns: npt.NDArray[np.float64],
    bar_durations: npt.NDArray[np.int64],
    time_durations: npt.NDArray[Any],
    barrier_hit: npt.NDArray[np.object_],
    uniqueness: npt.NDArray[np.float64] | None,
    sample_weights: npt.NDArray[np.float64] | None,
) -> pl.DataFrame:
    """Build the result DataFrame from labeling outputs.

    Parameters
    ----------
    data : pl.DataFrame
        Original input data
    event_indices : npt.NDArray
        Event indices
    labels : npt.NDArray
        Label values
    label_times : list
        When each label was determined
    label_prices : npt.NDArray
        Prices at label time
    label_returns : npt.NDArray
        Returns from event to label
    bar_durations : npt.NDArray
        Number of bars held
    time_durations : npt.NDArray
        Time elapsed
    barrier_hit : npt.NDArray
        Which barrier was hit
    uniqueness : npt.NDArray | None
        Uniqueness scores (if calculated)
    sample_weights : npt.NDArray | None
        Sample weights (if calculated)

    Returns
    -------
    pl.DataFrame
        Data with labeling columns added
    """
    calculate_uniqueness = uniqueness is not None

    if "event_time" in data.columns:
        # Event-based: add columns only for events
        label_dict: dict[str, Any] = {
            "event_index": event_indices,
            "label": labels,
            "label_time": label_times,
            "label_price": label_prices,
            "label_return": label_returns,
            "label_bars": bar_durations,
            "label_duration": time_durations,
            "barrier_hit": barrier_hit,
        }

        if calculate_uniqueness:
            label_dict["label_uniqueness"] = uniqueness
            label_dict["sample_weight"] = sample_weights

        label_data = pl.DataFrame(label_dict)

        result = (
            data.with_row_index("event_index")
            .join(
                label_data,
                on="event_index",
                how="left",
            )
            .drop("event_index")
        )
    else:
        # Regular: all rows get labels
        columns_to_add: dict[str, pl.Series] = {
            "label": pl.Series(labels, dtype=pl.Int32),
            "label_time": pl.Series(label_times),
            "label_price": pl.Series(label_prices, dtype=pl.Float64),
            "label_return": pl.Series(label_returns, dtype=pl.Float64),
            "label_bars": pl.Series(bar_durations, dtype=pl.Int64),
            "label_duration": pl.Series(time_durations),
            "barrier_hit": pl.Series(barrier_hit),
        }

        if calculate_uniqueness:
            columns_to_add["label_uniqueness"] = pl.Series(uniqueness, dtype=pl.Float64)
            columns_to_add["sample_weight"] = pl.Series(sample_weights, dtype=pl.Float64)

        result = data.with_columns(**columns_to_add)

    return result


def triple_barrier_labels(
    data: pl.DataFrame,
    config: BarrierConfig,
    price_col: str = "close",
    high_col: str | None = None,
    low_col: str | None = None,
    timestamp_col: str | None = None,
    calculate_uniqueness: bool = False,
    uniqueness_weight_scheme: Literal[
        "returns_uniqueness", "uniqueness_only", "returns_only", "equal"
    ] = "returns_uniqueness",
) -> pl.DataFrame:
    """Apply triple-barrier labeling to data.

    This method implements the generalized triple-barrier labeling technique
    used in financial machine learning. It labels price movements based on
    which barrier (upper, lower, or time) is touched first.

    Optionally calculates label uniqueness and sample weights to handle
    overlapping labels (De Prado's AFML Chapter 4).

    Parameters
    ----------
    data : pl.DataFrame
        Input data with price information and optionally event markers
    config : BarrierConfig
        Configuration for the barriers including:
        - upper_barrier: profit target (can be column name for dynamic)
        - lower_barrier: stop loss (can be column name for dynamic)
        - max_periods: time barrier (can be column name for dynamic)
        - side: position side for asymmetric barriers
        - trailing_stop: trailing stop percentage
    price_col : str, default "close"
        Name of the price column (used for entry price and return calculation)
    high_col : str, optional
        Name of the high price column for OHLC barrier checking.
        If None, defaults to price_col (close-only checking).
        When provided, enables more realistic barrier detection:
        - LONG positions: TP triggers on high >= target
        - SHORT positions: SL triggers on high >= stop
    low_col : str, optional
        Name of the low price column for OHLC barrier checking.
        If None, defaults to price_col (close-only checking).
        When provided, enables more realistic barrier detection:
        - LONG positions: SL triggers on low <= stop
        - SHORT positions: TP triggers on low <= target
    timestamp_col : str, optional
        Name of the timestamp column. If None, uses row index
    calculate_uniqueness : bool, default False
        If True, calculates label uniqueness scores and sample weights
        to handle overlapping labels (De Prado's method)
    uniqueness_weight_scheme : str, default "returns_uniqueness"
        How to calculate sample weights (only if calculate_uniqueness=True):
        - "returns_uniqueness": u_i * |r_i| (De Prado's recommendation)
        - "uniqueness_only": u_i only (statistical correction)
        - "returns_only": |r_i| only (economic significance)
        - "equal": uniform weights

    Returns
    -------
    pl.DataFrame
        Original data with added columns:
        - label: -1 (stop loss), 0 (timeout), 1 (profit target)
        - label_time: when the label was determined
        - label_price: price at label time
        - label_return: return from event to label
        - label_bars: number of bars from entry to exit (int)
        - label_duration: time elapsed from entry to exit (timedelta/None)
        - barrier_hit: which barrier was hit ("upper", "lower", "time")
        - label_uniqueness: uniqueness score (if calculate_uniqueness=True)
        - sample_weight: combined weight (if calculate_uniqueness=True)

    Notes
    -----
    Duration Calculations:
    - **label_bars**: Always calculated as the integer number of bars held.
      Equals `label_index - event_index`.
    - **label_duration**: Only calculated when `timestamp_col` is provided.
      Represents actual time elapsed, handling gaps in data correctly.
      For continuous data: duration ≈ bars * bar_interval
      For gapped data: duration > bars * typical_interval

    OHLC Barrier Checking:
    - When high_col and low_col are provided, barriers are checked against
      intra-bar price extremes for more realistic detection.
    - Without OHLC, barriers are checked against close prices only, which
      can miss intra-bar barrier touches.
    - Returns are always calculated using close prices regardless of
      how barriers are detected.

    Examples
    --------
    >>> # Fixed barriers (both values are POSITIVE distances)
    >>> config = BarrierConfig(
    ...     upper_barrier=0.02,  # 2% profit target (positive distance)
    ...     lower_barrier=0.01,  # 1% stop loss (positive distance)
    ...     max_periods=10,     # 10 periods max
    ...     side=1              # Long position: profit=102, stop=99 (for entry=100)
    ... )
    >>> labeled = triple_barrier_labels(df, config, timestamp_col="datetime")
    >>> print(labeled[["label", "label_return", "label_bars", "label_duration"]].head())

    >>> # OHLC-based barrier checking for more realistic detection
    >>> labeled = triple_barrier_labels(
    ...     df, config,
    ...     price_col="close",
    ...     high_col="high",
    ...     low_col="low",
    ...     timestamp_col="datetime"
    ... )

    >>> # Short position example
    >>> config = BarrierConfig(
    ...     upper_barrier=0.02,  # 2% profit target (positive distance)
    ...     lower_barrier=0.01,  # 1% stop loss (positive distance)
    ...     max_periods=10,     # 10 periods max
    ...     side=-1             # Short position: profit=98, stop=101 (for entry=100)
    ... )
    >>> labeled = triple_barrier_labels(df, config, timestamp_col="datetime")

    >>> # Duration analysis
    >>> config = BarrierConfig(upper_barrier=0.02, lower_barrier=0.01, max_periods=20)
    >>> labeled = triple_barrier_labels(df, config, timestamp_col="datetime")
    >>> print(f"Average bars held: {labeled['label_bars'].mean():.1f}")
    >>> print(f"Average time held: {labeled['label_duration'].mean()}")

    >>> # Dynamic barriers based on volatility
    >>> config = BarrierConfig(
    ...     upper_barrier="volatility_2x",  # Column name
    ...     lower_barrier="volatility_1x",  # Column name (always positive values)
    ...     max_periods=20
    ... )
    >>> labeled = triple_barrier_labels(df, config, timestamp_col="datetime")
    """
    # ==========================================================================
    # Step 1: Validate inputs and determine events
    # ==========================================================================
    if price_col not in data.columns:
        raise DataValidationError(f"Price column '{price_col}' not found in data")

    # Handle event-based vs regular labeling
    if "event_time" in data.columns:
        event_mask = data["event_time"].is_not_null()
        event_indices = np.where(event_mask.to_numpy())[0]

        if len(event_indices) == 0:
            return data.with_columns(
                label=pl.lit(None, dtype=pl.Int32),
                label_time=pl.lit(None, dtype=pl.Int64),
                label_price=pl.lit(None, dtype=pl.Float64),
                label_return=pl.lit(None, dtype=pl.Float64),
                weight=pl.lit(None, dtype=pl.Float64),
            )
    else:
        event_indices = np.arange(len(data))

    # ==========================================================================
    # Step 2: Extract price data (close, high, low)
    # ==========================================================================
    closes = data[price_col].to_numpy()

    if high_col is not None:
        if high_col not in data.columns:
            raise DataValidationError(f"High column '{high_col}' not found in data")
        highs = data[high_col].to_numpy()
    else:
        highs = closes

    if low_col is not None:
        if low_col not in data.columns:
            raise DataValidationError(f"Low column '{low_col}' not found in data")
        lows = data[low_col].to_numpy()
    else:
        lows = closes

    # ==========================================================================
    # Step 3: Prepare barrier arrays from config
    # ==========================================================================
    upper_barriers, lower_barriers, max_periods, sides, trailing_stops = _prepare_barrier_arrays(
        data, config, event_indices
    )

    # ==========================================================================
    # Step 4: Apply triple barrier labeling (core numba function)
    # ==========================================================================
    labels, label_indices, label_prices, label_returns, bar_durations = _apply_triple_barrier_nb(
        closes,
        highs,
        lows,
        event_indices,
        upper_barriers,
        lower_barriers,
        max_periods,
        sides,
        trailing_stops,
    )

    # ==========================================================================
    # Step 5: Calculate uniqueness and sample weights (optional)
    # ==========================================================================
    if calculate_uniqueness:
        uniqueness = calculate_label_uniqueness(
            event_indices=event_indices,
            label_indices=label_indices,
            n_bars=len(closes),
        )
        sample_weights = calculate_sample_weights(
            uniqueness=uniqueness,
            returns=label_returns,
            weight_scheme=uniqueness_weight_scheme,
        )
    else:
        uniqueness = None
        sample_weights = None

    # ==========================================================================
    # Step 6: Determine barrier hits and calculate durations
    # ==========================================================================
    barrier_hit = _determine_barrier_hits(labels)
    label_times, time_durations = _calculate_time_durations(
        data, event_indices, label_indices, timestamp_col
    )

    # ==========================================================================
    # Step 7: Build result DataFrame
    # ==========================================================================
    return _build_labeling_result(
        data=data,
        event_indices=event_indices,
        labels=labels,
        label_times=label_times,
        label_prices=label_prices,
        label_returns=label_returns,
        bar_durations=bar_durations,
        time_durations=time_durations,
        barrier_hit=barrier_hit,
        uniqueness=uniqueness,
        sample_weights=sample_weights,
    )


def compute_barrier_touches(
    prices: npt.NDArray[np.float64],
    upper_barrier: float,
    lower_barrier: float,
) -> dict[str, Any]:
    """Compute when barriers are first touched.

    Parameters
    ----------
    prices : npt.NDArray
        Array of prices
    upper_barrier : float
        Upper barrier price level
    lower_barrier : float
        Lower barrier price level

    Returns
    -------
    dict
        Dictionary with touch information
    """
    # Find first touch of each barrier
    upper_touches = np.where(prices >= upper_barrier)[0]
    lower_touches = np.where(prices <= lower_barrier)[0]

    first_upper = upper_touches[0] if len(upper_touches) > 0 else None
    first_lower = lower_touches[0] if len(lower_touches) > 0 else None

    # Determine which was touched first
    if first_upper is None and first_lower is None:
        first_touch = None
        barrier_hit = None
    elif first_upper is None:
        first_touch = first_lower
        barrier_hit = "lower"
    elif first_lower is None or first_upper < first_lower:
        first_touch = first_upper
        barrier_hit = "upper"
    else:
        first_touch = first_lower
        barrier_hit = "lower"

    return {
        "first_upper": first_upper,
        "first_lower": first_lower,
        "first_touch": first_touch,
        "barrier_hit": barrier_hit,
    }


def calculate_returns(
    entry_price: float,
    exit_prices: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate returns from entry to exit prices.

    Parameters
    ----------
    entry_price : float
        Entry price
    exit_prices : npt.NDArray
        Array of exit prices

    Returns
    -------
    npt.NDArray
        Array of returns
    """
    return (exit_prices - entry_price) / entry_price


def apply_triple_barrier(
    prices: npt.NDArray[np.float64],
    event_idx: int,
    upper_barrier: float,
    lower_barrier: float,
    max_period: int,
    side: int = 0,
) -> dict[str, Any]:
    """Apply triple barrier to a single event.

    Parameters
    ----------
    prices : npt.NDArray
        Array of all prices
    event_idx : int
        Index of the event
    upper_barrier : float
        Upper barrier percentage
    lower_barrier : float
        Lower barrier percentage
    max_period : int
        Maximum holding period
    side : int
        Position side

    Returns
    -------
    dict
        Labeling results
    """
    event_price = prices[event_idx]

    # Calculate barrier prices
    if side == 1:  # Long
        upper_price = event_price * (1 + upper_barrier)
        lower_price = event_price * (1 - abs(lower_barrier))
    elif side == -1:  # Short
        upper_price = event_price * (1 - upper_barrier)
        lower_price = event_price * (1 + abs(lower_barrier))
    else:  # Symmetric
        upper_price = event_price * (1 + upper_barrier)
        lower_price = event_price * (1 - abs(lower_barrier))

    # Scan forward
    end_idx = min(event_idx + max_period, len(prices))

    for i in range(event_idx + 1, end_idx):
        if prices[i] >= upper_price:
            return {
                "label": 1,
                "label_idx": i,
                "label_price": prices[i],
                "barrier_hit": "upper",
            }
        if prices[i] <= lower_price:
            return {
                "label": -1,
                "label_idx": i,
                "label_price": prices[i],
                "barrier_hit": "lower",
            }

    # Time barrier
    return {
        "label": 0,
        "label_idx": end_idx - 1,
        "label_price": prices[end_idx - 1] if end_idx - 1 < len(prices) else event_price,
        "barrier_hit": "time",
    }


def fixed_time_horizon_labels(
    data: pl.DataFrame,
    horizon: int = 1,
    method: str = "returns",
    price_col: str = "close",
) -> pl.DataFrame:
    """Generate forward-looking labels based on fixed time horizon.

    Creates labels by looking ahead a fixed number of periods and computing
    the return or direction of price movement. Commonly used for supervised
    learning in financial forecasting.

    Parameters
    ----------
    data : pl.DataFrame
        Input data with price information
    horizon : int, default 1
        Number of periods to look ahead
    method : str, default "returns"
        Labeling method:
        - "returns": (price[t+h] - price[t]) / price[t]
        - "log_returns": log(price[t+h] / price[t])
        - "binary": 1 if price[t+h] > price[t] else -1
    price_col : str, default "close"
        Name of the price column to use

    Returns
    -------
    pl.DataFrame
        Original data with additional label column.
        Last `horizon` values will be null (insufficient future data).

    Examples
    --------
    >>> # Simple returns over 5-period horizon
    >>> labeled = fixed_time_horizon_labels(df, horizon=5, method="returns")
    >>>
    >>> # Binary classification (up/down)
    >>> labeled = fixed_time_horizon_labels(df, horizon=1, method="binary")
    >>>
    >>> # Log returns for ML training
    >>> labeled = fixed_time_horizon_labels(df, horizon=10, method="log_returns")

    Notes
    -----
    This is a simple labeling method that:
    - Uses future information (forward-looking)
    - Cannot be used for live prediction (requires future data)
    - Best for supervised learning model training
    - Last `horizon` rows will have null labels

    References
    ----------
    .. [1] De Prado, M.L. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 3: Labeling.

    See Also
    --------
    triple_barrier_labels : Path-dependent labeling with profit/loss targets
    trend_scanning_labels : De Prado's trend scanning method
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if method not in ["returns", "log_returns", "binary"]:
        raise ValueError(f"Unknown method: {method}. Use 'returns', 'log_returns', or 'binary'")

    if price_col not in data.columns:
        raise DataValidationError(f"Column '{price_col}' not found in data")

    # Get price column
    prices = pl.col(price_col)
    future_prices = prices.shift(-horizon)

    # Compute label based on method
    if method == "returns":
        label = (future_prices - prices) / prices
        label_name = f"label_return_{horizon}p"
    elif method == "log_returns":
        label = (future_prices / prices).log()
        label_name = f"label_log_return_{horizon}p"
    elif method == "binary":
        # 1 if price goes up, -1 if down, null if no change or no data
        label = (
            pl.when(future_prices > prices)
            .then(1)
            .when(future_prices < prices)
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
        )
        label_name = f"label_direction_{horizon}p"

    # Add label column to data
    return data.with_columns(label.alias(label_name))


def trend_scanning_labels(
    data: pl.DataFrame,
    min_window: int = 5,
    max_window: int = 50,
    step: int = 1,
    price_col: str = "close",
) -> pl.DataFrame:
    """Generate labels using De Prado's trend scanning method.

    For each observation, fits linear trends over windows of varying lengths
    and selects the window with the highest absolute t-statistic. The label
    is assigned based on the trend direction (sign of the t-statistic).

    This method is more robust than fixed-horizon labeling as it adapts to
    the local trend structure in the data.

    Parameters
    ----------
    data : pl.DataFrame
        Input data with price information
    min_window : int, default 5
        Minimum window size to scan
    max_window : int, default 50
        Maximum window size to scan
    step : int, default 1
        Step size for window scanning
    price_col : str, default "close"
        Name of the price column to use

    Returns
    -------
    pl.DataFrame
        Original data with additional columns:
        - label: ±1 based on trend direction
        - t_value: t-statistic of the selected trend
        - optimal_window: window size with highest |t-value|

    Examples
    --------
    >>> # Scan windows from 5 to 50 bars
    >>> labeled = trend_scanning_labels(df, min_window=5, max_window=50)
    >>>
    >>> # Fast scanning with larger steps
    >>> labeled = trend_scanning_labels(df, min_window=10, max_window=100, step=5)

    Notes
    -----
    The trend scanning method:
    1. For each observation, scans forward with windows of varying lengths
    2. Fits a linear regression to each window
    3. Computes t-statistic for the slope coefficient
    4. Selects the window with highest absolute t-statistic
    5. Assigns label = sign(t-statistic)

    This approach:
    - Adapts to local trend structure
    - More robust than fixed horizons
    - Computationally expensive (O(n * m) where m = window range)

    References
    ----------
    .. [1] De Prado, M.L. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 18: Entropy Features (Section on Trend Scanning).

    See Also
    --------
    fixed_time_horizon_labels : Simple fixed-horizon labeling
    triple_barrier_labels : Path-dependent labeling with barriers
    """
    from scipy import stats

    if min_window < 2:
        raise ValueError("min_window must be at least 2")
    if max_window <= min_window:
        raise ValueError("max_window must be greater than min_window")
    if step < 1:
        raise ValueError("step must be at least 1")
    if price_col not in data.columns:
        raise DataValidationError(f"Column '{price_col}' not found in data")

    # Extract prices as numpy array for faster computation
    prices = data[price_col].to_numpy()
    n = len(prices)

    # Initialize result arrays
    labels = np.full(n, np.nan)
    t_values = np.full(n, np.nan)
    windows = np.full(n, np.nan)

    # Scan each observation
    for i in range(n - min_window):
        best_t = 0.0
        best_window = min_window

        # Scan windows of different lengths
        for window in range(min_window, min(max_window, n - i), step):
            # Extract window
            window_prices = prices[i : i + window]
            x = np.arange(window)
            y = window_prices

            # Fit linear regression
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                # Compute t-statistic
                t_stat = slope / std_err if std_err > 0 else 0.0

                # Keep window with highest |t|
                if abs(t_stat) > abs(best_t):
                    best_t = t_stat
                    best_window = window
            except (ValueError, RuntimeError):
                # Handle numerical issues
                continue

        # Assign label based on trend direction
        labels[i] = 1 if best_t > 0 else -1
        t_values[i] = best_t
        windows[i] = best_window

    # Add results to dataframe
    # Convert NaN to None for Polars compatibility
    label_series = pl.Series("label", labels)
    label_series = label_series.fill_nan(None).cast(pl.Int8)

    t_value_series = pl.Series("t_value", t_values)
    window_series = pl.Series("optimal_window", windows).fill_nan(None).cast(pl.Int32)

    return data.with_columns([label_series, t_value_series, window_series])


__all__ = [
    "apply_triple_barrier",
    "calculate_returns",
    "compute_barrier_touches",
    "triple_barrier_labels",
    "fixed_time_horizon_labels",
    "trend_scanning_labels",
    # De Prado's uniqueness weighting (AFML Ch. 4)
    "calculate_label_uniqueness",
    "calculate_sample_weights",
    "sequential_bootstrap",
]
