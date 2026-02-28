# mypy: disable-error-code="no-any-return,arg-type,call-arg,return-value,assignment"
"""Numba JIT-compiled operations for triple barrier labeling.

These are internal functions used by the triple barrier implementation.
Do not import directly - use the public API from labeling modules.
"""

import numpy as np
import numpy.typing as npt
from numba import jit


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
        # Treat side==0 (symmetric) as long-like for trailing stop
        if side == 1 or side == 0:
            return event_price * (1 - trailing_stop)
        return event_price * (1 + trailing_stop)
    # For disabled trailing stop, return inf that won't trigger
    return float(-np.inf) if side == 1 or side == 0 else float(np.inf)


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
        # Treat side==0 (symmetric) as long-like for trailing stop
        if side == 1 or side == 0:
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
    # Side==0 (symmetric) is treated as long-like for trailing stop
    lower_hit = (
        (side == 1 and (low_price <= lower_price or low_price <= trailing_stop_price))
        or (side == -1 and (high_price >= lower_price or high_price >= trailing_stop_price))
        or (side == 0 and (low_price <= lower_price or low_price <= trailing_stop_price))
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
    # Use max_period + 1 to include the vertical barrier bar (max holding period is inclusive)
    end_idx = min(event_idx + max_period + 1, n_prices)

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


__all__ = [
    "_calculate_barrier_prices",
    "_initialize_trailing_stop",
    "_update_trailing_stop",
    "_check_barrier_touch",
    "_calculate_label_return",
    "_process_single_event",
    "_apply_triple_barrier_nb",
    "_build_concurrency_nb",
]
