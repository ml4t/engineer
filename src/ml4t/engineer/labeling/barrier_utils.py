# mypy: disable-error-code="no-any-return,arg-type,call-arg,return-value,assignment"
"""Utility functions for barrier calculations.

Simple utility functions for barrier touch detection and returns calculation.
These are lower-level functions typically used for analysis or debugging.
"""

from typing import Any

import numpy as np
import numpy.typing as npt


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
    upper_touches = np.where(prices >= upper_barrier)[0]
    lower_touches = np.where(prices <= lower_barrier)[0]

    first_upper = upper_touches[0] if len(upper_touches) > 0 else None
    first_lower = lower_touches[0] if len(lower_touches) > 0 else None

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
    """Calculate returns from entry to exit prices."""
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

    if side == 1:  # Long
        upper_price = event_price * (1 + upper_barrier)
        lower_price = event_price * (1 - abs(lower_barrier))
    elif side == -1:  # Short
        upper_price = event_price * (1 - upper_barrier)
        lower_price = event_price * (1 + abs(lower_barrier))
    else:  # Symmetric
        upper_price = event_price * (1 + upper_barrier)
        lower_price = event_price * (1 - abs(lower_barrier))

    end_idx = min(event_idx + max_period, len(prices))

    for i in range(event_idx + 1, end_idx):
        if prices[i] >= upper_price:
            return {"label": 1, "label_idx": i, "label_price": prices[i], "barrier_hit": "upper"}
        if prices[i] <= lower_price:
            return {"label": -1, "label_idx": i, "label_price": prices[i], "barrier_hit": "lower"}

    return {
        "label": 0,
        "label_idx": end_idx - 1,
        "label_price": prices[end_idx - 1] if end_idx - 1 < len(prices) else event_price,
        "barrier_hit": "time",
    }


__all__ = [
    "compute_barrier_touches",
    "calculate_returns",
    "apply_triple_barrier",
]
