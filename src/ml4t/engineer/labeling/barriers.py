"""Barrier configurations for labeling.

Defines various barrier types for the generalized labeling framework.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any


@dataclass
class BarrierConfig:
    """Configuration for barrier-based labeling.

    All barrier distances are specified as POSITIVE values representing
    the distance from the entry price. The position side determines
    the direction of the barriers.

    Parameters
    ----------
    upper_barrier : float or str
        Upper barrier distance (positive value) or column name for dynamic barriers.
        For long positions: profit target above entry price.
        For short positions: profit target below entry price.
    lower_barrier : float or str
        Lower barrier distance (positive value) or column name for dynamic barriers.
        For long positions: stop loss below entry price.
        For short positions: stop loss above entry price.
    max_holding_period : int or str or timedelta
        Maximum holding period:
        - int: Number of bars
        - str: Duration string (e.g., '4h', '1d') or column name
        - timedelta: Python timedelta object

        Time-based values are converted to per-event bar counts during labeling.
    side : int or str or None
        Position side: 1 (long), -1 (short), 0/None (symmetric, assumes long-like)
    trailing_stop : float or str or None
        Trailing stop percentage (positive value) or column name
    weight_scheme : str or callable
        Weighting scheme: "equal", "returns", "time_decay", or custom function
    weight_decay_rate : float
        Decay rate for time-based weighting

    Examples
    --------
    >>> # Long position: profit at 102, stop at 99 (for entry at 100)
    >>> config = BarrierConfig(upper_barrier=0.02, lower_barrier=0.01, side=1)

    >>> # Short position: profit at 98, stop at 101 (for entry at 100)
    >>> config = BarrierConfig(upper_barrier=0.02, lower_barrier=0.01, side=-1)

    >>> # Time-based max holding period (4 hours)
    >>> config = BarrierConfig(
    ...     upper_barrier=0.02,
    ...     lower_barrier=0.01,
    ...     max_holding_period="4h",  # Duration string
    ...     side=1,
    ... )

    >>> # Using timedelta
    >>> from datetime import timedelta
    >>> config = BarrierConfig(
    ...     upper_barrier=0.02,
    ...     lower_barrier=0.01,
    ...     max_holding_period=timedelta(hours=4),
    ...     side=1,
    ... )
    """

    upper_barrier: float | str | None = None
    lower_barrier: float | str | None = None
    max_holding_period: int | str | timedelta = 10
    side: int | str | None = None
    trailing_stop: bool | float | str = False
    weight_scheme: str | Callable[..., Any] = "equal"
    weight_decay_rate: float = 0.1


@dataclass
class ATRBarrierConfig:
    """ATR-adjusted barrier configuration for volatility-adaptive labeling.

    Instead of fixed percentage thresholds, barriers are defined as multiples
    of the Average True Range (ATR), making them automatically adapt to
    volatility regimes.

    Innovation from ML4Trading Research Team (2024-2025).

    Parameters
    ----------
    upper_multiplier : float
        ATR multiplier for upper barrier (e.g., 2.0 = 2x ATR above entry)
    lower_multiplier : float
        ATR multiplier for lower barrier (e.g., 1.5 = 1.5x ATR below entry)
    atr_period : int
        Lookback period for ATR calculation (default: 14, Wilder standard)
    max_holding_period : int or str
        Maximum holding period in bars or column name
    side : int or None
        Position side: 1 (long), -1 (short), 0/None (symmetric)
    weight_scheme : str or callable
        Weighting scheme: "equal", "returns", "time_decay", or custom function
    weight_decay_rate : float
        Decay rate for time-based weighting

    Examples
    --------
    >>> # ATR-adjusted barriers: 2x ATR profit, 1.5x ATR stop
    >>> config = ATRBarrierConfig(
    ...     upper_multiplier=2.0,
    ...     lower_multiplier=1.5,
    ...     atr_period=14,
    ...     side=1
    ... )

    Notes
    -----
    ATR barriers automatically adapt to market conditions:
    - In low volatility (ATR=0.5%): barriers at ±1% and ±0.75%
    - In high volatility (ATR=2%): barriers at ±4% and ±3%

    This prevents overtrading in volatile markets and captures more signals
    in calm markets.

    References
    ----------
    Wilder, J. W. (1978). New Concepts in Technical Trading Systems.
    """

    upper_multiplier: float = 2.0
    lower_multiplier: float = 1.5
    atr_period: int = 14
    max_holding_period: int | str = 20
    side: int | None = None
    weight_scheme: str | Callable[..., Any] = "equal"
    weight_decay_rate: float = 0.1


__all__ = ["ATRBarrierConfig", "BarrierConfig"]
