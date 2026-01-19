# mypy: disable-error-code="misc,operator,assignment,arg-type"
"""Run bar sampler implementations.

AFML-compliant run bars per López de Prado Chapter 2.3.

Run bar threshold formula:
    E[θ_T] = E[T] × max{P[b=1], 1-P[b=1]}

Where:
    E[T] = EWMA of bar lengths (ticks per bar)
    P[b=1] = buy probability

CRITICAL: θ_T = max{Σ(all buys in bar), Σ(all sells in bar)}
    - NOT consecutive same-sided trades
    - NO reset on direction change within bar

Exports:
    TickRunBarSampler(expected_ticks_per_bar=50, alpha=0.1) -> BarSampler
        Run bars based on cumulative trade count.

    VolumeRunBarSampler(expected_ticks_per_bar=100, alpha=0.1) -> BarSampler
        Volume-weighted run bars.

    DollarRunBarSampler(expected_ticks_per_bar=100, alpha=0.1) -> BarSampler
        Dollar-weighted run bars.

Based on Advances in Financial Machine Learning by Marcos López de Prado.
"""

import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.bars.base import BarSampler
from ml4t.engineer.core.exceptions import DataValidationError


@jit(nopython=True, cache=True)
def _calculate_run_bars_nb(
    values: npt.NDArray[np.float64],
    sides: npt.NDArray[np.float64],
    initial_expected_t: float,
    initial_p_buy: float,
    alpha: float = 0.1,
    min_bars_warmup: int = 10,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Calculate AFML-compliant run bar indices.

    AFML Chapter 2.3 formula:
        θ_T = max{Σ(all buys in bar), Σ(all sells in bar)}
        E[θ_T] = E[T] × max{P[b=1], 1-P[b=1]}

    CRITICAL: This uses CUMULATIVE counts within the bar.
    Direction changes DO NOT reset the counts.

    Parameters
    ----------
    values : ndarray
        Array of values to accumulate (ticks=1, volumes, or dollar amounts)
    sides : ndarray
        Array of trade signs (+1 for buy, -1 for sell)
    initial_expected_t : float
        Initial expected ticks per bar (E[T])
    initial_p_buy : float
        Initial buy probability P[b=1]
    alpha : float
        EWMA decay factor for updating expectations
    min_bars_warmup : int
        Number of bars before starting EWMA updates

    Returns
    -------
    tuple of arrays
        (bar_indices, thetas, expected_thetas, expected_ts, p_buys,
         cumulative_buys, cumulative_sells)
    """
    n = len(values)

    # Pre-allocate output lists
    bar_indices = []
    thetas = []
    expected_thetas = []
    expected_ts = []
    p_buys = []
    cumulative_buys_out = []
    cumulative_sells_out = []

    # Initialize EWMA state
    expected_t = initial_expected_t
    p_buy = initial_p_buy

    # Within-bar accumulators - CUMULATIVE, not consecutive
    cumulative_buys = 0.0
    cumulative_sells = 0.0
    bar_tick_count = 0
    bar_buy_count = 0

    n_bars = 0

    for i in range(n):
        val = values[i]
        side = sides[i]
        is_buy = side > 0

        # Accumulate - NEVER reset on direction change
        bar_tick_count += 1
        if is_buy:
            cumulative_buys += val
            bar_buy_count += 1
        else:
            cumulative_sells += val

        # θ = max of cumulative buys and sells (NOT consecutive)
        theta = max(cumulative_buys, cumulative_sells)

        # Threshold: E[T] × max{P[b=1], 1-P[b=1]}
        expected_theta = expected_t * max(p_buy, 1 - p_buy)

        # Check if bar should be formed
        if theta >= expected_theta:
            # Record bar
            bar_indices.append(i)
            thetas.append(theta)
            expected_thetas.append(expected_theta)
            expected_ts.append(expected_t)
            p_buys.append(p_buy)
            cumulative_buys_out.append(cumulative_buys)
            cumulative_sells_out.append(cumulative_sells)

            n_bars += 1

            # Update EWMAs after warmup period
            if n_bars > min_bars_warmup:
                # Update E[T] - expected ticks per bar
                expected_t = alpha * bar_tick_count + (1 - alpha) * expected_t

                # Update P[b=1] - buy probability
                bar_p_buy = bar_buy_count / bar_tick_count if bar_tick_count > 0 else 0.5
                p_buy = alpha * bar_p_buy + (1 - alpha) * p_buy

            # Reset within-bar accumulators - ONLY at bar boundary
            cumulative_buys = 0.0
            cumulative_sells = 0.0
            bar_tick_count = 0
            bar_buy_count = 0

    return (
        np.array(bar_indices, dtype=np.int64),
        np.array(thetas, dtype=np.float64),
        np.array(expected_thetas, dtype=np.float64),
        np.array(expected_ts, dtype=np.float64),
        np.array(p_buys, dtype=np.float64),
        np.array(cumulative_buys_out, dtype=np.float64),
        np.array(cumulative_sells_out, dtype=np.float64),
    )


class TickRunBarSampler(BarSampler):
    """Sample bars based on cumulative tick runs (AFML-compliant).

    AFML Chapter 2.3 formula:
        θ_T = max{Σ(all buys in bar), Σ(all sells in bar)}
        E[θ_T] = E[T] × max{P[b=1], 1-P[b=1]}

    CRITICAL: Uses CUMULATIVE tick counts within the bar.
    Direction changes DO NOT reset the counts - only bar boundaries do.

    Parameters
    ----------
    expected_ticks_per_bar : int
        Expected number of ticks per bar (used to initialize E[T])
    initial_run_expectation : int, optional
        DEPRECATED. Use expected_ticks_per_bar instead.
    alpha : float, default 0.1
        EWMA decay factor for updating expectations
    initial_p_buy : float, default 0.5
        Initial buy probability P[b=1]
    min_bars_warmup : int, default 10
        Number of bars before starting EWMA updates

    Examples
    --------
    >>> sampler = TickRunBarSampler(expected_ticks_per_bar=100)
    >>> bars = sampler.sample(tick_data)

    References
    ----------
    .. [1] López de Prado, M. (2018). Advances in Financial Machine Learning.
           John Wiley & Sons. Chapter 2.3: Information-Driven Bars.
    """

    def __init__(
        self,
        expected_ticks_per_bar: int,
        initial_run_expectation: int | None = None,
        alpha: float = 0.1,
        initial_p_buy: float = 0.5,
        min_bars_warmup: int = 10,
    ):
        if expected_ticks_per_bar <= 0:
            raise ValueError("expected_ticks_per_bar must be positive")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= initial_p_buy <= 1:
            raise ValueError("initial_p_buy must be in [0, 1]")
        if min_bars_warmup < 0:
            raise ValueError("min_bars_warmup must be non-negative")

        if initial_run_expectation is not None:
            warnings.warn(
                "initial_run_expectation is deprecated and ignored. "
                "The AFML threshold E[T] × max{P[b=1], 1-P[b=1]} is computed dynamically.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.expected_ticks_per_bar = expected_ticks_per_bar
        self.initial_run_expectation = initial_run_expectation  # Keep for compat
        self.alpha = alpha
        self.initial_p_buy = initial_p_buy
        self.min_bars_warmup = min_bars_warmup

    def sample(
        self,
        data: pl.DataFrame,
        include_incomplete: bool = False,
    ) -> pl.DataFrame:
        """Sample tick run bars from data.

        Parameters
        ----------
        data : pl.DataFrame
            Tick data with columns: timestamp, price, volume, side
        include_incomplete : bool, default False
            Whether to include incomplete final bar

        Returns
        -------
        pl.DataFrame
            Sampled run bars with AFML diagnostic columns
        """
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Run bars require 'side' column")

        if len(data) == 0:
            return self._empty_run_bars_df()

        # Extract arrays - for tick runs, values are all 1.0
        n = len(data)
        values = np.ones(n, dtype=np.float64)
        volumes = data["volume"].to_numpy().astype(np.float64)
        sides = data["side"].to_numpy().astype(np.float64)

        # Calculate bar indices using AFML-compliant Numba function
        (
            bar_indices,
            thetas,
            expected_thetas,
            expected_ts,
            p_buys,
            cumulative_buys,
            cumulative_sells,
        ) = _calculate_run_bars_nb(
            values,
            sides,
            float(self.expected_ticks_per_bar),
            self.initial_p_buy,
            self.alpha,
            self.min_bars_warmup,
        )

        # Build bars
        bars = []
        start_idx = 0

        for i, end_idx in enumerate(bar_indices):
            bar_ticks = data.slice(start_idx, end_idx - start_idx + 1)
            bar_volumes = volumes[start_idx : end_idx + 1]
            bar_sides = sides[start_idx : end_idx + 1]

            buy_volume = float(np.sum(bar_volumes[bar_sides > 0]))
            sell_volume = float(np.sum(bar_volumes[bar_sides < 0]))

            bar = self._create_ohlcv_bar(
                bar_ticks,
                additional_cols={
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "run_length": int(thetas[i]),  # For backward compat
                    "expected_run": float(expected_thetas[i]),  # For backward compat
                    # AFML diagnostic columns
                    "theta": float(thetas[i]),
                    "expected_theta": float(expected_thetas[i]),
                    "expected_t": float(expected_ts[i]),
                    "p_buy": float(p_buys[i]),
                    "cumulative_buys": float(cumulative_buys[i]),
                    "cumulative_sells": float(cumulative_sells[i]),
                },
            )
            bars.append(bar)

            start_idx = end_idx + 1

        # Handle incomplete bar
        if include_incomplete and start_idx < len(data):
            bar_ticks = data.slice(start_idx)

            if len(bar_ticks) > 0:
                bar_volumes = volumes[start_idx:]
                bar_sides = sides[start_idx:]

                buy_vol = float(np.sum(bar_volumes[bar_sides > 0]))
                sell_vol = float(np.sum(bar_volumes[bar_sides < 0]))

                # Calculate current theta (cumulative, not consecutive)
                cum_buys = float(np.sum(bar_sides > 0))
                cum_sells = float(np.sum(bar_sides < 0))
                current_theta = max(cum_buys, cum_sells)

                last_expected_t = (
                    expected_ts[-1] if len(expected_ts) > 0 else float(self.expected_ticks_per_bar)
                )
                last_p_buy = p_buys[-1] if len(p_buys) > 0 else self.initial_p_buy
                expected_theta = last_expected_t * max(last_p_buy, 1 - last_p_buy)

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": buy_vol,
                        "sell_volume": sell_vol,
                        "run_length": int(current_theta),
                        "expected_run": float(expected_theta),
                        "theta": float(current_theta),
                        "expected_theta": float(expected_theta),
                        "expected_t": float(last_expected_t),
                        "p_buy": float(last_p_buy),
                        "cumulative_buys": float(cum_buys),
                        "cumulative_sells": float(cum_sells),
                    },
                )
                bars.append(bar)

        if not bars:
            return self._empty_run_bars_df()

        return pl.DataFrame(bars)

    def _empty_run_bars_df(self) -> pl.DataFrame:
        """Return empty DataFrame with correct schema."""
        return pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "tick_count": [],
                "buy_volume": [],
                "sell_volume": [],
                "run_length": [],
                "expected_run": [],
                "theta": [],
                "expected_theta": [],
                "expected_t": [],
                "p_buy": [],
                "cumulative_buys": [],
                "cumulative_sells": [],
            },
        )


class VolumeRunBarSampler(BarSampler):
    """Sample bars based on cumulative volume runs (AFML-compliant).

    AFML Chapter 2.3 formula with volume weighting:
        θ_T = max{Σ(buy volumes in bar), Σ(sell volumes in bar)}
        E[θ_T] = E[T] × max{P[b=1], 1-P[b=1]} × E[v]

    Where E[v] is estimated from the data.

    Parameters
    ----------
    expected_ticks_per_bar : int
        Expected number of ticks per bar
    initial_run_expectation : float, optional
        DEPRECATED. Threshold is computed dynamically.
    alpha : float, default 0.1
        EWMA decay factor
    initial_p_buy : float, default 0.5
        Initial buy probability P[b=1]
    min_bars_warmup : int, default 10
        Number of bars before starting EWMA updates

    Examples
    --------
    >>> sampler = VolumeRunBarSampler(expected_ticks_per_bar=100)
    >>> bars = sampler.sample(tick_data)
    """

    def __init__(
        self,
        expected_ticks_per_bar: int,
        initial_run_expectation: float | None = None,
        alpha: float = 0.1,
        initial_p_buy: float = 0.5,
        min_bars_warmup: int = 10,
    ):
        if expected_ticks_per_bar <= 0:
            raise ValueError("expected_ticks_per_bar must be positive")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= initial_p_buy <= 1:
            raise ValueError("initial_p_buy must be in [0, 1]")
        if min_bars_warmup < 0:
            raise ValueError("min_bars_warmup must be non-negative")

        if initial_run_expectation is not None:
            warnings.warn(
                "initial_run_expectation is deprecated and ignored. "
                "The AFML threshold is computed dynamically.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.expected_ticks_per_bar = expected_ticks_per_bar
        self.initial_run_expectation = initial_run_expectation
        self.alpha = alpha
        self.initial_p_buy = initial_p_buy
        self.min_bars_warmup = min_bars_warmup

    def sample(
        self,
        data: pl.DataFrame,
        include_incomplete: bool = False,
    ) -> pl.DataFrame:
        """Sample volume run bars from data."""
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Run bars require 'side' column")

        if len(data) == 0:
            return self._empty_run_bars_df()

        volumes = data["volume"].to_numpy().astype(np.float64)
        sides = data["side"].to_numpy().astype(np.float64)

        # Estimate initial E[v] for scaling the threshold
        warmup_size = min(1000, len(volumes))
        avg_volume = float(np.mean(volumes[:warmup_size]))

        # Scale expected_ticks_per_bar by average volume for volume-weighted runs
        initial_expected_t_scaled = float(self.expected_ticks_per_bar) * avg_volume

        # Calculate bar indices using AFML-compliant Numba function
        (
            bar_indices,
            thetas,
            expected_thetas,
            expected_ts,
            p_buys,
            cumulative_buys,
            cumulative_sells,
        ) = _calculate_run_bars_nb(
            volumes,  # Use volumes as values
            sides,
            initial_expected_t_scaled,
            self.initial_p_buy,
            self.alpha,
            self.min_bars_warmup,
        )

        # Build bars
        bars = []
        start_idx = 0

        for i, end_idx in enumerate(bar_indices):
            bar_ticks = data.slice(start_idx, end_idx - start_idx + 1)
            bar_volumes = volumes[start_idx : end_idx + 1]
            bar_sides = sides[start_idx : end_idx + 1]

            buy_volume = float(np.sum(bar_volumes[bar_sides > 0]))
            sell_volume = float(np.sum(bar_volumes[bar_sides < 0]))

            bar = self._create_ohlcv_bar(
                bar_ticks,
                additional_cols={
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "run_volume": float(thetas[i]),  # For backward compat
                    "expected_run": float(expected_thetas[i]),  # For backward compat
                    # AFML diagnostic columns
                    "theta": float(thetas[i]),
                    "expected_theta": float(expected_thetas[i]),
                    "expected_t": float(expected_ts[i]),
                    "p_buy": float(p_buys[i]),
                    "cumulative_buys": float(cumulative_buys[i]),
                    "cumulative_sells": float(cumulative_sells[i]),
                },
            )
            bars.append(bar)

            start_idx = end_idx + 1

        # Handle incomplete bar
        if include_incomplete and start_idx < len(data):
            bar_ticks = data.slice(start_idx)

            if len(bar_ticks) > 0:
                bar_volumes = volumes[start_idx:]
                bar_sides = sides[start_idx:]

                buy_vol = float(np.sum(bar_volumes[bar_sides > 0]))
                sell_vol = float(np.sum(bar_volumes[bar_sides < 0]))

                # Calculate current theta (cumulative volumes)
                current_theta = max(buy_vol, sell_vol)

                last_expected_t = (
                    expected_ts[-1] if len(expected_ts) > 0 else initial_expected_t_scaled
                )
                last_p_buy = p_buys[-1] if len(p_buys) > 0 else self.initial_p_buy
                expected_theta = last_expected_t * max(last_p_buy, 1 - last_p_buy)

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": buy_vol,
                        "sell_volume": sell_vol,
                        "run_volume": float(current_theta),
                        "expected_run": float(expected_theta),
                        "theta": float(current_theta),
                        "expected_theta": float(expected_theta),
                        "expected_t": float(last_expected_t),
                        "p_buy": float(last_p_buy),
                        "cumulative_buys": float(buy_vol),
                        "cumulative_sells": float(sell_vol),
                    },
                )
                bars.append(bar)

        if not bars:
            return self._empty_run_bars_df()

        return pl.DataFrame(bars)

    def _empty_run_bars_df(self) -> pl.DataFrame:
        """Return empty DataFrame with correct schema."""
        return pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "tick_count": [],
                "buy_volume": [],
                "sell_volume": [],
                "run_volume": [],
                "expected_run": [],
                "theta": [],
                "expected_theta": [],
                "expected_t": [],
                "p_buy": [],
                "cumulative_buys": [],
                "cumulative_sells": [],
            },
        )


class DollarRunBarSampler(BarSampler):
    """Sample bars based on cumulative dollar value runs (AFML-compliant).

    AFML Chapter 2.3 formula with dollar weighting:
        θ_T = max{Σ(buy dollars in bar), Σ(sell dollars in bar)}

    Parameters
    ----------
    expected_ticks_per_bar : int
        Expected number of ticks per bar
    initial_run_expectation : float, optional
        DEPRECATED. Threshold is computed dynamically.
    alpha : float, default 0.1
        EWMA decay factor
    initial_p_buy : float, default 0.5
        Initial buy probability P[b=1]
    min_bars_warmup : int, default 10
        Number of bars before starting EWMA updates

    Examples
    --------
    >>> sampler = DollarRunBarSampler(expected_ticks_per_bar=100)
    >>> bars = sampler.sample(tick_data)
    """

    def __init__(
        self,
        expected_ticks_per_bar: int,
        initial_run_expectation: float | None = None,
        alpha: float = 0.1,
        initial_p_buy: float = 0.5,
        min_bars_warmup: int = 10,
    ):
        if expected_ticks_per_bar <= 0:
            raise ValueError("expected_ticks_per_bar must be positive")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= initial_p_buy <= 1:
            raise ValueError("initial_p_buy must be in [0, 1]")
        if min_bars_warmup < 0:
            raise ValueError("min_bars_warmup must be non-negative")

        if initial_run_expectation is not None:
            warnings.warn(
                "initial_run_expectation is deprecated and ignored. "
                "The AFML threshold is computed dynamically.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.expected_ticks_per_bar = expected_ticks_per_bar
        self.initial_run_expectation = initial_run_expectation
        self.alpha = alpha
        self.initial_p_buy = initial_p_buy
        self.min_bars_warmup = min_bars_warmup

    def sample(
        self,
        data: pl.DataFrame,
        include_incomplete: bool = False,
    ) -> pl.DataFrame:
        """Sample dollar run bars from data."""
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Run bars require 'side' column")

        if len(data) == 0:
            return self._empty_run_bars_df()

        prices = data["price"].to_numpy().astype(np.float64)
        volumes = data["volume"].to_numpy().astype(np.float64)
        sides = data["side"].to_numpy().astype(np.float64)
        dollar_volumes = prices * volumes

        # Estimate initial E[dollar] for scaling
        warmup_size = min(1000, len(dollar_volumes))
        avg_dollar_volume = float(np.mean(dollar_volumes[:warmup_size]))

        # Scale expected_ticks_per_bar by average dollar volume
        initial_expected_t_scaled = float(self.expected_ticks_per_bar) * avg_dollar_volume

        # Calculate bar indices using AFML-compliant Numba function
        (
            bar_indices,
            thetas,
            expected_thetas,
            expected_ts,
            p_buys,
            cumulative_buys,
            cumulative_sells,
        ) = _calculate_run_bars_nb(
            dollar_volumes,  # Use dollar volumes as values
            sides,
            initial_expected_t_scaled,
            self.initial_p_buy,
            self.alpha,
            self.min_bars_warmup,
        )

        # Build bars
        bars = []
        start_idx = 0

        for i, end_idx in enumerate(bar_indices):
            bar_ticks = data.slice(start_idx, end_idx - start_idx + 1)
            bar_volumes = volumes[start_idx : end_idx + 1]
            bar_sides = sides[start_idx : end_idx + 1]
            bar_dollars = dollar_volumes[start_idx : end_idx + 1]

            buy_volume = float(np.sum(bar_volumes[bar_sides > 0]))
            sell_volume = float(np.sum(bar_volumes[bar_sides < 0]))
            total_dollars = float(np.sum(bar_dollars))
            total_volume = float(np.sum(bar_volumes))
            vwap = total_dollars / total_volume if total_volume > 0 else 0.0

            bar = self._create_ohlcv_bar(
                bar_ticks,
                additional_cols={
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "dollar_volume": total_dollars,
                    "vwap": vwap,
                    "run_dollars": float(thetas[i]),  # For backward compat
                    "expected_run": float(expected_thetas[i]),  # For backward compat
                    # AFML diagnostic columns
                    "theta": float(thetas[i]),
                    "expected_theta": float(expected_thetas[i]),
                    "expected_t": float(expected_ts[i]),
                    "p_buy": float(p_buys[i]),
                    "cumulative_buys": float(cumulative_buys[i]),
                    "cumulative_sells": float(cumulative_sells[i]),
                },
            )
            bars.append(bar)

            start_idx = end_idx + 1

        # Handle incomplete bar
        if include_incomplete and start_idx < len(data):
            bar_ticks = data.slice(start_idx)

            if len(bar_ticks) > 0:
                bar_volumes = volumes[start_idx:]
                bar_sides = sides[start_idx:]
                bar_dollars = dollar_volumes[start_idx:]

                buy_vol = float(np.sum(bar_volumes[bar_sides > 0]))
                sell_vol = float(np.sum(bar_volumes[bar_sides < 0]))
                total_dol = float(np.sum(bar_dollars))
                total_vol = float(np.sum(bar_volumes))
                vwap_val = total_dol / total_vol if total_vol > 0 else 0.0

                # Calculate current theta (cumulative dollars)
                buy_dollars = float(np.sum(bar_dollars[bar_sides > 0]))
                sell_dollars = float(np.sum(bar_dollars[bar_sides < 0]))
                current_theta = max(buy_dollars, sell_dollars)

                last_expected_t = (
                    expected_ts[-1] if len(expected_ts) > 0 else initial_expected_t_scaled
                )
                last_p_buy = p_buys[-1] if len(p_buys) > 0 else self.initial_p_buy
                expected_theta = last_expected_t * max(last_p_buy, 1 - last_p_buy)

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": buy_vol,
                        "sell_volume": sell_vol,
                        "dollar_volume": total_dol,
                        "vwap": vwap_val,
                        "run_dollars": float(current_theta),
                        "expected_run": float(expected_theta),
                        "theta": float(current_theta),
                        "expected_theta": float(expected_theta),
                        "expected_t": float(last_expected_t),
                        "p_buy": float(last_p_buy),
                        "cumulative_buys": float(buy_dollars),
                        "cumulative_sells": float(sell_dollars),
                    },
                )
                bars.append(bar)

        if not bars:
            return self._empty_run_bars_df()

        return pl.DataFrame(bars)

    def _empty_run_bars_df(self) -> pl.DataFrame:
        """Return empty DataFrame with correct schema."""
        return pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "tick_count": [],
                "buy_volume": [],
                "sell_volume": [],
                "dollar_volume": [],
                "vwap": [],
                "run_dollars": [],
                "expected_run": [],
                "theta": [],
                "expected_theta": [],
                "expected_t": [],
                "p_buy": [],
                "cumulative_buys": [],
                "cumulative_sells": [],
            },
        )


__all__ = [
    "DollarRunBarSampler",
    "TickRunBarSampler",
    "VolumeRunBarSampler",
]
