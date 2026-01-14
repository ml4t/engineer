# mypy: disable-error-code="misc,operator,assignment,arg-type"
"""Run bar sampler implementations.

Exports:
    TickRunBarSampler(initial_run=50, alpha=0.1) -> BarSampler
        Run bars based on consecutive trade count.

    VolumeRunBarSampler(initial_run=5000, alpha=0.1) -> BarSampler
        Volume-weighted run bars.

    DollarRunBarSampler(initial_run=500_000, alpha=0.1) -> BarSampler
        Dollar-weighted run bars.

Run bars sample when a sequence of trades becomes unusually long in terms of
consecutive buys or sells, indicating sustained one-sided market flow.

Based on Advances in Financial Machine Learning by Marcos López de Prado.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.bars.base import BarSampler
from ml4t.engineer.core.exceptions import DataValidationError


@jit(nopython=True, cache=True)
def _calculate_run_bars_nb(
    volumes: npt.NDArray[np.float64],
    sides: npt.NDArray[np.float64],
    initial_expectation: float,
    alpha: float = 0.1,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Calculate run bar indices using Numba.

    A run is a sequence of consecutive trades with the same sign (buy or sell).
    Bars are formed when the length of the current run exceeds an expected threshold.

    Parameters
    ----------
    volumes : npt.NDArray[np.float64]
        Array of volume values
    sides : npt.NDArray[np.float64]
        Array of trade signs (+1 for buy, -1 for sell)
    initial_expectation : float
        Initial expected run length threshold
    alpha : float, default 0.1
        EWMA decay factor for updating expectation

    Returns
    -------
    tuple of arrays
        (bar_end_indices, expected_runs, run_lengths)
    """
    n = len(volumes)
    bar_indices = []
    expected_runs = []
    run_lengths_out = []

    # Initialize
    expected_run = initial_expectation
    current_run_length = 0
    previous_side = 0.0  # No previous side yet

    for i in range(n):
        current_side = sides[i]

        # Check if we're continuing the run or starting a new one
        if current_side == previous_side or previous_side == 0.0:
            # Continue run
            current_run_length += 1
        else:
            # Direction changed - start new run
            current_run_length = 1

        # Check if current run exceeds expectation
        if current_run_length >= expected_run:
            # Record bar end
            bar_indices.append(i)
            run_lengths_out.append(current_run_length)
            expected_runs.append(expected_run)

            # Update expected run using EWMA
            # E[T] = alpha * T + (1 - alpha) * E[T]
            expected_run = alpha * current_run_length + (1 - alpha) * expected_run

            # Reset run
            current_run_length = 0
            previous_side = 0.0  # Reset so next tick starts new run
        else:
            previous_side = current_side

    return (
        np.array(bar_indices, dtype=np.int64),
        np.array(expected_runs, dtype=np.float64),
        np.array(run_lengths_out, dtype=np.int64),
    )


class TickRunBarSampler(BarSampler):
    """Sample bars based on consecutive tick runs.

    A tick run bar is formed when a sequence of consecutive buy or sell ticks
    becomes unusually long, indicating sustained directional pressure.

    Parameters
    ----------
    expected_ticks_per_bar : int
        Expected number of ticks per bar (for initialization)
    initial_run_expectation : int, optional
        Initial expected run length. If None, estimated from expected_ticks_per_bar
    alpha : float, default 0.1
        EWMA decay factor for updating expected run length

    Examples
    --------
    >>> sampler = TickRunBarSampler(expected_ticks_per_bar=100)
    >>> bars = sampler.sample(tick_data)

    References
    ----------
    .. [1] López de Prado, M. (2018). Advances in Financial Machine Learning.
           John Wiley & Sons. Chapter 2: Financial Data Structures.
    """

    def __init__(
        self,
        expected_ticks_per_bar: int,
        initial_run_expectation: int | None = None,
        alpha: float = 0.1,
    ):
        if expected_ticks_per_bar <= 0:
            raise ValueError("expected_ticks_per_bar must be positive")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")

        self.expected_ticks_per_bar = expected_ticks_per_bar
        self.initial_run_expectation = initial_run_expectation
        self.alpha = alpha

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
            Sampled run bars with run metrics
        """
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Run bars require 'side' column")

        if len(data) == 0:
            return self._empty_run_bars_df()

        # Estimate initial expectation if not provided
        if self.initial_run_expectation is None:
            # Assume runs are roughly 10-20% of expected ticks per bar
            self.initial_run_expectation = max(1, int(self.expected_ticks_per_bar * 0.15))

        # Extract arrays
        volumes = data["volume"].to_numpy()
        sides = data["side"].to_numpy()

        # Calculate bar indices using Numba
        bar_indices, expected_runs, run_lengths = _calculate_run_bars_nb(
            volumes,
            sides,
            float(self.initial_run_expectation),
            self.alpha,
        )

        # Build bars
        bars = []
        start_idx = 0

        for i, end_idx in enumerate(bar_indices):
            # Extract bar data
            bar_ticks = data.slice(start_idx, end_idx - start_idx + 1)

            # Calculate metrics
            bar_volumes = volumes[start_idx : end_idx + 1]
            bar_sides = sides[start_idx : end_idx + 1]

            buy_volume = float(np.sum(bar_volumes[bar_sides > 0]))
            sell_volume = float(np.sum(bar_volumes[bar_sides < 0]))

            # Create bar
            bar = self._create_ohlcv_bar(
                bar_ticks,
                additional_cols={
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "run_length": int(run_lengths[i]),
                    "expected_run": float(expected_runs[i]),
                },
            )
            bars.append(bar)

            start_idx = end_idx + 1

        # Handle incomplete final bar
        if include_incomplete and start_idx < len(data):
            bar_ticks = data.slice(start_idx)

            if len(bar_ticks) > 0:
                bar_volumes = volumes[start_idx:]
                bar_sides = sides[start_idx:]

                buy_vol = float(np.sum(bar_volumes[bar_sides > 0]))
                sell_vol = float(np.sum(bar_volumes[bar_sides < 0]))

                # Calculate current run length
                current_run_length = 1
                for j in range(1, len(bar_sides)):
                    if bar_sides[j] == bar_sides[j - 1]:
                        current_run_length += 1
                    else:
                        current_run_length = 1

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": buy_vol,
                        "sell_volume": sell_vol,
                        "run_length": current_run_length,
                        "expected_run": float(
                            expected_runs[-1]
                            if len(expected_runs) > 0
                            else self.initial_run_expectation
                        ),
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
            },
        )


class VolumeRunBarSampler(BarSampler):
    """Sample bars based on consecutive volume runs.

    Similar to tick run bars, but weighs runs by volume rather than tick count.

    Parameters
    ----------
    expected_ticks_per_bar : int
        Expected number of ticks per bar (for initialization)
    initial_run_expectation : float, optional
        Initial expected run length in volume terms
    alpha : float, default 0.1
        EWMA decay factor for updating expected run length

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
    ):
        if expected_ticks_per_bar <= 0:
            raise ValueError("expected_ticks_per_bar must be positive")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")

        self.expected_ticks_per_bar = expected_ticks_per_bar
        self.initial_run_expectation = initial_run_expectation
        self.alpha = alpha

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

        # Estimate initial expectation if not provided
        if self.initial_run_expectation is None:
            avg_volume = float(data["volume"].mean())
            self.initial_run_expectation = float(self.expected_ticks_per_bar * avg_volume * 0.15)

        # Use weighted volume as the run metric
        volumes = data["volume"].to_numpy()
        sides = data["side"].to_numpy()

        # For volume runs, we accumulate volume during a run
        bar_indices, expected_runs, run_volumes = self._calculate_volume_runs(
            volumes,
            sides,
            self.initial_run_expectation,
            self.alpha,
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
                    "run_volume": float(run_volumes[i]),
                    "expected_run": float(expected_runs[i]),
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

                # Calculate current run volume
                current_run_volume = 0.0
                prev_side = 0.0
                for j in range(len(bar_sides)):
                    if bar_sides[j] == prev_side or prev_side == 0.0:
                        current_run_volume += bar_volumes[j]
                    else:
                        current_run_volume = bar_volumes[j]
                    prev_side = bar_sides[j]

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": buy_vol,
                        "sell_volume": sell_vol,
                        "run_volume": current_run_volume,
                        "expected_run": float(
                            expected_runs[-1]
                            if len(expected_runs) > 0
                            else self.initial_run_expectation
                        ),
                    },
                )
                bars.append(bar)

        if not bars:
            return self._empty_run_bars_df()

        return pl.DataFrame(bars)

    def _calculate_volume_runs(
        self,
        volumes: npt.NDArray[np.float64],
        sides: npt.NDArray[np.float64],
        initial_expectation: float,
        alpha: float,
    ) -> tuple[list[int], list[float], list[float]]:
        """Calculate run bars based on cumulative volume in runs."""
        n = len(volumes)
        bar_indices = []
        expected_runs = []
        run_volumes_out = []

        expected_run = initial_expectation
        current_run_volume = 0.0
        previous_side = 0.0

        for i in range(n):
            current_side = sides[i]

            # Check if continuing run
            if current_side == previous_side or previous_side == 0.0:
                current_run_volume += volumes[i]
            else:
                # Direction changed
                current_run_volume = volumes[i]

            # Check if run exceeds expectation
            if current_run_volume >= expected_run:
                bar_indices.append(i)
                run_volumes_out.append(current_run_volume)
                expected_runs.append(expected_run)

                # Update expectation
                expected_run = alpha * current_run_volume + (1 - alpha) * expected_run

                # Reset
                current_run_volume = 0.0
                previous_side = 0.0
            else:
                previous_side = current_side

        return bar_indices, expected_runs, run_volumes_out

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
            },
        )


class DollarRunBarSampler(BarSampler):
    """Sample bars based on consecutive dollar value runs.

    Similar to volume run bars, but uses dollar value (price * volume) as the run metric.

    Parameters
    ----------
    expected_ticks_per_bar : int
        Expected number of ticks per bar (for initialization)
    initial_run_expectation : float, optional
        Initial expected run length in dollar terms
    alpha : float, default 0.1
        EWMA decay factor for updating expected run length

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
    ):
        if expected_ticks_per_bar <= 0:
            raise ValueError("expected_ticks_per_bar must be positive")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")

        self.expected_ticks_per_bar = expected_ticks_per_bar
        self.initial_run_expectation = initial_run_expectation
        self.alpha = alpha

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

        # Estimate initial expectation if not provided
        if self.initial_run_expectation is None:
            prices = data["price"].to_numpy()
            volumes = data["volume"].to_numpy()
            avg_dollar_volume = float(np.mean(prices * volumes))
            self.initial_run_expectation = self.expected_ticks_per_bar * avg_dollar_volume * 0.15

        # Calculate dollar runs
        prices = data["price"].to_numpy()
        volumes = data["volume"].to_numpy()
        sides = data["side"].to_numpy()
        dollar_volumes = prices * volumes

        bar_indices, expected_runs, run_dollars = self._calculate_dollar_runs(
            dollar_volumes,
            sides,
            self.initial_run_expectation,
            self.alpha,
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
                    "run_dollars": float(run_dollars[i]),
                    "expected_run": float(expected_runs[i]),
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

                # Calculate current run dollars
                current_run_dollars = 0.0
                prev_side = 0.0
                for j in range(len(bar_sides)):
                    if bar_sides[j] == prev_side or prev_side == 0.0:
                        current_run_dollars += bar_dollars[j]
                    else:
                        current_run_dollars = bar_dollars[j]
                    prev_side = bar_sides[j]

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": buy_vol,
                        "sell_volume": sell_vol,
                        "dollar_volume": total_dol,
                        "vwap": vwap_val,
                        "run_dollars": current_run_dollars,
                        "expected_run": float(
                            expected_runs[-1]
                            if len(expected_runs) > 0
                            else self.initial_run_expectation
                        ),
                    },
                )
                bars.append(bar)

        if not bars:
            return self._empty_run_bars_df()

        return pl.DataFrame(bars)

    def _calculate_dollar_runs(
        self,
        dollar_volumes: npt.NDArray[np.float64],
        sides: npt.NDArray[np.float64],
        initial_expectation: float,
        alpha: float,
    ) -> tuple[list[int], list[float], list[float]]:
        """Calculate run bars based on cumulative dollar volume in runs."""
        n = len(dollar_volumes)
        bar_indices = []
        expected_runs = []
        run_dollars_out = []

        expected_run = initial_expectation
        current_run_dollars = 0.0
        previous_side = 0.0

        for i in range(n):
            current_side = sides[i]

            # Check if continuing run
            if current_side == previous_side or previous_side == 0.0:
                current_run_dollars += dollar_volumes[i]
            else:
                # Direction changed
                current_run_dollars = dollar_volumes[i]

            # Check if run exceeds expectation
            if current_run_dollars >= expected_run:
                bar_indices.append(i)
                run_dollars_out.append(current_run_dollars)
                expected_runs.append(expected_run)

                # Update expectation
                expected_run = alpha * current_run_dollars + (1 - alpha) * expected_run

                # Reset
                current_run_dollars = 0.0
                previous_side = 0.0
            else:
                previous_side = current_side

        return bar_indices, expected_runs, run_dollars_out

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
            },
        )


__all__ = [
    "DollarRunBarSampler",
    "TickRunBarSampler",
    "VolumeRunBarSampler",
]
