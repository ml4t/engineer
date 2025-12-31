"""Imbalance bar sampler implementation."""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.bars.base import BarSampler
from ml4t.engineer.core.exceptions import DataValidationError


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _calculate_imbalance_bars_nb(
    volumes: npt.NDArray[np.float64],
    sides: npt.NDArray[np.float64],
    initial_expectation: float,
    alpha: float = 0.1,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate imbalance bar indices using Numba.

    Returns
    -------
    tuple of arrays
        (bar_end_indices, expected_imbalances, cumulative_thetas)
    """
    n = len(volumes)
    bar_indices = []
    expected_imbalances = []
    cumulative_thetas = []

    # Initialize
    expected_imbalance = initial_expectation
    cumulative_theta = 0.0

    for i in range(n):
        # Calculate signed volume (imbalance contribution)
        signed_volume = volumes[i] * sides[i]
        cumulative_theta += signed_volume

        # Check if we should create a new bar
        if abs(cumulative_theta) >= expected_imbalance:
            # Record bar end
            bar_indices.append(i)
            cumulative_thetas.append(cumulative_theta)

            # Store the threshold that was used to create this bar
            expected_imbalances.append(expected_imbalance)

            # Update expected imbalance using EWMA for the next bar
            # E[T] = alpha * |theta_t| + (1 - alpha) * E[T]
            expected_imbalance = alpha * abs(cumulative_theta) + (1 - alpha) * expected_imbalance

            # Reset cumulative theta
            cumulative_theta = 0.0

    return (
        np.array(bar_indices, dtype=np.int64),
        np.array(expected_imbalances, dtype=np.float64),
        np.array(cumulative_thetas, dtype=np.float64),
    )


class ImbalanceBarSampler(BarSampler):
    """Sample bars based on order flow imbalance.

    Imbalance bars sample when the cumulative signed volume (buy - sell)
    reaches a dynamically adjusted threshold. This captures periods of
    directional order flow.

    The threshold is updated using an exponentially weighted moving average
    (EWMA) of the absolute imbalance at each bar formation.

    Parameters
    ----------
    expected_ticks_per_bar : int
        Expected number of ticks per bar (for initialization)
    initial_expectation : float, optional
        Initial expected imbalance threshold. If None, estimated from
        expected_ticks_per_bar
    alpha : float, default 0.1
        EWMA decay factor for updating expected imbalance

    Examples
    --------
    >>> sampler = ImbalanceBarSampler(
    ...     expected_ticks_per_bar=100,
    ...     initial_expectation=5000
    ... )
    >>> bars = sampler.sample(tick_data)
    """

    def __init__(
        self,
        expected_ticks_per_bar: int,
        initial_expectation: float | None = None,
        alpha: float = 0.1,
    ):
        """Initialize imbalance bar sampler.

        Parameters
        ----------
        expected_ticks_per_bar : int
            Expected number of ticks per bar
        initial_expectation : float, optional
            Initial expected imbalance threshold
        alpha : float, default 0.1
            EWMA decay factor
        """
        if expected_ticks_per_bar <= 0:
            raise ValueError("expected_ticks_per_bar must be positive")

        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")

        self.expected_ticks_per_bar = expected_ticks_per_bar
        self.initial_expectation = initial_expectation
        self.alpha = alpha

    def sample(
        self,
        data: pl.DataFrame,
        include_incomplete: bool = False,
    ) -> pl.DataFrame:
        """Sample imbalance bars from data.

        Parameters
        ----------
        data : pl.DataFrame
            Tick data with columns: timestamp, price, volume, side
        include_incomplete : bool, default False
            Whether to include incomplete final bar

        Returns
        -------
        pl.DataFrame
            Sampled imbalance bars with flow metrics
        """
        # Validate input
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Imbalance bars require 'side' column")

        if len(data) == 0:
            return pl.DataFrame()

        # Extract arrays
        volumes = data["volume"].to_numpy()
        sides = data["side"].to_numpy()

        # Estimate initial expectation if not provided
        if self.initial_expectation is None:
            # Estimate as expected ticks * average volume * imbalance fraction
            avg_volume = np.mean(volumes[: min(1000, len(volumes))])
            imbalance_fraction = 0.1  # Assume 10% average imbalance
            self.initial_expectation = float(
                self.expected_ticks_per_bar * avg_volume * imbalance_fraction,
            )

        # Calculate bar indices using Numba
        (
            bar_indices,
            expected_imbalances,
            cumulative_thetas,
        ) = _calculate_imbalance_bars_nb(
            volumes,
            sides,
            self.initial_expectation,
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

            buy_volume: float = float(np.sum(bar_volumes[bar_sides > 0]))
            sell_volume: float = float(np.sum(bar_volumes[bar_sides < 0]))
            imbalance = buy_volume - sell_volume

            # Create bar
            bar = self._create_ohlcv_bar(
                bar_ticks,
                additional_cols={
                    "buy_volume": float(buy_volume),
                    "sell_volume": float(sell_volume),
                    "imbalance": float(imbalance),
                    "cumulative_theta": float(cumulative_thetas[i]),
                    "expected_imbalance": float(expected_imbalances[i]),
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

                buy_vol_incomplete: float = float(np.sum(bar_volumes[bar_sides > 0]))
                sell_vol_incomplete: float = float(np.sum(bar_volumes[bar_sides < 0]))
                imbalance_incomplete: float = buy_vol_incomplete - sell_vol_incomplete

                # Calculate current cumulative theta
                cumulative_theta: float = float(np.sum(bar_volumes * bar_sides))

                # Use last expected imbalance or initial
                expected_imbalance = (
                    expected_imbalances[-1]
                    if len(expected_imbalances) > 0
                    else self.initial_expectation
                )

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": float(buy_vol_incomplete),
                        "sell_volume": float(sell_vol_incomplete),
                        "imbalance": float(imbalance_incomplete),
                        "cumulative_theta": float(cumulative_theta),
                        "expected_imbalance": float(expected_imbalance)
                        if expected_imbalance is not None
                        else 0.0,
                    },
                )
                bars.append(bar)

        # Convert to DataFrame
        if not bars:
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
                    "imbalance": [],
                    "cumulative_theta": [],
                    "expected_imbalance": [],
                },
            )

        return pl.DataFrame(bars)


__all__ = ["ImbalanceBarSampler"]
