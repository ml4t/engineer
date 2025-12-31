"""Tick bar sampler implementation."""

import polars as pl

from ml4t.engineer.bars.base import BarSampler


class TickBarSampler(BarSampler):
    """Sample bars based on number of ticks.

    Tick bars sample the data every N ticks, providing a more stable
    sampling rate compared to time bars during varying market activity.

    Parameters
    ----------
    ticks_per_bar : int
        Number of ticks per bar

    Examples
    --------
    >>> sampler = TickBarSampler(ticks_per_bar=100)
    >>> bars = sampler.sample(tick_data)
    """

    def __init__(self, ticks_per_bar: int):
        """Initialize tick bar sampler.

        Parameters
        ----------
        ticks_per_bar : int
            Number of ticks per bar
        """
        if ticks_per_bar <= 0:
            raise ValueError("ticks_per_bar must be positive")

        self.ticks_per_bar = ticks_per_bar

    def sample(
        self,
        data: pl.DataFrame,
        include_incomplete: bool = False,
    ) -> pl.DataFrame:
        """Sample tick bars from data.

        Parameters
        ----------
        data : pl.DataFrame
            Tick data with columns: timestamp, price, volume
        include_incomplete : bool, default False
            Whether to include incomplete final bar

        Returns
        -------
        pl.DataFrame
            Sampled tick bars
        """
        # Validate input
        self._validate_data(data)

        if len(data) == 0:
            return pl.DataFrame()

        # Calculate bar indices
        n_ticks = len(data)
        n_complete_bars = n_ticks // self.ticks_per_bar

        bars = []

        # Process complete bars
        for i in range(n_complete_bars):
            start_idx = i * self.ticks_per_bar

            bar_ticks = data.slice(start_idx, self.ticks_per_bar)
            bar = self._create_ohlcv_bar(bar_ticks)
            bars.append(bar)

        # Handle incomplete final bar
        if include_incomplete and n_ticks % self.ticks_per_bar > 0:
            start_idx = n_complete_bars * self.ticks_per_bar
            bar_ticks = data.slice(start_idx)

            if len(bar_ticks) > 0:
                bar = self._create_ohlcv_bar(bar_ticks)
                bars.append(bar)

        # Convert to DataFrame
        if not bars:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(
                {
                    "timestamp": [],
                    "open": [],
                    "high": [],
                    "low": [],
                    "close": [],
                    "volume": [],
                    "tick_count": [],
                },
            )

        return pl.DataFrame(bars)


__all__ = ["TickBarSampler"]
