"""Volume and dollar bar sampler implementations."""

import numpy as np
import polars as pl

from ml4t.engineer.bars.base import BarSampler
from ml4t.engineer.core.exceptions import DataValidationError


class VolumeBarSampler(BarSampler):
    """Sample bars based on volume traded.

    Volume bars sample when the cumulative volume reaches a threshold,
    providing more samples during high activity periods.

    Parameters
    ----------
    volume_per_bar : float
        Target volume per bar

    Examples
    --------
    >>> sampler = VolumeBarSampler(volume_per_bar=10000)
    >>> bars = sampler.sample(tick_data)
    """

    def __init__(self, volume_per_bar: float):
        """Initialize volume bar sampler.

        Parameters
        ----------
        volume_per_bar : float
            Target volume per bar
        """
        if volume_per_bar <= 0:
            raise ValueError("volume_per_bar must be positive")

        self.volume_per_bar = volume_per_bar

    def sample(
        self,
        data: pl.DataFrame,
        include_incomplete: bool = False,
    ) -> pl.DataFrame:
        """Sample volume bars from data.

        Parameters
        ----------
        data : pl.DataFrame
            Tick data with columns: timestamp, price, volume, side
        include_incomplete : bool, default False
            Whether to include incomplete final bar

        Returns
        -------
        pl.DataFrame
            Sampled volume bars with buy/sell volume breakdown
        """
        # Validate input
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Volume bars require 'side' column")

        if len(data) == 0:
            return pl.DataFrame()

        # Convert to numpy for efficient processing
        volumes = data["volume"].to_numpy()
        sides = data["side"].to_numpy()

        bars = []
        current_volume: float = 0.0
        start_idx = 0

        for i in range(len(data)):
            current_volume += volumes[i]

            # Check if we've reached the threshold
            if current_volume >= self.volume_per_bar:
                # Extract bar data
                bar_ticks = data.slice(start_idx, i - start_idx + 1)

                # Calculate buy/sell volumes
                bar_sides = sides[start_idx : i + 1]
                bar_volumes = volumes[start_idx : i + 1]

                buy_vol: float = float(np.sum(bar_volumes[bar_sides > 0]))
                sell_vol: float = float(np.sum(bar_volumes[bar_sides < 0]))

                # Create bar
                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": float(buy_vol),
                        "sell_volume": float(sell_vol),
                    },
                )
                bars.append(bar)

                # Reset for next bar
                current_volume = 0
                start_idx = i + 1

        # Handle incomplete final bar
        if include_incomplete and start_idx < len(data):
            bar_ticks = data.slice(start_idx)

            if len(bar_ticks) > 0:
                bar_sides = sides[start_idx:]
                bar_volumes = volumes[start_idx:]

                buy_vol_incomplete: float = float(np.sum(bar_volumes[bar_sides > 0]))
                sell_vol_incomplete: float = float(np.sum(bar_volumes[bar_sides < 0]))

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": float(buy_vol_incomplete),
                        "sell_volume": float(sell_vol_incomplete),
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
                },
            )

        return pl.DataFrame(bars)


class DollarBarSampler(BarSampler):
    """Sample bars based on dollar value traded.

    Dollar bars sample when the cumulative dollar value (price * volume)
    reaches a threshold, providing adaptive sampling based on both
    price and volume.

    Parameters
    ----------
    dollars_per_bar : float
        Target dollar value per bar

    Examples
    --------
    >>> sampler = DollarBarSampler(dollars_per_bar=1_000_000)
    >>> bars = sampler.sample(tick_data)
    """

    def __init__(self, dollars_per_bar: float):
        """Initialize dollar bar sampler.

        Parameters
        ----------
        dollars_per_bar : float
            Target dollar value per bar
        """
        if dollars_per_bar <= 0:
            raise ValueError("dollars_per_bar must be positive")

        self.dollars_per_bar = dollars_per_bar

    def sample(
        self,
        data: pl.DataFrame,
        include_incomplete: bool = False,
    ) -> pl.DataFrame:
        """Sample dollar bars from data.

        Parameters
        ----------
        data : pl.DataFrame
            Tick data with columns: timestamp, price, volume
        include_incomplete : bool, default False
            Whether to include incomplete final bar

        Returns
        -------
        pl.DataFrame
            Sampled dollar bars with VWAP
        """
        # Validate input
        self._validate_data(data)

        if len(data) == 0:
            return pl.DataFrame()

        # Calculate dollar volumes
        prices = data["price"].to_numpy()
        volumes = data["volume"].to_numpy()
        dollar_volumes = prices * volumes

        bars = []
        current_dollars = 0
        start_idx = 0

        for i in range(len(data)):
            current_dollars += dollar_volumes[i]

            # Check if we've reached the threshold
            if current_dollars >= self.dollars_per_bar:
                # Extract bar data
                bar_ticks = data.slice(start_idx, i - start_idx + 1)

                # Calculate VWAP
                bar_prices = prices[start_idx : i + 1]
                bar_volumes = volumes[start_idx : i + 1]
                bar_dollars = dollar_volumes[start_idx : i + 1]

                total_volume: float = float(np.sum(bar_volumes))
                if total_volume > 0:
                    vwap = float(np.sum(bar_dollars) / total_volume)
                else:
                    vwap = float(np.mean(bar_prices))

                # Create bar
                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "dollar_volume": float(np.sum(bar_dollars)),
                        "vwap": float(vwap),
                    },
                )
                bars.append(bar)

                # Reset for next bar
                current_dollars = 0
                start_idx = i + 1

        # Handle incomplete final bar
        if include_incomplete and start_idx < len(data):
            bar_ticks = data.slice(start_idx)

            if len(bar_ticks) > 0:
                bar_prices = prices[start_idx:]
                bar_volumes = volumes[start_idx:]
                bar_dollars = dollar_volumes[start_idx:]

                total_vol_incomplete: float = float(np.sum(bar_volumes))
                if total_vol_incomplete > 0:
                    vwap_incomplete = float(np.sum(bar_dollars) / total_vol_incomplete)
                else:
                    vwap_incomplete = float(np.mean(bar_prices))

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "dollar_volume": float(np.sum(bar_dollars)),
                        "vwap": float(vwap_incomplete),
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
                    "dollar_volume": [],
                    "vwap": [],
                },
            )

        return pl.DataFrame(bars)


__all__ = ["DollarBarSampler", "VolumeBarSampler"]
