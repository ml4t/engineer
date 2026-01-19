# mypy: disable-error-code="misc,operator,assignment,arg-type"
"""
Vectorized bar samplers using Polars for high performance.

Exports:
    VolumeBarSamplerVectorized(volume_threshold=1000) -> BarSampler
        Volume bars using Numba-accelerated threshold detection.

    DollarBarSamplerVectorized(dollar_threshold=1_000_000) -> BarSampler
        Dollar bars using vectorized cumulative sum.

    TickBarSamplerVectorized(tick_threshold=100) -> BarSampler
        Tick bars with fixed trade count.

    ImbalanceBarSamplerVectorized(expected_imbalance=100) -> BarSampler
        Tick imbalance bars with EWMA-adaptive thresholds.

This module provides vectorized implementations of bar samplers that achieve
10,000+ rows/sec performance by leveraging Polars' columnar operations instead
of Python loops.

The key insight is using cumulative sums and groupings to identify bar boundaries
efficiently in a vectorized manner.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.bars.base import BarSampler
from ml4t.engineer.core.exceptions import DataValidationError


@jit(nopython=True)
def _assign_volume_bar_ids(
    volumes: npt.NDArray[np.float64],
    volume_threshold: float,
) -> npt.NDArray[np.int32]:
    """Assign bar IDs based on cumulative volume thresholds using Numba.

    This function is JIT-compiled for maximum performance when assigning
    bar IDs to rows based on when cumulative volume exceeds the threshold.

    Parameters
    ----------
    volumes : npt.NDArray[np.float64]
        Array of volume values for each row
    volume_threshold : float
        Volume threshold for creating new bars

    Returns
    -------
    npt.NDArray[np.float64]
        Array of bar IDs for each row
    """
    n_rows = len(volumes)
    bar_ids = np.zeros(n_rows, dtype=np.int32)

    current_volume = 0.0
    current_bar_id = 0

    for i in range(n_rows):
        current_volume += volumes[i]
        bar_ids[i] = current_bar_id

        if current_volume >= volume_threshold:
            current_volume = 0.0
            current_bar_id += 1

    return bar_ids


@jit(nopython=True)
def _assign_dollar_bar_ids(
    prices: npt.NDArray[np.float64],
    volumes: npt.NDArray[np.float64],
    dollar_threshold: float,
) -> npt.NDArray[np.int32]:
    """Assign bar IDs based on cumulative dollar volume thresholds using Numba.

    Parameters
    ----------
    prices : npt.NDArray[np.float64]
        Array of price values for each row
    volumes : npt.NDArray[np.float64]
        Array of volume values for each row
    dollar_threshold : float
        Dollar volume threshold for creating new bars

    Returns
    -------
    npt.NDArray[np.float64]
        Array of bar IDs for each row
    """
    n_rows = len(prices)
    bar_ids = np.zeros(n_rows, dtype=np.int32)

    current_dollar_volume = 0.0
    current_bar_id = 0

    for i in range(n_rows):
        current_dollar_volume += prices[i] * volumes[i]
        bar_ids[i] = current_bar_id

        if current_dollar_volume >= dollar_threshold:
            current_dollar_volume = 0.0
            current_bar_id += 1

    return bar_ids


class VolumeBarSamplerVectorized(BarSampler):
    """Vectorized volume bar sampler using Polars.

    This implementation replaces Python loops with vectorized Polars operations
    for dramatically improved performance on large datasets.

    Parameters
    ----------
    volume_per_bar : float
        Target volume per bar
    """

    def __init__(self, volume_per_bar: float):
        if volume_per_bar <= 0:
            raise ValueError("volume_per_bar must be positive")
        self.volume_per_bar = volume_per_bar

    def sample(self, data: pl.DataFrame, include_incomplete: bool = False) -> pl.DataFrame:
        """Sample volume bars using vectorized operations."""
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Volume bars require 'side' column")

        if len(data) == 0:
            return pl.DataFrame()

        # Step 1: Identify bar boundaries using Numba-optimized function
        volumes = data["volume"].to_numpy()

        # Use JIT-compiled function for 10-100x speedup
        bar_ids = _assign_volume_bar_ids(volumes, self.volume_per_bar)

        # Add bar IDs to dataframe
        df_with_bars = data.with_columns([pl.Series("bar_id", bar_ids)])

        # Filter out incomplete final bar if requested
        if not include_incomplete:
            # Check if last bar is complete
            last_bar_id = bar_ids[-1]
            last_bar_volume = df_with_bars.filter(pl.col("bar_id") == last_bar_id)["volume"].sum()
            if last_bar_volume < self.volume_per_bar:
                df_with_bars = df_with_bars.filter(pl.col("bar_id") < last_bar_id)
                if df_with_bars.is_empty():
                    return self._empty_volume_bars_df()

        # Step 2: Group by bar_id and aggregate using vectorized operations
        bars = (
            df_with_bars.group_by("bar_id", maintain_order=True)
            .agg(
                [
                    # OHLCV aggregations
                    pl.col("timestamp").first().alias("timestamp"),
                    pl.col("price").first().alias("open"),
                    pl.col("price").max().alias("high"),
                    pl.col("price").min().alias("low"),
                    pl.col("price").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                    pl.len().alias("tick_count"),
                    # Buy/sell volume breakdown using vectorized operations
                    pl.col("volume")
                    .filter(pl.col("side") > 0)
                    .sum()
                    .fill_null(0)
                    .alias("buy_volume"),
                    pl.col("volume")
                    .filter(pl.col("side") < 0)
                    .sum()
                    .fill_null(0)
                    .alias("sell_volume"),
                ],
            )
            .sort("bar_id")
            .drop("bar_id")
        )

        return bars

    def _empty_volume_bars_df(self) -> pl.DataFrame:
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
            },
        )


class DollarBarSamplerVectorized(BarSampler):
    """Vectorized dollar bar sampler using Polars.

    Parameters
    ----------
    dollars_per_bar : float
        Target dollar value per bar
    """

    def __init__(self, dollars_per_bar: float):
        if dollars_per_bar <= 0:
            raise ValueError("dollars_per_bar must be positive")
        self.dollars_per_bar = dollars_per_bar

    def sample(self, data: pl.DataFrame, include_incomplete: bool = False) -> pl.DataFrame:
        """Sample dollar bars using vectorized operations."""
        self._validate_data(data)

        if len(data) == 0:
            return pl.DataFrame()

        # Step 1: Calculate dollar volumes and identify bar boundaries
        prices = data["price"].to_numpy()
        volumes = data["volume"].to_numpy()
        dollar_volumes = prices * volumes

        # Use JIT-compiled function for 10-100x speedup
        bar_ids = _assign_dollar_bar_ids(prices, volumes, self.dollars_per_bar)

        # Add bar IDs and dollar volume to dataframe
        df_with_bars = data.with_columns(
            [pl.Series("bar_id", bar_ids), pl.Series("dollar_volume", dollar_volumes)],
        )

        # Filter out incomplete final bar if requested
        if not include_incomplete:
            # Check if last bar is complete
            last_bar_id = bar_ids[-1]
            last_bar_dollars = df_with_bars.filter(pl.col("bar_id") == last_bar_id)[
                "dollar_volume"
            ].sum()
            if last_bar_dollars < self.dollars_per_bar:
                df_with_bars = df_with_bars.filter(pl.col("bar_id") < last_bar_id)
                if df_with_bars.is_empty():
                    return self._empty_dollar_bars_df()

        # Step 2: Vectorized aggregation by bar_id
        bars = (
            df_with_bars.group_by("bar_id", maintain_order=True)
            .agg(
                [
                    # OHLCV aggregations
                    pl.col("timestamp").first().alias("timestamp"),
                    pl.col("price").first().alias("open"),
                    pl.col("price").max().alias("high"),
                    pl.col("price").min().alias("low"),
                    pl.col("price").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                    pl.len().alias("tick_count"),
                    # Dollar volume and VWAP calculation
                    pl.col("dollar_volume").sum().alias("dollar_volume"),
                    (pl.col("dollar_volume").sum() / pl.col("volume").sum()).alias("vwap"),
                ],
            )
            .sort("bar_id")
            .drop("bar_id")
        )

        return bars

    def _empty_dollar_bars_df(self) -> pl.DataFrame:
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
                "dollar_volume": [],
                "vwap": [],
            },
        )


class TickBarSamplerVectorized(BarSampler):
    """Vectorized tick bar sampler using Polars.

    Parameters
    ----------
    ticks_per_bar : int
        Number of ticks per bar
    """

    def __init__(self, ticks_per_bar: int):
        if ticks_per_bar <= 0:
            raise ValueError("ticks_per_bar must be positive")
        self.ticks_per_bar = ticks_per_bar

    def sample(self, data: pl.DataFrame, include_incomplete: bool = False) -> pl.DataFrame:
        """Sample tick bars using vectorized operations."""
        self._validate_data(data)

        if len(data) == 0:
            return pl.DataFrame()

        # Step 1: Add row indices and calculate bar IDs
        df_with_bars = data.with_row_index("row_idx").with_columns(
            [
                # Bar group IDs using integer division
                (pl.col("row_idx") // self.ticks_per_bar).cast(pl.Int32).alias("bar_id"),
            ],
        )

        # Filter out incomplete final bar if requested
        if not include_incomplete:
            # Calculate the maximum complete bar ID
            total_rows = len(df_with_bars)
            max_complete_bar = (total_rows // self.ticks_per_bar) - 1

            if max_complete_bar >= 0:
                df_with_bars = df_with_bars.filter(pl.col("bar_id") <= max_complete_bar)
            else:
                return self._empty_tick_bars_df()

        # Step 2: Vectorized aggregation by bar_id
        bars = (
            df_with_bars.group_by("bar_id", maintain_order=True)
            .agg(
                [
                    # OHLCV aggregations
                    pl.col("timestamp").first().alias("timestamp"),
                    pl.col("price").first().alias("open"),
                    pl.col("price").max().alias("high"),
                    pl.col("price").min().alias("low"),
                    pl.col("price").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                    pl.len().alias("tick_count"),
                ],
            )
            .sort("bar_id")
            .drop("bar_id")
        )

        return bars

    def _empty_tick_bars_df(self) -> pl.DataFrame:
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
            },
        )


class ImbalanceBarSamplerVectorized(BarSampler):
    """Vectorized imbalance bar sampler with AFML-compliant adaptive thresholds.

    This implementation uses vectorized operations for the main logic while
    keeping the adaptive threshold calculation efficient.

    AFML Threshold Formula:
        E[θ_T] = E[T] × |2v⁺ - E[v]|

    Where:
        E[T] = EWMA of bar lengths (ticks per bar)
        v⁺ = P[b=1] × E[v|b=1] = expected buy volume contribution
        E[v] = unconditional mean volume per tick

    Parameters
    ----------
    expected_ticks_per_bar : int
        Expected number of ticks per bar (initializes E[T])
    initial_expectation : float, optional
        DEPRECATED. The AFML threshold is computed dynamically.
    alpha : float, default 0.1
        EWMA decay factor for updating expectations
    initial_p_buy : float, default 0.5
        Initial buy probability P[b=1]
    min_bars_warmup : int, default 10
        Number of bars before starting EWMA updates
    """

    def __init__(
        self,
        expected_ticks_per_bar: int,
        initial_expectation: float | None = None,
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

        self.expected_ticks_per_bar = expected_ticks_per_bar
        self.initial_expectation = initial_expectation  # Deprecated, kept for backward compat
        self.alpha = alpha
        self.initial_p_buy = initial_p_buy
        self.min_bars_warmup = min_bars_warmup

        # Will be estimated from data
        self._initial_v_buy: float | None = None
        self._initial_v: float | None = None

    def sample(self, data: pl.DataFrame, include_incomplete: bool = False) -> pl.DataFrame:
        """Sample imbalance bars using vectorized operations where possible."""
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Imbalance bars require 'side' column")

        if len(data) == 0:
            return pl.DataFrame()

        # Extract arrays
        volumes = data["volume"].to_numpy().astype(np.float64)
        sides = data["side"].to_numpy().astype(np.float64)

        # Estimate initial values from data if not already set
        warmup_size = min(1000, len(volumes))
        warmup_volumes = volumes[:warmup_size]
        warmup_sides = sides[:warmup_size]

        if self._initial_v is None:
            self._initial_v = float(np.mean(warmup_volumes))

        if self._initial_v_buy is None:
            buy_mask = warmup_sides > 0
            if np.any(buy_mask):
                self._initial_v_buy = float(np.mean(warmup_volumes[buy_mask]))
            else:
                self._initial_v_buy = self._initial_v

        # Use the AFML-compliant Numba function for finding bar boundaries
        from .imbalance import _calculate_imbalance_bars_nb

        (
            bar_indices,
            expected_thetas,
            cumulative_thetas,
            expected_ts,
            p_buys,
            v_pluses,
            e_vs,
        ) = _calculate_imbalance_bars_nb(
            volumes,
            sides,
            float(self.expected_ticks_per_bar),
            self.initial_p_buy,
            self._initial_v_buy,
            self._initial_v,
            self.alpha,
            self.min_bars_warmup,
        )

        if len(bar_indices) == 0:
            if include_incomplete and len(data) > 0:
                # Return single incomplete bar
                return self._create_incomplete_imbalance_bar(data, volumes, sides)
            return self._empty_imbalance_bars_df()

        # Vectorized bar creation using bar boundaries
        bars_data = []
        start_idx = 0

        for i, end_idx in enumerate(bar_indices):
            # Create bar ID column for this segment
            segment_length = end_idx - start_idx + 1
            bar_segment = data.slice(start_idx, segment_length).with_columns(
                [
                    pl.lit(i).alias("bar_id"),
                    pl.lit(expected_thetas[i]).alias("expected_imbalance"),
                    pl.lit(cumulative_thetas[i]).alias("cumulative_theta"),
                    # AFML diagnostic columns
                    pl.lit(expected_ts[i]).alias("expected_t"),
                    pl.lit(p_buys[i]).alias("p_buy"),
                    pl.lit(v_pluses[i]).alias("v_plus"),
                    pl.lit(e_vs[i]).alias("e_v"),
                ],
            )
            bars_data.append(bar_segment)
            start_idx = end_idx + 1

        # Handle incomplete bar
        if include_incomplete and start_idx < len(data):
            # Calculate current cumulative theta for incomplete bar
            incomplete_volumes = volumes[start_idx:]
            incomplete_sides = sides[start_idx:]
            incomplete_theta = float(np.sum(incomplete_volumes * incomplete_sides))

            # Use last values or initial
            last_expected_t = (
                expected_ts[-1] if len(expected_ts) > 0 else float(self.expected_ticks_per_bar)
            )
            last_p_buy = p_buys[-1] if len(p_buys) > 0 else self.initial_p_buy
            last_v_plus = v_pluses[-1] if len(v_pluses) > 0 else last_p_buy * self._initial_v_buy
            last_e_v = e_vs[-1] if len(e_vs) > 0 else self._initial_v
            incomplete_expected = last_expected_t * abs(2 * last_v_plus - last_e_v)

            incomplete_segment = data.slice(start_idx).with_columns(
                [
                    pl.lit(len(bar_indices)).alias("bar_id"),
                    pl.lit(incomplete_expected).alias("expected_imbalance"),
                    pl.lit(incomplete_theta).alias("cumulative_theta"),
                    pl.lit(last_expected_t).alias("expected_t"),
                    pl.lit(last_p_buy).alias("p_buy"),
                    pl.lit(last_v_plus).alias("v_plus"),
                    pl.lit(last_e_v).alias("e_v"),
                ],
            )
            bars_data.append(incomplete_segment)

        if not bars_data:
            return self._empty_imbalance_bars_df()

        # Combine all segments and group by bar_id
        combined_data = pl.concat(bars_data)

        bars = (
            combined_data.group_by("bar_id", maintain_order=True)
            .agg(
                [
                    # OHLCV aggregations
                    pl.col("timestamp").first().alias("timestamp"),
                    pl.col("price").first().alias("open"),
                    pl.col("price").max().alias("high"),
                    pl.col("price").min().alias("low"),
                    pl.col("price").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                    pl.len().alias("tick_count"),
                    # Imbalance-specific metrics using vectorized operations
                    pl.col("volume")
                    .filter(pl.col("side") > 0)
                    .sum()
                    .fill_null(0)
                    .alias("buy_volume"),
                    pl.col("volume")
                    .filter(pl.col("side") < 0)
                    .sum()
                    .fill_null(0)
                    .alias("sell_volume"),
                    pl.col("expected_imbalance").first().alias("expected_imbalance"),
                    pl.col("cumulative_theta").first().alias("cumulative_theta"),
                    # AFML diagnostic columns
                    pl.col("expected_t").first().alias("expected_t"),
                    pl.col("p_buy").first().alias("p_buy"),
                    pl.col("v_plus").first().alias("v_plus"),
                    pl.col("e_v").first().alias("e_v"),
                ],
            )
            .with_columns(
                [
                    # Calculate imbalance
                    (pl.col("buy_volume") - pl.col("sell_volume")).alias("imbalance"),
                ],
            )
            .sort("bar_id")
            .drop("bar_id")
        )

        return bars

    def _create_incomplete_imbalance_bar(
        self,
        data: pl.DataFrame,
        volumes: np.ndarray,
        sides: np.ndarray,
    ) -> pl.DataFrame:
        """Create single incomplete bar for all remaining data."""
        # Compute cumulative theta
        cumulative_theta = float(np.sum(volumes * sides))

        # Use initial values for AFML parameters
        expected_t = float(self.expected_ticks_per_bar)
        p_buy = self.initial_p_buy
        v_plus = (
            p_buy * self._initial_v_buy if self._initial_v_buy else p_buy * float(np.mean(volumes))
        )
        e_v = self._initial_v if self._initial_v else float(np.mean(volumes))
        expected_imbalance = expected_t * abs(2 * v_plus - e_v)

        return data.select(
            [
                pl.col("timestamp").first().alias("timestamp"),
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.len().alias("tick_count"),
                pl.col("volume").filter(pl.col("side") > 0).sum().fill_null(0).alias("buy_volume"),
                pl.col("volume").filter(pl.col("side") < 0).sum().fill_null(0).alias("sell_volume"),
                pl.lit(expected_imbalance).alias("expected_imbalance"),
                pl.lit(cumulative_theta).alias("cumulative_theta"),
                # AFML diagnostic columns
                pl.lit(expected_t).alias("expected_t"),
                pl.lit(p_buy).alias("p_buy"),
                pl.lit(v_plus).alias("v_plus"),
                pl.lit(e_v).alias("e_v"),
            ],
        ).with_columns(
            [
                (pl.col("buy_volume") - pl.col("sell_volume")).alias("imbalance"),
            ],
        )

    def _empty_imbalance_bars_df(self) -> pl.DataFrame:
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
                "imbalance": [],
                "cumulative_theta": [],
                "expected_imbalance": [],
                # AFML diagnostic columns
                "expected_t": [],
                "p_buy": [],
                "v_plus": [],
                "e_v": [],
            },
        )


__all__ = [
    "DollarBarSamplerVectorized",
    "ImbalanceBarSamplerVectorized",
    "TickBarSamplerVectorized",
    "VolumeBarSamplerVectorized",
]
