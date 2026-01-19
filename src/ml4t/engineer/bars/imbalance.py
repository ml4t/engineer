"""Imbalance bar sampler implementations.

AFML-compliant imbalance bars per López de Prado Chapter 2.3.

Two types implemented:

1. **Tick Imbalance Bars (TIBs)**:
   - θ = Σ b_t (sum of trade signs)
   - E[θ_T] = E[T] × |2P[b=1] - 1|

2. **Volume Imbalance Bars (VIBs)**:
   - θ = Σ b_t × v_t (volume-weighted)
   - E[θ_T] = E[T] × |2v⁺ - E[v]|

Where:
    E[T] = EWMA of bar lengths (ticks per bar)
    P[b=1] = probability of buy
    v⁺ = P[b=1] × E[v|b=1] = expected buy volume contribution
    E[v] = unconditional mean volume per tick
"""

import warnings

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
    initial_expected_t: float,
    initial_p_buy: float,
    initial_v_buy: float,
    initial_v: float,
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
    """Calculate AFML-compliant volume imbalance bar indices.

    AFML Chapter 2.3 formula:
        E[θ_T] = E[T] × |2v⁺ - E[v]|

    Where:
        E[T] = EWMA of bar lengths (ticks per bar)
        v⁺ = P[b=1] × E[v|b=1] = expected buy volume contribution
        E[v] = unconditional mean volume per tick

    Parameters
    ----------
    volumes : ndarray
        Array of volume values
    sides : ndarray
        Array of trade signs (+1 for buy, -1 for sell)
    initial_expected_t : float
        Initial expected ticks per bar (E[T])
    initial_p_buy : float
        Initial buy probability P[b=1]
    initial_v_buy : float
        Initial expected buy volume E[v|b=1]
    initial_v : float
        Initial unconditional mean volume E[v]
    alpha : float
        EWMA decay factor for updating expectations
    min_bars_warmup : int
        Number of bars before starting EWMA updates

    Returns
    -------
    tuple of arrays
        (bar_indices, expected_thetas, cumulative_thetas, expected_ts, p_buys, v_pluses, e_vs)
    """
    n = len(volumes)

    # Pre-allocate output lists
    bar_indices = []
    expected_thetas = []
    cumulative_thetas = []
    expected_ts = []
    p_buys = []
    v_pluses = []
    e_vs = []

    # Initialize EWMA state
    expected_t = initial_expected_t
    p_buy = initial_p_buy
    v_buy = initial_v_buy
    v_all = initial_v

    # Within-bar accumulators
    cumulative_theta = 0.0
    bar_tick_count = 0
    bar_buy_count = 0
    bar_buy_volume = 0.0
    bar_total_volume = 0.0

    n_bars = 0

    for i in range(n):
        vol = volumes[i]
        side = sides[i]
        is_buy = side > 0

        # Accumulate signed volume (imbalance)
        cumulative_theta += vol * side
        bar_tick_count += 1
        bar_total_volume += vol

        if is_buy:
            bar_buy_count += 1
            bar_buy_volume += vol

        # Compute AFML threshold: E[T] × |2v⁺ - E[v]|
        v_plus = p_buy * v_buy
        expected_theta = expected_t * abs(2 * v_plus - v_all)

        # Check if bar should be formed
        if abs(cumulative_theta) >= expected_theta:
            # Record bar
            bar_indices.append(i)
            expected_thetas.append(expected_theta)
            cumulative_thetas.append(cumulative_theta)
            expected_ts.append(expected_t)
            p_buys.append(p_buy)
            v_pluses.append(v_plus)
            e_vs.append(v_all)

            n_bars += 1

            # Update EWMAs after warmup period
            if n_bars > min_bars_warmup:
                # Update E[T] - expected ticks per bar
                expected_t = alpha * bar_tick_count + (1 - alpha) * expected_t

                # Update P[b=1] - buy probability
                bar_p_buy = bar_buy_count / bar_tick_count if bar_tick_count > 0 else 0.5
                p_buy = alpha * bar_p_buy + (1 - alpha) * p_buy

                # Update E[v|b=1] - conditional mean buy volume
                if bar_buy_count > 0:
                    bar_v_buy = bar_buy_volume / bar_buy_count
                    v_buy = alpha * bar_v_buy + (1 - alpha) * v_buy

                # Update E[v] - unconditional mean volume
                bar_v = bar_total_volume / bar_tick_count if bar_tick_count > 0 else v_all
                v_all = alpha * bar_v + (1 - alpha) * v_all

            # Reset within-bar accumulators
            cumulative_theta = 0.0
            bar_tick_count = 0
            bar_buy_count = 0
            bar_buy_volume = 0.0
            bar_total_volume = 0.0

    return (
        np.array(bar_indices, dtype=np.int64),
        np.array(expected_thetas, dtype=np.float64),
        np.array(cumulative_thetas, dtype=np.float64),
        np.array(expected_ts, dtype=np.float64),
        np.array(p_buys, dtype=np.float64),
        np.array(v_pluses, dtype=np.float64),
        np.array(e_vs, dtype=np.float64),
    )


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _calculate_tick_imbalance_bars_nb(
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
]:
    """Calculate AFML-compliant tick imbalance bar indices.

    AFML Chapter 2.3 formula for Tick Imbalance Bars:
        θ_T = Σ b_t (sum of trade signs)
        E[θ_T] = E[T] × |2P[b=1] - 1|

    Parameters
    ----------
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
        (bar_indices, expected_thetas, cumulative_thetas, expected_ts, p_buys)
    """
    n = len(sides)

    # Pre-allocate output lists
    bar_indices = []
    expected_thetas = []
    cumulative_thetas = []
    expected_ts = []
    p_buys = []

    # Initialize EWMA state
    expected_t = initial_expected_t
    p_buy = initial_p_buy

    # Within-bar accumulators
    cumulative_theta = 0.0
    bar_tick_count = 0
    bar_buy_count = 0

    n_bars = 0

    for i in range(n):
        side = sides[i]
        is_buy = side > 0

        # Accumulate signed ticks (tick imbalance)
        cumulative_theta += side
        bar_tick_count += 1

        if is_buy:
            bar_buy_count += 1

        # Compute AFML threshold: E[T] × |2P[b=1] - 1|
        expected_theta = expected_t * abs(2 * p_buy - 1)

        # Check if bar should be formed
        if abs(cumulative_theta) >= expected_theta:
            # Record bar
            bar_indices.append(i)
            expected_thetas.append(expected_theta)
            cumulative_thetas.append(cumulative_theta)
            expected_ts.append(expected_t)
            p_buys.append(p_buy)

            n_bars += 1

            # Update EWMAs after warmup period
            if n_bars > min_bars_warmup:
                # Update E[T] - expected ticks per bar
                expected_t = alpha * bar_tick_count + (1 - alpha) * expected_t

                # Update P[b=1] - buy probability
                bar_p_buy = bar_buy_count / bar_tick_count if bar_tick_count > 0 else 0.5
                p_buy = alpha * bar_p_buy + (1 - alpha) * p_buy

            # Reset within-bar accumulators
            cumulative_theta = 0.0
            bar_tick_count = 0
            bar_buy_count = 0

    return (
        np.array(bar_indices, dtype=np.int64),
        np.array(expected_thetas, dtype=np.float64),
        np.array(cumulative_thetas, dtype=np.float64),
        np.array(expected_ts, dtype=np.float64),
        np.array(p_buys, dtype=np.float64),
    )


class TickImbalanceBarSampler(BarSampler):
    """Sample bars based on tick count imbalance (AFML-compliant TIBs).

    Tick Imbalance Bars (TIBs) sample when the cumulative signed tick count
    (number of buys - number of sells) reaches a dynamically adjusted threshold.

    AFML Threshold Formula:
        θ = Σ b_t (sum of trade signs)
        E[θ_T] = E[T] × |2P[b=1] - 1|

    Where:
        E[T] = EWMA of bar lengths (ticks per bar)
        P[b=1] = probability of buy

    This produces bar counts comparable to tick bars (both count ticks),
    unlike Volume Imbalance Bars which have thresholds scaled by volume.

    Parameters
    ----------
    expected_ticks_per_bar : int
        Expected number of ticks per bar (used to initialize E[T])
    alpha : float, default 0.1
        EWMA decay factor for updating expectations
    initial_p_buy : float, default 0.5
        Initial buy probability P[b=1]
    min_bars_warmup : int, default 10
        Number of bars before starting EWMA updates

    Examples
    --------
    >>> sampler = TickImbalanceBarSampler(
    ...     expected_ticks_per_bar=1000,
    ...     alpha=0.1
    ... )
    >>> bars = sampler.sample(tick_data)

    References
    ----------
    .. [1] López de Prado, M. (2018). Advances in Financial Machine Learning.
           John Wiley & Sons. Chapter 2.3: Information-Driven Bars.
    """

    def __init__(
        self,
        expected_ticks_per_bar: int,
        alpha: float = 0.1,
        initial_p_buy: float = 0.5,
        min_bars_warmup: int = 10,
    ):
        """Initialize tick imbalance bar sampler.

        Parameters
        ----------
        expected_ticks_per_bar : int
            Expected number of ticks per bar
        alpha : float, default 0.1
            EWMA decay factor
        initial_p_buy : float, default 0.5
            Initial buy probability P[b=1]
        min_bars_warmup : int, default 10
            Number of bars before starting EWMA updates
        """
        if expected_ticks_per_bar <= 0:
            raise ValueError("expected_ticks_per_bar must be positive")

        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")

        if not 0 <= initial_p_buy <= 1:
            raise ValueError("initial_p_buy must be in [0, 1]")

        if min_bars_warmup < 0:
            raise ValueError("min_bars_warmup must be non-negative")

        self.expected_ticks_per_bar = expected_ticks_per_bar
        self.alpha = alpha
        self.initial_p_buy = initial_p_buy
        self.min_bars_warmup = min_bars_warmup

    def sample(
        self,
        data: pl.DataFrame,
        include_incomplete: bool = False,
    ) -> pl.DataFrame:
        """Sample tick imbalance bars from data.

        Parameters
        ----------
        data : pl.DataFrame
            Tick data with columns: timestamp, price, volume, side
        include_incomplete : bool, default False
            Whether to include incomplete final bar

        Returns
        -------
        pl.DataFrame
            Sampled tick imbalance bars with AFML diagnostic columns:
            - expected_t: E[T] at bar formation
            - p_buy: P[b=1] at bar formation
            - expected_imbalance: AFML threshold E[θ_T]
            - cumulative_theta: Actual tick imbalance at bar formation
        """
        # Validate input
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Tick imbalance bars require 'side' column")

        if len(data) == 0:
            return self._empty_tick_imbalance_bars_df()

        # Extract arrays
        volumes = data["volume"].to_numpy().astype(np.float64)
        sides = data["side"].to_numpy().astype(np.float64)

        # Estimate initial P[b=1] from warmup data
        warmup_size = min(1000, len(sides))
        warmup_sides = sides[:warmup_size]
        estimated_p_buy = float(np.mean(warmup_sides > 0))

        # Use provided initial_p_buy or estimated
        p_buy_init = self.initial_p_buy if self.initial_p_buy != 0.5 else estimated_p_buy

        # Calculate bar indices using AFML-compliant Numba function
        (
            bar_indices,
            expected_thetas,
            cumulative_thetas,
            expected_ts,
            p_buys,
        ) = _calculate_tick_imbalance_bars_nb(
            sides,
            float(self.expected_ticks_per_bar),
            p_buy_init,
            self.alpha,
            self.min_bars_warmup,
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
            buy_count = int(np.sum(bar_sides > 0))
            sell_count = int(np.sum(bar_sides < 0))

            # Create bar with AFML diagnostic columns
            bar = self._create_ohlcv_bar(
                bar_ticks,
                additional_cols={
                    "buy_volume": float(buy_volume),
                    "sell_volume": float(sell_volume),
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    "tick_imbalance": buy_count - sell_count,
                    "cumulative_theta": float(cumulative_thetas[i]),
                    "expected_imbalance": float(expected_thetas[i]),
                    # AFML diagnostic columns
                    "expected_t": float(expected_ts[i]),
                    "p_buy": float(p_buys[i]),
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
                buy_count_incomplete = int(np.sum(bar_sides > 0))
                sell_count_incomplete = int(np.sum(bar_sides < 0))

                # Calculate current cumulative theta (tick count)
                cumulative_theta: float = float(np.sum(bar_sides))

                # Use last values or initial
                last_expected_t = (
                    expected_ts[-1] if len(expected_ts) > 0 else float(self.expected_ticks_per_bar)
                )
                last_p_buy = p_buys[-1] if len(p_buys) > 0 else p_buy_init
                expected_imbalance = last_expected_t * abs(2 * last_p_buy - 1)

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": float(buy_vol_incomplete),
                        "sell_volume": float(sell_vol_incomplete),
                        "buy_count": buy_count_incomplete,
                        "sell_count": sell_count_incomplete,
                        "tick_imbalance": buy_count_incomplete - sell_count_incomplete,
                        "cumulative_theta": float(cumulative_theta),
                        "expected_imbalance": float(expected_imbalance),
                        "expected_t": float(last_expected_t),
                        "p_buy": float(last_p_buy),
                    },
                )
                bars.append(bar)

        # Convert to DataFrame
        if not bars:
            return self._empty_tick_imbalance_bars_df()

        return pl.DataFrame(bars)

    def _empty_tick_imbalance_bars_df(self) -> pl.DataFrame:
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
                "buy_count": [],
                "sell_count": [],
                "tick_imbalance": [],
                "cumulative_theta": [],
                "expected_imbalance": [],
                "expected_t": [],
                "p_buy": [],
            },
        )


class ImbalanceBarSampler(BarSampler):
    """Sample bars based on order flow imbalance (AFML-compliant).

    Imbalance bars sample when the cumulative signed volume (buy - sell)
    reaches a dynamically adjusted threshold based on AFML Chapter 2.3.

    AFML Threshold Formula:
        E[θ_T] = E[T] × |2v⁺ - E[v]|

    Where:
        E[T] = EWMA of bar lengths (ticks per bar)
        v⁺ = P[b=1] × E[v|b=1] = expected buy volume contribution
        E[v] = unconditional mean volume per tick

    Parameters
    ----------
    expected_ticks_per_bar : int
        Expected number of ticks per bar (used to initialize E[T])
    initial_expectation : float, optional
        DEPRECATED. Use expected_ticks_per_bar instead.
    alpha : float, default 0.1
        EWMA decay factor for updating expectations
    initial_p_buy : float, default 0.5
        Initial buy probability P[b=1]
    min_bars_warmup : int, default 10
        Number of bars before starting EWMA updates

    Examples
    --------
    >>> sampler = ImbalanceBarSampler(
    ...     expected_ticks_per_bar=100,
    ...     alpha=0.1
    ... )
    >>> bars = sampler.sample(tick_data)

    References
    ----------
    .. [1] López de Prado, M. (2018). Advances in Financial Machine Learning.
           John Wiley & Sons. Chapter 2.3: Information-Driven Bars.
    """

    def __init__(
        self,
        expected_ticks_per_bar: int,
        initial_expectation: float | None = None,
        alpha: float = 0.1,
        initial_p_buy: float = 0.5,
        min_bars_warmup: int = 10,
    ):
        """Initialize imbalance bar sampler.

        Parameters
        ----------
        expected_ticks_per_bar : int
            Expected number of ticks per bar
        initial_expectation : float, optional
            DEPRECATED. Use expected_ticks_per_bar instead.
        alpha : float, default 0.1
            EWMA decay factor
        initial_p_buy : float, default 0.5
            Initial buy probability P[b=1]
        min_bars_warmup : int, default 10
            Number of bars before starting EWMA updates
        """
        if expected_ticks_per_bar <= 0:
            raise ValueError("expected_ticks_per_bar must be positive")

        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")

        if not 0 <= initial_p_buy <= 1:
            raise ValueError("initial_p_buy must be in [0, 1]")

        if min_bars_warmup < 0:
            raise ValueError("min_bars_warmup must be non-negative")

        if initial_expectation is not None:
            warnings.warn(
                "initial_expectation is deprecated and ignored. "
                "The AFML threshold E[T] × |2v⁺ - E[v]| is computed dynamically.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.expected_ticks_per_bar = expected_ticks_per_bar
        self.initial_expectation = initial_expectation  # Keep for backward compat
        self.alpha = alpha
        self.initial_p_buy = initial_p_buy
        self.min_bars_warmup = min_bars_warmup

        # Will be estimated from data
        self._initial_v_buy: float | None = None
        self._initial_v: float | None = None

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
            Sampled imbalance bars with AFML diagnostic columns:
            - expected_t: E[T] at bar formation
            - p_buy: P[b=1] at bar formation
            - v_plus: v⁺ = P[b=1] × E[v|b=1] at bar formation
            - e_v: E[v] at bar formation
            - expected_imbalance: AFML threshold E[θ_T]
            - cumulative_theta: Actual imbalance at bar formation
        """
        # Validate input
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Imbalance bars require 'side' column")

        if len(data) == 0:
            return self._empty_imbalance_bars_df()

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

        # Calculate bar indices using AFML-compliant Numba function
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

            # Create bar with AFML diagnostic columns
            bar = self._create_ohlcv_bar(
                bar_ticks,
                additional_cols={
                    "buy_volume": float(buy_volume),
                    "sell_volume": float(sell_volume),
                    "imbalance": float(imbalance),
                    "cumulative_theta": float(cumulative_thetas[i]),
                    "expected_imbalance": float(expected_thetas[i]),
                    # AFML diagnostic columns
                    "expected_t": float(expected_ts[i]),
                    "p_buy": float(p_buys[i]),
                    "v_plus": float(v_pluses[i]),
                    "e_v": float(e_vs[i]),
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

                # Use last values or initial
                last_expected_t = (
                    expected_ts[-1] if len(expected_ts) > 0 else float(self.expected_ticks_per_bar)
                )
                last_p_buy = p_buys[-1] if len(p_buys) > 0 else self.initial_p_buy
                last_v_plus = (
                    v_pluses[-1] if len(v_pluses) > 0 else last_p_buy * self._initial_v_buy
                )
                last_e_v = e_vs[-1] if len(e_vs) > 0 else self._initial_v
                expected_imbalance = last_expected_t * abs(2 * last_v_plus - last_e_v)

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": float(buy_vol_incomplete),
                        "sell_volume": float(sell_vol_incomplete),
                        "imbalance": float(imbalance_incomplete),
                        "cumulative_theta": float(cumulative_theta),
                        "expected_imbalance": float(expected_imbalance),
                        "expected_t": float(last_expected_t),
                        "p_buy": float(last_p_buy),
                        "v_plus": float(last_v_plus),
                        "e_v": float(last_e_v),
                    },
                )
                bars.append(bar)

        # Convert to DataFrame
        if not bars:
            return self._empty_imbalance_bars_df()

        return pl.DataFrame(bars)

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
                "expected_t": [],
                "p_buy": [],
                "v_plus": [],
                "e_v": [],
            },
        )


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _calculate_fixed_tick_imbalance_bars_nb(
    sides: npt.NDArray[np.float64],
    threshold: float,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Calculate fixed-threshold tick imbalance bar indices.

    Simple, stable algorithm with no adaptation.

    Parameters
    ----------
    sides : ndarray
        Array of trade signs (+1 for buy, -1 for sell)
    threshold : float
        Fixed imbalance threshold (bar forms when |Σ b_t| >= threshold)

    Returns
    -------
    tuple of arrays
        (bar_indices, cumulative_thetas)
    """
    n = len(sides)

    bar_indices = []
    cumulative_thetas = []

    cumulative_theta = 0.0

    for i in range(n):
        cumulative_theta += sides[i]

        if abs(cumulative_theta) >= threshold:
            bar_indices.append(i)
            cumulative_thetas.append(cumulative_theta)
            cumulative_theta = 0.0

    return (
        np.array(bar_indices, dtype=np.int64),
        np.array(cumulative_thetas, dtype=np.float64),
    )


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _calculate_fixed_volume_imbalance_bars_nb(
    volumes: npt.NDArray[np.float64],
    sides: npt.NDArray[np.float64],
    threshold: float,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Calculate fixed-threshold volume imbalance bar indices.

    Simple, stable algorithm with no adaptation.

    Parameters
    ----------
    volumes : ndarray
        Array of volume values
    sides : ndarray
        Array of trade signs (+1 for buy, -1 for sell)
    threshold : float
        Fixed imbalance threshold (bar forms when |Σ b_t × v_t| >= threshold)

    Returns
    -------
    tuple of arrays
        (bar_indices, cumulative_thetas)
    """
    n = len(volumes)

    bar_indices = []
    cumulative_thetas = []

    cumulative_theta = 0.0

    for i in range(n):
        cumulative_theta += volumes[i] * sides[i]

        if abs(cumulative_theta) >= threshold:
            bar_indices.append(i)
            cumulative_thetas.append(cumulative_theta)
            cumulative_theta = 0.0

    return (
        np.array(bar_indices, dtype=np.int64),
        np.array(cumulative_thetas, dtype=np.float64),
    )


class FixedTickImbalanceBarSampler(BarSampler):
    """Sample bars using fixed tick imbalance threshold.

    Unlike the adaptive AFML algorithm, this uses a fixed threshold that
    doesn't change during sampling. This avoids the threshold spiral issue
    that occurs with adaptive algorithms when order flow is imbalanced.

    **Recommended for production use** - more stable and predictable than
    the adaptive version.

    Parameters
    ----------
    threshold : int
        Fixed imbalance threshold. Bar forms when |Σ b_t| >= threshold.
        Typical values: 50-500 depending on desired bar frequency.

    Calibration
    -----------
    To calibrate threshold for N bars per day:
        1. Compute historical |mean imbalance| per tick
        2. threshold ≈ ticks_per_day / N × |2P[b=1] - 1|

    Or empirically: test a range and pick threshold giving desired bar count.

    Examples
    --------
    >>> sampler = FixedTickImbalanceBarSampler(threshold=100)
    >>> bars = sampler.sample(tick_data)

    Notes
    -----
    Advantages over adaptive (AFML) algorithm:
    - No threshold spiral with imbalanced order flow
    - Predictable bar count based on imbalance statistics
    - No feedback loops - stable by construction
    - Works consistently across all market conditions
    """

    def __init__(self, threshold: int):
        """Initialize fixed tick imbalance bar sampler.

        Parameters
        ----------
        threshold : int
            Fixed imbalance threshold (positive integer)
        """
        if threshold <= 0:
            raise ValueError("threshold must be positive")

        self.threshold = threshold

    def sample(
        self,
        data: pl.DataFrame,
        include_incomplete: bool = False,
    ) -> pl.DataFrame:
        """Sample fixed tick imbalance bars from data.

        Parameters
        ----------
        data : pl.DataFrame
            Tick data with columns: timestamp, price, volume, side
        include_incomplete : bool, default False
            Whether to include incomplete final bar

        Returns
        -------
        pl.DataFrame
            Sampled tick imbalance bars
        """
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Tick imbalance bars require 'side' column")

        if len(data) == 0:
            return self._empty_bars_df()

        # Extract arrays
        volumes = data["volume"].to_numpy().astype(np.float64)
        sides = data["side"].to_numpy().astype(np.float64)

        # Calculate bar indices
        bar_indices, cumulative_thetas = _calculate_fixed_tick_imbalance_bars_nb(
            sides, float(self.threshold)
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
            buy_count = int(np.sum(bar_sides > 0))
            sell_count = int(np.sum(bar_sides < 0))

            bar = self._create_ohlcv_bar(
                bar_ticks,
                additional_cols={
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    "tick_imbalance": buy_count - sell_count,
                    "cumulative_theta": float(cumulative_thetas[i]),
                    "threshold": float(self.threshold),
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

                buy_volume = float(np.sum(bar_volumes[bar_sides > 0]))
                sell_volume = float(np.sum(bar_volumes[bar_sides < 0]))
                buy_count = int(np.sum(bar_sides > 0))
                sell_count = int(np.sum(bar_sides < 0))
                cumulative_theta = float(np.sum(bar_sides))

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": buy_volume,
                        "sell_volume": sell_volume,
                        "buy_count": buy_count,
                        "sell_count": sell_count,
                        "tick_imbalance": buy_count - sell_count,
                        "cumulative_theta": cumulative_theta,
                        "threshold": float(self.threshold),
                    },
                )
                bars.append(bar)

        if not bars:
            return self._empty_bars_df()

        return pl.DataFrame(bars)

    def _empty_bars_df(self) -> pl.DataFrame:
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
                "buy_count": [],
                "sell_count": [],
                "tick_imbalance": [],
                "cumulative_theta": [],
                "threshold": [],
            },
        )


class FixedVolumeImbalanceBarSampler(BarSampler):
    """Sample bars using fixed volume imbalance threshold.

    Unlike the adaptive AFML algorithm, this uses a fixed threshold that
    doesn't change during sampling. This avoids instability issues that
    occur with adaptive algorithms.

    **Recommended for production use** - more stable and predictable than
    the adaptive version.

    Parameters
    ----------
    threshold : float
        Fixed volume imbalance threshold. Bar forms when |Σ b_t × v_t| >= threshold.
        Typical values: 10,000-1,000,000 depending on stock and desired frequency.

    Calibration
    -----------
    To calibrate threshold for N bars per day:
        1. Compute historical |mean signed volume| per tick
        2. threshold ≈ ticks_per_day / N × E[|signed_volume|]

    Or empirically: test a range and pick threshold giving desired bar count.

    Examples
    --------
    >>> sampler = FixedVolumeImbalanceBarSampler(threshold=50000)
    >>> bars = sampler.sample(tick_data)

    Notes
    -----
    Advantages over adaptive (AFML) algorithm:
    - No threshold spiral or collapse
    - Predictable bar count based on volume imbalance statistics
    - No feedback loops - stable by construction
    - Works consistently across all market conditions
    """

    def __init__(self, threshold: float):
        """Initialize fixed volume imbalance bar sampler.

        Parameters
        ----------
        threshold : float
            Fixed volume imbalance threshold (positive)
        """
        if threshold <= 0:
            raise ValueError("threshold must be positive")

        self.threshold = threshold

    def sample(
        self,
        data: pl.DataFrame,
        include_incomplete: bool = False,
    ) -> pl.DataFrame:
        """Sample fixed volume imbalance bars from data.

        Parameters
        ----------
        data : pl.DataFrame
            Tick data with columns: timestamp, price, volume, side
        include_incomplete : bool, default False
            Whether to include incomplete final bar

        Returns
        -------
        pl.DataFrame
            Sampled volume imbalance bars
        """
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Volume imbalance bars require 'side' column")

        if len(data) == 0:
            return self._empty_bars_df()

        # Extract arrays
        volumes = data["volume"].to_numpy().astype(np.float64)
        sides = data["side"].to_numpy().astype(np.float64)

        # Calculate bar indices
        bar_indices, cumulative_thetas = _calculate_fixed_volume_imbalance_bars_nb(
            volumes, sides, float(self.threshold)
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
                    "volume_imbalance": buy_volume - sell_volume,
                    "cumulative_theta": float(cumulative_thetas[i]),
                    "threshold": float(self.threshold),
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

                buy_volume = float(np.sum(bar_volumes[bar_sides > 0]))
                sell_volume = float(np.sum(bar_volumes[bar_sides < 0]))
                cumulative_theta = float(np.sum(bar_volumes * bar_sides))

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": buy_volume,
                        "sell_volume": sell_volume,
                        "volume_imbalance": buy_volume - sell_volume,
                        "cumulative_theta": cumulative_theta,
                        "threshold": float(self.threshold),
                    },
                )
                bars.append(bar)

        if not bars:
            return self._empty_bars_df()

        return pl.DataFrame(bars)

    def _empty_bars_df(self) -> pl.DataFrame:
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
                "volume_imbalance": [],
                "cumulative_theta": [],
                "threshold": [],
            },
        )


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _calculate_window_tick_imbalance_bars_nb(
    sides: npt.NDArray[np.float64],
    initial_expected_t: int,
    bar_window: int,
    tick_window: int,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Calculate window-based tick imbalance bar indices.

    Alternative to α-based EWMA that uses rolling windows:
    - E[T] = mean of last `bar_window` bar lengths
    - P[b=1] = mean of last `tick_window` tick signs
    - Old data falls out of windows → bounded drift (no spiral)

    Parameters
    ----------
    sides : ndarray
        Array of trade signs (+1 for buy, -1 for sell)
    initial_expected_t : int
        Initial expected ticks per bar before first bar forms
    bar_window : int
        Number of recent bars to average for E[T]
    tick_window : int
        Number of recent ticks to average for P[b=1]

    Returns
    -------
    tuple of arrays
        (bar_indices, expected_thetas, cumulative_thetas, expected_ts, p_buys)
    """
    n = len(sides)

    bar_indices = []
    expected_thetas = []
    cumulative_thetas = []
    expected_ts = []
    p_buys = []

    # Track recent bar lengths for E[T] calculation
    recent_bar_lengths: list[int] = []

    # Initialize state
    expected_t = float(initial_expected_t)
    cumulative_theta = 0.0
    bar_start_idx = 0

    # Track whether we've exited warmup
    warmup_complete = False

    for i in range(n):
        cumulative_theta += sides[i]

        # Wait for tick_window to fill before forming bars
        # This ensures P[b=1] estimate is stable before threshold checking
        if i < tick_window - 1:
            continue

        # On first tick after warmup, reset bar tracking
        if not warmup_complete:
            warmup_complete = True
            cumulative_theta = sides[i]  # Start fresh with just this tick
            bar_start_idx = i

        # Calculate P[b=1] from recent ticks (rolling window)
        window_start = i - tick_window + 1
        window_sides = sides[window_start : i + 1]
        p_buy = float(np.sum(window_sides > 0)) / tick_window

        # Calculate expected imbalance threshold
        expected_imbalance = expected_t * abs(2 * p_buy - 1)

        # Check if bar should form (only if threshold is meaningful)
        if expected_imbalance > 0 and abs(cumulative_theta) >= expected_imbalance:
            bar_indices.append(i)
            expected_thetas.append(expected_imbalance)
            cumulative_thetas.append(cumulative_theta)
            expected_ts.append(expected_t)
            p_buys.append(p_buy)

            # Update E[T] from recent bar lengths
            bar_length = i - bar_start_idx + 1
            recent_bar_lengths.append(bar_length)
            if len(recent_bar_lengths) > bar_window:
                recent_bar_lengths.pop(0)

            # E[T] = mean of recent bar lengths
            if len(recent_bar_lengths) > 0:
                expected_t = float(np.mean(np.array(recent_bar_lengths)))

            # Reset for next bar
            cumulative_theta = 0.0
            bar_start_idx = i + 1

    return (
        np.array(bar_indices, dtype=np.int64),
        np.array(expected_thetas, dtype=np.float64),
        np.array(cumulative_thetas, dtype=np.float64),
        np.array(expected_ts, dtype=np.float64),
        np.array(p_buys, dtype=np.float64),
    )


class WindowTickImbalanceBarSampler(BarSampler):
    """Sample tick imbalance bars using window-based estimation.

    Alternative to α-based EWMA that uses rolling windows instead of
    exponential decay for parameter estimation.

    Key difference from α-based version:
    - E[T] computed from rolling mean of last N bar lengths
    - P[b=1] computed from rolling mean of last M tick signs
    - Old data falls out of windows → bounded adaptation → no threshold spiral

    Parameters
    ----------
    initial_expected_t : int
        Initial expected ticks per bar (before first bar forms)
    bar_window : int, default 10
        Number of recent bars to average for E[T] estimation
    tick_window : int, default 1000
        Number of recent ticks to average for P[b=1] estimation

    Examples
    --------
    >>> sampler = WindowTickImbalanceBarSampler(
    ...     initial_expected_t=1000,
    ...     bar_window=10,    # E[T] from last 10 bars
    ...     tick_window=5000, # P[b=1] from last 5000 ticks
    ... )
    >>> bars = sampler.sample(tick_data)

    Notes
    -----
    Recommended settings:
    - bar_window: 5-20 (small, since bar count is limited)
    - tick_window: 1000-10000 (large, for stable P[b=1] estimate)
    - initial_expected_t: Rough estimate of ticks per bar
    """

    def __init__(
        self,
        initial_expected_t: int,
        bar_window: int = 10,
        tick_window: int = 1000,
    ):
        if initial_expected_t <= 0:
            raise ValueError("initial_expected_t must be positive")
        if bar_window <= 0:
            raise ValueError("bar_window must be positive")
        if tick_window <= 0:
            raise ValueError("tick_window must be positive")

        self.initial_expected_t = initial_expected_t
        self.bar_window = bar_window
        self.tick_window = tick_window

    def sample(
        self,
        data: pl.DataFrame,
        include_incomplete: bool = False,
    ) -> pl.DataFrame:
        """Sample window-based tick imbalance bars from data."""
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Tick imbalance bars require 'side' column")

        if len(data) == 0:
            return self._empty_bars_df()

        # Extract arrays
        volumes = data["volume"].to_numpy().astype(np.float64)
        sides = data["side"].to_numpy().astype(np.float64)

        # Calculate bar indices
        (
            bar_indices,
            expected_thetas,
            cumulative_thetas,
            expected_ts,
            p_buys,
        ) = _calculate_window_tick_imbalance_bars_nb(
            sides,
            self.initial_expected_t,
            self.bar_window,
            self.tick_window,
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
            buy_count = int(np.sum(bar_sides > 0))
            sell_count = int(np.sum(bar_sides < 0))

            bar = self._create_ohlcv_bar(
                bar_ticks,
                additional_cols={
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    "tick_imbalance": buy_count - sell_count,
                    "cumulative_theta": float(cumulative_thetas[i]),
                    "expected_imbalance": float(expected_thetas[i]),
                    "expected_t": float(expected_ts[i]),
                    "p_buy": float(p_buys[i]),
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

                buy_volume = float(np.sum(bar_volumes[bar_sides > 0]))
                sell_volume = float(np.sum(bar_volumes[bar_sides < 0]))
                buy_count = int(np.sum(bar_sides > 0))
                sell_count = int(np.sum(bar_sides < 0))

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": buy_volume,
                        "sell_volume": sell_volume,
                        "buy_count": buy_count,
                        "sell_count": sell_count,
                        "tick_imbalance": buy_count - sell_count,
                        "cumulative_theta": float(np.sum(bar_sides)),
                        "expected_imbalance": (
                            float(expected_thetas[-1]) if expected_thetas else 0.0
                        ),
                        "expected_t": (
                            float(expected_ts[-1])
                            if expected_ts
                            else float(self.initial_expected_t)
                        ),
                        "p_buy": float(p_buys[-1]) if p_buys else 0.5,
                    },
                )
                bars.append(bar)

        if not bars:
            return self._empty_bars_df()

        return pl.DataFrame(bars)

    def _empty_bars_df(self) -> pl.DataFrame:
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
                "buy_count": [],
                "sell_count": [],
                "tick_imbalance": [],
                "cumulative_theta": [],
                "expected_imbalance": [],
                "expected_t": [],
                "p_buy": [],
            },
        )


@jit(nopython=True, cache=True)  # type: ignore[misc]
def _calculate_window_volume_imbalance_bars_nb(
    volumes: npt.NDArray[np.float64],
    sides: npt.NDArray[np.float64],
    initial_expected_t: int,
    bar_window: int,
    tick_window: int,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Calculate window-based volume imbalance bar indices.

    Alternative to α-based EWMA that uses rolling windows:
    - E[T] = mean of last `bar_window` bar lengths
    - (2v⁺ - E[v]) = mean of last `tick_window` signed volumes
    - Old data falls out of windows → bounded drift (no spiral)

    Parameters
    ----------
    volumes : ndarray
        Array of volume values
    sides : ndarray
        Array of trade signs (+1 for buy, -1 for sell)
    initial_expected_t : int
        Initial expected ticks per bar before first bar forms
    bar_window : int
        Number of recent bars to average for E[T]
    tick_window : int
        Number of recent ticks to average for imbalance

    Returns
    -------
    tuple of arrays
        (bar_indices, expected_thetas, cumulative_thetas, expected_ts, imbalance_factors)
    """
    n = len(volumes)

    bar_indices = []
    expected_thetas = []
    cumulative_thetas = []
    expected_ts = []
    imbalance_factors = []

    # Track recent bar lengths for E[T] calculation
    recent_bar_lengths: list[int] = []

    # Initialize state
    expected_t = float(initial_expected_t)
    cumulative_theta = 0.0
    bar_start_idx = 0

    # Track whether we've exited warmup
    warmup_complete = False

    for i in range(n):
        signed_volume = volumes[i] * sides[i]
        cumulative_theta += signed_volume

        # Wait for tick_window to fill before forming bars
        # This ensures imbalance estimate is stable before threshold checking
        if i < tick_window - 1:
            continue

        # On first tick after warmup, reset bar tracking
        if not warmup_complete:
            warmup_complete = True
            cumulative_theta = signed_volume  # Start fresh with just this tick
            bar_start_idx = i

        # Calculate imbalance factor from recent signed volumes (rolling window)
        # AFML: (2v⁺ - E[v]) = mean of b_t × v_t
        window_start = i - tick_window + 1
        window_signed_vols = volumes[window_start : i + 1] * sides[window_start : i + 1]
        imbalance_factor = abs(np.mean(window_signed_vols))

        # Calculate expected imbalance threshold
        expected_imbalance = expected_t * imbalance_factor

        # Check if bar should form (only if threshold is meaningful)
        if expected_imbalance > 0 and abs(cumulative_theta) >= expected_imbalance:
            bar_indices.append(i)
            expected_thetas.append(expected_imbalance)
            cumulative_thetas.append(cumulative_theta)
            expected_ts.append(expected_t)
            imbalance_factors.append(imbalance_factor)

            # Update E[T] from recent bar lengths
            bar_length = i - bar_start_idx + 1
            recent_bar_lengths.append(bar_length)
            if len(recent_bar_lengths) > bar_window:
                recent_bar_lengths.pop(0)

            # E[T] = mean of recent bar lengths
            if len(recent_bar_lengths) > 0:
                expected_t = float(np.mean(np.array(recent_bar_lengths)))

            # Reset for next bar
            cumulative_theta = 0.0
            bar_start_idx = i + 1

    return (
        np.array(bar_indices, dtype=np.int64),
        np.array(expected_thetas, dtype=np.float64),
        np.array(cumulative_thetas, dtype=np.float64),
        np.array(expected_ts, dtype=np.float64),
        np.array(imbalance_factors, dtype=np.float64),
    )


class WindowVolumeImbalanceBarSampler(BarSampler):
    """Sample volume imbalance bars using window-based estimation.

    Alternative to α-based EWMA that uses rolling windows instead of
    exponential decay for parameter estimation.

    Key difference from α-based version:
    - E[T] computed from rolling mean of last N bar lengths
    - Imbalance factor computed from rolling mean of last M signed volumes
    - Old data falls out of windows → bounded adaptation → no threshold spiral

    Parameters
    ----------
    initial_expected_t : int
        Initial expected ticks per bar (before first bar forms)
    bar_window : int, default 10
        Number of recent bars to average for E[T] estimation
    tick_window : int, default 1000
        Number of recent ticks to average for imbalance estimation

    Examples
    --------
    >>> sampler = WindowVolumeImbalanceBarSampler(
    ...     initial_expected_t=5000,
    ...     bar_window=10,    # E[T] from last 10 bars
    ...     tick_window=5000, # Imbalance from last 5000 ticks
    ... )
    >>> bars = sampler.sample(tick_data)
    """

    def __init__(
        self,
        initial_expected_t: int,
        bar_window: int = 10,
        tick_window: int = 1000,
    ):
        if initial_expected_t <= 0:
            raise ValueError("initial_expected_t must be positive")
        if bar_window <= 0:
            raise ValueError("bar_window must be positive")
        if tick_window <= 0:
            raise ValueError("tick_window must be positive")

        self.initial_expected_t = initial_expected_t
        self.bar_window = bar_window
        self.tick_window = tick_window

    def sample(
        self,
        data: pl.DataFrame,
        include_incomplete: bool = False,
    ) -> pl.DataFrame:
        """Sample window-based volume imbalance bars from data."""
        self._validate_data(data)

        if "side" not in data.columns:
            raise DataValidationError("Volume imbalance bars require 'side' column")

        if len(data) == 0:
            return self._empty_bars_df()

        # Extract arrays
        volumes = data["volume"].to_numpy().astype(np.float64)
        sides = data["side"].to_numpy().astype(np.float64)

        # Calculate bar indices
        (
            bar_indices,
            expected_thetas,
            cumulative_thetas,
            expected_ts,
            imbalance_factors,
        ) = _calculate_window_volume_imbalance_bars_nb(
            volumes,
            sides,
            self.initial_expected_t,
            self.bar_window,
            self.tick_window,
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
                    "volume_imbalance": buy_volume - sell_volume,
                    "cumulative_theta": float(cumulative_thetas[i]),
                    "expected_imbalance": float(expected_thetas[i]),
                    "expected_t": float(expected_ts[i]),
                    "imbalance_factor": float(imbalance_factors[i]),
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

                buy_volume = float(np.sum(bar_volumes[bar_sides > 0]))
                sell_volume = float(np.sum(bar_volumes[bar_sides < 0]))

                bar = self._create_ohlcv_bar(
                    bar_ticks,
                    additional_cols={
                        "buy_volume": buy_volume,
                        "sell_volume": sell_volume,
                        "volume_imbalance": buy_volume - sell_volume,
                        "cumulative_theta": float(np.sum(bar_volumes * bar_sides)),
                        "expected_imbalance": (
                            float(expected_thetas[-1]) if expected_thetas else 0.0
                        ),
                        "expected_t": (
                            float(expected_ts[-1])
                            if expected_ts
                            else float(self.initial_expected_t)
                        ),
                        "imbalance_factor": (
                            float(imbalance_factors[-1]) if imbalance_factors else 0.0
                        ),
                    },
                )
                bars.append(bar)

        if not bars:
            return self._empty_bars_df()

        return pl.DataFrame(bars)

    def _empty_bars_df(self) -> pl.DataFrame:
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
                "volume_imbalance": [],
                "cumulative_theta": [],
                "expected_imbalance": [],
                "expected_t": [],
                "imbalance_factor": [],
            },
        )


__all__ = [
    "ImbalanceBarSampler",
    "TickImbalanceBarSampler",
    "FixedTickImbalanceBarSampler",
    "FixedVolumeImbalanceBarSampler",
    "WindowTickImbalanceBarSampler",
    "WindowVolumeImbalanceBarSampler",
]
