# mypy: disable-error-code="union-attr"
"""
STOCHF - Stochastic Fast.

The Fast Stochastic Oscillator is the raw %K and %D without additional smoothing.
It's more sensitive to price changes than the regular Stochastic.

STOCHF is identical to STOCH with no additional smoothing (fastk_period=1).
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.features.utils.ma_types import apply_ma


@njit  # type: ignore[misc]
def stochf_fastk_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    fastk_period: int,
) -> npt.NDArray[np.float64]:
    """Calculate Fast Stochastic %K using Numba for performance."""
    n = len(close)
    fastk = np.full(n, np.nan)

    # Calculate %K close
    for i in range(fastk_period - 1, n):
        # Find highest high and lowest low in the period
        period_high = high[i - fastk_period + 1]
        period_low = low[i - fastk_period + 1]

        for j in range(i - fastk_period + 2, i + 1):
            period_high = max(period_high, high[j])
            period_low = min(period_low, low[j])

        # Calculate %K
        hl_diff = period_high - period_low
        if abs(hl_diff) < 1e-10:
            fastk[i] = 0.0  # TA-Lib returns 0 when no range
        else:
            fastk[i] = 100.0 * (close[i] - period_low) / hl_diff

    return fastk


def stochf_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    fastk_period: int,
    fastd_period: int,
    fastd_matype: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate Fast Stochastic using Numba for performance.

    Parameters
    ----------
    high, low, close : npt.NDArray
        Price arrays
    fastk_period : int
        Period for %K
    fastd_period : int
        Period for %D smoothing
    fastd_matype : int, default 0
        Moving average type for %D (0=SMA, 1=EMA, etc.)

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (%K, %D) arrays
    """
    n = len(close)

    # Calculate intermediate %K close
    intermediate_k = stochf_fastk_numba(high, low, close, fastk_period)

    # Calculate %D using specified MA type on intermediate %K
    fastd = apply_ma(intermediate_k, fastd_period, fastd_matype)

    # TA-Lib output starts at this index
    output_start = fastk_period + fastd_period - 2

    # Copy intermediate %K to output %K starting from output_start
    fastk = np.full(n, np.nan)
    fastk[output_start:] = intermediate_k[output_start:]

    return fastk, fastd


@feature(
    name="stochf",
    category="momentum",
    description="Stochastic Fast - fast version without smoothing",
    lookback=0,
    value_range=(0.0, 100.0),
    normalized=True,
    formula="",
    ta_lib_compatible=True,
)
def stochf(
    high: npt.NDArray[np.float64] | pl.Expr | str,
    low: npt.NDArray[np.float64] | pl.Expr | str,
    close: npt.NDArray[np.float64] | pl.Expr | str,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
    return_pair: bool = False,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | pl.Expr:
    """
    Calculate Fast Stochastic Oscillator.

    The Fast Stochastic is more sensitive than the regular Stochastic,
    providing earlier signals but with more noise.

    Parameters
    ----------
    high : array-like or Polars expression
        High close
    low : array-like or Polars expression
        Low close
    close : array-like or Polars expression
        Close close
    fastk_period : int, default 5
        Period for %K calculation
    fastd_period : int, default 3
        Period for %D smoothing
    fastd_matype : int, default 0
        Moving average type for %D:
        0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 8=T3
    return_pair : bool, default False
        If True, return (%K, %D) tuple. If False, return only %K.

    Returns
    -------
    npt.NDArray[np.float64] or tuple or pl.Expr
        Fast Stochastic %K, or (%K, %D) if return_pair=True

    Notes
    -----
    Fast Stochastic formulas:
    %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
    %D = SMA(%K, fastd_period)

    The difference from regular STOCH is that there's no additional smoothing
    of %K before calculating %D.

    Examples
    --------
    >>> import numpy as np
    >>> import ml4t.engineer.features.ta as ta as qta
    >>>
    >>> high = np.array([110, 112, 111, 113, 115, 114, 116])
    >>> low = np.array([108, 109, 107, 110, 112, 111, 113])
    >>> close = np.array([109, 111, 108, 112, 114, 112, 115])
    >>>
    >>> # Get %K only
    >>> fastk = qta.stochf(high, low, close, 5, 3)
    >>>
    >>> # Get both %K and %D
    >>> fastk, fastd = qta.stochf(high, low, close, 5, 3, return_pair=True)

    Using with Polars:
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "high": high, "low": low, "close": close
    ... })
    >>> result = df.with_columns([
    ...     qta.stochf("high", "low", "close", 5, 3).alias("fastk")
    ... ])
    """
    if isinstance(high, pl.Expr | str):
        # Return Polars expression using map_batches for complex calculations
        if isinstance(high, str):
            high = pl.col(high)
        if isinstance(low, str):
            low = pl.col(low)
        if isinstance(close, str):
            close = pl.col(close)
        # Type narrowing: at this point all must be Expr
        assert isinstance(low, pl.Expr) and isinstance(close, pl.Expr)

        if return_pair:
            # For pairs, we need to use struct to return multiple columns
            def calc_stochf_pair(s: pl.Series) -> dict[str, pl.Series]:
                if isinstance(s[0], dict):
                    # When using struct, we get a list of dicts
                    h = np.array([x["high"] for x in s])
                    lo = np.array([x["low"] for x in s])
                    c = np.array([x["close"] for x in s])
                else:
                    # Direct array input
                    h = s.struct.field("high").to_numpy()
                    lo = s.struct.field("low").to_numpy()
                    c = s.struct.field("close").to_numpy()
                fastk, fastd = stochf_numba(h, lo, c, fastk_period, fastd_period, fastd_matype)
                return {"fastk": pl.Series(fastk), "fastd": pl.Series(fastd)}

            return pl.struct(
                [high.alias("high"), low.alias("low"), close.alias("close")],
            ).map_batches(
                calc_stochf_pair,
                return_dtype=pl.Struct(
                    [pl.Field("fastk", pl.Float64), pl.Field("fastd", pl.Float64)],
                ),
            )

        def calc_stochf_k(s: pl.Series) -> pl.Series:
            if isinstance(s[0], dict):
                # When using struct, we get a list of dicts
                h = np.array([x["high"] for x in s])
                lo = np.array([x["low"] for x in s])
                c = np.array([x["close"] for x in s])
            else:
                # Direct array input
                h = s.struct.field("high").to_numpy()
                lo = s.struct.field("low").to_numpy()
                c = s.struct.field("close").to_numpy()
            fastk, _ = stochf_numba(h, lo, c, fastk_period, fastd_period, fastd_matype)
            return pl.Series(fastk)

        return pl.struct(
            [high.alias("high"), low.alias("low"), close.alias("close")],
        ).map_batches(calc_stochf_k, return_dtype=pl.Float64)

    # Handle numpy arrays
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("high, low, and close must have the same length")

    if fastk_period <= 0:
        raise ValueError("fastk_period must be > 0")

    if fastd_period <= 0:
        raise ValueError("fastd_period must be > 0")

    if len(high) < fastk_period + fastd_period - 1:
        if return_pair:
            return np.full(len(high), np.nan), np.full(len(high), np.nan)
        return np.full(len(high), np.nan)

    fastk, fastd = stochf_numba(high, low, close, fastk_period, fastd_period, fastd_matype)

    if return_pair:
        return fastk, fastd
    return fastk


# Make available at package level
__all__ = ["stochf"]
