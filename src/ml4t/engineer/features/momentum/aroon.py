# mypy: disable-error-code="no-any-return"
"""
Aroon Indicators (AROON, AROONOSC) - TA-Lib compatible implementation.

Aroon indicators help identify when trends are likely to change direction.
They measure how long it has been since the highest high and lowest low.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def aroon_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    timeperiod: int = 14,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    AROON calculation exactly replicating TA-Lib algorithm.

    This is a line-by-line translation of the TA-Lib C implementation
    from ta_AROON.c, following the exact same logic and variable naming.

    Parameters
    ----------
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close
    timeperiod : int, default 14
        Number of periods for calculation

    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        (aroon_down, aroon_up) - Note: TA-Lib returns down first, then up
    """
    n = len(high)
    if len(low) != n:
        return np.full(n, np.nan), np.full(n, np.nan)

    aroon_down = np.full(n, np.nan)
    aroon_up = np.full(n, np.nan)

    # TA-Lib validation: need at least timeperiod data points
    if n < timeperiod:
        return aroon_down, aroon_up

    # Following TA-Lib algorithm exactly
    # Move up the start index if there is not enough initial data
    start_idx = (
        timeperiod  # In TA-Lib: if( startIdx < optInTimePeriod ) startIdx = optInTimePeriod;
    )
    end_idx = n - 1

    # Make sure there is still something to evaluate
    if start_idx > end_idx:
        return aroon_down, aroon_up

    # Initialize TA-Lib variables
    out_idx = 0
    today = start_idx
    trailing_idx = start_idx - timeperiod
    lowest_idx = -1
    highest_idx = -1
    lowest = 0.0
    highest = 0.0
    factor = 100.0 / timeperiod

    while today <= end_idx:
        # Keep track of the lowestIdx (TA-Lib algorithm)
        tmp = low[today]
        if lowest_idx < trailing_idx:
            lowest_idx = trailing_idx
            lowest = low[lowest_idx]
            i = lowest_idx
            while i < today:  # Changed from <= to < since we increment first
                i += 1
                tmp = low[i]
                if tmp <= lowest:
                    lowest_idx = i
                    lowest = tmp
        elif tmp <= lowest:
            lowest_idx = today
            lowest = tmp

        # Keep track of the highestIdx (TA-Lib algorithm)
        tmp = high[today]
        if highest_idx < trailing_idx:
            highest_idx = trailing_idx
            highest = high[highest_idx]
            i = highest_idx
            while i < today:  # Changed from <= to < since we increment first
                i += 1
                tmp = high[i]
                if tmp >= highest:
                    highest_idx = i
                    highest = tmp
        elif tmp >= highest:
            highest_idx = today
            highest = tmp

        # Calculate Aroon close (exact TA-Lib formulas)
        aroon_up[today] = factor * (timeperiod - (today - highest_idx))
        aroon_down[today] = factor * (timeperiod - (today - lowest_idx))

        out_idx += 1
        trailing_idx += 1
        today += 1

    return aroon_down, aroon_up


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def aroonosc_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    timeperiod: int = 14,
) -> npt.NDArray[np.float64]:
    """
    AROONOSC calculation: Aroon Up - Aroon Down.

    Parameters
    ----------
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close
    timeperiod : int, default 14
        Number of periods for calculation

    Returns
    -------
    npt.NDArray
        Aroon Oscillator close
    """
    aroon_down, aroon_up = aroon_numba(high, low, timeperiod)
    return aroon_up - aroon_down


def aroon_polars(high_column: str, low_column: str, timeperiod: int = 14) -> pl.Expr:
    """
    AROON using Polars - delegates to Numba for exact TA-Lib compatibility.

    Returns a struct with 'down' and 'up' fields.

    Parameters
    ----------
    high_column : str
        Column name for high close
    low_column : str
        Column name for low close
    timeperiod : int, default 14
        Number of periods for calculation

    Returns
    -------
    pl.Expr
        Polars expression returning struct with 'down' and 'up' fields
    """
    return pl.struct([high_column, low_column]).map_batches(
        lambda s: pl.DataFrame(
            {
                "down": aroon_numba(
                    s.struct.field(high_column).to_numpy(),
                    s.struct.field(low_column).to_numpy(),
                    timeperiod,
                )[0],
                "up": aroon_numba(
                    s.struct.field(high_column).to_numpy(),
                    s.struct.field(low_column).to_numpy(),
                    timeperiod,
                )[1],
            },
        ).to_struct(""),
        return_dtype=pl.Struct(
            [pl.Field("down", pl.Float64), pl.Field("up", pl.Float64)],
        ),
    )


def aroonosc_polars(high_column: str, low_column: str, timeperiod: int = 14) -> pl.Expr:
    """
    AROONOSC using Polars - delegates to Numba for exact TA-Lib compatibility.

    Parameters
    ----------
    high_column : str
        Column name for high close
    low_column : str
        Column name for low close
    timeperiod : int, default 14
        Number of periods for calculation

    Returns
    -------
    pl.Expr
        Polars expression for Aroon Oscillator
    """
    return pl.struct([high_column, low_column]).map_batches(
        lambda s: pl.Series(
            aroonosc_numba(
                s.struct.field(high_column).to_numpy(),
                s.struct.field(low_column).to_numpy(),
                timeperiod,
            ),
        ),
        return_dtype=pl.Float64,
    )


@feature(
    name="aroon",
    category="momentum",
    description="Aroon - identifies trend changes and strength",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def aroon(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 14,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | pl.Expr:
    """
    Aroon indicators exactly matching TA-Lib.

    The Aroon indicators measure how long it has been since the highest high
    and lowest low occurred over a given period. They help identify when
    trends are likely to change direction.

    Parameters
    ----------
    high : array-like or str
        High close or column name
    low : array-like or str
        Low close or column name
    timeperiod : int, default 14
        Number of periods for calculation
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    tuple of arrays or Polars expression
        (aroon_down, aroon_up) for NumPy
        Struct with 'down' and 'up' fields for Polars

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10])
    >>> low = np.array([9, 10, 11, 12, 13, 14, 13, 12, 11, 10, 9])
    >>> aroon_down, aroon_up = aroon(high, low, timeperiod=5)

    Notes
    -----
    - Aroon Up = 100 * (timeperiod - periods_since_highest_high) / timeperiod
    - Aroon Down = 100 * (timeperiod - periods_since_lowest_low) / timeperiod
    - Values range from 0 to 100
    - High Aroon Up (>70) suggests uptrend
    - High Aroon Down (>70) suggests downtrend
    - When both are high, indicates consolidation
    - When both are low, indicates trend weakness
    """
    # Validate parameters
    if timeperiod < 2:
        raise InvalidParameterError(f"timeperiod must be >= 2, got {timeperiod}")

    if isinstance(high, str) and isinstance(low, str):
        # Column names provided for Polars
        return aroon_polars(high, low, timeperiod)

    # Convert to numpy if needed
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()

    # Ensure both arrays have the same length
    if len(high) != len(low):
        raise ValueError(
            f"high and low must have the same length. Got {len(high)} and {len(low)}",
        )

    return tuple(aroon_numba(high, low, timeperiod))


@feature(
    name="aroonosc",
    category="momentum",
    description="Aroon Oscillator - difference between Aroon Up and Aroon Down",
    lookback="timeperiod",
    normalized=True,
    value_range=(-100.0, 100.0),
    formula="AroonOsc = AroonUp - AroonDown",
    ta_lib_compatible=True,
    input_type="HL",
    parameters={"timeperiod": 14},
    tags=["oscillator", "trend"],
)
def aroonosc(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Aroon Oscillator exactly matching TA-Lib.

    The Aroon Oscillator is simply Aroon Up minus Aroon Down. It oscillates
    between -100 and +100, with positive close indicating upward momentum
    and negative close indicating downward momentum.

    Parameters
    ----------
    high : array-like or str
        High close or column name
    low : array-like or str
        Low close or column name
    timeperiod : int, default 14
        Number of periods for calculation
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        Aroon Oscillator close

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10])
    >>> low = np.array([9, 10, 11, 12, 13, 14, 13, 12, 11, 10, 9])
    >>> aroon_osc = aroonosc(high, low, timeperiod=5)

    Notes
    -----
    - Aroon Oscillator = Aroon Up - Aroon Down
    - Values range from -100 to +100
    - Positive close indicate upward momentum
    - Negative close indicate downward momentum
    - Zero line crossovers can signal trend changes
    """
    # Validate parameters
    if timeperiod < 2:
        raise InvalidParameterError(f"timeperiod must be >= 2, got {timeperiod}")

    if isinstance(high, str) and isinstance(low, str):
        # Column names provided for Polars
        return aroonosc_polars(high, low, timeperiod)

    # Convert to numpy if needed
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()

    # Ensure both arrays have the same length
    if len(high) != len(low):
        raise ValueError(
            f"high and low must have the same length. Got {len(high)} and {len(low)}",
        )

    return aroonosc_numba(high, low, timeperiod)


# Export all functions
__all__ = [
    "aroon",
    "aroon_numba",
    "aroon_polars",
    "aroonosc",
    "aroonosc_numba",
    "aroonosc_polars",
]
