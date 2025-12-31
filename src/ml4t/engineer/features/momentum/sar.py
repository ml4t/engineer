# mypy: disable-error-code="no-any-return"
"""
Parabolic SAR (Stop and Reverse) - TA-Lib compatible implementation.

The Parabolic SAR is a trend-following indicator that provides exit points
for long or short positions. It was developed by J. Welles Wilder Jr.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def sar_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> npt.NDArray[np.float64]:
    """
    Parabolic SAR calculation exactly replicating TA-Lib algorithm.

    This is a line-by-line translation of the TA-Lib C implementation
    from ta_SAR.c, following the exact same logic and variable naming.

    Parameters
    ----------
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close
    acceleration : float, default 0.02
        Acceleration factor
    maximum : float, default 0.2
        Maximum acceleration factor

    Returns
    -------
    npt.NDArray
        Parabolic SAR close
    """
    n = len(high)
    if len(low) != n:
        return np.full(n, np.nan)

    result = np.full(n, np.nan)

    # TA-Lib requires at least 2 data points (startIdx = 1)
    if n < 2:
        return result

    # Following TA-Lib algorithm exactly
    start_idx = 1  # TA-Lib: if( startIdx < 1 ) startIdx = 1;
    end_idx = n - 1

    # Make sure there is still something to evaluate
    if start_idx > end_idx:
        return result

    # Make sure the acceleration and maximum are coherent
    af = acceleration
    if af > maximum:
        af = acceleration = maximum

    # Identify if the initial direction is long or short using MINUS_DM
    # This matches TA-Lib's logic exactly
    dm_minus = low[start_idx - 1] - low[start_idx]
    dm_minus = max(dm_minus, 0)

    is_long = 0 if dm_minus > 0 else 1  # 0=Short, 1=Long (default)

    out_idx = 0
    today_idx = start_idx

    # Write the first SAR (TA-Lib algorithm)
    new_high = high[today_idx - 1]
    new_low = low[today_idx - 1]

    if is_long == 1:
        ep = high[today_idx]  # Extreme point
        sar = new_low
    else:
        ep = low[today_idx]
        sar = new_high

    # Cheat on the newLow and newHigh for the first iteration (TA-Lib comment)
    new_low = low[today_idx]
    new_high = high[today_idx]

    while today_idx <= end_idx:
        prev_low = new_low
        prev_high = new_high
        new_low = low[today_idx]
        new_high = high[today_idx]
        today_idx += 1

        if is_long == 1:
            # Switch to short if the low penetrates the SAR value
            if new_low <= sar:
                # Switch and Override the SAR with the ep
                is_long = 0
                sar = ep

                # Make sure the override SAR is within yesterday's and today's range
                sar = max(sar, prev_high)
                sar = max(sar, new_high)

                # Output the override SAR
                result[today_idx - 1] = sar

                # Adjust af and ep
                af = acceleration
                ep = new_low

                # Calculate the new SAR
                sar = sar + af * (ep - sar)

                # Make sure the new SAR is within yesterday's and today's range
                sar = max(sar, prev_high)
                sar = max(sar, new_high)
            else:
                # No switch - Output the SAR (was calculated in the previous iteration)
                result[today_idx - 1] = sar

                # Adjust af and ep
                if new_high > ep:
                    ep = new_high
                    af += acceleration
                    af = min(af, maximum)

                # Calculate the new SAR
                sar = sar + af * (ep - sar)

                # Make sure the new SAR is within yesterday's and today's range
                sar = min(sar, prev_low)
                sar = min(sar, new_low)
        # Switch to long if the high penetrates the SAR value
        elif new_high >= sar:
            # Switch and Override the SAR with the ep
            is_long = 1
            sar = ep

            # Make sure the override SAR is within yesterday's and today's range
            sar = min(sar, prev_low)
            sar = min(sar, new_low)

            # Output the override SAR
            result[today_idx - 1] = sar

            # Adjust af and ep
            af = acceleration
            ep = new_high

            # Calculate the new SAR
            sar = sar + af * (ep - sar)

            # Make sure the new SAR is within yesterday's and today's range
            sar = min(sar, prev_low)
            sar = min(sar, new_low)
        else:
            # No switch - Output the SAR (was calculated in the previous iteration)
            result[today_idx - 1] = sar

            # Adjust af and ep
            if new_low < ep:
                ep = new_low
                af += acceleration
                af = min(af, maximum)

            # Calculate the new SAR
            sar = sar + af * (ep - sar)

            # Make sure the new SAR is within yesterday's and today's range
            sar = max(sar, prev_high)
            sar = max(sar, new_high)

        out_idx += 1

    return result


def sar_polars(
    high_column: str,
    low_column: str,
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> pl.Expr:
    """
    Parabolic SAR using Polars - delegates to Numba for exact TA-Lib compatibility.

    Parameters
    ----------
    high_column : str
        Column name for high close
    low_column : str
        Column name for low close
    acceleration : float, default 0.02
        Acceleration factor
    maximum : float, default 0.2
        Maximum acceleration factor

    Returns
    -------
    pl.Expr
        Polars expression for Parabolic SAR
    """
    return pl.struct([high_column, low_column]).map_batches(
        lambda s: pl.Series(
            sar_numba(
                s.struct.field(high_column).to_numpy(),
                s.struct.field(low_column).to_numpy(),
                acceleration,
                maximum,
            ),
        ),
        return_dtype=pl.Float64,
    )


@feature(
    name="sar",
    category="momentum",
    description="SAR - Parabolic Stop and Reverse",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def sar(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Parabolic SAR (Stop and Reverse) exactly matching TA-Lib.

    The Parabolic SAR is a trend-following indicator developed by J. Welles Wilder Jr.
    It provides potential reversal points by following price trends. When the trend
    is up, the SAR is below the price, and when the trend is down, the SAR is above.

    Parameters
    ----------
    high : array-like or str
        High close or column name
    low : array-like or str
        Low close or column name
    acceleration : float, default 0.02
        Acceleration factor - how quickly the SAR follows price
    maximum : float, default 0.2
        Maximum acceleration factor - caps the acceleration
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        Parabolic SAR close

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10])
    >>> low = np.array([9, 10, 11, 12, 13, 14, 13, 12, 11, 10, 9])
    >>> sar_values = sar(high, low, acceleration=0.02, maximum=0.2)

    Notes
    -----
    - The first valid SAR value appears at index 1 (second bar)
    - Initial direction is determined by comparing directional movement
    - SAR switches sides when price penetrates the SAR level
    - Acceleration factor increases each time a new extreme is reached
    - SAR is constrained to not penetrate the previous two bars' range
    """
    # Validate parameters
    if acceleration <= 0:
        raise InvalidParameterError(f"acceleration must be > 0, got {acceleration}")
    if maximum <= 0:
        raise InvalidParameterError(f"maximum must be > 0, got {maximum}")
    if acceleration > maximum:
        raise InvalidParameterError(
            f"acceleration ({acceleration}) must be <= maximum ({maximum})",
        )

    if isinstance(high, str) and isinstance(low, str):
        # Column names provided for Polars
        return sar_polars(high, low, acceleration, maximum)

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

    return sar_numba(high, low, acceleration, maximum)


# Export all functions
__all__ = ["sar", "sar_numba", "sar_polars"]
