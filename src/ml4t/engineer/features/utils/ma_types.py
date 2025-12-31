"""
Moving Average Type Dispatcher for TA-Lib compatibility.

Maps TA-Lib MA type codes to ml4t.engineer implementations.
"""

from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from ml4t.engineer.features.trend.dema import dema_numba
from ml4t.engineer.features.trend.ema import ema_numba
from ml4t.engineer.features.trend.kama import kama_numba
from ml4t.engineer.features.trend.sma import sma_numba
from ml4t.engineer.features.trend.t3 import t3_numba
from ml4t.engineer.features.trend.tema import tema_numba
from ml4t.engineer.features.trend.trima import trima_numba
from ml4t.engineer.features.trend.wma import wma_numba

# Type alias for NDArray
NDArrayFloat: TypeAlias = npt.NDArray[np.float64]


def apply_ma(
    close: npt.NDArray[np.float64],
    period: int,
    matype: int = 0,
) -> npt.NDArray[np.float64]:
    """
    Apply moving average based on TA-Lib matype code.

    Parameters
    ----------
    close : npt.NDArray
        Input close
    period : int
        Period for moving average
    matype : int, default 0
        Moving average type:
        - 0 = SMA (Simple Moving Average)
        - 1 = EMA (Exponential Moving Average)
        - 2 = WMA (Weighted Moving Average)
        - 3 = DEMA (Double Exponential Moving Average)
        - 4 = TEMA (Triple Exponential Moving Average)
        - 5 = TRIMA (Triangular Moving Average)
        - 6 = KAMA (Kaufman Adaptive Moving Average)
        - 7 = MAMA (MESA Adaptive Moving Average) - NOT IMPLEMENTED
        - 8 = T3 (Triple Exponential Moving Average)

    Returns
    -------
    npt.NDArray
        Moving average close

    Raises
    ------
    ValueError
        If matype is not supported

    Notes
    -----
    MAMA (matype=7) is not implemented as it requires additional parameters
    beyond period. Use KAMA (matype=6) as an adaptive alternative.
    """
    if matype == 0:
        return cast(NDArrayFloat, sma_numba(close, period))
    elif matype == 1:
        return cast(NDArrayFloat, ema_numba(close, period))
    elif matype == 2:
        return cast(NDArrayFloat, wma_numba(close, period))
    elif matype == 3:
        return cast(NDArrayFloat, dema_numba(close, period))
    elif matype == 4:
        return cast(NDArrayFloat, tema_numba(close, period))
    elif matype == 5:
        return cast(NDArrayFloat, trima_numba(close, period))
    elif matype == 6:
        return cast(NDArrayFloat, kama_numba(close, timeperiod=period))
    elif matype == 7:
        raise ValueError(
            "MAMA (matype=7) is not supported. "
            "MAMA requires additional parameters (fastlimit, slowlimit). "
            "Use KAMA (matype=6) as an adaptive alternative."
        )
    elif matype == 8:
        return cast(NDArrayFloat, t3_numba(close, period))
    else:
        raise ValueError(
            f"Invalid matype={matype}. Must be 0-8 (excluding 7). See docstring for valid MA types."
        )


__all__ = ["apply_ma"]
