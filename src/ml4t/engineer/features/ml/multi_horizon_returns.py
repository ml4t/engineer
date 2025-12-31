import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_list_length,
)


@feature(
    name="multi_horizon_returns",
    category="ml",
    description="Multi-Horizon Returns - returns at multiple horizons",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def multi_horizon_returns(
    close: pl.Expr | str,
    horizons: list[int] | None = None,
    log_returns: bool = False,
) -> dict[str, pl.Expr]:
    """Calculate returns at multiple horizons for multi-task learning.

    Creates return features at different time horizons, useful for
    models that predict multiple time periods simultaneously.

    Parameters
    ----------
    close : pl.Expr | str
        Close price column
    horizons : List[int], optional
        List of horizons in minutes (default: [1, 5, 10, 30, 60])
    log_returns : bool, default False
        Whether to calculate log returns

    Returns
    -------
    dict[str, pl.Expr]
        Dictionary of return expressions for each horizon

    Raises
    ------
    ValueError
        If horizons list is empty or contains non-positive close
    TypeError
        If horizons contains non-integers or log_returns is not boolean
    """
    if horizons is None:
        horizons = [1, 5, 10, 30, 60]

    # Validate inputs
    validate_list_length(horizons, min_length=1, name="horizons")
    for i, h in enumerate(horizons):
        if not isinstance(h, int):
            raise TypeError(f"horizons[{i}] must be an integer, got {type(h).__name__}")
        if h <= 0:
            raise ValueError(f"horizons[{i}] must be positive, got {h}")
    if not isinstance(log_returns, bool):
        raise TypeError(
            f"log_returns must be a boolean, got {type(log_returns).__name__}",
        )

    close = pl.col(close) if isinstance(close, str) else close

    returns = {}
    for h in horizons:
        if log_returns:
            returns[f"log_ret_{h}m"] = (close / close.shift(h)).log()
        else:
            returns[f"ret_{h}m"] = close / close.shift(h) - 1

    return returns
