import numpy as np
import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="volatility_adjusted_returns",
    category="ml",
    description="Volatility Adjusted Returns - risk-adjusted returns",
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def volatility_adjusted_returns(
    returns: pl.Expr | str,
    volatility: pl.Expr | str | None = None,
    vol_lookback: int = 20,
    annualize: bool = False,
) -> pl.Expr:
    """Calculate volatility-adjusted returns (Sharpe-like ratio).

    Normalizes returns by volatility to create risk-adjusted features.

    Parameters
    ----------
    returns : pl.Expr | str
        Returns column
    volatility : pl.Expr | str or None, default None
        Volatility column. When omitted, volatility is calculated from returns.
    vol_lookback : int, default 20
        Lookback period for volatility calculation
    annualize : bool, default False
        Whether to annualize the ratio

    Returns
    -------
    pl.Expr
        Volatility-adjusted returns

    Raises
    ------
    ValueError
        If vol_lookback is not positive
    TypeError
        If vol_lookback is not an integer or annualize is not boolean
    """
    # Validate inputs
    validate_window(vol_lookback, min_window=1, name="vol_lookback")
    if not isinstance(annualize, bool):
        raise TypeError(f"annualize must be a boolean, got {type(annualize).__name__}")

    returns = pl.col(returns) if isinstance(returns, str) else returns

    if isinstance(volatility, str):
        volatility = pl.col(volatility)
    elif volatility is None:
        # Calculate rolling volatility
        volatility = returns.rolling_std(vol_lookback)

    # Adjust returns
    adj_returns = returns / (volatility + 1e-10)

    if annualize:
        adj_returns = adj_returns * np.sqrt(252)

    return adj_returns
