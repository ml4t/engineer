import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_positive,
)


@feature(
    name="ewma_volatility",
    category="volatility",
    description="EWMA Volatility - exponentially weighted moving average of variance",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def ewma_volatility(
    close: pl.Expr | str = "close",
    span: int = 120,
    normalize: bool = False,
    mu: float = -5.0,
    sigma: float = 2.0,
) -> pl.Expr:
    """
    Calculate EWMA (Exponentially Weighted Moving Average) volatility.

    Computes volatility as sqrt(EWMA of squared returns), with optional
    tanh normalization for ML applications.

    Parameters
    ----------
    close : pl.Expr | str, default "close"
        Close price column or expression
    span : int, default 120
        EWMA span in bars (alpha = 1/span). Common close:
        - 120 bars (2 hours on minute data)
        - 1440 bars (1 day on minute data)
        - 28800 bars (20 days on minute data)
    normalize : bool, default False
        Whether to apply tanh normalization to [-1, 1] range
    mu : float, default -5.0
        Mean for tanh normalization (only used if normalize=True)
    sigma : float, default 2.0
        Std dev for tanh normalization (only used if normalize=True)

    Returns
    -------
    pl.Expr
        EWMA volatility expression

    Examples
    --------
    >>> # Basic EWMA volatility
    >>> df = df.with_columns(
    ...     ewma_volatility("close", span=120).alias("ewma_vol_2h")
    ... )
    >>>
    >>> # Normalized for ML
    >>> df = df.with_columns(
    ...     ewma_volatility("close", span=1440, normalize=True).alias("ewma_vol_1d_norm")
    ... )

    References
    ----------
    .. [1] RiskMetrics Technical Document (1996). J.P. Morgan/Reuters.
    """
    validate_positive(span, name="span")

    close_col = pl.col(close) if isinstance(close, str) else close

    # Calculate returns and square them
    returns = close_col.pct_change()
    returns_sq = returns**2

    # EWMA of squared returns (alpha = 1/span)
    alpha = 1.0 / span
    ewma_sq = returns_sq.ewm_mean(alpha=alpha, adjust=False)

    # Square root to get volatility
    vol = ewma_sq.sqrt()

    if normalize:
        # Tanh normalization: tanh((log(vol + eps) - mu) / sigma)
        epsilon = 1e-10
        vol_norm = ((vol + epsilon).log() - mu) / sigma
        return vol_norm.tanh()
    else:
        return vol
