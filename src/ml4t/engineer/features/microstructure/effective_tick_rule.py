import polars as pl

from ml4t.engineer.core.decorators import feature


@feature(
    name="effective_tick_rule",
    category="microstructure",
    description="Effective Tick Rule - infers trade direction from price changes",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def effective_tick_rule(close: pl.Expr | str) -> pl.Expr:
    """Calculate effective tick rule for trade classification.

    Classifies trades as buyer or seller initiated based on price movements.
    Returns: 1 for buy, -1 for sell, 0 for unchanged.

    Parameters
    ----------
    close : pl.Expr | str
        Close price column

    Returns
    -------
    pl.Expr
        Trade classification
    """
    close = pl.col(close) if isinstance(close, str) else close

    # Current price change
    price_change = close.diff()

    # Tick rule: compare with previous price
    tick = pl.when(price_change > 0).then(1).when(price_change < 0).then(-1).otherwise(0)

    # Use previous tick if current is zero (effective tick rule)
    tick_filled = tick.fill_null(strategy="forward").fill_null(0)

    return tick_filled
