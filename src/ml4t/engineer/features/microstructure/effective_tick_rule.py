import polars as pl

from ml4t.engineer.core.decorators import feature


@feature(
    name="effective_tick_rule",
    category="microstructure",
    description="Effective Tick Rule - infers trade direction from price changes",
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def effective_tick_rule(close: pl.Expr | str) -> pl.Expr:
    """Calculate effective tick rule for trade classification.

    Classifies trades as buyer or seller initiated based on price movements.
    An uptick is +1 and a downtick is -1. A zero price change keeps the previous
    nonzero sign. The value is 0 when no earlier price change exists or an
    adjacent price is missing.

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

    # Compute differences before casting to preserve small Decimal price changes.
    price_change = close.diff().cast(pl.Float64).fill_nan(None)
    # A zero change is null here so forward-fill can recover the previous sign.
    # A missing price has a null change too, and stays unclassified (0).
    direction = pl.when(price_change > 0).then(1).when(price_change < 0).then(-1).otherwise(None)
    carried = direction.fill_null(strategy="forward").fill_null(0)
    return pl.when(price_change.is_null()).then(0).otherwise(carried)
