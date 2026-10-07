import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.features.microstructure.effective_tick_rule import effective_tick_rule

# Main feature function


@feature(
    name="order_flow_imbalance",
    category="microstructure",
    description="Order Flow Imbalance - imbalance between buy and sell orders",
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def order_flow_imbalance(
    volume: pl.Expr | str,
    close: pl.Expr | str,
    use_tick_rule: bool = True,
) -> pl.Expr:
    """Calculate order flow imbalance.

    Estimates the imbalance between buying and selling pressure.
    When bid/ask volume is not available, uses tick rule classification.

    Parameters
    ----------
    volume : pl.Expr | str
        Volume column
    close : pl.Expr | str
        Close price column (for tick rule)
    use_tick_rule : bool, default True
        Whether to use tick rule for trade classification

    Returns
    -------
    pl.Expr
        Order flow imbalance (-1 to 1)
    """
    volume = pl.col(volume) if isinstance(volume, str) else volume

    if use_tick_rule:
        # Classify trades using tick rule
        trade_sign = effective_tick_rule(close)

        # Signed volume
        signed_volume = volume * trade_sign

        # Order flow imbalance
        buy_volume = pl.when(signed_volume > 0).then(signed_volume).otherwise(0)
        sell_volume = pl.when(signed_volume < 0).then(-signed_volume).otherwise(0)

        total_volume = buy_volume + sell_volume
        imbalance = (
            pl.when(total_volume > 0).then((buy_volume - sell_volume) / total_volume).otherwise(0)
        )
    else:
        # If no classification available, return 0
        imbalance = pl.lit(0)

    return imbalance
