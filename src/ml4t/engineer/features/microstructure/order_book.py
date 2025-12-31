"""Order book features for quote-level data.

These features require bid/ask price and size data, typically from:
- Level 1 quotes (best bid/ask)
- Level 2/3 order book snapshots
- Pre-aggregated quote data (e.g., ALGOSeek TAQ)
"""

import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import validate_window

__all__ = [
    "bid_ask_imbalance",
    "book_depth_ratio",
    "weighted_mid_price",
]


@feature(
    name="bid_ask_imbalance",
    category="microstructure",
    description="Order book imbalance measuring buy vs sell pressure from resting orders",
    lookback="period",
    normalized=True,
    value_range=(-1.0, 1.0),
    formula="(bid_size - ask_size) / (bid_size + ask_size)",
    ta_lib_compatible=False,
    input_type="quote",
    references=["Cont et al. (2014). The Price Impact of Order Book Events"],
    tags=["microstructure", "order-book", "imbalance", "liquidity"],
)
def bid_ask_imbalance(
    bid_size: pl.Expr | str = "bid_size",
    ask_size: pl.Expr | str = "ask_size",
    period: int = 1,
) -> pl.Expr:
    """Calculate bid-ask imbalance from order book sizes.

    Bid-ask imbalance measures the relative difference between buy and sell
    pressure in the order book. Positive values indicate more buying pressure
    (larger bid size), while negative values indicate more selling pressure
    (larger ask size).

    Mathematical Formula
    --------------------
    .. math::

        \\text{Imbalance} = \\frac{\\text{bid\\_size} - \\text{ask\\_size}}{\\text{bid\\_size} + \\text{ask\\_size}}

    When period > 1, a rolling mean is applied to smooth the signal.

    Parameters
    ----------
    bid_size : pl.Expr | str
        Size (quantity) at best bid, or total bid depth if using multiple levels.
    ask_size : pl.Expr | str
        Size (quantity) at best ask, or total ask depth if using multiple levels.
    period : int, default 1
        Rolling window for smoothing. Use 1 for raw imbalance, larger values
        for signal smoothing.

    Returns
    -------
    pl.Expr
        Bid-ask imbalance in range [-1, 1]:
        - +1: All size on bid side (extreme buying pressure)
        - 0: Equal bid and ask size (balanced)
        - -1: All size on ask side (extreme selling pressure)

    Interpretation
    --------------
    **Value Range**: [-1, 1] (normalized)

    **Signal Guidelines**:
    - **Positive imbalance (>0.2)**: More resting bids, potential upward pressure
    - **Negative imbalance (<-0.2)**: More resting asks, potential downward pressure
    - **Near zero**: Balanced book, no directional pressure

    **Common Use Cases**:
    - **Short-term direction prediction**: Imbalance often precedes price moves
    - **Trade timing**: Execute buys when imbalance negative (more supply)
    - **Market making**: Adjust quotes based on book imbalance
    - **Feature for ML models**: Strong predictor of next-tick returns

    References
    ----------
    .. [1] Cont, R., Kukanov, A., & Stoikov, S. (2014). "The Price Impact of
           Order Book Events". *Journal of Financial Econometrics*, 12(1), 47-88.

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.features.microstructure import bid_ask_imbalance
    >>>
    >>> df = pl.DataFrame({
    ...     "bid_size": [1000, 1200, 800, 1500],
    ...     "ask_size": [900, 1000, 1200, 500],
    ... })
    >>> result = df.with_columns(
    ...     bid_ask_imbalance("bid_size", "ask_size").alias("imbalance")
    ... )
    """
    validate_window(period, min_window=1, name="period")

    bid = pl.col(bid_size) if isinstance(bid_size, str) else bid_size
    ask = pl.col(ask_size) if isinstance(ask_size, str) else ask_size

    total = bid + ask
    imbalance = pl.when(total > 0).then((bid - ask) / total).otherwise(0.0)

    if period > 1:
        return imbalance.rolling_mean(period)
    return imbalance


@feature(
    name="book_depth_ratio",
    category="microstructure",
    description="Ratio of bid depth to total depth, measuring relative buying interest",
    lookback=0,
    normalized=True,
    value_range=(0.0, 1.0),
    formula="bid_depth / (bid_depth + ask_depth)",
    ta_lib_compatible=False,
    input_type="quote",
    references=["Cao et al. (2009). The Information Content of an Open Limit Order Book"],
    tags=["microstructure", "order-book", "depth", "liquidity"],
)
def book_depth_ratio(
    bid_depth: pl.Expr | str = "bid_size",
    ask_depth: pl.Expr | str = "ask_size",
) -> pl.Expr:
    """Calculate ratio of bid depth to total book depth.

    This metric shows what proportion of total order book depth is on the
    bid side. Unlike imbalance which ranges [-1, 1], this ratio ranges [0, 1]
    which may be more intuitive for some applications.

    Mathematical Formula
    --------------------
    .. math::

        \\text{Depth Ratio} = \\frac{\\text{bid\\_depth}}{\\text{bid\\_depth} + \\text{ask\\_depth}}

    Parameters
    ----------
    bid_depth : pl.Expr | str
        Total depth (size) on bid side. Can be single level or sum of multiple levels.
    ask_depth : pl.Expr | str
        Total depth (size) on ask side. Can be single level or sum of multiple levels.

    Returns
    -------
    pl.Expr
        Depth ratio in range [0, 1]:
        - 1.0: All depth on bid side
        - 0.5: Equal depth on both sides
        - 0.0: All depth on ask side

    Interpretation
    --------------
    **Value Range**: [0, 1]

    **Signal Guidelines**:
    - **>0.6**: Heavy bid-side depth, buying interest dominates
    - **~0.5**: Balanced book
    - **<0.4**: Heavy ask-side depth, selling interest dominates

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.features.microstructure import book_depth_ratio
    >>>
    >>> df = pl.DataFrame({
    ...     "bid_depth": [5000, 6000, 4000],
    ...     "ask_depth": [5000, 4000, 8000],
    ... })
    >>> result = df.with_columns(
    ...     book_depth_ratio("bid_depth", "ask_depth").alias("depth_ratio")
    ... )
    >>> # Returns [0.5, 0.6, 0.333...]
    """
    bid = pl.col(bid_depth) if isinstance(bid_depth, str) else bid_depth
    ask = pl.col(ask_depth) if isinstance(ask_depth, str) else ask_depth

    total = bid + ask
    return (
        pl.when(total > 0).then(bid / total).otherwise(0.5)  # Default to balanced when no depth
    )


@feature(
    name="weighted_mid_price",
    category="microstructure",
    description="Volume-weighted midpoint using bid/ask sizes as weights",
    lookback=0,
    normalized=False,
    formula="(bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)",
    ta_lib_compatible=False,
    input_type="quote",
    references=["Stoikov, S. (2018). The Micro-Price: A High-Frequency Estimator of Future Prices"],
    tags=["microstructure", "order-book", "mid-price", "fair-value"],
)
def weighted_mid_price(
    bid_price: pl.Expr | str = "bid_price",
    ask_price: pl.Expr | str = "ask_price",
    bid_size: pl.Expr | str = "bid_size",
    ask_size: pl.Expr | str = "ask_size",
) -> pl.Expr:
    """Calculate volume-weighted mid-price (micro-price).

    The weighted mid-price (also called micro-price) adjusts the simple midpoint
    by weighting each side by the opposing side's depth. This gives a better
    estimate of fair value when the book is imbalanced.

    Intuition: If there's more depth on the bid side, the price is more likely
    to move up (bids absorb selling), so fair value is closer to the ask.

    Mathematical Formula
    --------------------
    .. math::

        \\text{Weighted Mid} = \\frac{P_{bid} \\cdot Q_{ask} + P_{ask} \\cdot Q_{bid}}{Q_{bid} + Q_{ask}}

    This can be rewritten as:

    .. math::

        \\text{Weighted Mid} = \\text{Mid} + \\frac{\\text{Imbalance}}{2} \\cdot \\text{Spread}

    where Imbalance = (bid_size - ask_size) / (bid_size + ask_size).

    Parameters
    ----------
    bid_price : pl.Expr | str
        Best bid price.
    ask_price : pl.Expr | str
        Best ask price.
    bid_size : pl.Expr | str
        Size at best bid.
    ask_size : pl.Expr | str
        Size at best ask.

    Returns
    -------
    pl.Expr
        Weighted mid-price. Always between bid_price and ask_price.
        Closer to ask when bid_size > ask_size (imbalance positive).

    Interpretation
    --------------
    **Value Range**: [bid_price, ask_price]

    **Relationship to Simple Mid**:
    - When bid_size = ask_size: weighted_mid = simple_mid = (bid + ask) / 2
    - When bid_size > ask_size: weighted_mid > simple_mid (closer to ask)
    - When bid_size < ask_size: weighted_mid < simple_mid (closer to bid)

    **Common Use Cases**:
    - **Fair value estimation**: Better than simple mid for pricing
    - **Market making**: Quote around weighted mid for better fills
    - **Execution benchmarking**: More accurate reference price
    - **Returns calculation**: Use weighted mid for microstructure research

    References
    ----------
    .. [1] Stoikov, S. (2018). "The Micro-Price: A High-Frequency Estimator
           of Future Prices". *Quantitative Finance*, 18(12), 1959-1966.

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.features.microstructure import weighted_mid_price
    >>>
    >>> df = pl.DataFrame({
    ...     "bid_price": [100.00, 100.00],
    ...     "ask_price": [100.02, 100.02],
    ...     "bid_size": [1000, 2000],  # More bid depth in row 2
    ...     "ask_size": [1000, 1000],
    ... })
    >>> result = df.with_columns(
    ...     weighted_mid_price("bid_price", "ask_price", "bid_size", "ask_size").alias("wmid")
    ... )
    >>> # Row 1: wmid = 100.01 (balanced, equals simple mid)
    >>> # Row 2: wmid = 100.0133... (bid-heavy, weighted toward ask)
    """
    bp = pl.col(bid_price) if isinstance(bid_price, str) else bid_price
    ap = pl.col(ask_price) if isinstance(ask_price, str) else ask_price
    bs = pl.col(bid_size) if isinstance(bid_size, str) else bid_size
    asz = pl.col(ask_size) if isinstance(ask_size, str) else ask_size

    total_size = bs + asz

    # When total_size is 0, fall back to simple midpoint
    simple_mid = (bp + ap) / 2
    weighted_mid = (bp * asz + ap * bs) / total_size

    return pl.when(total_size > 0).then(weighted_mid).otherwise(simple_mid)
