import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="amihud_illiquidity",
    category="microstructure",
    description="Amihud Illiquidity - measures price impact per dollar of trading volume",
    lookback="period",
    normalized=False,
    formula="ILLIQ = mean(|R_t| / DVOL_t)",
    ta_lib_compatible=False,
    input_type="returns_volume_price",
    references=["Amihud (2002). Illiquidity and stock returns"],
    tags=["microstructure", "liquidity", "price-impact", "transaction-costs"],
)
def amihud_illiquidity(
    returns: pl.Expr | str,
    volume: pl.Expr | str,
    price: pl.Expr | str,
    period: int = 20,
) -> pl.Expr:
    """Calculate the Amihud Illiquidity measure, a proxy for price impact costs.

    The Amihud (2002) illiquidity ratio quantifies the average price impact
    per dollar of trading volume. It captures how much prices move in response
    to order flow, serving as a widely-used proxy for market liquidity and
    transaction costs. Higher values indicate lower liquidity (greater price
    impact per dollar traded). This measure has become one of the most popular
    liquidity proxies in empirical asset pricing due to its simplicity and the
    fact that it requires only daily data.

    Mathematical Formula
    --------------------
    .. math::

        \\text{ILLIQ}_{i,t} = \\frac{1}{D_{i,t}} \\sum_{d=1}^{D_{i,t}} \\frac{|R_{i,d,t}|}{DVOL_{i,d,t}}

    where:

    - :math:`\\text{ILLIQ}_{i,t}` : Amihud illiquidity for asset i in period t
    - :math:`D_{i,t}` : Number of trading days in period t
    - :math:`R_{i,d,t}` : Daily return on day d (absolute value taken)
    - :math:`DVOL_{i,d,t}` : Daily dollar volume = Price × Volume

    **Scaling**: This implementation multiplies by 10^6 to express the measure
    as "price impact per million dollars traded" for better readability.

    **Economic Interpretation**: The Amihud ratio measures the percentage price
    change associated with one dollar of trading volume. It is based on the
    idea that in illiquid markets, a given volume of trades causes larger
    price movements.

    Parameters
    ----------
    returns : pl.Expr | str
        Returns column (absolute value will be computed). Daily returns
        are typical, but any frequency works.
    volume : pl.Expr | str
        Trading volume (number of shares/contracts traded per period)
    price : pl.Expr | str
        Price used to compute dollar volume. Typically the closing price
        or average of bid-ask midpoint.
    period : int, default 20
        Rolling window period (number of days/periods to average).
        Common values: 20 (monthly), 60 (quarterly), 252 (annual).
        Larger periods provide more stable estimates but less sensitivity
        to regime changes.

    Returns
    -------
    pl.Expr
        Amihud illiquidity measure (scaled by 10^6). Higher values indicate
        lower liquidity (higher transaction costs). Values are comparable
        across assets when using the same period.

    Raises
    ------
    ValueError
        If period < 1
    TypeError
        If period is not an integer

    Interpretation
    --------------
    **Value Range**: Unbounded, typically 0 to 100+ (after 10^6 scaling)

    **Signal Guidelines**:
    - **Low ILLIQ (<25th percentile)**: High liquidity, low transaction costs,
      favorable trading conditions
    - **High ILLIQ (>75th percentile)**: Low liquidity, high transaction costs,
      adverse market conditions
    - **Rising ILLIQ**: Deteriorating liquidity, increasing trading costs
    - **Falling ILLIQ**: Improving liquidity, decreasing trading costs

    **Common Use Cases**:
    - **Liquidity Risk Assessment**: Identify less liquid stocks for risk management
    - **Transaction Cost Estimation**: Predict execution costs before trading
    - **Asset Pricing**: Test whether illiquidity commands a return premium
    - **Portfolio Construction**: Avoid or underweight illiquid securities
    - **Market Timing**: Adjust strategy based on market-wide liquidity conditions
    - **Performance Attribution**: Separate alpha from liquidity-driven returns

    **Empirical Findings** (Amihud 2002):
    - Strong positive relationship between illiquidity and expected returns
    - Illiquidity premium is time-varying and countercyclical
    - Cross-sectional illiquidity explains significant variation in returns

    **Limitations**:
    - Requires price and volume data (not applicable to all assets)
    - Sensitive to outliers and market microstructure noise
    - Assumes linear price impact (may not hold for very large trades)
    - Does not distinguish between buyer- and seller-initiated trades
    - Less reliable for thinly traded securities with frequent zero-volume days
    - Historical measure; may not predict future liquidity during crises

    References
    ----------
    .. [1] Amihud, Y. (2002). "Illiquidity and stock returns: cross-section
           and time-series effects". *Journal of Financial Markets*, 5(1), 31-56.
           DOI: 10.1016/S1386-4181(01)00024-6
    .. [2] Goyenko, R. Y., Holden, C. W., & Trzcinka, C. A. (2009).
           "Do liquidity measures measure liquidity?". *Journal of Financial
           Economics*, 92(2), 153-181.
    .. [3] Hasbrouck, J. (2009). "Trading costs and returns for US equities:
           Estimating effective costs from daily data". *Journal of Finance*,
           64(3), 1445-1477.

    Examples
    --------
    Basic usage with daily data:

    >>> import polars as pl
    >>> from ml4t.engineer.features.microstructure import amihud_illiquidity
    >>>
    >>> # Sample daily stock data
    >>> df = pl.DataFrame({
    ...     "returns": [0.01, -0.008, 0.015, -0.005, 0.012, 0.003, -0.01, 0.007],
    ...     "volume": [1000000, 850000, 1200000, 950000, 1100000, 800000, 900000, 1050000],
    ...     "close": [100, 99.2, 100.7, 100.2, 101.4, 101.7, 100.7, 101.4]
    ... })
    >>>
    >>> # Calculate 5-day Amihud illiquidity
    >>> result = df.with_columns(
    ...     amihud_illiquidity("returns", "volume", "close", period=5).alias("illiq")
    ... )
    >>> print(result["illiq"])

    Monthly illiquidity (20-day rolling):

    >>> # Standard monthly measure
    >>> result = df.with_columns(
    ...     amihud_illiquidity("returns", "volume", "close", period=20).alias("illiq_20d")
    ... )

    Cross-sectional liquidity analysis:

    >>> # Compare illiquidity across multiple stocks
    >>> # Assuming df has columns: ticker, date, returns, volume, close
    >>> result = df.group_by("ticker").agg([
    ...     pl.col("returns"),
    ...     pl.col("volume"),
    ...     pl.col("close"),
    ... ]).with_columns(
    ...     amihud_illiquidity("returns", "volume", "close", period=20).alias("illiq")
    ... )

    Liquidity-adjusted portfolio weighting:

    >>> # Underweight illiquid stocks
    >>> result = df.with_columns([
    ...     amihud_illiquidity("returns", "volume", "close", period=20).alias("illiq"),
    ... ]).with_columns(
    ...     (1.0 / pl.col("illiq")).alias("liquidity_weight")  # Inverse weighting
    ... )
    """
    # Validate inputs
    validate_window(period, min_window=1, name="period")

    returns = pl.col(returns) if isinstance(returns, str) else returns
    volume = pl.col(volume) if isinstance(volume, str) else volume
    price = pl.col(price) if isinstance(price, str) else price

    # Calculate dollar volume
    dollar_volume = price * volume

    # Amihud = |return| / dollar_volume
    # Use 1e-6 multiplier for readability (expressing as return per million dollars)
    # Protect against division by zero
    amihud = (
        pl.when(dollar_volume.abs() > 1e-10)
        .then(
            returns.abs() / dollar_volume * 1e6,
        )
        .otherwise(None)
        .rolling_mean(period)
    )

    return amihud
