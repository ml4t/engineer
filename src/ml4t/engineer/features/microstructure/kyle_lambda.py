import polars as pl

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.validation import (
    validate_window,
)


@feature(
    name="kyle_lambda",
    category="microstructure",
    description="Kyle's Lambda - measures price impact per unit of order flow",
    lookback="period",
    normalized=False,
    formula="lambda = cov(r, sqrt(|v|)) / var(sqrt(|v|))",
    ta_lib_compatible=False,
    input_type="returns_volume",
    references=["Kyle (1985). Continuous Auctions and Insider Trading"],
    tags=["microstructure", "liquidity", "price-impact", "market-depth"],
)
def kyle_lambda(
    returns: pl.Expr | str,
    volume: pl.Expr | str,
    period: int = 20,
    method: str = "ratio",
) -> pl.Expr:
    """Calculate Kyle's Lambda, measuring the price impact of order flow.

    Kyle's Lambda quantifies market depth by estimating how much prices move
    per unit of order flow. It originated from Kyle's (1985) sequential auction
    model of informed trading, where λ represents the price concession informed
    traders must pay to execute their orders. Higher λ indicates lower liquidity
    (greater price impact per dollar traded).

    Mathematical Formula
    --------------------
    The true Kyle's Lambda is estimated via OLS regression:

    .. math::

        r_{i,n} = \\lambda_i \\cdot S_{i,n} + \\varepsilon_{i,n}

    where:

    - :math:`r_{i,n}` : Stock return for period n (percentage)
    - :math:`\\lambda_i` : Kyle's Lambda (slope coefficient, price impact)
    - :math:`S_{i,n}` : Signed square-root dollar volume = :math:`\\sum_k \\text{sign}(v_{k,n}) \\sqrt{|v_{k,n}|}`
    - :math:`v_{k,n}` : Signed dollar volume for trade k in period n
    - :math:`\\varepsilon_{i,n}` : Error term

    **Current Implementation Note**: This function provides a simplified ratio-based
    approximation (method="ratio"). The true regression-based estimator (method="regression")
    matching Kyle's original formulation is planned for future release.

    **Ratio Approximation** (current default):

    .. math::

        \\lambda_{\\text{ratio}} = \\frac{|r_t|}{v_t / \\bar{v}}

    where :math:`\\bar{v}` is the rolling mean volume for normalization.

    Parameters
    ----------
    returns : pl.Expr | str
        Returns column (absolute value will be computed). Typically log returns
        or percentage returns over the measurement period.
    volume : pl.Expr | str
        Volume column. Signed volume (buy volume - sell volume) is preferred
        for true Kyle's Lambda; unsigned volume works with ratio approximation.
    period : int, default 20
        Rolling window period for estimation. Typical values: 20 (daily for
        monthly estimate), 60 (3-month), 252 (annual). Larger periods provide
        more stable estimates but reduce responsiveness.
    method : str, default "ratio"
        Estimation method:
        - "ratio": Fast approximation using |return| / normalized_volume
        - "regression": True Kyle's Lambda via OLS (not yet implemented)

    Returns
    -------
    pl.Expr
        Kyle's Lambda estimate. Higher values indicate lower liquidity
        (greater price impact). Scaled to be comparable across assets
        when using the same period.

    Raises
    ------
    ValueError
        If period < 1 or method is not in ["ratio", "regression"]
    NotImplementedError
        If method="regression" is requested (planned for future release)

    Interpretation
    --------------
    **Value Range**: Unbounded, typically 0 to 1000+ (scaled by implementation)

    **Signal Guidelines**:
    - **Low λ (<25th percentile)**: High liquidity, low price impact,
      favorable for large orders
    - **High λ (>75th percentile)**: Low liquidity, high price impact,
      adverse selection risk
    - **Increasing λ**: Deteriorating liquidity, markets becoming less deep
    - **Decreasing λ**: Improving liquidity, easier execution

    **Common Use Cases**:
    - **Transaction Cost Estimation**: Predict market impact before trading
    - **Liquidity Risk Management**: Monitor market depth for portfolio positions
    - **Optimal Execution**: Size orders based on current λ to minimize impact
    - **Regime Detection**: Identify liquidity crises (spikes in λ)
    - **Cross-Asset Comparison**: Compare execution quality across markets

    **Limitations**:
    - Assumes linear price impact (may not hold for very large orders)
    - Sensitive to microstructure noise in high-frequency data
    - Ratio approximation is simplified; true regression method preferred
    - Requires signed volume for accurate estimation of informed trading
    - Historical estimate; may not predict future impact during regime shifts

    References
    ----------
    .. [1] Kyle, A. S. (1985). "Continuous Auctions and Insider Trading".
           *Econometrica*, 53(6), 1315-1335. DOI: 10.2307/1913210
    .. [2] Goyenko, R. Y., Holden, C. W., & Trzcinka, C. A. (2009).
           "Do liquidity measures measure liquidity?". *Journal of Financial
           Economics*, 92(2), 153-181.

    Examples
    --------
    Basic usage with default ratio approximation:

    >>> import polars as pl
    >>> from ml4t.engineer.features.microstructure import kyle_lambda
    >>>
    >>> # Sample data with returns and volume
    >>> df = pl.DataFrame({
    ...     "returns": [0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.012, 0.005],
    ...     "volume": [1000000, 950000, 1200000, 800000, 1100000, 1050000, 900000, 1150000]
    ... })
    >>>
    >>> # Calculate Kyle's Lambda with 5-period window
    >>> result = df.with_columns(
    ...     kyle_lambda("returns", "volume", period=5).alias("lambda")
    ... )
    >>> print(result["lambda"])

    Advanced usage with longer estimation window:

    >>> # For monthly data, use longer window (e.g., 60 days)
    >>> result = df.with_columns(
    ...     kyle_lambda("returns", "volume", period=60).alias("lambda_60d")
    ... )

    Using in liquidity monitoring:

    >>> # Monitor liquidity regime
    >>> result = df.with_columns([
    ...     kyle_lambda("returns", "volume", period=20).alias("lambda_20d"),
    ...     kyle_lambda("returns", "volume", period=20).alias("lambda_20d").pct_change().alias("lambda_change")
    ... ])
    >>> # High lambda_change indicates rapid liquidity deterioration
    """
    # Validate inputs
    validate_window(period, min_window=1, name="period")
    if method not in ["ratio", "regression"]:
        raise ValueError(
            f"Unknown method: {method}. Supported methods: ['ratio', 'regression']",
        )

    if method == "regression":
        raise NotImplementedError(
            "Kyle's Lambda via regression (true OLS-based method) is not yet implemented. "
            "Please use method='ratio' for the ratio-based approximation, or wait for "
            "the regression implementation in a future release. See FUNCTIONALITY_INVENTORY.md "
            "for status and roadmap.",
        )

    returns = pl.col(returns) if isinstance(returns, str) else returns
    volume = pl.col(volume) if isinstance(volume, str) else volume

    # Ratio method with volume normalization
    # Protect against division by zero
    vol_normalized = volume / volume.rolling_mean(period)
    lambda_ratio = (
        pl.when(vol_normalized.abs() > 1e-10)
        .then(
            returns.abs() / vol_normalized,
        )
        .otherwise(None)
        .rolling_mean(period)
    )

    return lambda_ratio
