# microstructure/ - 12 Market Microstructure Metrics

Academic market microstructure features. Validated against research papers.

## Price Impact

### kyle_lambda(data, period=20) -> DataFrame
Kyle's Lambda. Price impact coefficient. Higher = less liquid.
Reference: Kyle (1985)

### amihud_illiquidity(data, period=20) -> DataFrame
Amihud illiquidity ratio. |return| / dollar_volume.
Reference: Amihud (2002)

### hasbrouck_lambda(data, period=20) -> DataFrame
Hasbrouck's Lambda. Alternative price impact measure.

## Spread Estimation

### roll_spread(data, period=20) -> DataFrame
Roll implied spread. From serial covariance of returns.
Reference: Roll (1984)

### corwin_schultz(data) -> DataFrame
Corwin-Schultz spread. From high-low prices.
Reference: Corwin & Schultz (2012)

### effective_spread(data) -> DataFrame
Effective spread estimate.

## Information Asymmetry

### vpin(data, volume_bucket_size=50) -> DataFrame
Volume-Synchronized PIN. Probability of informed trading.
Reference: Easley, López de Prado, O'Hara (2012)

### order_flow_imbalance(data, period=20) -> DataFrame
Order flow imbalance. Buy vs sell pressure.

### trade_classification(data) -> DataFrame
Lee-Ready trade classification. Tick rule + quote rule.

## Volume Analysis

### volume_weighted_price_momentum(data, period=20) -> DataFrame
VWAP-based momentum.

### realized_spread(data, period=20) -> DataFrame
Realized spread from price reversals.

### price_impact_ratio(data, period=20) -> DataFrame
Price impact per unit volume.

## Common Parameters

- `data`: pl.DataFrame with OHLCV columns
- `period`: Rolling window for estimation
- All require volume data for meaningful results
