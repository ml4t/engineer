# Technical Indicators

ML4T Engineer provides 107+ technical indicators across 10 categories.

## Momentum Indicators (31)

| Name | Description | TA-Lib Compatible |
|------|-------------|-------------------|
| `rsi` | Relative Strength Index | Yes |
| `macd` | Moving Average Convergence/Divergence | Yes |
| `stoch` | Stochastic Oscillator | Yes |
| `cci` | Commodity Channel Index | Yes |
| `willr` | Williams %R | Yes |
| `adx` | Average Directional Index | Yes |
| `mfi` | Money Flow Index | Yes |
| `roc` | Rate of Change | Yes |
| `mom` | Momentum | Yes |
| `trix` | Triple Exponential Average | Yes |

## Trend Indicators (10)

| Name | Description | TA-Lib Compatible |
|------|-------------|-------------------|
| `sma` | Simple Moving Average | Yes |
| `ema` | Exponential Moving Average | Yes |
| `wma` | Weighted Moving Average | Yes |
| `dema` | Double EMA | Yes |
| `tema` | Triple EMA | Yes |
| `kama` | Kaufman Adaptive MA | Yes |
| `t3` | Triple Exponential T3 | Yes |

## Volatility Indicators (15)

| Name | Description |
|------|-------------|
| `atr` | Average True Range (TA-Lib) |
| `natr` | Normalized ATR (TA-Lib) |
| `bollinger_bands` | Bollinger Bands (TA-Lib) |
| `yang_zhang` | Yang-Zhang volatility (most efficient) |
| `parkinson` | Parkinson high-low volatility |
| `garman_klass` | Garman-Klass OHLC volatility |
| `rogers_satchell` | Rogers-Satchell drift-adjusted |
| `realized` | Realized volatility |
| `ewma` | EWMA volatility |
| `garch` | GARCH(1,1) forecast |

## Microstructure (12)

| Name | Description |
|------|-------------|
| `kyle_lambda` | Kyle's Lambda (price impact) |
| `amihud_illiquidity` | Amihud illiquidity ratio |
| `vpin` | Volume-Synchronized PIN |
| `roll_spread` | Roll implied spread |
| `corwin_schultz` | Corwin-Schultz high-low spread |
| `hasbrouck_lambda` | Hasbrouck's Lambda |

## ML Features (11)

| Name | Description |
|------|-------------|
| `fractional_diff` | Fractionally differenced series |
| `lag` | Lagged values |
| `rolling_stats` | Rolling statistics |
| `entropy` | Shannon entropy |
| `hurst` | Hurst exponent |
| `autocorr` | Autocorrelation |

## Usage Examples

```python
from ml4t.engineer import compute_features

# Single indicator
result = compute_features(df, ["rsi"])

# Multiple indicators
result = compute_features(df, ["rsi", "macd", "bollinger_bands"])

# Custom parameters
result = compute_features(df, [
    {"name": "rsi", "params": {"period": 20}},
    {"name": "sma", "params": {"period": 50}},
])
```
