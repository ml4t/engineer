# features/ - Technical Indicators

120 features in 10 categories. All use `@feature` decorator for registry.

## Categories

| Category | Count | Directory | Key Functions |
|----------|-------|-----------|---------------|
| momentum | 31 | `momentum/` | rsi, macd, adx, stoch, cci, mfi, willr, aroon, sar |
| trend | 10 | `trend/` | sma, ema, wma, dema, tema, kama, t3 |
| volatility | 15 | `volatility/` | atr, yang_zhang, bollinger_bands, parkinson, garch |
| microstructure | 12 | `microstructure/` | kyle_lambda, amihud, vpin, roll_spread |
| ml | 11 | `ml/` | lag, cyclical_encode, rolling_entropy, fourier |
| statistics | 8 | `statistics/` | stddev, var, linear_regression, tsf |
| volume | 3 | `volume/` | obv, ad, adosc |
| price_transform | 5 | `price_transform/` | avgprice, typprice, medprice, wclprice |
| math | 3 | `math/` | max_, min_, sum_ (O(n) optimized) |
| regime | 3 | `regime.py` | hurst_exponent, choppiness_index, variance_ratio |

## Standalone Files

| File | Lines | Purpose |
|------|-------|---------|
| `risk.py` | 748 | Risk metrics: CVaR, Sortino, max_drawdown, information_ratio |
| `regime.py` | 537 | Regime detection: Hurst, choppiness, fractal efficiency |
| `cross_asset.py` | 654 | Multi-asset: beta_to_market, rolling_correlation |
| `fdiff.py` | 383 | Fractional differentiation: fractional_diff, find_min_d |
| `composite.py` | 312 | Feature combinations |

## Navigation

| Category | Detail |
|----------|--------|
| momentum | [momentum/AGENT.md](momentum/AGENT.md) |
| trend | [trend/AGENT.md](trend/AGENT.md) |
| volatility | [volatility/AGENT.md](volatility/AGENT.md) |
| microstructure | [microstructure/AGENT.md](microstructure/AGENT.md) |
| ml | [ml/AGENT.md](ml/AGENT.md) |
| statistics | [statistics/AGENT.md](statistics/AGENT.md) |
| volume | [volume/AGENT.md](volume/AGENT.md) |
| price_transform | [price_transform/AGENT.md](price_transform/AGENT.md) |
| math | [math/AGENT.md](math/AGENT.md) |

## Common Signature

All features follow this pattern:

```python
def feature_name(
    data: pl.DataFrame,
    period: int = 14,
    price_col: str = "close",
) -> pl.DataFrame:
    """Returns DataFrame with original data + new column(s)."""
```

## TA-Lib Compatibility

60 features are validated against TA-Lib at 1e-6 tolerance:
- All momentum indicators (RSI, MACD, ADX, STOCH, etc.)
- All trend indicators (SMA, EMA, WMA, etc.)
- Basic volatility (ATR, NATR, Bollinger)
- Volume indicators (OBV, A/D)
- Statistics (STDDEV, VAR, regression family)
