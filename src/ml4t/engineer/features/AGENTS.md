# features/ - Technical Indicators and Feature Helpers

This directory contains the registry-backed indicator implementations used by
`compute_features()` plus a small set of standalone feature helpers.

## Categories Registered in `compute_features()`

| Category | Count | Location | Examples |
|----------|-------|----------|----------|
| momentum | 31 | `momentum/` | `rsi`, `macd`, `stoch`, `adx`, `mfi` |
| trend | 10 | `trend/` | `sma`, `ema`, `wma`, `kama`, `donchian_channels` |
| volatility | 15 | `volatility/` | `atr`, `bollinger_bands`, `yang_zhang_volatility`, `garch_forecast` |
| microstructure | 15 | `microstructure/` | `kyle_lambda`, `amihud_illiquidity`, `realized_spread` |
| ml | 14 | `ml/` plus `fdiff.py` | lag features, entropy, Fourier, percentile-rank features |
| statistics | 14 | `statistics/` | `stddev`, `var`, `linearreg`, `structural_break` |
| volume | 3 | `volume/` | `obv`, `ad`, `adosc` |
| price_transform | 5 | `price_transform/` | `avgprice`, `typprice`, `medprice`, `wclprice`, `midprice` |
| math | 3 | `math/` | rolling `max_`, `min_`, `sum_` |
| regime | 4 | `regime.py` | `hurst_exponent`, `choppiness_index`, `fractal_efficiency` |
| risk | 6 | `risk.py` | drawdown, downside deviation, tail ratio |

## Standalone Helpers

These are not registry entries but remain important:

| File | Purpose |
|------|---------|
| `cross_asset.py` | multi-asset and panel features such as rolling correlation and beta |
| `fdiff.py` | fractional differencing utilities such as `ffdiff()` and `find_optimal_d()` |
| `composite.py` | composite indicators and z-score combinations |

## Navigation

| Category | AGENTS |
|----------|--------|
| momentum | [momentum/AGENTS.md](momentum/AGENTS.md) |
| trend | [trend/AGENTS.md](trend/AGENTS.md) |
| volatility | [volatility/AGENTS.md](volatility/AGENTS.md) |
| microstructure | [microstructure/AGENTS.md](microstructure/AGENTS.md) |
| ml | [ml/AGENTS.md](ml/AGENTS.md) |
| statistics | [statistics/AGENTS.md](statistics/AGENTS.md) |
| volume | [volume/AGENTS.md](volume/AGENTS.md) |
| price_transform | [price_transform/AGENTS.md](price_transform/AGENTS.md) |
| math | [math/AGENTS.md](math/AGENTS.md) |

## Current Usage Pattern

```python
from ml4t.engineer import compute_features

result = compute_features(
    df,
    [
        "rsi",
        {"name": "sma", "params": {"period": 50}},
        {"name": "bollinger_bands", "params": {"period": 20, "std_dev": 2.5}},
    ],
)
```

## Trust Signals

- 120 total features across 11 categories
- 60 TA-Lib validated indicators at `1e-6` tolerance
- registry metadata supports discovery by category, normalization, and TA-Lib parity
