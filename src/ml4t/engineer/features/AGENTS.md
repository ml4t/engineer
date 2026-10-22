# features/ - Technical Indicators and Feature Helpers

This directory contains the registry-backed indicator implementations used by
`compute_features()` plus a small set of standalone feature helpers.

## Categories Registered in `compute_features()`

| Category | Count | Location | Examples |
|----------|-------|----------|----------|
| momentum | 36 | `momentum/`, `rolling.py` | `rsi`, `rsi_cutler`, `absolute_percent_change`, `finite_macd_signal` |
| trend | 15 | `trend/`, `rolling.py` | `sma`, `ema`, `finite_mean`, `finite_ewma` |
| volatility | 18 | `volatility/`, `rolling.py` | `atr`, `bollinger_bands`, `annualized_volatility` |
| microstructure | 15 | `microstructure/` | `kyle_lambda`, `amihud_illiquidity`, `realized_spread` |
| ml | 14 | `ml/` plus `fdiff.py` | lag features, entropy, Fourier, percentile-rank features |
| statistics | 15 | `statistics/`, `rolling.py` | `stddev`, `var`, `finite_ewm_std`, `structural_break` |
| volume | 4 | `volume/`, `rolling.py` | `obv`, `ad`, `adosc`, `average_dollar_volume` |
| price_transform | 5 | `price_transform/` | `avgprice`, `typprice`, `medprice`, `wclprice`, `midprice` |
| math | 3 | `math/` | rolling `max_`, `min_`, `sum_` |
| regime | 4 | `regime.py` | `hurst_exponent`, `choppiness_index`, `fractal_efficiency` |
| risk | 7 | `risk.py`, `rolling.py` | drawdown, downside deviation, tail ratio |

## Standalone Helpers

These are not registry entries but remain important:

| File | Purpose |
|------|---------|
| `cross_asset.py` | multi-asset and panel features such as rolling correlation and beta |
| `paired.py` | full-window paired correlation/beta/regression with explicit missing-count limits |
| `events.py` | supplied known event weekday counts, separate from exchange-session counts |
| `selection.py` | eligible contemporary group transforms, stable selectors and boolean windows |
| `preparation.py` | decision-time eligibility over supplied prepared histories and universe facts |
| `inputs.py` | optional market/loading/classification container conversion and known-at preparation |
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
    calendar=None,
)
```

This example explicitly counts observation rows. Omitting `calendar` selects shared
feed metadata, then NYSE, and requires a declared complete daily or intraday grid. Use
`FeedSpec` declarations for expected bars; do not infer frequency or holidays from gaps.
Numerical kernels keep their documented annualization conventions independently of
calendar selection. `sma`, `stddev` and `realized_volatility` also support session/elapsed
windows through the public API. Reuse their existing finite kernels over causal original
bars, with schedule ordinals for sessions and open-left physical UTC intervals for elapsed
durations. Do not resample, infer missing sessions or reinterpret recursive indicators.

## Trust Signals

- 136 total features across 11 categories
- 60 TA-Lib validated indicators at `1e-6` tolerance
- registry metadata supports discovery by category, normalization, and TA-Lib parity
