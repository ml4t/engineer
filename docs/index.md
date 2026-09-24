# ML4T Engineer

Feature engineering, labeling, alternative bars, and leakage-safe datasets for
financial ML.

`ml4t-engineer` is the feature-engineering layer in the ML4T stack. It sits between
`ml4t-data`, which prepares canonical datasets, and `ml4t-diagnostic`, which evaluates
signals and models.

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } __First successful workflow__

    ---

    Install the released package, compute three features from synthetic data, and
    verify the result.
    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-check-decagram:{ .lg .middle } __Task guides__

    ---

    Compute features, create labels, sample bars, and build leakage-safe datasets.
    [:octicons-arrow-right-24: Features](user-guide/features.md)

-   :material-label:{ .lg .middle } __Exact API reference__

    ---

    Look up released functions, classes, signatures, and supported options.
    [:octicons-arrow-right-24: API Reference](api/index.md)

-   :material-book-open-variant:{ .lg .middle } __Checked book notebooks__

    ---

    Open commit-pinned notebooks and see whether each one calls Engineer, teaches the
    method manually, or illustrates a related workflow.
    [:octicons-arrow-right-24: Book Guide](book-guide/index.md)

</div>

## Quick Example

You have OHLCV data and need a feature matrix:

<!-- ml4t-exec -->
```python
from datetime import date, timedelta

import polars as pl
from ml4t.engineer import compute_features

close = [100.0 + i * 0.1 + (i % 7) * 0.2 for i in range(100)]
df = pl.DataFrame({
    "timestamp": [date(2024, 1, 1) + timedelta(days=i) for i in range(100)],
    "open": close,
    "high": [price + 1.0 for price in close],
    "low": [price - 1.0 for price in close],
    "close": close,
    "volume": [100_000 + i * 100 for i in range(100)],
})
features = compute_features(df, ["rsi", "macd", "atr"])

assert {"rsi", "macd", "atr"} <= set(features.columns)
assert features.select(["rsi", "macd", "atr"]).drop_nulls().height > 0
```

The call appends three indicator columns to the input DataFrame. The assertions check
that the columns exist and contain values after their rolling warmup windows.

## Core Workflows

### 1. Add supervised labels

You have features. You need targets for a classification model:

```python
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels

config = LabelingConfig.triple_barrier(
    upper_barrier=0.02, lower_barrier=0.01, max_holding_period=20,
)
labels = triple_barrier_labels(features, config=config)
```

This produces standardized label columns such as `label`, `label_return`, and
`barrier_hit` for supervised learning workflows.

### 2. Build train/test data without leakage

You have features and labels. You need train/test splits with train-only scaling:

```python
from ml4t.engineer import create_dataset_builder

builder = create_dataset_builder(
    features=labels.select(["rsi", "macd", "atr"]),
    labels=labels["label"],
    dates=labels["timestamp"],
    scaler="robust",
)
X_train, X_test, y_train, y_test = builder.train_test_split(train_size=0.8)
```

This keeps preprocessing statistics on the training window only, which is the default
you want for time-series ML.

### 3. Use non-time bars when time bars are the wrong abstraction

You have trade data and need bars tied to market activity instead of clock time:

```python
from ml4t.engineer.bars import VolumeBarSampler

sampler = VolumeBarSampler(volume_per_bar=50_000)
volume_bars = sampler.sample(trades_df)
```

This turns raw trade prints into OHLCV bars that are easier to use in downstream
feature and labeling pipelines.

### 4. Preserve memory while making series stationary

You need a stationary series but do not want to erase signal with first differences:

```python
from ml4t.engineer.features.fdiff import find_optimal_d, ffdiff

result = find_optimal_d(df["close"])
ffd_close = ffdiff(df["close"], d=result["optimal_d"])
```

The parameter search requires the `stats` extra. The [Fractional Differencing guide](user-guide/fractional-differencing.md) explains the statistical check and the core-only `ffdiff()` transform.

## Documentation Entry Points

- [Quickstart](getting-started/quickstart.md) for the first feature-computation workflow
- [Features](user-guide/features.md) for the core computation API
- [Labeling](user-guide/labeling.md) for supervised targets and sample weighting
- [Book Guide](book-guide/index.md) for chapter, notebook, and case-study mapping
- [API Reference](api/index.md) for exact interfaces

## Feature Catalog

Use this as reference once you know what kind of signal you want to build.

| Category | Count | Examples |
|----------|-------|----------|
| Momentum | 31 | RSI, MACD, Stochastic, CCI, ADX, MFI |
| Microstructure | 15 | Kyle Lambda, VPIN, Amihud, Roll spread |
| Volatility | 15 | ATR, Bollinger, Yang-Zhang, Parkinson |
| Statistics | 14 | Variance, Linear Regression, Correlation |
| ML | 14 | Fractional Diff, Entropy, Lag features |
| Trend | 10 | SMA, EMA, WMA, DEMA, TEMA, KAMA |
| Risk | 6 | Max Drawdown, Sortino, CVaR |
| Price Transform | 5 | Typical Price, Weighted Close |
| Regime | 4 | Hurst Exponent, Choppiness Index |
| Volume | 3 | OBV, AD, ADOSC |
| Math | 3 | MAX, MIN, SUM |

Standalone cross-asset utilities such as beta, rolling correlation, and
cointegration are also available outside the 120-feature registry.

## Installation

```bash
pip install ml4t-engineer
```

If you use `uv`, `uv pip install ml4t-engineer` is equivalent. See
[Installation](getting-started/installation.md) for environment details and optional
TA-Lib setup.

## Part of the ML4T Library Suite

```text
ml4t-data → ml4t-engineer → ml4t-diagnostic → ml4t-backtest → ml4t-live
```

`ml4t-engineer` is where raw market data becomes reusable research inputs: features,
labels, alternative bars, and leakage-safe training datasets.
