# ML4T Engineer

Turn OHLCV and tick data into model-ready features, labels, and sampling schemes
without rewriting the same research code in every notebook and pipeline.

`ml4t-engineer` is the feature-engineering layer in the ML4T stack. It sits between
`ml4t-data`, which prepares canonical datasets, and `ml4t-diagnostic`, which
evaluates signals and models. Start here if you want a working workflow quickly, use
the [Book Guide](book-guide/index.md) to map notebooks to production APIs, and use the
[API Reference](api/index.md) when you need exact interfaces.

Chapters 7-10 of *Machine Learning for Trading, Third Edition* develop many of these
methods manually in notebooks. This library packages those computations as tested,
reusable functions. See the [Book Guide](book-guide/index.md) to map notebook code to
library calls.

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } __120 Features, One Call__

    ---

    Momentum, volatility, microstructure, trend, and other feature families through
    `compute_features(df, indicators)`.
    [:octicons-arrow-right-24: Features](user-guide/features.md)

-   :material-check-decagram:{ .lg .middle } __60 TA-Lib Validated__

    ---

    Indicators tested against TA-Lib to `1e-6` tolerance so notebook and pipeline
    outputs stay aligned.
    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-label:{ .lg .middle } __Labels, Bars, and Leakage Control__

    ---

    Triple-barrier labels, alternative bars, preprocessing, and dataset splitting in
    the same workflow.
    [:octicons-arrow-right-24: Labeling](user-guide/labeling.md)

-   :material-book-open-variant:{ .lg .middle } __Book to Production__

    ---

    The book teaches the methods step by step. This library turns them into reusable
    calls for research and scheduled pipelines.
    [:octicons-arrow-right-24: Book Guide](book-guide/index.md)

</div>

## Quick Example

You have OHLCV data and need a feature matrix:

```python
import polars as pl
from ml4t.engineer import compute_features

df = pl.read_parquet("spy_daily.parquet")
features = compute_features(df, ["rsi", "macd", "atr"])

print(features.select("rsi_14", "macd", "atr_14").tail(3))
```

That single call appends validated indicator columns to the same DataFrame you will
pass downstream into labeling, preprocessing, and model training.

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
    features=labels.select(["rsi_14", "macd", "atr_14"]),
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

This is the standard bridge from Chapter 9’s fractional differencing workflow to a
reusable production transform.

## Documentation Entry Points

- [Quickstart](getting-started/quickstart.md) for a working feature and labeling run
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
| Cross-Asset | 10 | Beta, Correlation, Cointegration |
| Risk | 6 | Max Drawdown, Sortino, CVaR |
| Price Transform | 5 | Typical Price, Weighted Close |
| Regime | 4 | Hurst Exponent, Choppiness Index |
| Volume | 3 | OBV, AD, ADOSC |
| Math | 3 | MAX, MIN, SUM |

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
