# ml4t-engineer

Feature engineering for financial ML: compute indicators, create labels, sample
alternative bars, and prepare leakage-safe datasets for model training.

## Quick Start

```python
from ml4t.engineer import compute_features, create_dataset_builder
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels

features = compute_features(df, ["rsi", "macd", "atr"])
labels = triple_barrier_labels(
    features,
    config=LabelingConfig.triple_barrier(
        upper_barrier=0.02, lower_barrier=0.01, max_holding_period=20,
    ),
)
builder = create_dataset_builder(
    features=labels.select(["rsi", "macd", "atr"]),
    labels=labels["label"],
    dates=labels["timestamp"],
    scaler="robust",
)
```

## Directory Map

| Path | Purpose | Key Surfaces |
|------|---------|--------------|
| `src/ml4t/engineer/features/` | Feature computation and discovery metadata | `compute_features()`, `feature_catalog`, `FeatureCatalog` |
| `src/ml4t/engineer/labeling/` | Supervised label generation and sample weighting | `triple_barrier_labels()`, `atr_triple_barrier_labels()`, `rolling_percentile_binary_labels()` |
| `src/ml4t/engineer/bars/` | Tick, volume, dollar, imbalance, and run bars | `TickBarSampler`, `VolumeBarSampler`, `DollarBarSampler` |
| `src/ml4t/engineer/dataset.py` | Leakage-safe dataset preparation | `MLDatasetBuilder`, `create_dataset_builder()` |
| `src/ml4t/engineer/preprocessing.py` | Train-only scalers and pipelines | `StandardScaler`, `RobustScaler`, `PreprocessingPipeline` |
| `src/ml4t/engineer/config/` | Reusable config objects | `LabelingConfig`, `PreprocessingConfig`, `DataContractConfig` |

## Public Entry Points

```python
from ml4t.engineer import (
    compute_features,
    create_dataset_builder,
    feature_catalog,
    FeatureCatalog,
    StandardScaler,
    RobustScaler,
)
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import (
    triple_barrier_labels,
    atr_triple_barrier_labels,
    rolling_percentile_binary_labels,
    fixed_time_horizon_labels,
    trend_scanning_labels,
)
from ml4t.engineer.bars import (
    TickBarSampler,
    VolumeBarSampler,
    DollarBarSampler,
    FixedTickImbalanceBarSampler,
    FixedVolumeImbalanceBarSampler,
)
```

## Core Workflows

- `compute_features()` appends indicator columns to an OHLCV DataFrame or LazyFrame
- `LabelingConfig` plus labeling functions create reusable supervised targets
- `MLDatasetBuilder` handles train/test splitting and train-only scaling
- sampler classes in `bars/` turn trade data into non-time bars for downstream use

## Trust Signals

- 120 features across 11 categories
- 60 TA-Lib validated indicators at `1e-6` tolerance
- ~50,000 labels/second for triple-barrier workflows
- shared book integration via `docs/book-guide/index.md`

## Navigation

See [src/ml4t/engineer/AGENTS.md](src/ml4t/engineer/AGENTS.md) for package-level
module orientation and subdirectory entry points.
