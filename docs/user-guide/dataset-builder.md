# Dataset Builder

`MLDatasetBuilder` provides leakage-safe dataset preparation for ML training. It handles train/test splitting, automatic scaling (fit on train only), and cross-validation integration with proper fold-level preprocessing.

> **Book**: *ML for Trading, 3rd ed.* — Ch7 `10_ml4t_library_ecosystem.py` demonstrates `MLDatasetBuilder` with triple-barrier labels: features + labels in, scaled train/test split out. Ch7 `02_preprocessing_pipeline.py` covers the underlying preprocessing concepts.

## Basic Usage

```python
from ml4t.engineer import create_dataset_builder

builder = create_dataset_builder(
    features=features_df,       # pl.DataFrame of feature columns
    labels=labels_series,       # pl.Series of target labels
    dates=dates_series,         # Optional: pl.Series of timestamps
    scaler="standard",          # "standard", "minmax", "robust", or None
)
```

### Train/Test Split

```python
X_train, X_test, y_train, y_test = builder.train_test_split(
    train_size=0.8,
    shuffle=False,              # Keep False for time series!
    random_state=None,
)
```

When a scaler is set, `train_test_split` automatically:

1. Fits the scaler on `X_train` only
2. Transforms both `X_train` and `X_test` using training statistics
3. Returns scaled DataFrames

This prevents information leakage by construction.

### Dataset Info

```python
info = builder.info
# DatasetInfo(
#     n_samples=2516,
#     n_features=45,
#     feature_names=["rsi", "macd", "atr", ...],
#     label_name="label",
#     has_dates=True,
# )
```

## Scaler Configuration

```python
from ml4t.engineer.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Via string shorthand
builder = create_dataset_builder(features, labels, scaler="standard")
builder = create_dataset_builder(features, labels, scaler="robust")
builder = create_dataset_builder(features, labels, scaler="minmax")
builder = create_dataset_builder(features, labels, scaler=None)  # No scaling

# Via scaler instance (custom parameters)
builder = create_dataset_builder(
    features, labels,
    scaler=RobustScaler(quantile_range=(10.0, 90.0)),
)

# Change scaler after construction
builder.set_scaler(MinMaxScaler(feature_range=(0, 1)))
builder.set_scaler(None)  # Disable scaling
```

## Cross-Validation Integration

`MLDatasetBuilder` integrates with any splitter that follows the `SplitterProtocol` (compatible with ml4t-diagnostic's `WalkForwardCV` and `CombinatorialCV`).

```python
for fold in builder.split(cv=splitter):
    # Each fold has properly scaled train/test data
    fold.X_train       # pl.DataFrame (scaled with train-only stats)
    fold.X_test        # pl.DataFrame (scaled with train stats)
    fold.y_train       # pl.Series
    fold.y_test        # pl.Series
    fold.fold_number   # int
    fold.scaler        # Fitted BaseScaler (or None)
    fold.train_indices # np.ndarray
    fold.test_indices  # np.ndarray

    # Convert to numpy for sklearn/lightgbm
    X_np, X_test_np, y_np, y_test_np = fold.to_numpy()

    # Train your model
    model.fit(X_np, y_np)
    preds = model.predict(X_test_np)
```

Each fold gets its own scaler instance, fitted independently on that fold's training data. This is the correct behavior for time-series cross-validation where the training window shifts.

### FoldResult

The `FoldResult` dataclass returned by each iteration:

```python
@dataclass
class FoldResult:
    X_train: pl.DataFrame
    X_test: pl.DataFrame
    y_train: pl.Series
    y_test: pl.Series
    train_indices: NDArray[np.intp]
    test_indices: NDArray[np.intp]
    fold_number: int
    scaler: BaseScaler | None = None

    def to_numpy(self) -> tuple[NDArray, NDArray, NDArray, NDArray]
```

## Percentile Computation

For creating training-only thresholds (e.g., percentile-based labels):

```python
# Feature percentiles (training data only)
cutoffs = builder.get_feature_percentiles(
    train_idx=train_indices,
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
)

# Label percentiles (for discretizing continuous targets)
label_cutoffs = builder.compute_label_percentiles(
    train_idx=train_indices,
    n_quantiles=5,
)
```

These methods ensure percentile thresholds are computed from training data only, preventing look-ahead bias.

## Format Conversion

```python
# To numpy (raw, no scaling applied)
X_np, y_np = builder.to_numpy()

# To pandas
X_pd, y_pd = builder.to_pandas()
```

## Factory Function

The `create_dataset_builder` factory provides convenient scaler configuration:

```python
from ml4t.engineer import create_dataset_builder

builder = create_dataset_builder(
    features=features_df,
    labels=labels_series,
    dates=dates_series,           # Optional timestamps
    scaler="standard",            # str, BaseScaler, PreprocessingConfig, or None
)
```

The `scaler` parameter accepts:

| Value | Effect |
|-------|--------|
| `"standard"` | StandardScaler with defaults |
| `"minmax"` | MinMaxScaler with (0, 1) range |
| `"robust"` | RobustScaler with IQR |
| `None` | No scaling |
| `BaseScaler` instance | Custom scaler with your parameters |
| `PreprocessingConfig` | Config object that creates the scaler |

## End-to-End Example

```python
import polars as pl
from ml4t.engineer import compute_features, create_dataset_builder
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels

# 1. Compute features
df = pl.read_parquet("spy_daily.parquet")
features_df = compute_features(df, ["rsi", "macd", "atr", "bollinger_bands"])

# 2. Create labels
config = LabelingConfig.triple_barrier(
    upper_barrier=0.02, lower_barrier=0.01, max_holding_period=20,
)
labeled = triple_barrier_labels(features_df, config=config)

# 3. Build dataset
feature_cols = [c for c in features_df.columns if c not in df.columns]
builder = create_dataset_builder(
    features=labeled.select(feature_cols),
    labels=labeled["label"],
    dates=labeled["timestamp"],
    scaler="robust",
)

# 4. Train/test split with automatic scaling
X_train, X_test, y_train, y_test = builder.train_test_split(train_size=0.8)
```
