# Preprocessing

ML4T Engineer provides sklearn-compatible scalers built on Polars for leakage-safe feature preprocessing.

## Scalers

All scalers follow the sklearn pattern: `fit()` on training data, `transform()` on any data. This prevents information leakage from test data into training.

### StandardScaler

Z-score normalization: output has mean=0, std=1.

```python
from ml4t.engineer.preprocessing import StandardScaler

scaler = StandardScaler(
    columns=None,          # None = all numeric columns
    with_mean=True,        # Center to zero mean
    with_std=True,         # Scale to unit variance
    ddof=1,                # Delta degrees of freedom
)

# Fit on training data
train_scaled = scaler.fit_transform(train_df)

# Transform test data using training statistics
test_scaled = scaler.transform(test_df)
```

Best for: Approximately Gaussian features. Default choice for most ML models.

### MinMaxScaler

Scale features to a bounded range (default [0, 1]).

```python
from ml4t.engineer.preprocessing import MinMaxScaler

scaler = MinMaxScaler(
    columns=None,
    feature_range=(0.0, 1.0),   # Target range
)

train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)
```

Best for: Neural networks expecting bounded input, or when preserving zero values matters.

### RobustScaler

IQR-based scaling that's resistant to outliers.

```python
from ml4t.engineer.preprocessing import RobustScaler

scaler = RobustScaler(
    columns=None,
    with_centering=True,            # Subtract median
    with_scaling=True,              # Scale by IQR
    quantile_range=(25.0, 75.0),    # IQR range
)

train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)
```

Best for: Financial data with fat tails, outliers, or extreme values.

### When to Use Each Scaler

| Scaler | Use When | Sensitive To |
|--------|----------|-------------|
| `StandardScaler` | Data is approximately Gaussian | Outliers |
| `MinMaxScaler` | Need bounded 0-1 range | Outliers |
| `RobustScaler` | Data has outliers or fat tails | Nothing (robust) |

For financial data, `RobustScaler` is generally the safest default due to fat-tailed return distributions.

## Leakage Prevention

The critical rule: **fit on training data only, transform everything**.

```python
# CORRECT: fit on train, transform both
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df)
X_test = scaler.transform(test_df)     # Uses train statistics

# WRONG: fitting on all data leaks test information
scaler = StandardScaler()
X_all = scaler.fit_transform(all_data)  # Leaks test statistics!
```

### Scaler State

After fitting, inspect the learned statistics:

```python
scaler.is_fitted           # True after fit/fit_transform
scaler.fitted_columns      # ["rsi", "macd", "atr", ...]
scaler.statistics          # {"rsi": {"mean": 52.3, "std": 15.1}, ...}
```

### Serialization

Save and reload fitted scalers:

```python
# Save
state = scaler.to_dict()

# Reload
scaler = StandardScaler.from_dict(state)
```

### Cloning

Create an unfitted copy with the same parameters:

```python
new_scaler = scaler.clone()  # Same params, unfitted
```

## PreprocessingPipeline

For multi-step preprocessing, chain transforms:

```python
from ml4t.engineer.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline.from_recommendations({
    "rsi_14": {"transform": "standardize", "confidence": 0.9},
    "volume": {"transform": "log", "confidence": 0.8},
    "returns": {"transform": "winsorize", "confidence": 0.85},
})

train_transformed = pipeline.fit_transform(train_df)
test_transformed = pipeline.transform(test_df)
```

### Available Transform Types

| Transform | Description |
|-----------|-------------|
| `NONE` | No transformation |
| `LOG` | Log transform (for skewed data) |
| `SQRT` | Square root transform |
| `STANDARDIZE` | Z-score normalization |
| `NORMALIZE` | Min-max scaling |
| `WINSORIZE` | Clip extreme values |
| `DIFF` | First difference |

## Integration with MLDatasetBuilder

The preprocessing module integrates with `MLDatasetBuilder` for a leakage-safe end-to-end workflow. See the [Dataset Builder guide](dataset-builder.md) for details.

```python
from ml4t.engineer import create_dataset_builder

builder = create_dataset_builder(
    features=features_df,
    labels=labels_series,
    scaler="robust",   # "standard", "minmax", "robust", or None
)

# Scaling happens automatically during train/test split
X_train, X_test, y_train, y_test = builder.train_test_split(train_size=0.8)
```

> **Book**: *ML for Trading, 3rd ed.* — Ch7 `02_preprocessing_pipeline.py` demonstrates split-aware preprocessing with StandardScaler in a feature preparation pipeline. See also the [ML-Readiness guide](ml-readiness.md) for which features need preprocessing vs. which are ML-ready out of the box.
