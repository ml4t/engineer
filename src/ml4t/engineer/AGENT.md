# ml4t.engineer Package

Package-level navigation for feature engineering library.

## Top-Level Modules

| Module | Lines | Purpose | Key Functions |
|--------|-------|---------|---------------|
| `api.py` | 445 | Main API | `compute_features()`, `list_features()`, `describe_feature()` |
| `preprocessing.py` | 800 | Scalers, transforms | `StandardScaler`, `RobustScaler`, `Preprocessor` |
| `dataset.py` | 630 | Dataset creation | `FeatureDataset`, `create_dataset()` |
| `__init__.py` | - | Package exports | Re-exports from api.py |

## Subdirectories

| Directory | Purpose | AGENT.md |
|-----------|---------|----------|
| `features/` | 107 technical indicators (10 categories) | [features/AGENT.md](features/AGENT.md) |
| `labeling/` | ML label generation methods | [labeling/AGENT.md](labeling/AGENT.md) |
| `bars/` | Alternative bar sampling | [bars/AGENT.md](bars/AGENT.md) |
| `core/` | Registry, types, validation, decorators | See below |
| `config/` | Pydantic v2 configurations | See below |
| `outcome/` | IC analysis, drift detection | See below |
| `selection/` | Feature selection methods | See below |
| `store/` | DuckDB feature storage | See below |
| `validation/` | Cross-validation utilities | See below |
| `visualization/` | Plotly charts | See below |

## core/ - Foundation

| File | Purpose | Key Exports |
|------|---------|-------------|
| `registry.py` | Feature metadata registry | `FeatureRegistry`, `FeatureMetadata`, `get_registry()` |
| `decorators.py` | @feature decorator | `feature()` |
| `types.py` | Type definitions | `PriceDataFrame`, `FeatureSpec` |
| `schemas.py` | Polars schemas | `OHLCV_SCHEMA` |
| `validation.py` | Input validation | `validate_ohlcv()`, `validate_window()` |
| `exceptions.py` | Custom exceptions | `DataValidationError`, `InvalidParameterError` |

## config/ - Configurations

| File | Purpose | Key Classes |
|------|---------|-------------|
| `feature_config.py` | Feature specifications | `FeatureConfig`, `FeatureSpec` |
| `labeling.py` | Labeling config | `LabelingConfig`, `BarrierConfig` |
| `preprocessing_config.py` | Scaler config | `PreprocessingConfig`, `ScalerConfig` |
| `base.py` | Base config classes | `BaseConfig` |

## outcome/ - Analysis

| File | Lines | Purpose |
|------|-------|---------|
| `drift.py` | 1808 | Feature drift detection, PSI, concept drift |
| `feature_outcome.py` | 1016 | IC analysis, importance scoring |
| `plot_ic.py` | 430 | IC visualization |
| `plot_importance.py` | 419 | Importance charts |

## selection/ - Feature Selection

| File | Purpose |
|------|---------|
| `systematic.py` | Correlation-based, variance threshold, mutual info |

## store/ - Storage

| File | Purpose |
|------|---------|
| `offline.py` | DuckDB-based feature storage |

## Common Patterns

### Adding a New Feature

```python
from ml4t.engineer.core.decorators import feature

@feature(name="my_indicator", category="momentum")
def my_indicator(data: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """One-line description."""
    # Implementation using Polars expressions
    return data.with_columns(result.alias(f"my_indicator_{period}"))
```

### Using Registry

```python
from ml4t.engineer.core.registry import get_registry

registry = get_registry()
all_features = registry.list_features()
momentum = registry.list_features(category="momentum")
metadata = registry.get("rsi")
```

## File Size Tiers (for refactoring reference)

**Very Large (>1000 lines)** - Consider splitting:
- `outcome/drift.py` (1808)
- `labeling/core.py` (1694)
- `outcome/feature_outcome.py` (1016)

**Large (500-1000 lines)**:
- `config/feature_config.py`, `preprocessing.py`, `features/risk.py`
- `selection/systematic.py`, `features/ml/rolling_entropy.py`
- `bars/run.py`, `features/cross_asset.py`, `dataset.py`
