# Installation

Use this page when you want to install `ml4t-engineer`, verify the package surface,
and then move directly into the first workflow.

## Requirements

- Python 3.12, 3.13, 3.14, or 3.15
- Polars 0.20+

## Install from PyPI

```bash
pip install ml4t-engineer
```

If you standardize on `uv`, `uv pip install ml4t-engineer` is equivalent.

## Install from Source

```bash
git clone https://github.com/ml4t/engineer.git
cd ml4t-engineer
pip install -e .
```

## Optional Dependencies

### TA-Lib (for validation)

Some indicators can be validated against TA-Lib. The `ta` extra requires TA-Lib 0.8.2
or later in the 0.8 series, whose wheels include the native library and support Python 3.15:

```bash
uv pip install "ml4t-engineer[ta]"
```

Use the [upstream installation guide](https://github.com/TA-Lib/ta-lib-python#installation)
when your platform requires a source build. Comparisons must declare matching moving-average
types: Engineer's APO/PPO default remains SMA (`matype=0`), while TA-Lib 0.8 defaults to EMA.

### Statistical diagnostics

`find_optimal_d()` and `fdiff_diagnostics()` use Statsmodels for the augmented
Dickey-Fuller test:

```bash
pip install "ml4t-engineer[stats]"
```

The core `ffdiff()` transform does not require this extra.

## Verify Installation

```python
from ml4t.engineer import feature_catalog

# Check available features
all_features = feature_catalog.list()
print(f"Total features: {len(all_features)}")
print(f"Categories: {feature_catalog.categories()}")
```

Expected output:
```
Total features: 120
Categories: ['math', 'microstructure', 'ml', 'momentum', 'price_transform', 'regime', 'risk', 'statistics', 'trend', 'volatility', 'volume']
```

## Next Steps

- Read [Quickstart](quickstart.md) for the first feature-computation workflow.
- Read the [Book Guide](../book-guide/index.md) if you are coming from the book or
  case studies.
- Use the [API Reference](../api/index.md) once you need exact object locations.
