# Installation

Use this page when you want to install `ml4t-engineer`, verify the package surface,
and then move directly into the first workflow.

## Requirements

- Python 3.12 or higher
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

Some indicators can be validated against TA-Lib. Install TA-Lib if you need this:

```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Ubuntu/Debian
sudo apt-get install libta-lib-dev
pip install TA-Lib

# Windows
# Download from https://www.ta-lib.org/
pip install TA-Lib
```

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

- Read [Quickstart](quickstart.md) for a first working feature and labeling example.
- Read the [Book Guide](../book-guide/index.md) if you are coming from the book or
  case studies.
- Use the [API Reference](../api/index.md) once you need exact object locations.
