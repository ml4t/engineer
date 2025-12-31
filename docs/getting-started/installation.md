# Installation

## Requirements

- Python 3.9 or higher
- Polars 0.20+

## Install from PyPI

```bash
pip install ml4t-engineer
```

## Install from Source

```bash
git clone https://github.com/stefan-jansen/ml4t-engineer.git
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
from ml4t.engineer import list_features, list_categories

# Check available features
print(f"Total features: {len(list_features())}")
print(f"Categories: {list_categories()}")
```

Expected output:
```
Total features: 107
Categories: ['math', 'microstructure', 'ml', 'momentum', 'price_transform', 'statistics', 'trend', 'volatility', 'volume']
```
