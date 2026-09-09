# ml4t-engineer

[![Python 3.12-3.14](https://img.shields.io/badge/python-3.12--3.14-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/ml4t-engineer)](https://pypi.org/project/ml4t-engineer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Feature engineering, labeling, alternative bars, and leakage-safe datasets for
financial ML.

`ml4t-engineer` provides 120 registry features across 11 categories, path-dependent
and fixed-horizon labeling, activity-based bar sampling, and train-only preprocessing.
The core interface uses Polars DataFrames.

## Installation

`ml4t-engineer` supports Python 3.12, 3.13, and 3.14 on Linux, macOS, and Windows.

```bash
pip install ml4t-engineer
```

The quick start uses only core dependencies. Optional extras provide TA-Lib validation,
DuckDB and PyArrow storage, market calendars, visualization, statistics, and ML tools:

```bash
pip install "ml4t-engineer[ta]"
pip install "ml4t-engineer[store]"
pip install "ml4t-engineer[calendars]"
pip install "ml4t-engineer[viz]"
pip install "ml4t-engineer[stats]"
pip install "ml4t-engineer[ml]"
```

TA-Lib requires its native library. The core package does not require an external
service, credentials, or special hardware. Python 3.15 is not supported while the
active Polars compatibility exception applies.

## Quick Start

<!-- ml4t-exec -->
```python
from datetime import date, timedelta

import polars as pl
from ml4t.engineer import compute_features

close = [100.0 + i * 0.1 + (i % 7) * 0.2 for i in range(100)]
ohlcv = pl.DataFrame(
    {
        "timestamp": [date(2024, 1, 1) + timedelta(days=i) for i in range(100)],
        "open": close,
        "high": [price + 1.0 for price in close],
        "low": [price - 1.0 for price in close],
        "close": close,
        "volume": [100_000 + i * 100 for i in range(100)],
    }
)

features = compute_features(ohlcv, ["rsi", "macd", "atr"])

assert {"rsi", "macd", "atr"} <= set(features.columns)
assert features.height == ohlcv.height
```

`compute_features()` returns the input columns with the requested feature columns
appended. Use the feature registry to inspect categories and parameters before building
larger pipelines.

## Supported Workflows

- Technical, volatility, risk, microstructure, statistical, and ML-oriented features
- Triple-barrier, ATR-barrier, percentile, fixed-horizon, trend-scanning, and meta-labels
- Tick, volume, dollar, imbalance, and run bars
- Train/test splitting with train-only scaling
- Feature metadata search and discovery

See the [documentation](https://www.ml4trading.io/docs/engineer/) for tutorials,
task-oriented guides, explanations, and the API reference. Report defects and request
changes through [GitHub Issues](https://github.com/ml4t/engineer/issues).

## Related Libraries

- [`ml4t-specs`](https://github.com/ml4t/specs) defines the shared market-data and
  artifact contracts used by this package.
- [`ml4t-data`](https://github.com/ml4t/data) supplies validated market data for feature
  computation.
- [`ml4t-diagnostic`](https://github.com/ml4t/diagnostic) evaluates features, labels, and
  model signals produced from engineered datasets.

## Development

```bash
git clone https://github.com/ml4t/engineer.git
cd engineer
uv sync --dev --extra docs --extra ta --extra store --extra viz
uv run ruff check src/ tests/ examples/ scripts/
uv run ruff format --check src/ tests/ examples/ scripts/
uv run ty check
uv run pytest tests/ -q
uv build
uv run mkdocs build --strict
```

Pull requests must also pass the supported Python and operating-system matrix,
dependency and vulnerability review, clean-wheel installation, documented workflow
tests, and ecosystem qualification.

## Project Information

- [Documentation](https://www.ml4trading.io/docs/engineer/)
- [Issue tracker](https://github.com/ml4t/engineer/issues)
- [Releases and changelog](https://github.com/ml4t/engineer/releases)
- [License](LICENSE)
