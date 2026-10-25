# Installation

Use this page when you want to install `ml4t-engineer`, verify the package surface,
and then move directly into the first workflow.

## Requirements

- Python 3.12, 3.13, 3.14, or 3.15
- Polars 2.0+

The published 0.2.0 release supports these Python versions and the `calendar` argument.
Published 0.1.6
supports Python 3.12-3.14 and does not accept the new `calendar` argument. Use its
[version-matched quick start](https://github.com/ml4t/engineer/tree/v0.1.6#quick-start)
for that release.

## Install from PyPI

```bash
pip install ml4t-engineer
```

If you standardize on `uv`, `uv pip install ml4t-engineer` is equivalent.

## Install from Source

```bash
git clone https://github.com/ml4t/engineer.git
cd engineer
uv sync --no-dev
```

Run the examples with the created environment, for example `uv run --no-sync python`.

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

### Feature store

The `store` extra adds DuckDB and PyArrow; the core quick start does not need it.
Our Python 3.15 qualification used DuckDB 1.5.6 source builds on Linux, macOS and Windows.
If no matching wheel is available, install the
[upstream native build prerequisites](https://duckdb.org/docs/current/dev/building/python)
before enabling the extra. Source builds need a C++ toolchain and can take tens of minutes.

Within the source checkout, enable the extra and verify both imports:

```bash
uv sync --no-dev --extra store
uv run --no-sync python -c "import duckdb; import ml4t.engineer.store; print(duckdb.__version__)"
```

### Statistical diagnostics

`find_optimal_d()` and `fdiff_diagnostics()` use Statsmodels for the augmented
Dickey-Fuller test:

```bash
pip install "ml4t-engineer[stats]"
```

The core `ffdiff()` transform does not require this extra.

### Optional ML dependencies

The `ml` extra installs LightGBM and SHAP:

```bash
uv pip install "ml4t-engineer[ml]"
```

On macOS, LightGBM also needs the OpenMP runtime. Install it before importing the ML
dependencies, including when using the `dev` or `all` extras:

```bash
brew install libomp
```

See the [LightGBM Python installation guide](https://github.com/lightgbm-org/LightGBM/blob/v4.7.0/python-package/README.rst#installation)
for native dependency details.

## Verify Installation

```python
from ml4t.engineer import feature_catalog

# Check available features
all_features = feature_catalog.list()
print(f"Total features: {len(all_features)}")
print(f"Categories: {feature_catalog.categories()}")
```

This prints the features and categories available in your installed release.

## Next Steps

- Read [Quickstart](quickstart.md) for the first feature-computation workflow.
- Read the [Book Guide](../book-guide/index.md) if you are coming from the book or
  case studies.
- Use the [API Reference](../api/index.md) once you need exact object locations.
