# Golden File Tests

This directory contains regression tests that compare current feature calculations
against stored "golden" reference outputs.

## Purpose

Golden file testing provides:
1. **Regression protection**: Detect unintended changes to feature calculations
2. **Documentation**: Captured outputs serve as reference implementations
3. **Reproducibility**: Known inputs → known outputs for debugging

## Structure

```
golden/
├── README.md           # This file
├── __init__.py
├── generate_golden.py  # Script to regenerate golden files
├── test_golden.py      # Pytest test comparing against golden files
└── data/               # Stored reference outputs
    ├── sma_20_spy.parquet
    ├── rsi_14_spy.parquet
    └── ...
```

## Usage

### Running Golden Tests

```bash
# Run golden file tests
python -m pytest validation/golden/ -v

# Run with verbose diff output
python -m pytest validation/golden/ -v --tb=long
```

### Regenerating Golden Files

**Only regenerate if you intentionally changed feature calculations!**

```bash
# Regenerate all golden files
python validation/golden/generate_golden.py

# Regenerate specific feature
python validation/golden/generate_golden.py --feature sma
```

### Adding New Golden Tests

1. Add feature to `FEATURES` dict in `generate_golden.py`
2. Run regeneration script
3. Commit new `.parquet` files
4. Add test case to `test_golden.py`

## Golden File Format

Files are stored as Parquet for efficient storage and exact reproducibility:

```python
# Reading golden file
import polars as pl
golden = pl.read_parquet("validation/golden/data/sma_20_spy.parquet")
```

## Tolerance

Golden tests use **exact match** by default (tolerance=0).

If a change is intentional, the test will fail with a clear diff showing
what changed.
