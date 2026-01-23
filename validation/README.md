# ML4T Engineer Validation Suite

Cross-framework validation tests verifying ml4t-engineer calculations.

## Feature Coverage

| Category | TA-Lib Compatible | Test Functions | Validation Method | Confidence |
|----------|-------------------|----------------|-------------------|------------|
| Momentum | 29 | 30+ | TA-Lib C library | **100%** |
| Trend | 9 | 15+ | TA-Lib C library | **100%** |
| Volatility | 4 | 10+ | TA-Lib C library | **100%** |
| Volume | 3 | 8+ | TA-Lib C library | **100%** |
| Statistics | 7 | 12+ | TA-Lib C library | **100%** |
| Price Transform | 5 | 10+ | TA-Lib C library | **100%** |
| Math | 3 | 5+ | TA-Lib C library | **100%** |
| **Labeling** | - | 65+ | AFML formulas + mlfinpy | **100%** |
| Microstructure | - | 30+ | Unit tests | Unit tests only |
| ML utilities | - | 20+ | N/A (utilities) | N/A |

**Total TA-Lib Compatible**: 60 features (marked `ta_lib_compatible=True` in registry)

### Labeling Validation (De Prado)

Two validation approaches:

#### 1. AFML Book Formulas (`vs_afml/`)

Direct verification against mathematical formulas from "Advances in Financial Machine Learning":

| Module | Tests | What's Validated |
|--------|-------|------------------|
| Triple Barrier | 25+ | Upper/lower/time barriers, returns, OHLC checking |
| Meta-Labeling | 15+ | Binary labels, threshold, bet sizing |
| Sample Weights | 20+ | Concurrency, uniqueness, sequential bootstrap |

These tests use **deterministic synthetic cases** where correct answers are known by construction.

#### 2. mlfinpy Comparison (`vs_mlfinpy/`)

Comparison against the [mlfinpy](https://mlfinpy.readthedocs.io/) reference implementation:

| Test File | What's Compared |
|-----------|-----------------|
| test_triple_barrier.py | Label distribution, return calculations |
| test_sample_weights.py | Indicator matrix, uniqueness, sequential bootstrap |

**Note**: mlfinpy requires separate venv due to numba version conflict.

### TA-Lib Validation

95 test functions in `tests/test_talib_*.py` validate against TA-Lib C library with 1e-6 precision.

## Directory Structure

```
validation/
├── README.md           # This file
├── conftest.py         # Shared fixtures
├── requirements.txt    # Dependencies
├── vs_afml/            # AFML book formula tests
│   ├── test_triple_barrier_formulas.py
│   ├── test_meta_labeling_formulas.py
│   └── test_sample_weight_formulas.py
├── vs_mlfinpy/         # mlfinpy comparison (requires separate venv)
│   ├── test_triple_barrier.py
│   └── test_sample_weights.py
├── vs_talib/           # TA-Lib tests (main validation in tests/)
└── golden/             # Regression test snapshots
```

## Running Validation

```bash
# AFML formula tests (primary labeling validation)
uv run pytest validation/vs_afml/ -v

# TA-Lib validation
uv run pytest tests/test_talib_*.py -v

# Golden file regression tests
uv run pytest validation/golden/ -v
```

### mlfinpy Tests (Optional)

mlfinpy requires numba 0.60.0 which conflicts with ml4t-engineer's numba 0.63.1.

```bash
# Create separate venv for mlfinpy
python3 -m venv .venv-mlfinpy
source .venv-mlfinpy/bin/activate

# Install mlfinpy and ml4t-engineer
pip install mlfinpy pandas numpy pytest pytest-cov
pip install -e .

# Run mlfinpy comparison tests
pytest validation/vs_mlfinpy/ -v

# Deactivate when done
deactivate
```

**Note**: Some sample weight tests are skipped due to a pandas 2.x compatibility
bug in mlfinpy's `get_ind_matrix` function. The triple barrier and concurrency
tests run and pass successfully.

## Tolerance Standards

- **TA-Lib**: 1e-6 relative tolerance
- **AFML formulas**: 1e-10 absolute tolerance for deterministic cases
- **mlfinpy**: 20% tolerance for stochastic comparisons

## References

- López de Prado, M. (2018). "Advances in Financial Machine Learning". Wiley.
- [mlfinpy documentation](https://mlfinpy.readthedocs.io/)
- [TA-Lib documentation](https://ta-lib.org/)
