# Fractional Differencing

Fractional differencing (FFD) produces stationary time series while preserving long-range memory — the key insight from De Prado (2018, Chapter 5). Standard first-differencing (d=1) achieves stationarity but destroys predictive signal; fractional differencing finds the minimum d that passes stationarity tests.

## The Memory-Stationarity Tradeoff

| Differencing Degree | Stationarity | Memory Preserved | ML Utility |
|---------------------|-------------|------------------|------------|
| d = 0 (original) | Non-stationary | 100% | Poor (unit root) |
| 0 < d < 0.5 | May be stationary | High | Optimal zone |
| d = 0.5 | Borderline | Moderate | Acceptable |
| d = 1 (first diff) | Stationary | ~0% | Poor (signal lost) |

The goal: find the smallest d where the ADF test rejects the null hypothesis of a unit root (p-value < 0.05).

## Core Functions

### `ffdiff` — Apply Fractional Differencing

```python
from ml4t.engineer.features.fdiff import ffdiff

# As a Polars expression (chainable)
result = df.with_columns(
    ffdiff("close", d=0.4, threshold=1e-5).alias("close_ffd")
)

# Or on a Series directly
ffd_series = ffdiff(df["close"], d=0.4)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `close` | `pl.Series \| pl.Expr \| str` | required | Price series or column name |
| `d` | `float` | required | Differencing degree (0 to 2) |
| `threshold` | `float` | `1e-5` | Minimum weight magnitude (truncation) |

**How it works**: FFD applies a weighted sum of lagged values where weights are derived from the fractional binomial expansion. Weights decay geometrically, and the `threshold` parameter truncates negligibly small weights for efficiency. Weights are cached via `@lru_cache` and the inner loop is Numba-accelerated.

### `find_optimal_d` — Find Minimum Stationary d

```python
from ml4t.engineer.features.fdiff import find_optimal_d

result = find_optimal_d(
    close=df["close"],          # Must be a pl.Series (not Expr)
    d_range=(0.0, 1.0),        # Search range
    step=0.01,                  # Step size (finer = slower)
    adf_pvalue_threshold=0.05,  # ADF significance level
)

print(result)
# {"optimal_d": 0.35, "adf_pvalue": 0.023, "correlation": 0.97}
```

**Returns** a dict with:

| Key | Description |
|-----|-------------|
| `optimal_d` | Smallest d passing ADF test |
| `adf_pvalue` | ADF p-value at optimal d |
| `correlation` | Correlation between original and FFD series |

A high correlation (>0.90) means most of the predictive information is preserved.

### `fdiff_diagnostics` — Full Diagnostic Report

```python
from ml4t.engineer.features.fdiff import fdiff_diagnostics

diag = fdiff_diagnostics(
    close=df["close"],
    d=0.4,
    threshold=1e-5,
)

print(diag)
# {"d": 0.4, "adf_statistic": -3.21, "adf_pvalue": 0.019,
#  "correlation": 0.96, "n_weights": 127, "weight_sum": 0.998}
```

## Step-by-Step Workflow

### 1. Find Optimal d

```python
import polars as pl
from ml4t.engineer.features.fdiff import find_optimal_d, ffdiff

# Load price data
df = pl.read_parquet("spy_daily.parquet")

# Find minimum d for stationarity
result = find_optimal_d(df["close"], step=0.01)
optimal_d = result["optimal_d"]
print(f"Optimal d: {optimal_d}, ADF p-value: {result['adf_pvalue']:.4f}")
print(f"Correlation with original: {result['correlation']:.4f}")
```

### 2. Apply Fractional Differencing

```python
# Apply FFD with optimal d
df = df.with_columns(
    ffdiff("close", d=optimal_d).alias("close_ffd")
)
```

### 3. Validate with ADF Test

```python
from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(df["close_ffd"].drop_nulls().to_numpy())
print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.6f}")
# Should be < 0.05
```

### 4. Use as ML Feature

```python
# FFD series is now stationary but retains memory
# Use alongside other features
from ml4t.engineer import compute_features

features = compute_features(df, ["rsi", "macd", "atr"])
features = features.with_columns(
    ffdiff("close", d=optimal_d).alias("close_ffd")
)
```

## Via compute_features

Fractional differencing is also available through the standard `compute_features` API:

```python
result = compute_features(df, [
    {"name": "fractional_diff", "params": {"d": 0.4}},
])
```

Or find the optimal d and apply it:

```python
result = compute_features(df, [
    {"name": "ffdiff_optimal", "params": {"adf_pvalue_threshold": 0.05}},
])
```

## Asset-Class Guidelines

Typical optimal d values (these are starting points — always validate on your data):

| Asset Class | Typical d Range | Notes |
|-------------|----------------|-------|
| Equity indices (SPY, QQQ) | 0.3 - 0.5 | Strong trend component |
| Individual stocks | 0.3 - 0.6 | Higher variance, may need larger d |
| Futures (ES, NQ) | 0.2 - 0.4 | Session structure affects ADF |
| FX pairs | 0.3 - 0.5 | Mean-reverting pairs may need lower d |
| Crypto | 0.4 - 0.7 | High volatility, regime-dependent |

### Multi-Asset Application

For multi-asset pipelines, compute optimal d per asset:

```python
for symbol in ["SPY", "QQQ", "IWM"]:
    asset_data = df.filter(pl.col("symbol") == symbol)
    result = find_optimal_d(asset_data["close"])
    print(f"{symbol}: d={result['optimal_d']}, corr={result['correlation']:.3f}")
```

## Performance

- Weights computed once and cached (`@lru_cache`)
- Inner loop Numba-accelerated
- Weight truncation via `threshold` limits computation window
- Typical: 383 lines of implementation for the full module

> **Book**: *ML for Trading, 3rd ed.* — Ch9 `03_fractional_differencing.py` applies FFD to equity data with ADF test validation and memory-stationarity analysis. The ETFs and US Equities Panel case studies use FFD in their `04_temporal.py` notebooks as a standard feature preparation step.

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 5: Fractionally Differentiated Features.
- Hosking, J.R.M. (1981). Fractional differencing. *Biometrika*, 68(1), 165-176.
