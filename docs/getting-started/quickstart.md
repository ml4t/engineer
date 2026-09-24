# Quickstart

Use synthetic OHLCV data to compute three features with the released package. This
workflow needs no credentials, external service, optional dependency, or special
hardware.

## Install

Install `ml4t-engineer` in a Python 3.12, 3.13, or 3.14 environment:

```bash
pip install ml4t-engineer
```

## Compute a feature matrix

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

added = [name for name in features.columns if name not in ohlcv.columns]
print(f"rows={features.height}")
print(f"added={added}")

assert features.height == 100
assert added == ["rsi", "macd", "atr"]
assert features.select(added).drop_nulls().height > 0
```

Expected result:

```text
rows=100
added=['rsi', 'macd', 'atr']
```

`compute_features()` preserves the input rows and columns, then appends the requested
features. Rolling features contain null values during their warmup windows. The final
assertion verifies that the three features produce values after warmup.

## Continue with your task

- [Compute and configure features](../user-guide/features.md)
- [Inspect available features](../user-guide/discovery.md)
- [Create supervised labels](../user-guide/labeling.md)
- [Sample alternative bars](../user-guide/bars.md)
- [Build leakage-safe train/test data](../user-guide/dataset-builder.md)
- [Apply train-only preprocessing](../user-guide/preprocessing.md)
- [Apply fractional differencing](../user-guide/fractional-differencing.md)
- [Look up exact signatures](../api/index.md)
- [Open matching book notebooks](../book-guide/index.md)
