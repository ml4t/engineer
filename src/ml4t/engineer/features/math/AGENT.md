# math/ - 3 Math Operators

Rolling min/max/sum with O(n) complexity using monotonic deque.

## Functions

### max_(data, period=30, price_col="close") -> DataFrame
Rolling maximum. O(n) using monotonic decreasing deque.
Output column: `max_{period}`

### min_(data, period=30, price_col="close") -> DataFrame
Rolling minimum. O(n) using monotonic increasing deque.
Output column: `min_{period}`

### sum_(data, period=30, price_col="close") -> DataFrame
Rolling sum. Simple rolling sum.
Output column: `sum_{period}`

## Performance Note

These use monotonic deque algorithm for O(n) complexity instead of naive O(n*k).
Critical for large datasets with long periods.

## Common Usage

```python
# Donchian Channel
df = max_(df, period=20, price_col="high")
df = min_(df, period=20, price_col="low")
df = df.with_columns([
    ((pl.col("max_20") + pl.col("min_20")) / 2).alias("donchian_mid")
])
```

## Note on Naming

Functions use trailing underscore (`max_`, `min_`, `sum_`) to avoid conflict with Python builtins.
