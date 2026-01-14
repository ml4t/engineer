# price_transform/ - 5 Price Transformations

Price averaging methods. All TA-Lib compatible.

## Average Prices

### avgprice(data) -> DataFrame
Average Price. (O + H + L + C) / 4

### typprice(data) -> DataFrame
Typical Price. (H + L + C) / 3

### medprice(data) -> DataFrame
Median Price. (H + L) / 2

### wclprice(data) -> DataFrame
Weighted Close Price. (H + L + 2*C) / 4

### midprice(data, period=14) -> DataFrame
Mid Price. (highest_high + lowest_low) / 2 over period.

## Common Usage

```python
# Use typical price for indicator calculation
df = typprice(df)
# Then use typ_price column as input
df = sma(df, price_col="typ_price", period=20)
```

## Notes

- No parameters needed except midprice (requires period)
- Output column names: avg_price, typ_price, med_price, wcl_price, mid_price
