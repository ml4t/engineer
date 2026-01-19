"""Example of using fractional differencing with real crypto data."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller

from ml4t.engineer import pipeline
from ml4t.engineer.features.fdiff import fdiff_diagnostics, ffdiff, find_optimal_d

# Load real crypto data
print("Loading BTC futures data...")
df = pl.read_parquet("data/crypto/futures/BTC.parquet")
print(f"Data shape: {df.shape}")
print(f"Date range: {df['date_time'].min()} to {df['date_time'].max()}")

# Take a subset for faster computation
df = df.slice(0, 10000)  # First 10k minutes

# Check if the close price is stationary
close_prices = df["close"].to_numpy()
adf_result = adfuller(close_prices[~np.isnan(close_prices)], autolag="AIC")
print(f"\nOriginal series ADF test p-value: {adf_result[1]:.6f}")
print("Series is", "stationary" if adf_result[1] < 0.05 else "non-stationary")

# Find optimal d
print("\nSearching for optimal d value...")
optimal_result = find_optimal_d(
    df["close"],
    d_range=(0.0, 1.0),
    step=0.05,
    adf_pvalue_threshold=0.05,
)

print(f"Optimal d: {optimal_result['optimal_d']:.3f}")
print(f"ADF p-value at optimal d: {optimal_result['adf_pvalue']:.6f}")
print(f"Correlation with original: {optimal_result['correlation']:.3f}")

# Apply fractional differencing with different d values
print("\nApplying fractional differencing with different d values...")
d_values = [0.3, 0.5, 0.7, optimal_result["optimal_d"]]

results = {}
for d in d_values:
    # Apply FFD
    ffd_series = ffdiff(df["close"], d=d)

    # Get diagnostics
    diag = fdiff_diagnostics(df["close"], d=d)

    results[d] = {"series": ffd_series, "diagnostics": diag}

    print(f"\nd = {d:.1f}:")
    print(f"  ADF p-value: {diag['adf_pvalue']:.6f}")
    print(f"  Correlation: {diag['correlation']:.3f}")
    print(f"  Num weights: {diag['n_weights']}")

# Create visualization
print("\nCreating visualization...")
fig, axes = plt.subplots(len(d_values) + 1, 1, figsize=(12, 10))

# Original series
axes[0].plot(df["close"].to_numpy()[:1000], "b-", alpha=0.7)
axes[0].set_title("Original BTC Price (non-stationary)")
axes[0].set_ylabel("Price")

# FFD transformed series
for i, d in enumerate(d_values):
    ax = axes[i + 1]
    ffd_data = results[d]["series"].to_numpy()[:1000]

    # Skip initial NaN values for plotting
    valid_idx = ~np.isnan(ffd_data)
    ax.plot(np.where(valid_idx)[0], ffd_data[valid_idx], "r-", alpha=0.7)

    title = f"FFD with d={d:.2f} (p-value: {results[d]['diagnostics']['adf_pvalue']:.4f}, "
    title += f"corr: {results[d]['diagnostics']['correlation']:.3f})"
    ax.set_title(title)
    ax.set_ylabel("FFD Value")

axes[-1].set_xlabel("Time (minutes)")

plt.tight_layout()
plt.savefig("examples/fdiff_results.png", dpi=150)
print("Saved visualization to examples/fdiff_results.png")

# Demonstrate pipeline integration
print("\n" + "=" * 50)
print("Pipeline Integration Example")
print("=" * 50)

# Create a pipeline with multiple transformations
pipeline = pipeline.Pipeline(
    steps=[
        # Calculate returns
        ("returns", lambda df: df.with_columns(returns=pl.col("close").pct_change())),
        # Add fractionally differenced price
        (
            "ffd_optimal",
            lambda df: df.with_columns(
                close_ffd=ffdiff("close", d=optimal_result["optimal_d"]),
            ),
        ),
        # Add another FFD with fixed d=0.5
        ("ffd_half", lambda df: df.with_columns(close_ffd_half=ffdiff("close", d=0.5))),
        # Calculate rolling volatility of FFD series
        (
            "volatility",
            lambda df: df.with_columns(
                ffd_volatility=pl.col("close_ffd").rolling_std(window_size=60),
            ),
        ),
    ],
)

# Run pipeline
result_df = pipeline.run(df)
print("\nPipeline output columns:", result_df.columns)

# Show sample of results
print("\nSample of pipeline results:")
print(
    result_df.select(
        [
            "date_time",
            "close",
            "returns",
            "close_ffd",
            "close_ffd_half",
            "ffd_volatility",
        ],
    ).slice(100, 5),
)

print("\n✅ Fractional differencing example completed successfully!")
