"""Example of using the generalized triple-barrier labeling method."""

from datetime import datetime

import matplotlib.pyplot as plt
import polars as pl

from ml4t.engineer import pipeline
from ml4t.engineer.labeling import BarrierConfig, triple_barrier_labels

# Load real crypto data
print("Loading BTC futures data...")
df = pl.read_parquet("data/crypto/futures/BTC.parquet")
print(f"Data shape: {df.shape}")

# Take a subset for demonstration
df = df.slice(0, 1000)  # First 1000 minutes

# Add volatility estimate for dynamic barriers
df = df.with_columns(
    volatility=pl.col("close").rolling_std(window_size=20) / pl.col("close"),
)

# Example 1: Fixed barriers
print("\n" + "=" * 50)
print("Example 1: Fixed Barriers")
print("=" * 50)

config_fixed = BarrierConfig(
    upper_barrier=0.01,  # 1% profit target
    lower_barrier=-0.005,  # 0.5% stop loss
    max_holding_period=30,  # 30 minutes max
)

result_fixed = triple_barrier_labels(
    df,
    config_fixed,
    price_col="close",
    timestamp_col="date_time",
)

# Show label distribution
label_counts = result_fixed.group_by("label").len().sort("label")
print("\nLabel distribution:")
print(label_counts)

# Example 2: Dynamic barriers based on volatility
print("\n" + "=" * 50)
print("Example 2: Dynamic Barriers (Volatility-based)")
print("=" * 50)

# Create dynamic barriers: 2x volatility for upper, 1x for lower
df = df.with_columns(
    upper_barrier_dynamic=(2 * pl.col("volatility")).fill_null(0.01),
    lower_barrier_dynamic=(-1 * pl.col("volatility")).fill_null(-0.005),
)

config_dynamic = BarrierConfig(
    upper_barrier="upper_barrier_dynamic",
    lower_barrier="lower_barrier_dynamic",
    max_holding_period=30,
)

result_dynamic = triple_barrier_labels(
    df,
    config_dynamic,
    price_col="close",
    timestamp_col="date_time",
)

# Show barrier hit distribution
barrier_counts = result_dynamic.group_by("barrier_hit").len().sort("barrier_hit")
print("\nBarrier hit distribution:")
print(barrier_counts)

# Example 3: Trailing stop
print("\n" + "=" * 50)
print("Example 3: With Trailing Stop")
print("=" * 50)

config_trailing = BarrierConfig(
    upper_barrier=0.02,  # 2% profit target
    lower_barrier=-0.01,  # 1% stop loss
    max_holding_period=60,  # 60 minutes max
    trailing_stop=0.005,  # 0.5% trailing stop
)

result_trailing = triple_barrier_labels(
    df,
    config_trailing,
    price_col="close",
    timestamp_col="date_time",
)

# Analyze returns by label
print("\nReturn statistics by label:")
for label in [-1, 0, 1]:
    label_data = result_trailing.filter(pl.col("label") == label)
    if len(label_data) > 0:
        returns = label_data["label_return"]
        print(f"\nLabel {label}:")
        print(f"  Count: {len(label_data)}")
        print(f"  Mean return: {returns.mean():.4%}")
        print(f"  Std return: {returns.std():.4%}")

# Example 4: Duration Analysis
print("\n" + "=" * 50)
print("Example 4: Duration Analysis (Bars vs Time)")
print("=" * 50)

config_duration = BarrierConfig(
    upper_barrier=0.02,  # 2% profit target
    lower_barrier=-0.01,  # 1% stop loss
    max_holding_period=60,  # 60 minutes max
)

result_duration = triple_barrier_labels(
    df,
    config_duration,
    price_col="close",
    timestamp_col="date_time",
)

# Duration statistics
print("\nDuration Statistics:")
print(f"Average bars held: {result_duration['label_bars'].mean():.1f}")
print(f"Max bars held: {result_duration['label_bars'].max()}")
print(f"Min bars held: {result_duration['label_bars'].min()}")

# Analyze time duration
time_durations_seconds = (
    result_duration.select(pl.col("label_duration").dt.total_seconds()).to_series().to_list()
)
avg_seconds = sum(s for s in time_durations_seconds if s is not None) / len(
    [s for s in time_durations_seconds if s is not None]
)
print(f"Average time held: {avg_seconds / 60:.1f} minutes")

# Show sample of duration data
print("\nSample duration data (first 10 labels):")
duration_sample = result_duration.filter(pl.col("label").is_not_null()).head(10)
print(
    duration_sample.select(
        [
            "date_time",
            "label",
            "label_bars",
            "label_duration",
            "label_return",
            "barrier_hit",
        ]
    )
)

# Duration by barrier hit type
print("\nAverage duration by barrier hit:")
for barrier_type in ["upper", "lower", "time"]:
    barrier_data = result_duration.filter(pl.col("barrier_hit") == barrier_type)
    if len(barrier_data) > 0:
        avg_bars = barrier_data["label_bars"].mean()
        print(f"  {barrier_type:6s}: {avg_bars:.1f} bars on average")

# Show relationship between duration and returns
print("\nDuration vs Returns Analysis:")
profitable = result_duration.filter(pl.col("label_return") > 0)
unprofitable = result_duration.filter(pl.col("label_return") < 0)
print(
    f"  Profitable trades: avg {profitable['label_bars'].mean():.1f} bars, "
    f"avg return {profitable['label_return'].mean():.4%}"
)
print(
    f"  Unprofitable trades: avg {unprofitable['label_bars'].mean():.1f} bars, "
    f"avg return {unprofitable['label_return'].mean():.4%}"
)

# Visualize a sample of labeling results
print("\n" + "=" * 50)
print("Visualization")
print("=" * 50)

# Take first 200 rows for visualization
viz_data = result_fixed.head(200)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot prices with entry/exit points
ax1.plot(viz_data["close"].to_numpy(), "b-", alpha=0.7, label="Close Price")

# Mark entry and exit points
for i in range(len(viz_data)):
    if viz_data["label"][i] is not None:
        # Entry point
        ax1.plot(i, viz_data["close"][i], "ko", markersize=6)

        # Exit point
        exit_idx = viz_data["label_time"][i]
        if isinstance(exit_idx, datetime):
            # Find the index corresponding to this timestamp
            matches = viz_data.with_row_index().filter(pl.col("date_time") == exit_idx)
            if len(matches) > 0:
                exit_idx = matches["index"][0]
            else:
                continue

        if viz_data["label"][i] == 1:
            ax1.plot(
                exit_idx,
                viz_data["label_price"][i],
                "g^",
                markersize=8,
                label="Profit",
            )
        elif viz_data["label"][i] == -1:
            ax1.plot(
                exit_idx,
                viz_data["label_price"][i],
                "rv",
                markersize=8,
                label="Loss",
            )
        else:
            ax1.plot(
                exit_idx,
                viz_data["label_price"][i],
                "bs",
                markersize=8,
                label="Timeout",
            )

ax1.set_ylabel("Price")
ax1.set_title("BTC Price with Triple-Barrier Labels")
ax1.grid(True, alpha=0.3)

# Plot returns
returns = viz_data["label_return"].to_numpy()
colors = ["red" if r < 0 else "green" if r > 0 else "blue" for r in returns]
ax2.bar(range(len(returns)), returns, color=colors, alpha=0.6)
ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax2.axhline(
    y=config_fixed.upper_barrier,
    color="green",
    linestyle="--",
    alpha=0.5,
    label="Upper Barrier",
)
ax2.axhline(
    y=config_fixed.lower_barrier,
    color="red",
    linestyle="--",
    alpha=0.5,
    label="Lower Barrier",
)
ax2.set_xlabel("Time (minutes)")
ax2.set_ylabel("Return")
ax2.set_title("Labeled Returns")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("examples/labeling_results.png", dpi=150)
print("Saved visualization to examples/labeling_results.png")

# Pipeline integration example
print("\n" + "=" * 50)
print("Pipeline Integration")
print("=" * 50)

# Create a pipeline that combines multiple transformations
pipeline = pipeline.Pipeline(
    steps=[
        # Add returns
        ("returns", lambda df: df.with_columns(returns=pl.col("close").pct_change())),
        # Add volatility
        (
            "volatility",
            lambda df: df.with_columns(
                volatility_20=pl.col("returns").rolling_std(window_size=20),
            ),
        ),
        # Add dynamic barriers
        (
            "barriers",
            lambda df: df.with_columns(
                upper_barrier=2 * pl.col("volatility_20"),
                lower_barrier=-1 * pl.col("volatility_20"),
            ),
        ),
        # Apply labeling
        (
            "labeling",
            lambda df: triple_barrier_labels(
                df,
                BarrierConfig(
                    upper_barrier="upper_barrier",
                    lower_barrier="lower_barrier",
                    max_holding_period=30,
                ),
                price_col="close",
                timestamp_col="date_time",
            ),
        ),
        # Filter to labeled events only
        ("filter", lambda df: df.filter(pl.col("label").is_not_null())),
    ],
)

# Run pipeline
pipeline_result = pipeline.run(df)
print(f"\nPipeline output shape: {pipeline_result.shape}")
print("\nSample of pipeline results:")
print(
    pipeline_result.select(
        [
            "date_time",
            "close",
            "returns",
            "volatility_20",
            "upper_barrier",
            "lower_barrier",
            "label",
            "label_return",
        ],
    ).head(10),
)

print("\n✅ Labeling example completed successfully!")
