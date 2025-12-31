"""Example of using qfeatures pipeline with TA indicators.

This example demonstrates how to use the new pipeline API to chain
multiple technical analysis indicators together.
"""

import ml4t_features as qf
import polars as pl
from ml4t.engineer.features.ta import create_ta_pipeline_step, rsi, sma, ta_step

# Create sample data
data = pl.DataFrame(
    {
        "event_time": pl.datetime_range(
            start=pl.datetime(2024, 1, 1),
            end=pl.datetime(2024, 4, 9),  # 100 days
            interval="1d",
            eager=True,
        ),
        "asset_id": "BTC",
        "open": [100.0 + i * 0.5 + (i % 10) for i in range(100)],
        "high": [101.0 + i * 0.5 + (i % 10) + 1 for i in range(100)],
        "low": [99.0 + i * 0.5 + (i % 10) - 1 for i in range(100)],
        "close": [100.0 + i * 0.5 + (i % 10) + 0.5 for i in range(100)],
        "volume": [1000000.0 + i * 10000 for i in range(100)],
    },
)

print("Sample data shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

# Method 1: Using the pipeline with custom functions
print("\n" + "=" * 50)
print("Method 1: Pipeline with custom functions")
print("=" * 50)

pipeline1 = qf.pipeline.Pipeline(
    steps=[
        # Calculate returns
        ("returns", lambda df: df.with_columns(returns=pl.col("close").pct_change())),
        # Add 20-day SMA
        (
            "sma_20",
            lambda df: df.with_columns(
                sma_20=sma(values="close", period=20).alias("sma_20"),
            ),
        ),
        # Add RSI
        (
            "rsi_14",
            lambda df: df.with_columns(
                rsi_14=rsi(values="close", period=14).alias("rsi_14"),
            ),
        ),
    ],
)

result1 = pipeline1.run(data)
print("\nPipeline 1 result columns:", result1.columns)
print("\nLast few rows with indicators:")
print(result1.select(["event_time", "close", "returns", "sma_20", "rsi_14"]).tail())

# Method 2: Using ta_step adapter
print("\n" + "=" * 50)
print("Method 2: Pipeline with ta_step adapter")
print("=" * 50)

# Create pipeline steps using the adapter
sma_step = ta_step(sma, output_name="sma_50", values="close", period=50)
rsi_step = ta_step(rsi, output_name="rsi_21", values="close", period=21)

pipeline2 = qf.pipeline.Pipeline(
    steps=[
        ("sma", sma_step),
        ("rsi", rsi_step),
    ],
)

result2 = pipeline2.run(data)
print("\nPipeline 2 result columns:", result2.columns)
print("\nSample of results:")
print(result2.select(["event_time", "close", "sma_50", "rsi_21"]).slice(50, 5))

# Method 3: Using create_ta_pipeline_step
print("\n" + "=" * 50)
print("Method 3: Pipeline with create_ta_pipeline_step")
print("=" * 50)

pipeline3 = qf.pipeline.Pipeline(
    steps=[
        (
            "ema_12",
            create_ta_pipeline_step(
                "ema",
                output_name="ema_12",
                values="close",
                period=12,
            ),
        ),
        (
            "ema_26",
            create_ta_pipeline_step(
                "ema",
                output_name="ema_26",
                values="close",
                period=26,
            ),
        ),
        (
            "macd",
            lambda df: df.with_columns(
                macd_line=(pl.col("ema_12") - pl.col("ema_26")).alias("macd_line"),
            ),
        ),
    ],
)

result3 = pipeline3.run(data)
print("\nPipeline 3 result columns:", result3.columns)
print("\nMACD calculation results:")
print(result3.select(["event_time", "close", "ema_12", "ema_26", "macd_line"]).tail(10))

# Demonstrate intermediate results access
print("\n" + "=" * 50)
print("Accessing intermediate pipeline results")
print("=" * 50)

intermediate = pipeline3.get_intermediate_result("ema_12")
print("\nResult after 'ema_12' step:")
print(intermediate.columns)

# Show that the pipeline is working correctly
print("\n" + "=" * 50)
print("Summary Statistics")
print("=" * 50)

final_result = result3.select(
    [
        pl.col("close").mean().alias("avg_close"),
        pl.col("ema_12").mean().alias("avg_ema_12"),
        pl.col("ema_26").mean().alias("avg_ema_26"),
        pl.col("macd_line").mean().alias("avg_macd"),
        pl.col("macd_line").std().alias("std_macd"),
    ],
)

print(final_result)

print("\n✅ Pipeline example completed successfully!")
