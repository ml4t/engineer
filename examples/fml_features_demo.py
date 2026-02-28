"""
Comprehensive demonstration of Financial Machine Learning (FML) features.

This example shows how to use all the new qfeatures capabilities:
1. Information-driven bars for better sampling
2. Fractional differencing for stationarity with memory
3. Triple-barrier labeling for ML-ready targets
4. Pipeline integration for end-to-end workflows
"""

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from ml4t.engineer import pipeline
from ml4t.engineer.bars import ImbalanceBarSampler, VolumeBarSampler
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.features import fractional_diff
from ml4t.engineer.labeling import triple_barrier_labels


def load_or_create_data():
    """Load crypto data or create synthetic data for demonstration."""
    # Try to load real data first
    data_path = Path("data/crypto/futures/BTC.parquet")

    if data_path.exists():
        print("Loading BTC futures data...")
        df = pl.read_parquet(data_path)
        # Ensure we have required columns
        if "close" not in df.columns and "Close" in df.columns:
            df = df.rename({"Close": "close"})
        if "volume" not in df.columns and "Volume" in df.columns:
            df = df.rename({"Volume": "volume"})
        return df.head(50000)  # Use first 50k rows for speed

    # Create synthetic tick data
    print("Creating synthetic market data...")
    n = 50000
    base_time = datetime(2024, 1, 1)

    # Generate realistic price movement with trends
    np.random.seed(42)
    returns = np.random.standard_t(df=3, size=n) * 0.001  # Fat tails
    trend = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.0001  # Cyclical trend
    returns = returns + trend

    prices = 30000 * np.exp(np.cumsum(returns))

    # Generate volume correlated with volatility
    volatility = np.abs(returns)
    volumes = np.random.lognormal(
        mean=np.log(1000),
        sigma=0.5 + 10 * volatility,
    ).astype(int)

    # Generate buy/sell pressure
    momentum = np.convolve(returns, np.ones(20) / 20, mode="same")
    buy_prob = 0.5 + np.clip(momentum * 100, -0.3, 0.3)
    sides = np.where(np.random.rand(n) < buy_prob, 1, -1)

    return pl.DataFrame(
        {
            "timestamp": [base_time + timedelta(minutes=i) for i in range(n)],
            "open": prices * (1 + np.random.randn(n) * 0.0001),
            "high": prices * (1 + np.abs(np.random.randn(n)) * 0.0002),
            "low": prices * (1 - np.abs(np.random.randn(n)) * 0.0002),
            "close": prices,
            "volume": volumes,
            "price": prices,  # For bar sampling
            "side": sides,  # For imbalance bars
        },
    )


def demonstrate_bars(data):
    """Demonstrate information-driven bar sampling."""
    print("\n" + "=" * 60)
    print("1. INFORMATION-DRIVEN BARS")
    print("=" * 60)

    # Create different bar types
    print("\nCreating volume bars...")
    volume_sampler = VolumeBarSampler(volume_per_bar=100_000)
    volume_bars = volume_sampler.sample(data)

    print(f"Original ticks: {len(data):,}")
    print(f"Volume bars: {len(volume_bars):,}")
    print(f"Compression ratio: {len(data) / len(volume_bars):.1f}x")

    # Analyze bar properties
    returns = volume_bars.select(pl.col("close").pct_change().alias("returns"))[
        "returns"
    ].drop_nulls()

    print("\nVolume bar statistics:")
    print(f"  Mean return: {returns.mean():.4%}")
    print(f"  Std return: {returns.std():.4%}")
    print(f"  Skewness: {returns.skew():.3f}")
    print(f"  Kurtosis: {returns.kurtosis():.3f}")

    # Create imbalance bars
    print("\nCreating imbalance bars...")
    imbalance_sampler = ImbalanceBarSampler(
        expected_ticks_per_bar=100,
        initial_expectation=10000,
        alpha=0.2,
    )
    imbalance_bars = imbalance_sampler.sample(data)

    print(f"Imbalance bars: {len(imbalance_bars):,}")
    print(f"Avg ticks per bar: {imbalance_bars['tick_count'].mean():.1f}")

    return volume_bars, imbalance_bars


def demonstrate_fdiff(bars):
    """Demonstrate fractional differencing."""
    print("\n" + "=" * 60)
    print("2. FRACTIONAL DIFFERENCING")
    print("=" * 60)

    prices = bars["close"]

    # Find optimal differencing order
    print("\nFinding optimal differencing order...")
    d_opt, adf_stats = fractional_diff.find_min_ffd_order(
        prices,
        significance_level=0.05,
        max_d=1.0,
    )

    print(f"Optimal d: {d_opt:.3f}")
    print(f"ADF statistic at d={d_opt:.3f}: {adf_stats[d_opt]:.3f}")

    # Apply fractional differencing
    ffd_prices = fractional_diff(prices, d=d_opt, threshold=0.01)

    # Compare properties
    print("\nOriginal series:")
    print(f"  Mean: {prices.mean():.2f}")
    print(f"  Std: {prices.std():.2f}")
    print(f"  Min: {prices.min():.2f}")
    print(f"  Max: {prices.max():.2f}")

    print("\nFractionally differenced series:")
    print(f"  Mean: {ffd_prices.mean():.6f}")
    print(f"  Std: {ffd_prices.std():.6f}")
    print(f"  Min: {ffd_prices.min():.6f}")
    print(f"  Max: {ffd_prices.max():.6f}")

    # Add to bars dataframe
    bars_with_ffd = bars.with_columns(ffd_close=ffd_prices)

    return bars_with_ffd, d_opt


def demonstrate_labeling(bars_with_ffd):
    """Demonstrate triple-barrier labeling."""
    print("\n" + "=" * 60)
    print("3. TRIPLE-BARRIER LABELING")
    print("=" * 60)

    # Calculate volatility for dynamic barriers
    bars_with_vol = bars_with_ffd.with_columns(
        returns=pl.col("close").pct_change(),
        volatility=pl.col("close").pct_change().rolling_std(window_size=20),
    )

    # Create dynamic barriers
    bars_with_barriers = bars_with_vol.with_columns(
        upper_barrier=(2 * pl.col("volatility")).fill_null(0.02),
        lower_barrier=(-1 * pl.col("volatility")).fill_null(-0.01),
    )

    # Apply triple-barrier labeling
    config = LabelingConfig.triple_barrier(
        upper_barrier="upper_barrier",
        lower_barrier="lower_barrier",
        max_holding_period=10,
        trailing_stop=False,
    )

    labeled_bars = triple_barrier_labels(
        bars_with_barriers,
        config,
        price_col="close",
        timestamp_col="timestamp",
    )

    # Analyze labels
    label_counts = labeled_bars.group_by("label").count().sort("label")
    print("\nLabel distribution:")
    for row in label_counts.iter_rows():
        label, count = row
        pct = count / len(labeled_bars) * 100
        label_name = {-1: "Loss", 0: "Timeout", 1: "Profit"}.get(label, "Unknown")
        print(f"  {label_name}: {count:,} ({pct:.1f}%)")

    # Analyze barrier hits
    barrier_counts = labeled_bars.group_by("barrier_hit").count().sort("barrier_hit")
    print("\nBarrier hit distribution:")
    for row in barrier_counts.iter_rows():
        barrier, count = row
        if barrier is not None:
            pct = count / labeled_bars.filter(pl.col("barrier_hit").is_not_null()).height * 100
            print(f"  {barrier}: {count:,} ({pct:.1f}%)")

    return labeled_bars


def demonstrate_pipeline(data):
    """Demonstrate end-to-end pipeline."""
    print("\n" + "=" * 60)
    print("4. PIPELINE INTEGRATION")
    print("=" * 60)

    # Create a complete FML pipeline
    fml_pipeline = pipeline.Pipeline(
        steps=[
            # Step 1: Create volume bars
            (
                "volume_bars",
                lambda df: VolumeBarSampler(volume_per_bar=50_000).sample(df),
            ),
            # Step 2: Add returns and volatility
            (
                "features",
                lambda df: df.with_columns(
                    [
                        pl.col("close").pct_change().alias("returns"),
                        pl.col("volume").rolling_mean(window_size=10).alias("volume_ma"),
                        pl.col("close").rolling_std(window_size=20).alias("volatility_20"),
                    ],
                ),
            ),
            # Step 3: Add fractionally differenced price
            (
                "ffd",
                lambda df: df.with_columns(
                    ffd_close=fractional_diff(df["close"], d=0.3, threshold=0.01),
                ),
            ),
            # Step 4: Create dynamic barriers
            (
                "barriers",
                lambda df: df.with_columns(
                    [
                        (2 * pl.col("volatility_20")).alias("upper_barrier"),
                        (-1 * pl.col("volatility_20")).alias("lower_barrier"),
                    ],
                ),
            ),
            # Step 5: Apply labeling
            (
                "labeling",
                lambda df: triple_barrier_labels(
                    df,
                    LabelingConfig.triple_barrier(
                        upper_barrier="upper_barrier",
                        lower_barrier="lower_barrier",
                        max_holding_period=10,
                    ),
                    price_col="close",
                    timestamp_col="timestamp",
                ),
            ),
            # Step 6: Filter to labeled events only
            ("filter", lambda df: df.filter(pl.col("label").is_not_null())),
            # Step 7: Select ML-ready features
            (
                "select",
                lambda df: df.select(
                    [
                        "timestamp",
                        "ffd_close",
                        "returns",
                        "volume_ma",
                        "volatility_20",
                        "label",
                        "label_return",
                    ],
                ),
            ),
        ],
    )

    # Run pipeline
    print("\nRunning pipeline...")
    ml_ready_data = fml_pipeline.run(data)

    print("\nPipeline results:")
    print(f"  Input rows: {len(data):,}")
    print(f"  Output rows: {len(ml_ready_data):,}")
    print(f"  Features: {', '.join(ml_ready_data.columns)}")

    # Show sample
    print("\nSample of ML-ready data:")
    print(ml_ready_data.head())

    return ml_ready_data


def plot_bar_sampling(ax, data, volume_bars):
    """Plot bar sampling comparison."""
    ax.plot(data["close"][:1000], alpha=0.3, label="Raw ticks")

    # Resample bars to tick indices for comparison
    n_ticks = min(1000, len(data))
    n_volume_bars = int(n_ticks * len(volume_bars) / len(data))
    volume_indices = np.linspace(0, n_ticks - 1, n_volume_bars).astype(int)

    ax.plot(
        volume_indices,
        volume_bars["close"][:n_volume_bars],
        "o-",
        alpha=0.7,
        label="Volume bars",
    )
    ax.set_title("Information-Driven Sampling")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_returns_distribution(ax, data, volume_bars):
    """Plot returns distribution comparison."""
    tick_returns = data.select(pl.col("close").pct_change())["close"].drop_nulls()
    bar_returns = volume_bars.select(pl.col("close").pct_change())["close"].drop_nulls()

    ax.hist(tick_returns, bins=50, alpha=0.5, density=True, label="Tick returns")
    ax.hist(bar_returns, bins=50, alpha=0.5, density=True, label="Bar returns")
    ax.set_title("Returns Distribution")
    ax.set_xlabel("Returns")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_imbalance_characteristics(ax, imbalance_bars):
    """Plot imbalance bar characteristics."""
    scatter = ax.scatter(
        imbalance_bars["tick_count"][:100],
        imbalance_bars["imbalance"][:100],
        c=imbalance_bars["expected_imbalance"][:100],
        cmap="viridis",
        alpha=0.6,
    )
    ax.set_title("Imbalance Bar Dynamics")
    ax.set_xlabel("Ticks per Bar")
    ax.set_ylabel("Order Flow Imbalance")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Expected Threshold")


def plot_fractional_differencing(ax, volume_bars, labeled_bars, d_opt):
    """Plot fractional differencing comparison."""
    ax.plot(volume_bars["close"][:200], label="Original", alpha=0.7)
    if "ffd_close" in labeled_bars.columns:
        ffd_normalized = (
            labeled_bars["ffd_close"][:200]
            * volume_bars["close"][:200].std()
            / labeled_bars["ffd_close"][:200].std()
            + volume_bars["close"][:200].mean()
        )
        ax.plot(ffd_normalized, label=f"FFD (d={d_opt:.3f})", alpha=0.7)
    ax.set_title("Fractional Differencing")
    ax.set_xlabel("Bar Number")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_label_distribution(ax, labeled_bars):
    """Plot triple-barrier label distribution."""
    label_counts = labeled_bars.group_by("label").count().sort("label")
    labels = []
    counts = []
    for row in label_counts.iter_rows():
        label, count = row
        if label is not None:
            labels.append(
                {-1: "Loss", 0: "Timeout", 1: "Profit"}.get(label, str(label)),
            )
            counts.append(count)

    colors = ["red", "gray", "green"][: len(labels)]
    ax.bar(labels, counts, color=colors)
    ax.set_title("Triple-Barrier Labels")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)


def plot_feature_correlation(ax, labeled_bars):
    """Plot feature correlation heatmap."""
    numeric_cols = ["returns", "volatility", "ffd_close", "label"]
    available_cols = [col for col in numeric_cols if col in labeled_bars.columns]

    if len(available_cols) >= 2:
        corr_data = labeled_bars.select(available_cols).drop_nulls()
        if len(corr_data) > 0:
            corr_matrix = np.corrcoef(corr_data.to_numpy().T)
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                xticklabels=available_cols,
                yticklabels=available_cols,
                ax=ax,
                cmap="coolwarm",
                center=0,
            )
            ax.set_title("Feature Correlations")


def create_visualizations(data, volume_bars, imbalance_bars, labeled_bars, d_opt):
    """Create comprehensive visualizations."""
    print("\n" + "=" * 60)
    print("5. VISUALIZATIONS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Financial Machine Learning Features Demonstration", fontsize=16)

    # 1. Bar sampling comparison
    plot_bar_sampling(axes[0, 0], data, volume_bars)

    # 2. Returns distribution
    plot_returns_distribution(axes[0, 1], data, volume_bars)

    # 3. Imbalance bar characteristics
    plot_imbalance_characteristics(axes[0, 2], imbalance_bars)

    # 4. Fractional differencing
    plot_fractional_differencing(axes[1, 0], volume_bars, labeled_bars, d_opt)

    # 5. Label distribution
    plot_label_distribution(axes[1, 1], labeled_bars)

    # 6. Feature correlations
    plot_feature_correlation(axes[1, 2], labeled_bars)

    plt.tight_layout()
    plt.savefig("examples/fml_features_demo.png", dpi=150)
    print("Saved visualization to examples/fml_features_demo.png")


def main():
    """Run the complete FML features demonstration."""
    print("Financial Machine Learning Features Demonstration")
    print("=" * 60)

    # Load data
    data = load_or_create_data()
    print(f"\nLoaded {len(data):,} rows of market data")

    # 1. Information-driven bars
    volume_bars, imbalance_bars = demonstrate_bars(data)

    # 2. Fractional differencing
    bars_with_ffd, d_opt = demonstrate_fdiff(volume_bars)

    # 3. Triple-barrier labeling
    labeled_bars = demonstrate_labeling(bars_with_ffd)

    # 4. Pipeline integration
    demonstrate_pipeline(data)

    # 5. Create visualizations
    create_visualizations(data, volume_bars, imbalance_bars, labeled_bars, d_opt)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"""
The qfeatures library provides essential tools for financial machine learning:

1. **Information-Driven Bars**: Better sampling that captures market microstructure
   - Volume bars: Sample by traded volume
   - Dollar bars: Sample by dollar volume
   - Imbalance bars: Sample by order flow imbalance

2. **Fractional Differencing**: Achieve stationarity while preserving memory
   - Optimal d = {d_opt:.3f} for this data
   - Maintains more information than integer differencing

3. **Triple-Barrier Labeling**: Create directional labels for ML
   - Dynamic barriers based on volatility
   - Handles path-dependency and time limits
   - Reduces noise in labels

4. **Pipeline Integration**: Combine all features seamlessly
   - Declarative API for complex workflows
   - Handles point-in-time correctness
   - Ready for production use

These features address key challenges in applying ML to finance:
- Non-stationarity of price series
- Irregular sampling of market data
- Label noise and path-dependency
- Feature engineering complexity

Next steps:
- Use ml_ready_data to train ML models
- Experiment with different bar types and parameters
- Add more features (technical indicators, microstructure)
- Implement walk-forward validation
""",
    )

    print("\n✅ Demonstration completed successfully!")


if __name__ == "__main__":
    main()
