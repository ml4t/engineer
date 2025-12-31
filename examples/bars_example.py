"""Example demonstrating information-driven bar sampling with real SPY tick data."""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from ml4t.engineer.bars import (
    DollarBarSampler,
    ImbalanceBarSampler,
    TickBarSampler,
    VolumeBarSampler,
)


def load_spy_tick_data():
    """Load SPY tick data from the data directory."""
    data_path = Path("data/equities/SPY")

    # Check what files are available
    if not data_path.exists():
        print(f"Data directory {data_path} not found!")
        return None

    # Look for parquet or CSV files
    parquet_files = list(data_path.glob("*.parquet"))
    csv_files = list(data_path.glob("*.csv"))

    if parquet_files:
        print(f"Loading {parquet_files[0]}")
        return pl.read_parquet(parquet_files[0])
    if csv_files:
        print(f"Loading {csv_files[0]}")
        return pl.read_csv(csv_files[0])
    print(f"No data files found in {data_path}")
    return None


def create_synthetic_tick_data():
    """Create synthetic tick data for demonstration if real data isn't available."""
    from datetime import datetime, timedelta

    import numpy as np

    print("Creating synthetic tick data for demonstration...")

    n = 10000
    base_time = datetime(2024, 1, 1, 9, 30)  # Market open

    # Generate realistic intraday price movement
    np.random.seed(42)

    # Price with intraday patterns
    t = np.linspace(0, 6.5, n)  # 6.5 hours of trading
    trend = 0.001 * t  # Slight upward trend
    seasonality = 0.002 * np.sin(2 * np.pi * t / 6.5)  # U-shaped intraday pattern
    noise = np.random.randn(n) * 0.0001

    prices = 450 * (1 + trend + seasonality + noise.cumsum())

    # Volume with intraday patterns (high at open/close)
    volume_pattern = 2 - np.abs(t - 3.25) / 3.25  # U-shaped volume
    volumes = np.random.poisson(100 * volume_pattern).astype(int)

    # Buy/sell pressure (more balanced with slight momentum)
    momentum = np.sin(2 * np.pi * t / 2)  # Oscillating momentum
    buy_prob = 0.5 + 0.1 * momentum + 0.1 * np.random.randn(n)
    buy_prob = np.clip(buy_prob, 0.1, 0.9)
    sides = np.where(np.random.rand(n) < buy_prob, 1, -1)

    return pl.DataFrame(
        {
            "timestamp": [base_time + timedelta(seconds=i * 2.34) for i in range(n)],
            "price": prices,
            "volume": volumes,
            "side": sides,
        },
    )


def sample_bars(tick_data: pl.DataFrame) -> dict:
    """Sample different types of bars from tick data."""
    bars = {}

    # 1. Tick Bars
    print("\n1. Tick Bars (every 100 ticks)")
    tick_sampler = TickBarSampler(ticks_per_bar=100)
    bars["tick"] = tick_sampler.sample(tick_data)
    print(f"   Created {len(bars['tick'])} bars")
    print(f"   Avg ticks per bar: {bars['tick']['tick_count'].mean():.1f}")

    # 2. Volume Bars
    print("\n2. Volume Bars (10,000 shares per bar)")
    volume_sampler = VolumeBarSampler(volume_per_bar=10_000)
    bars["volume"] = volume_sampler.sample(tick_data)
    print(f"   Created {len(bars['volume'])} bars")
    print(f"   Avg volume per bar: {bars['volume']['volume'].mean():.0f}")
    print(
        f"   Buy/Sell ratio: {bars['volume']['buy_volume'].sum() / bars['volume']['sell_volume'].sum():.2f}",
    )

    # 3. Dollar Bars
    print("\n3. Dollar Bars ($1M per bar)")
    dollar_sampler = DollarBarSampler(dollars_per_bar=1_000_000)
    bars["dollar"] = dollar_sampler.sample(tick_data)
    print(f"   Created {len(bars['dollar'])} bars")
    print(f"   Avg dollar volume: ${bars['dollar']['dollar_volume'].mean():,.0f}")
    print(
        f"   VWAP range: ${bars['dollar']['vwap'].min():.2f} to ${bars['dollar']['vwap'].max():.2f}",
    )

    # 4. Imbalance Bars
    print("\n4. Imbalance Bars")
    imbalance_sampler = ImbalanceBarSampler(
        expected_ticks_per_bar=50,
        initial_expectation=5000,
        alpha=0.2,
    )
    bars["imbalance"] = imbalance_sampler.sample(tick_data)
    print(f"   Created {len(bars['imbalance'])} bars")
    print(f"   Avg ticks per bar: {bars['imbalance']['tick_count'].mean():.1f}")
    print(
        f"   Imbalance range: {bars['imbalance']['imbalance'].min():.0f} to {bars['imbalance']['imbalance'].max():.0f}",
    )

    return bars


def create_visualizations(bars: dict) -> None:
    """Create visualizations for different bar types."""
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Information-Driven Bar Sampling Comparison", fontsize=16)

    # Extract bars
    tick_bars = bars["tick"]
    volume_bars = bars["volume"]
    dollar_bars = bars["dollar"]
    imbalance_bars = bars["imbalance"]

    # Plot 1: Bar counts over time
    ax = axes[0, 0]
    bar_types = ["Tick", "Volume", "Dollar", "Imbalance"]
    bar_counts = [
        len(tick_bars),
        len(volume_bars),
        len(dollar_bars),
        len(imbalance_bars),
    ]
    ax.bar(bar_types, bar_counts, color=["blue", "green", "orange", "red"])
    ax.set_ylabel("Number of Bars")
    ax.set_title("Bar Count by Type")
    ax.grid(True, alpha=0.3)

    # Plot 2: Average ticks per bar
    ax = axes[0, 1]
    avg_ticks = [
        tick_bars["tick_count"].mean(),
        volume_bars["tick_count"].mean(),
        dollar_bars["tick_count"].mean(),
        imbalance_bars["tick_count"].mean(),
    ]
    ax.bar(bar_types, avg_ticks, color=["blue", "green", "orange", "red"])
    ax.set_ylabel("Average Ticks per Bar")
    ax.set_title("Information Content by Bar Type")
    ax.grid(True, alpha=0.3)

    # Plot 3: Price series comparison (first 100 bars)
    ax = axes[1, 0]
    n_bars = min(
        100,
        len(tick_bars),
        len(volume_bars),
        len(dollar_bars),
        len(imbalance_bars),
    )

    ax.plot(tick_bars["close"][:n_bars], label="Tick Bars", alpha=0.7)
    ax.plot(volume_bars["close"][:n_bars], label="Volume Bars", alpha=0.7)
    ax.plot(dollar_bars["close"][:n_bars], label="Dollar Bars", alpha=0.7)
    ax.plot(imbalance_bars["close"][:n_bars], label="Imbalance Bars", alpha=0.7)

    ax.set_xlabel("Bar Number")
    ax.set_ylabel("Close Price")
    ax.set_title("Price Series Comparison (First 100 Bars)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Imbalance bar characteristics
    ax = axes[1, 1]
    if len(imbalance_bars) > 0:
        scatter = ax.scatter(
            imbalance_bars["tick_count"],
            imbalance_bars["imbalance"],
            c=imbalance_bars["expected_imbalance"],
            cmap="viridis",
            alpha=0.6,
        )
        ax.set_xlabel("Ticks per Bar")
        ax.set_ylabel("Imbalance")
        ax.set_title("Imbalance Bar Characteristics")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Expected Imbalance")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Run the bar sampling examples."""
    print("Information-Driven Bar Sampling Example")
    print("=" * 50)

    # Load data
    tick_data = load_spy_tick_data()
    if tick_data is None:
        tick_data = create_synthetic_tick_data()

    print(f"\nLoaded {len(tick_data)} ticks")
    print(
        f"Time range: {tick_data['timestamp'].min()} to {tick_data['timestamp'].max()}",
    )
    print(
        f"Price range: ${tick_data['price'].min():.2f} to ${tick_data['price'].max():.2f}",
    )

    # Sample different bar types
    bars = sample_bars(tick_data)

    # Create visualizations
    create_visualizations(bars)

    # Show statistics
    print_bar_statistics(bars)

    print("\n✅ Example completed successfully!")


def print_bar_statistics(bars: dict) -> None:
    """Print statistics for each bar type."""
    print("\n" + "=" * 50)
    print("Bar Statistics Summary")
    print("=" * 50)

    for name, bar_data in bars.items():
        if len(bar_data) > 0:
            returns = bar_data.select(pl.col("close").pct_change().alias("returns"))["returns"]
            returns_clean = returns.drop_nulls()

            print(f"\n{name.capitalize()} Bars:")
            print(f"  Number of bars: {len(bar_data)}")
            print(f"  Avg return: {returns_clean.mean():.4%}")
            print(f"  Std return: {returns_clean.std():.4%}")
            print(f"  Return skewness: {returns_clean.skew():.3f}")
            print(f"  Return kurtosis: {returns_clean.kurtosis():.3f}")


if __name__ == "__main__":
    main()
