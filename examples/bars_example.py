"""Run the public information-driven bar samplers on deterministic tick data."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ml4t.engineer.bars import (
    DollarBarSampler,
    ImbalanceBarSampler,
    TickBarSampler,
    VolumeBarSampler,
)


def make_ticks(n_rows: int = 5_000) -> pl.DataFrame:
    """Create valid tick data with persistent but changing order-flow imbalance."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.0003, n_rows)
    prices = 100.0 * np.exp(np.cumsum(returns))
    volumes = rng.integers(50, 250, n_rows)
    sides = np.where((np.arange(n_rows) % 10) < 6, 1, -1)
    start = datetime(2024, 1, 2, 9, 30)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(seconds=i) for i in range(n_rows)],
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )


def main() -> None:
    """Sample four bar types and print their basic contracts."""
    ticks = make_ticks()
    samplers = {
        "tick": TickBarSampler(ticks_per_bar=100),
        "volume": VolumeBarSampler(volume_per_bar=15_000),
        "dollar": DollarBarSampler(dollars_per_bar=1_500_000),
        "imbalance": ImbalanceBarSampler(
            expected_ticks_per_bar=100,
            alpha=0.2,
            initial_p_buy=0.6,
        ),
    }

    print(f"input_ticks={len(ticks)}")
    for name, sampler in samplers.items():
        bars = sampler.sample(ticks)
        required = {"timestamp", "open", "high", "low", "close", "volume", "tick_count"}
        assert required <= set(bars.columns)
        assert len(bars) > 0
        print(
            f"{name}: bars={len(bars)} "
            f"mean_ticks={bars['tick_count'].mean():.1f} "
            f"last_close={bars['close'][-1]:.4f}"
        )

    print("bars_example=pass")


if __name__ == "__main__":
    main()
