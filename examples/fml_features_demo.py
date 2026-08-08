"""Combine event-driven bars, fractional differencing, and barrier labels."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ml4t.engineer.bars import ImbalanceBarSampler, VolumeBarSampler
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.features.fdiff import ffdiff
from ml4t.engineer.labeling import triple_barrier_labels


def make_ticks(n_rows: int = 20_000) -> pl.DataFrame:
    """Create deterministic tick data for the complete financial-ML flow."""
    rng = np.random.default_rng(19)
    returns = rng.standard_t(df=5, size=n_rows) * 0.0004
    prices = 30_000.0 * np.exp(np.cumsum(returns))
    volumes = rng.integers(1, 100, n_rows)
    sides = np.where((np.arange(n_rows) % 20) < 11, 1, -1)
    start = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(seconds=i) for i in range(n_rows)],
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )


def main() -> None:
    """Run the supported finance-specific APIs in sequence."""
    ticks = make_ticks()
    volume_bars = VolumeBarSampler(volume_per_bar=5_000).sample(ticks)
    imbalance_bars = ImbalanceBarSampler(
        expected_ticks_per_bar=100,
        alpha=0.2,
        initial_p_buy=0.55,
    ).sample(ticks)
    assert len(volume_bars) > 100
    assert len(imbalance_bars) > 0

    features = volume_bars.with_columns(
        close_ffd=ffdiff("close", d=0.5, threshold=1e-3),
        volatility=pl.col("close").pct_change().rolling_std(20),
    ).with_columns(
        upper_distance=(2.0 * pl.col("volatility")).fill_null(0.02),
        lower_distance=pl.col("volatility").fill_null(0.01),
    )
    labeled = triple_barrier_labels(
        features,
        LabelingConfig.triple_barrier(
            upper_barrier="upper_distance",
            lower_barrier="lower_distance",
            max_holding_period=20,
        ),
        price_col="close",
        high_col="high",
        low_col="low",
        timestamp_col="timestamp",
    )

    assert "close_ffd" in labeled.columns
    assert set(labeled["label"].unique()) <= {-1, 0, 1}
    print(f"ticks={len(ticks)}")
    print(f"volume_bars={len(volume_bars)} imbalance_bars={len(imbalance_bars)}")
    print(f"labels={labeled['label'].value_counts().sort('label').to_dicts()}")
    print("fml_features_demo=pass")


if __name__ == "__main__":
    main()
