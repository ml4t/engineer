"""Exercise fixed, dynamic, and trailing triple-barrier labels."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels


def make_ohlc(n_rows: int = 500) -> pl.DataFrame:
    """Create valid deterministic OHLC data."""
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, n_rows)))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.001, 0.006, n_rows)
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    start = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=i) for i in range(n_rows)],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def apply_labels(data: pl.DataFrame, config: LabelingConfig) -> pl.DataFrame:
    """Apply a triple-barrier config with full OHLC execution."""
    return triple_barrier_labels(
        data,
        config,
        price_col="close",
        open_col="open",
        high_col="high",
        low_col="low",
        timestamp_col="timestamp",
    )


def main() -> None:
    """Run three supported labeling configurations."""
    data = make_ohlc().with_columns(volatility=pl.col("close").pct_change().rolling_std(20))
    data = data.with_columns(
        upper_distance=(2.0 * pl.col("volatility")).fill_null(0.01),
        lower_distance=pl.col("volatility").fill_null(0.005),
    )

    configs = {
        "fixed": LabelingConfig.triple_barrier(
            upper_barrier=0.01,
            lower_barrier=0.005,
            max_holding_period=30,
        ),
        "dynamic": LabelingConfig.triple_barrier(
            upper_barrier="upper_distance",
            lower_barrier="lower_distance",
            max_holding_period=30,
        ),
        "trailing": LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=60,
            trailing_stop=0.005,
        ),
    }

    for name, config in configs.items():
        result = apply_labels(data, config)
        assert set(result["label"].unique()) <= {-1, 0, 1}
        assert (result["label_bars"] >= 0).all()
        counts = result.group_by("barrier_hit").len().sort("barrier_hit")
        print(f"{name}: {counts.to_dicts()}")

    print("labeling_example=pass")


if __name__ == "__main__":
    main()
