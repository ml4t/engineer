"""Run feature computation, labeling, and leakage-safe dataset preparation."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ml4t.engineer import compute_features, create_dataset_builder
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels


def make_market_data(n_rows: int = 1_000) -> pl.DataFrame:
    """Create deterministic daily OHLCV data."""
    rng = np.random.default_rng(42)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n_rows)))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.002, 0.01, n_rows)
    start = datetime(2020, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(n_rows)],
            "open": open_,
            "high": np.maximum(open_, close) * (1.0 + spread),
            "low": np.minimum(open_, close) * (1.0 - spread),
            "close": close,
            "volume": rng.integers(100_000, 2_000_000, n_rows),
        }
    )


def main() -> None:
    """Build a scaled chronological train/test dataset."""
    market = make_market_data()
    engineered = compute_features(
        market,
        [
            {"name": "rsi", "params": {"period": 14}, "output": "rsi_14"},
            {"name": "sma", "params": {"period": 20}, "output": "sma_20"},
            {"name": "ema", "params": {"period": 20}, "output": "ema_20"},
            {"name": "atr", "params": {"period": 14}, "output": "atr_14"},
            {
                "name": "bollinger_bands",
                "params": {"period": 20},
                "output": "bollinger_20",
            },
        ],
    )
    labels = triple_barrier_labels(
        engineered,
        LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=20,
        ),
        price_col="close",
        high_col="high",
        low_col="low",
        timestamp_col="timestamp",
    )

    feature_columns = [name for name in labels.columns if name not in market.columns]
    feature_columns = [
        name
        for name in feature_columns
        if name
        not in {
            "label",
            "label_time",
            "label_price",
            "label_return",
            "label_bars",
            "label_duration",
            "barrier_hit",
        }
    ]
    usable = labels.drop_nulls(feature_columns)
    builder = create_dataset_builder(
        features=usable.select(feature_columns),
        labels=usable["label"],
        dates=usable["timestamp"],
        scaler="robust",
    )
    X_train, X_test, y_train, y_test = builder.train_test_split(
        train_size=0.8,
        shuffle=False,
    )

    assert X_train.width == len(feature_columns)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert np.isfinite(X_train["rsi_14"].to_numpy()).all()
    print(f"features={feature_columns}")
    print(f"train={X_train.shape} test={X_test.shape}")
    print(f"train_labels={y_train.value_counts().sort('label').to_dicts()}")
    print("complete_workflow_example=pass")


if __name__ == "__main__":
    main()
