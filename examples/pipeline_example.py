"""Compose the current feature API with train-only preprocessing."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ml4t.engineer import compute_features
from ml4t.engineer.preprocessing import StandardScaler


def make_market_data(n_rows: int = 200) -> pl.DataFrame:
    """Create a deterministic daily OHLCV frame."""
    rng = np.random.default_rng(11)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n_rows)))
    open_ = np.r_[close[0], close[:-1]]
    start = datetime(2023, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=i) for i in range(n_rows)],
            "open": open_,
            "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995,
            "close": close,
            "volume": rng.integers(100_000, 500_000, n_rows),
        }
    )


def main() -> None:
    """Compute configured features and fit scaling on the training partition."""
    data = make_market_data()
    engineered = compute_features(
        data,
        [
            {"name": "sma", "params": {"period": 20}, "output": "sma_20"},
            {"name": "ema", "params": {"period": 12}, "output": "ema_12"},
            {"name": "rsi", "params": {"period": 14}, "output": "rsi_14"},
            {"name": "atr", "params": {"period": 14}, "output": "atr_14"},
        ],
    )
    feature_columns = ["sma_20", "ema_12", "rsi_14", "atr_14"]
    usable = engineered.drop_nulls(feature_columns)
    split = int(len(usable) * 0.8)
    train = usable[:split].select(feature_columns)
    test = usable[split:].select(feature_columns)

    scaler = StandardScaler(columns=feature_columns)
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    assert train_scaled.shape == train.shape
    assert test_scaled.shape == test.shape
    print(f"train_rows={len(train_scaled)} test_rows={len(test_scaled)}")
    print(f"fitted_columns={scaler.fitted_columns}")
    print("pipeline_example=pass")


if __name__ == "__main__":
    main()
