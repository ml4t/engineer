"""Integration tests: OHLCV → features → labeling → preprocessing → MLDatasetBuilder.

These tests verify the full pipeline a book reader would use, end-to-end.
No mocking — real computations with synthetic but realistic data.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.config import LabelingConfig, PreprocessingConfig
from ml4t.engineer.dataset import MLDatasetBuilder
from ml4t.engineer.labeling import (
    atr_triple_barrier_labels,
    fixed_time_horizon_labels,
    triple_barrier_labels,
)
from ml4t.engineer.preprocessing import RobustScaler, StandardScaler


@pytest.fixture
def ohlcv_data():
    """Realistic synthetic OHLCV data for a single asset (200 bars)."""
    np.random.seed(42)
    n = 200
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]

    # Generate correlated OHLCV with realistic structure
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.abs(np.random.randn(n) * 1000 + 5000)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_.tolist(),
            "high": high.tolist(),
            "low": low.tolist(),
            "close": close.tolist(),
            "volume": volume.tolist(),
        }
    )


@pytest.fixture
def panel_data():
    """Multi-asset OHLCV data for panel testing (2 assets, 100 bars each)."""
    np.random.seed(123)
    n = 100

    frames = []
    for symbol in ["AAPL", "MSFT"]:
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
        high = close + np.abs(np.random.randn(n) * 0.2)
        low = close - np.abs(np.random.randn(n) * 0.2)
        open_ = close + np.random.randn(n) * 0.05
        volume = np.abs(np.random.randn(n) * 500 + 3000)

        frames.append(
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "symbol": [symbol] * n,
                    "open": open_.tolist(),
                    "high": high.tolist(),
                    "low": low.tolist(),
                    "close": close.tolist(),
                    "volume": volume.tolist(),
                }
            )
        )
    return pl.concat(frames)


class TestFixedHorizonPipeline:
    """Test: OHLCV → fixed_time_horizon_labels → scaler → numpy."""

    def test_bar_based_labels_to_numpy(self, ohlcv_data):
        # Label
        labeled = fixed_time_horizon_labels(
            ohlcv_data,
            horizon=5,
            method="returns",
            price_col="close",
            group_col=[],
        )
        assert "label_return_5p" in labeled.columns
        assert labeled.shape[0] == ohlcv_data.shape[0]

        # The last 5 rows should be null (no future data)
        label_col = labeled["label_return_5p"]
        assert label_col[-1] is None

    def test_time_based_labels(self, ohlcv_data):
        labeled = fixed_time_horizon_labels(
            ohlcv_data,
            horizon="2h",
            method="returns",
            price_col="close",
            timestamp_col="timestamp",
            group_col=[],
        )
        assert any("label_return" in c for c in labeled.columns)

    def test_binary_labels(self, ohlcv_data):
        labeled = fixed_time_horizon_labels(
            ohlcv_data,
            horizon=1,
            method="binary",
            price_col="close",
            group_col=[],
        )
        label_col = "label_direction_1p"
        assert label_col in labeled.columns
        # Non-null values should be in {-1, 0, 1}
        non_null = labeled.filter(pl.col(label_col).is_not_null())[label_col]
        assert set(non_null.to_list()).issubset({-1, 0, 1})

    def test_panel_data_labels(self, panel_data):
        labeled = fixed_time_horizon_labels(
            panel_data,
            horizon=5,
            method="returns",
            price_col="close",
            group_col="symbol",
            timestamp_col="timestamp",
        )
        assert labeled.shape[0] == panel_data.shape[0]
        # Both assets should have labels
        per_asset = labeled.group_by("symbol").agg(
            pl.col("label_return_5p").is_not_null().sum().alias("non_null_count")
        )
        for row in per_asset.iter_rows(named=True):
            assert row["non_null_count"] > 0


class TestTripleBarrierPipeline:
    """Test: OHLCV → triple_barrier_labels → preprocessing → numpy."""

    def test_basic_labeling(self, ohlcv_data):
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=20,
        )
        labeled = triple_barrier_labels(
            ohlcv_data,
            config=config,
            price_col="close",
            timestamp_col="timestamp",
            group_col=[],
        )
        assert "label" in labeled.columns
        assert "label_return" in labeled.columns
        assert "barrier_hit" in labeled.columns

        # Labels should be in {-1, 0, 1}
        non_null = labeled.filter(pl.col("label").is_not_null())["label"]
        assert set(non_null.to_list()).issubset({-1, 0, 1})

    def test_atr_labeling(self, ohlcv_data):
        labeled = atr_triple_barrier_labels(
            ohlcv_data,
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
            atr_period=14,
            max_holding_bars=30,
            price_col="close",
            timestamp_col="timestamp",
            group_col=[],
        )
        assert "label" in labeled.columns
        assert "atr" in labeled.columns
        assert "upper_barrier_distance" in labeled.columns
        assert "lower_barrier_distance" in labeled.columns

    def test_panel_triple_barrier(self, panel_data):
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=15,
        )
        labeled = triple_barrier_labels(
            panel_data,
            config=config,
            price_col="close",
            timestamp_col="timestamp",
            group_col="symbol",
        )
        assert labeled.shape[0] == panel_data.shape[0]
        # Both assets should have labels
        per_asset = labeled.group_by("symbol").agg(
            pl.col("label").is_not_null().sum().alias("label_count")
        )
        for row in per_asset.iter_rows(named=True):
            assert row["label_count"] > 0


class TestPreprocessingPipeline:
    """Test: labeled data → scaler → numpy arrays."""

    def test_standard_scaler_fit_transform(self, ohlcv_data):
        scaler = StandardScaler(columns=["close", "volume"])
        scaled = scaler.fit_transform(ohlcv_data)

        # Scaled columns should have ~0 mean and ~1 std
        close_mean = scaled["close"].mean()
        close_std = scaled["close"].std()
        assert abs(close_mean) < 0.01
        assert abs(close_std - 1.0) < 0.01

        # Non-scaled columns should be unchanged
        assert scaled["timestamp"].to_list() == ohlcv_data["timestamp"].to_list()
        assert scaled["open"].to_list() == ohlcv_data["open"].to_list()

    def test_robust_scaler(self, ohlcv_data):
        scaler = RobustScaler(columns=["close", "volume"])
        scaled = scaler.fit_transform(ohlcv_data)
        assert scaled.shape == ohlcv_data.shape

    def test_scaler_from_config(self, ohlcv_data):
        config = PreprocessingConfig.robust(quantile_range=(10.0, 90.0))
        scaler = config.create_scaler()
        assert scaler is not None
        scaled = scaler.fit_transform(ohlcv_data)
        assert scaled.shape == ohlcv_data.shape

    def test_scaler_preserves_column_order(self, ohlcv_data):
        scaler = StandardScaler(columns=["volume", "close"])
        scaled = scaler.fit_transform(ohlcv_data)
        assert scaled.columns == ohlcv_data.columns


class TestMLDatasetBuilder:
    """Test: full pipeline → MLDatasetBuilder → numpy arrays."""

    def test_basic_dataset_build(self, ohlcv_data):
        # Add a simple feature and label
        data = ohlcv_data.with_columns(
            (pl.col("close").pct_change()).alias("returns"),
            (pl.col("close").shift(-1) > pl.col("close")).cast(pl.Int8).alias("label"),
        )
        clean = data.drop_nulls(subset=["returns", "label"])

        features = clean.select(["close", "volume", "returns"])
        labels = clean["label"]

        builder = MLDatasetBuilder(features=features, labels=labels)

        # train_test_split returns Polars DataFrames/Series
        X_train, X_test, y_train, y_test = builder.train_test_split(train_size=0.75)
        assert isinstance(X_train, pl.DataFrame)
        assert isinstance(y_train, pl.Series)
        assert X_train.shape[1] == 3  # 3 features
        assert len(X_train) + len(X_test) == len(clean)

    def test_to_numpy(self, ohlcv_data):
        """Test conversion to numpy arrays."""
        data = ohlcv_data.with_columns(
            (pl.col("close").pct_change()).alias("returns"),
            (pl.col("close").shift(-1) > pl.col("close")).cast(pl.Int8).alias("label"),
        )
        clean = data.drop_nulls(subset=["returns", "label"])

        features = clean.select(["close", "volume", "returns"])
        labels = clean["label"]

        builder = MLDatasetBuilder(features=features, labels=labels)
        X, y = builder.to_numpy()
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape == (len(clean), 3)

    def test_dataset_with_scaler(self, ohlcv_data):
        data = ohlcv_data.with_columns(
            (pl.col("close").pct_change()).alias("returns"),
            (pl.col("close").shift(-5) > pl.col("close")).cast(pl.Int8).alias("label"),
        )
        clean = data.drop_nulls(subset=["returns", "label"])

        features = clean.select(["close", "volume", "returns"])
        labels = clean["label"]

        builder = MLDatasetBuilder(features=features, labels=labels)
        builder.set_scaler(StandardScaler(columns=["close", "volume"]))

        X_train, X_test, y_train, y_test = builder.train_test_split(train_size=0.75)
        assert X_train.shape[1] == 3

    def test_dataset_dates(self, ohlcv_data):
        """Test that dates are preserved through the builder."""
        data = ohlcv_data.with_columns(
            (pl.col("close").pct_change()).alias("returns"),
            (pl.col("close").shift(-1) > pl.col("close")).cast(pl.Int8).alias("label"),
        )
        clean = data.drop_nulls(subset=["returns", "label"])

        features = clean.select(["close", "volume", "returns"])
        labels = clean["label"]
        dates = clean["timestamp"]

        builder = MLDatasetBuilder(features=features, labels=labels, dates=dates)
        assert builder.dates is not None
        assert len(builder.dates) == len(features)


class TestEndToEndPipeline:
    """Full end-to-end: OHLCV → label → preprocess → dataset → numpy."""

    def test_full_pipeline_fixed_horizon(self, ohlcv_data):
        """Complete pipeline a book reader would follow."""
        # Step 1: Label with fixed horizon
        labeled = fixed_time_horizon_labels(
            ohlcv_data,
            horizon=5,
            method="returns",
            price_col="close",
            group_col=[],
        )
        assert "label_return_5p" in labeled.columns

        # Step 2: Add features
        enriched = labeled.with_columns(
            (pl.col("close").pct_change()).alias("returns_1"),
            (pl.col("close").pct_change(5)).alias("returns_5"),
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("hl_range"),
        )

        # Step 3: Clean nulls and build dataset
        feature_cols = ["returns_1", "returns_5", "hl_range"]
        label_col = "label_return_5p"
        clean = enriched.drop_nulls(subset=feature_cols + [label_col])

        features = clean.select(feature_cols)
        labels = clean[label_col]

        builder = MLDatasetBuilder(features=features, labels=labels)
        X_train, X_test, y_train, y_test = builder.train_test_split(train_size=0.75)

        assert isinstance(X_train, pl.DataFrame)
        assert X_train.shape[1] == 3
        assert X_train.shape[0] > 0

        # Verify numpy conversion works
        X_np, y_np = builder.to_numpy()
        assert isinstance(X_np, np.ndarray)
        assert X_np.shape[1] == 3

    def test_full_pipeline_triple_barrier(self, ohlcv_data):
        """Triple barrier → features → scaler → dataset."""
        # Step 1: Triple barrier labeling
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=20,
        )
        labeled = triple_barrier_labels(
            ohlcv_data,
            config=config,
            price_col="close",
            timestamp_col="timestamp",
            group_col=[],
        )

        # Step 2: Add features
        enriched = labeled.with_columns(
            (pl.col("close").pct_change()).alias("returns"),
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_pct"),
            (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("volume_ratio"),
        )

        # Step 3: Clean and build dataset
        feature_cols = ["returns", "range_pct", "volume_ratio"]
        clean = enriched.drop_nulls(subset=feature_cols + ["label"])

        features = clean.select(feature_cols)
        labels = clean["label"]

        builder = MLDatasetBuilder(features=features, labels=labels)
        builder.set_scaler(StandardScaler(columns=feature_cols))

        X_train, X_test, y_train, y_test = builder.train_test_split(train_size=0.75)
        assert X_train.shape[1] == 3
        assert X_train.shape[0] > 0
        # Labels should be in {-1, 0, 1}
        assert set(y_train.to_numpy().astype(int)).issubset({-1, 0, 1})


class TestNaNValidation:
    """Test that NaN validation catches bad data early."""

    def test_triple_barrier_rejects_nan_prices(self, ohlcv_data):
        from ml4t.engineer.core.exceptions import DataValidationError

        bad_data = ohlcv_data.with_columns(
            pl.when(pl.col("close").is_first_distinct())
            .then(float("nan"))
            .otherwise(pl.col("close"))
            .alias("close")
        )
        config = LabelingConfig.triple_barrier(
            upper_barrier=0.02,
            lower_barrier=0.01,
            max_holding_period=20,
        )
        with pytest.raises(DataValidationError, match="null/NaN"):
            triple_barrier_labels(
                bad_data,
                config=config,
                price_col="close",
                timestamp_col="timestamp",
                group_col=[],
            )

    def test_fixed_horizon_rejects_nan_prices(self, ohlcv_data):
        from ml4t.engineer.core.exceptions import DataValidationError

        bad_data = ohlcv_data.with_columns(pl.lit(None, dtype=pl.Float64).alias("close"))
        with pytest.raises(DataValidationError, match="null/NaN"):
            fixed_time_horizon_labels(
                bad_data,
                horizon=5,
                price_col="close",
                group_col=[],
            )
