"""Tests for rolling percentile-based binary labels."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.config import DataContractConfig, LabelingConfig
from ml4t.engineer.labeling.percentile_labels import (
    compute_label_statistics,
    rolling_percentile_binary_labels,
    rolling_percentile_multi_labels,
)


@pytest.fixture
def sample_price_data() -> pl.DataFrame:
    """Generate sample price data for testing."""
    np.random.seed(42)
    n = 500
    # Create trending price data
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    return pl.DataFrame(
        {
            "close": prices,
            "session_date": [f"2024-01-{(i // 100) + 1:02d}" for i in range(n)],
        }
    )


class TestRollingPercentileBinaryLabels:
    """Tests for rolling_percentile_binary_labels function."""

    def test_basic_long_labels(self, sample_price_data: pl.DataFrame) -> None:
        """Test basic long label generation."""
        result = rolling_percentile_binary_labels(
            sample_price_data,
            horizon=10,
            percentile=90,
            direction="long",
            lookback_window=100,
        )

        # Check that expected columns are added
        assert "forward_return_10" in result.columns
        assert "threshold_p90_h10" in result.columns
        assert "label_long_p90_h10" in result.columns

        # Labels should be 0 or 1 (or null)
        labels = result["label_long_p90_h10"].drop_nulls()
        assert all(label in [0, 1] for label in labels)

    def test_basic_short_labels(self, sample_price_data: pl.DataFrame) -> None:
        """Test basic short label generation."""
        result = rolling_percentile_binary_labels(
            sample_price_data,
            horizon=10,
            percentile=10,
            direction="short",
            lookback_window=100,
        )

        # Check that expected columns are added
        assert "forward_return_10" in result.columns
        assert "threshold_p10_h10" in result.columns
        assert "label_short_p10_h10" in result.columns

        # Labels should be 0 or 1 (or null)
        labels = result["label_short_p10_h10"].drop_nulls()
        assert all(label in [0, 1] for label in labels)

    def test_session_aware_returns(self, sample_price_data: pl.DataFrame) -> None:
        """Test session-aware forward return computation."""
        result = rolling_percentile_binary_labels(
            sample_price_data,
            horizon=10,
            percentile=90,
            direction="long",
            lookback_window=100,
            session_col="session_date",
        )

        # Should have forward returns column
        assert "forward_return_10" in result.columns

    def test_invalid_direction_raises(self, sample_price_data: pl.DataFrame) -> None:
        """Test that invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="Invalid direction"):
            rolling_percentile_binary_labels(
                sample_price_data,
                horizon=10,
                percentile=90,
                direction="invalid",  # type: ignore
                lookback_window=100,
            )

    def test_custom_min_samples(self, sample_price_data: pl.DataFrame) -> None:
        """Test custom min_samples parameter."""
        result = rolling_percentile_binary_labels(
            sample_price_data,
            horizon=10,
            percentile=90,
            direction="long",
            lookback_window=100,
            min_samples=50,
        )

        # Should complete without error
        assert "label_long_p90_h10" in result.columns

    def test_high_percentile_long_labels(self, sample_price_data: pl.DataFrame) -> None:
        """Test high percentile creates sparse long labels."""
        result = rolling_percentile_binary_labels(
            sample_price_data,
            horizon=10,
            percentile=95,
            direction="long",
            lookback_window=100,
        )

        labels = result["label_long_p95_h10"].drop_nulls()
        # With 95th percentile, should have ~5% positive labels
        positive_rate = labels.mean()
        # Allow some tolerance due to rolling window effects
        assert 0 < positive_rate < 0.30

    def test_low_percentile_short_labels(self, sample_price_data: pl.DataFrame) -> None:
        """Test low percentile creates sparse short labels."""
        result = rolling_percentile_binary_labels(
            sample_price_data,
            horizon=10,
            percentile=5,
            direction="short",
            lookback_window=100,
        )

        labels = result["label_short_p5_h10"].drop_nulls()
        # With 5th percentile, should have ~5% positive labels
        positive_rate = labels.mean()
        assert 0 < positive_rate < 0.30

    def test_threshold_uses_only_historical_forward_returns(self) -> None:
        """Rolling threshold must exclude current row forward return to avoid leakage."""
        data = pl.DataFrame({"close": [100.0, 101.0, 103.0, 102.0, 104.0]})

        result = rolling_percentile_binary_labels(
            data,
            horizon=1,
            percentile=100,
            direction="long",
            lookback_window=1,
            min_samples=1,
        )

        forward = result["forward_return_1"].to_list()
        threshold = result["threshold_p100_h1"].to_list()

        assert threshold[0] is None
        assert threshold[1] == pytest.approx(forward[0])
        assert threshold[2] == pytest.approx(forward[1])
        assert threshold[3] == pytest.approx(forward[2])

        # With lookback=1 and p100, leakage would make all non-null labels == 1.
        # Historical-only threshold correctly yields at least one 0.
        labels = result["label_long_p100_h1"].drop_nulls().to_list()
        assert 0 in labels

    def test_bar_threshold_waits_for_outcome_realization(self) -> None:
        """Changing an unresolved future outcome must not alter an earlier threshold."""
        base = pl.DataFrame({"close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 105.0]})
        changed = (
            base.with_row_index()
            .with_columns(
                pl.when(pl.col("index") == 5).then(110.0).otherwise(pl.col("close")).alias("close")
            )
            .drop("index")
        )

        kwargs = {
            "horizon": 3,
            "percentile": 50,
            "direction": "long",
            "lookback_window": 1,
            "min_samples": 1,
        }
        base_result = rolling_percentile_binary_labels(base, **kwargs)
        changed_result = rolling_percentile_binary_labels(changed, **kwargs)

        assert base_result["forward_return_3"][3] == changed_result["forward_return_3"][3]
        assert base_result["threshold_p50_h3"][3] == changed_result["threshold_p50_h3"][3]
        assert base_result["label_long_p50_h3"][3] == changed_result["label_long_p50_h3"][3]

    def test_time_threshold_waits_for_irregular_outcome_realization(self) -> None:
        """Time horizons must align history to the matched outcome timestamp."""
        from datetime import datetime, timedelta

        start = datetime(2024, 1, 1, 9, 30)
        timestamps = [start + timedelta(minutes=offset) for offset in (0, 1, 3, 4, 6)]
        base = pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0, 100.0, 100.0, 100.0, 100.0],
            }
        )
        changed = (
            base.with_row_index()
            .with_columns(
                pl.when(pl.col("index") == 4).then(110.0).otherwise(pl.col("close")).alias("close")
            )
            .drop("index")
        )

        kwargs = {
            "horizon": "3m",
            "percentile": 100,
            "direction": "long",
            "lookback_window": "10m",
            "min_samples": 1,
        }
        base_result = rolling_percentile_binary_labels(base, **kwargs)
        changed_result = rolling_percentile_binary_labels(changed, **kwargs)

        target_index = 3
        assert (
            base_result["threshold_p100_h3m"][target_index]
            == changed_result["threshold_p100_h3m"][target_index]
        )

    def test_time_threshold_includes_all_outcomes_realized_at_same_time(self) -> None:
        """Rolling thresholds must retain duplicate realization timestamps."""
        from datetime import datetime, timedelta

        start = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "timestamp": [start, start + timedelta(minutes=1), start + timedelta(minutes=3)],
                "close": [100.0, 200.0, 300.0],
            }
        )

        low = rolling_percentile_binary_labels(
            data,
            horizon="2m",
            percentile=0,
            lookback_window="10m",
            min_samples=1,
        )
        high = rolling_percentile_binary_labels(
            data,
            horizon="2m",
            percentile=100,
            lookback_window="10m",
            min_samples=1,
        )

        assert low["threshold_p0_h2m"][2] == pytest.approx(0.5)
        assert high["threshold_p100_h2m"][2] == pytest.approx(2.0)

    def test_bar_threshold_isolated_by_panel_and_session(self) -> None:
        """Outcome availability must respect both asset and session boundaries."""
        data = pl.DataFrame(
            {
                "symbol": ["A"] * 6 + ["B"] * 6,
                "session": ["s1"] * 3 + ["s2"] * 3 + ["s1"] * 3 + ["s2"] * 3,
                "close": [
                    100.0,
                    101.0,
                    102.0,
                    200.0,
                    202.0,
                    204.0,
                    1000.0,
                    990.0,
                    980.0,
                    500.0,
                    495.0,
                    490.0,
                ],
            }
        )

        result = rolling_percentile_binary_labels(
            data,
            horizon=2,
            percentile=50,
            lookback_window=10,
            min_samples=1,
            session_col="session",
            group_col="symbol",
        )

        a = result.filter(pl.col("symbol") == "A")
        b = result.filter(pl.col("symbol") == "B")
        assert a["threshold_p50_h2"][2] == pytest.approx(0.02)
        assert b["threshold_p50_h2"][2] == pytest.approx(-0.02)

    def test_time_based_threshold_uses_only_historical_forward_returns(self) -> None:
        """Time-based rolling threshold should also use shifted forward returns."""
        from datetime import datetime

        ts = pl.datetime_range(
            start=datetime(2024, 1, 1, 9, 30, 0),
            end=datetime(2024, 1, 1, 9, 34, 0),
            interval="1m",
            eager=True,
        )
        data = pl.DataFrame(
            {
                "timestamp": ts,
                "close": [100.0, 101.0, 103.0, 102.0, 104.0],
            }
        )

        result = rolling_percentile_binary_labels(
            data,
            horizon="1m",
            percentile=100,
            direction="long",
            lookback_window="2m",
            min_samples=1,
        )

        forward = result["forward_return_1m"].to_list()
        threshold = result["threshold_p100_h1m"].to_list()

        assert threshold[0] is None
        assert threshold[1] == pytest.approx(forward[0])

    def test_uses_config_column_contract_for_panel_data(self) -> None:
        """Config-driven price/timestamp/group mapping should be honored."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "ts": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "ticker": ["A", "B", "A", "B"],
                "px": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        config = LabelingConfig(
            method="percentile",
            price_col="px",
            timestamp_col="ts",
            group_col="ticker",
        )

        result = rolling_percentile_binary_labels(
            data,
            horizon=1,
            percentile=50,
            direction="long",
            lookback_window=1,
            min_samples=1,
            config=config,
        )

        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)
        assert a0["forward_return_1"] == pytest.approx(0.01)
        assert b0["forward_return_1"] == pytest.approx(-0.001)

    def test_uses_shared_contract_column_mapping_for_panel_data(self) -> None:
        """DataContract-driven price/timestamp/group mapping should be honored."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "ts": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "ticker": ["A", "B", "A", "B"],
                "px": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        contract = DataContractConfig(timestamp_col="ts", symbol_col="ticker", price_col="px")

        result = rolling_percentile_binary_labels(
            data,
            horizon=1,
            percentile=50,
            direction="long",
            lookback_window=1,
            min_samples=1,
            contract=contract,
        )

        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)
        assert a0["forward_return_1"] == pytest.approx(0.01)
        assert b0["forward_return_1"] == pytest.approx(-0.001)

    def test_uses_nested_data_contract_from_config_for_panel_data(self) -> None:
        """LabelingConfig.data_contract should drive percentile mapping."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "ts": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "ticker": ["A", "B", "A", "B"],
                "px": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        contract = DataContractConfig(timestamp_col="ts", symbol_col="ticker", price_col="px")
        config = LabelingConfig(method="percentile", data_contract=contract)

        result = rolling_percentile_binary_labels(
            data,
            horizon=1,
            percentile=50,
            direction="long",
            lookback_window=1,
            min_samples=1,
            config=config,
        )

        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)
        assert a0["forward_return_1"] == pytest.approx(0.01)
        assert b0["forward_return_1"] == pytest.approx(-0.001)


class TestRollingPercentileMultiLabels:
    """Tests for rolling_percentile_multi_labels function."""

    def test_multiple_horizons(self, sample_price_data: pl.DataFrame) -> None:
        """Test generating labels for multiple horizons."""
        result = rolling_percentile_multi_labels(
            sample_price_data,
            horizons=[10, 20],
            percentiles=[90],
            direction="long",
            lookback_window=100,
        )

        # Should have columns for both horizons
        assert "label_long_p90_h10" in result.columns
        assert "label_long_p90_h20" in result.columns

    def test_multiple_percentiles(self, sample_price_data: pl.DataFrame) -> None:
        """Test generating labels for multiple percentiles."""
        result = rolling_percentile_multi_labels(
            sample_price_data,
            horizons=[10],
            percentiles=[90, 95],
            direction="long",
            lookback_window=100,
        )

        # Should have columns for both percentiles
        assert "label_long_p90_h10" in result.columns
        assert "label_long_p95_h10" in result.columns

    def test_multiple_horizons_and_percentiles(self, sample_price_data: pl.DataFrame) -> None:
        """Test generating labels for multiple horizons and percentiles."""
        result = rolling_percentile_multi_labels(
            sample_price_data,
            horizons=[10, 20],
            percentiles=[90, 95],
            direction="long",
            lookback_window=100,
        )

        # Should have 4 label columns (2 horizons × 2 percentiles)
        label_cols = [c for c in result.columns if c.startswith("label_")]
        assert len(label_cols) == 4

    def test_distinct_float_percentiles_have_distinct_columns(
        self, sample_price_data: pl.DataFrame
    ) -> None:
        """Accepted float configurations must retain separate identities."""
        result = rolling_percentile_multi_labels(
            sample_price_data,
            horizons=[2],
            percentiles=[50.1, 50.9],
            lookback_window=10,
        )

        assert "threshold_p50p1_h2" in result.columns
        assert "threshold_p50p9_h2" in result.columns
        assert "label_long_p50p1_h2" in result.columns
        assert "label_long_p50p9_h2" in result.columns

    @pytest.mark.parametrize(
        ("horizons", "percentiles", "message"),
        [
            ([], [50], "horizons"),
            ([2], [], "percentiles"),
            ([2], [50, 50.0], "duplicate percentile"),
            (["1h", "1H"], [50], "duplicate horizon"),
        ],
    )
    def test_rejects_empty_or_duplicate_requests(
        self,
        sample_price_data: pl.DataFrame,
        horizons,
        percentiles,
        message,
    ) -> None:
        """Multi-label validation must fail atomically before computation."""
        with pytest.raises(ValueError, match=message):
            rolling_percentile_multi_labels(
                sample_price_data,
                horizons=horizons,
                percentiles=percentiles,
                lookback_window=10,
            )

    @pytest.mark.parametrize("percentile", [-0.1, 100.1, np.nan, np.inf, -np.inf])
    def test_rejects_invalid_percentiles(
        self, sample_price_data: pl.DataFrame, percentile: float
    ) -> None:
        """Percentiles must be finite values in the closed unit-percent range."""
        with pytest.raises(ValueError, match="percentile"):
            rolling_percentile_binary_labels(
                sample_price_data,
                horizon=2,
                percentile=percentile,
                lookback_window=10,
            )

    def test_uses_shared_contract_column_mapping(self) -> None:
        """Multi-label API should pass DataContractConfig through to binary calls."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "ts": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "ticker": ["A", "B", "A", "B"],
                "px": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        contract = DataContractConfig(timestamp_col="ts", symbol_col="ticker", price_col="px")

        result = rolling_percentile_multi_labels(
            data,
            horizons=[1],
            percentiles=[50],
            direction="long",
            lookback_window=1,
            contract=contract,
        )

        assert "forward_return_1" in result.columns


class TestComputeLabelStatistics:
    """Tests for compute_label_statistics function."""

    def test_basic_statistics(self) -> None:
        """Test basic label statistics computation."""
        df = pl.DataFrame(
            {
                "label": [1, 0, 1, 0, 0, 1, None, None],
            }
        )

        stats = compute_label_statistics(df, "label")

        assert stats["total_bars"] == 8
        assert stats["positive_labels"] == 3
        assert stats["negative_labels"] == 3
        assert stats["null_labels"] == 2
        assert stats["positive_rate"] == 50.0
        assert stats["null_rate"] == 25.0

    def test_all_nulls(self) -> None:
        """Test statistics with all null labels."""
        df = pl.DataFrame(
            {
                "label": [None, None, None],
            }
        )

        stats = compute_label_statistics(df, "label")

        assert stats["total_bars"] == 3
        assert stats["positive_labels"] == 0
        assert stats["negative_labels"] == 0
        assert stats["null_labels"] == 3
        assert stats["positive_rate"] == 0.0
        assert stats["null_rate"] == 100.0

    def test_no_nulls(self) -> None:
        """Test statistics with no nulls."""
        df = pl.DataFrame(
            {
                "label": [1, 0, 1, 1, 0],
            }
        )

        stats = compute_label_statistics(df, "label")

        assert stats["total_bars"] == 5
        assert stats["null_labels"] == 0
        assert stats["null_rate"] == 0.0
        assert stats["positive_rate"] == 60.0
