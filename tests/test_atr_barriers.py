"""
Tests for ATR-adjusted triple barrier labeling.

Focuses on practical usage and integration testing.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.config import DataContractConfig, LabelingConfig
from ml4t.engineer.core.exceptions import DataValidationError
from ml4t.engineer.labeling.atr_barriers import atr_triple_barrier_labels


@pytest.fixture
def sample_ohlc():
    """Sample OHLC data with realistic price movements."""
    np.random.seed(42)
    n = 100
    base_price = 100.0

    # Generate realistic price series
    returns = np.random.normal(0.0001, 0.01, n)
    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_price = np.roll(close, 1)
    open_price[0] = base_price

    timestamps = pl.datetime_range(
        eager=True,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        time_zone="UTC",
    )

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        }
    )


class TestBasicFunctionality:
    """Test basic ATR barrier labeling functionality."""

    def test_basic_long_labels(self, sample_ohlc):
        """Test basic long position labeling."""
        result = atr_triple_barrier_labels(
            sample_ohlc,
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
            atr_period=14,
            max_holding_bars=20,
            side=1,
        )

        # Check output structure
        assert "atr" in result.columns
        assert "upper_barrier_distance" in result.columns
        assert "lower_barrier_distance" in result.columns
        assert "label" in result.columns
        assert "label_time" in result.columns

        # Check label values
        labels = result["label"].drop_nulls()
        assert set(labels.unique()) <= {-1, 0, 1}

        # Check ATR is positive where not null
        atr_values = result["atr"].drop_nulls()
        if len(atr_values) > 0:
            assert (atr_values > 0).all()

    def test_basic_short_labels(self, sample_ohlc):
        """Test basic short position labeling."""
        result = atr_triple_barrier_labels(
            sample_ohlc,
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
            atr_period=14,
            max_holding_bars=20,
            side=-1,  # Short
        )

        assert "label" in result.columns
        labels = result["label"].drop_nulls()
        assert set(labels.unique()) <= {-1, 0, 1}

    def test_barrier_distances_are_positive(self, sample_ohlc):
        """Test that barrier distances are positive."""
        result = atr_triple_barrier_labels(
            sample_ohlc,
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
            atr_period=14,
        )

        # Barrier distances should be positive
        upper = result["upper_barrier_distance"].drop_nulls()
        lower = result["lower_barrier_distance"].drop_nulls()

        if len(upper) > 0:
            assert (upper > 0).all()
        if len(lower) > 0:
            assert (lower > 0).all()

    def test_multiple_ratios(self, sample_ohlc):
        """Test different TP/SL ratios."""
        ratios = [(2.0, 1.0), (3.0, 1.0), (1.5, 1.0), (2.0, 2.0)]

        for tp_mult, sl_mult in ratios:
            result = atr_triple_barrier_labels(
                sample_ohlc,
                atr_tp_multiple=tp_mult,
                atr_sl_multiple=sl_mult,
                max_holding_bars=20,
            )

            # Should complete without error
            assert len(result) == len(sample_ohlc)
            assert "label" in result.columns


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_ohlc_columns(self):
        """Should raise error if OHLC columns are missing."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1)],
                "close": [100.0],
            }
        )

        with pytest.raises(DataValidationError, match="Missing columns"):
            atr_triple_barrier_labels(df)

    def test_missing_timestamp_column(self):
        """Should raise error if timestamp column is missing."""
        df = pl.DataFrame(
            {
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
            }
        )

        with pytest.raises(DataValidationError, match="Timestamp column"):
            atr_triple_barrier_labels(df)

    def test_short_data_series(self):
        """Handle data shorter than ATR period."""
        timestamps = pl.datetime_range(
            eager=True,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
            interval="1d",
            time_zone="UTC",
        )

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )

        result = atr_triple_barrier_labels(
            df,
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
            atr_period=14,  # Longer than data
        )

        # Should complete without error
        assert len(result) == 5


class TestDynamicParameters:
    """Test dynamic side and holding period support."""

    def test_dynamic_side_column(self, sample_ohlc):
        """Test dynamic side from column."""
        df = sample_ohlc.with_columns(
            pl.Series("side_signal", [1 if i % 2 == 0 else -1 for i in range(len(sample_ohlc))]),
        )

        result = atr_triple_barrier_labels(
            df,
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
            side="side_signal",
        )

        assert len(result) == len(df)
        assert "label" in result.columns

    def test_dynamic_holding_period(self, sample_ohlc):
        """Test dynamic holding period from column."""
        df = sample_ohlc.with_columns(
            pl.Series("holding_bars", [10 + (i % 20) for i in range(len(sample_ohlc))]),
        )

        result = atr_triple_barrier_labels(
            df,
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
            max_holding_bars="holding_bars",
        )

        assert len(result) == len(df)
        assert "label" in result.columns


class TestIntegration:
    """Test integration with existing labeling API."""

    def test_produces_expected_columns(self, sample_ohlc):
        """ATR barriers should produce expected output columns."""
        result = atr_triple_barrier_labels(
            sample_ohlc,
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
        )

        # Check essential columns exist
        essential_cols = ["label", "label_time", "label_return", "label_bars", "label_duration"]
        for col in essential_cols:
            assert col in result.columns

    def test_trailing_stop_support(self, sample_ohlc):
        """Test trailing stop works with ATR barriers."""
        result = atr_triple_barrier_labels(
            sample_ohlc,
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
            trailing_stop=True,
        )

        assert len(result) == len(sample_ohlc)
        assert "label" in result.columns

    def test_config_preserves_numeric_trailing_stop(self, sample_ohlc, monkeypatch):
        """Numeric trailing_stop in LabelingConfig should not be coerced to bool."""
        captured: dict[str, float | bool | str] = {}

        def _stub_triple_barrier(data, config, **_kwargs):  # type: ignore[no-untyped-def]
            captured["trailing_stop"] = config.trailing_stop
            return data.with_columns(label=pl.lit(0, dtype=pl.Int32))

        monkeypatch.setattr(
            "ml4t.engineer.labeling.atr_barriers.triple_barrier_labels",
            _stub_triple_barrier,
        )

        config = LabelingConfig.atr_barrier(
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
            atr_period=14,
            max_holding_period=5,
            trailing_stop=0.025,
        )
        _ = atr_triple_barrier_labels(sample_ohlc, config=config)

        assert captured["trailing_stop"] == pytest.approx(0.025)

    def test_no_max_holding_period(self, sample_ohlc):
        """Test with no maximum holding period."""
        result = atr_triple_barrier_labels(
            sample_ohlc,
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
            max_holding_bars=None,
        )

        assert len(result) == len(sample_ohlc)
        assert "label" in result.columns

    def test_uses_config_column_contract(self):
        """Column mapping should come from LabelingConfig when omitted in call."""
        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "ts": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "ticker": ["A", "B", "A", "B"],
                "open": [100.0, 1000.0, 100.0, 1000.0],
                "high": [100.1, 1000.1, 101.1, 999.1],
                "low": [99.9, 999.9, 100.9, 998.9],
                "close": [100.0, 1000.0, 101.0, 999.0],
                "px": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        config = LabelingConfig.atr_barrier(
            atr_tp_multiple=50.0,
            atr_sl_multiple=50.0,
            atr_period=1,
            max_holding_period=1,
            price_col="px",
            timestamp_col="ts",
            group_col="ticker",
        )

        result = atr_triple_barrier_labels(data, config=config)
        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)
        assert a0["label_price"] == pytest.approx(101.0)
        assert b0["label_price"] == pytest.approx(999.0)

    def test_uses_shared_contract_column_mapping(self):
        """Column mapping should come from DataContractConfig when config is omitted."""
        base = datetime(2024, 1, 1, 9, 30)
        data = pl.DataFrame(
            {
                "ts": [base, base, base + timedelta(minutes=1), base + timedelta(minutes=1)],
                "ticker": ["A", "B", "A", "B"],
                "open": [100.0, 1000.0, 100.0, 1000.0],
                "high": [100.1, 1000.1, 101.1, 999.1],
                "low": [99.9, 999.9, 100.9, 998.9],
                "close": [100.0, 1000.0, 101.0, 999.0],
                "px": [100.0, 1000.0, 101.0, 999.0],
            }
        )
        contract = DataContractConfig(timestamp_col="ts", symbol_col="ticker", price_col="px")

        result = atr_triple_barrier_labels(
            data,
            atr_tp_multiple=50.0,
            atr_sl_multiple=50.0,
            atr_period=1,
            max_holding_bars=1,
            contract=contract,
        )
        a0 = result.filter((pl.col("ticker") == "A") & (pl.col("ts") == base)).row(0, named=True)
        b0 = result.filter((pl.col("ticker") == "B") & (pl.col("ts") == base)).row(0, named=True)
        assert a0["label_price"] == pytest.approx(101.0)
        assert b0["label_price"] == pytest.approx(999.0)


class TestDocumentationExamples:
    """Verify documentation examples work correctly."""

    def test_basic_example(self):
        """Test basic example from docstring."""
        timestamps = pl.datetime_range(
            eager=True,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 31),
            interval="1d",
            time_zone="UTC",
        )

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "high": np.linspace(101, 131, 31),
                "low": np.linspace(99, 129, 31),
                "close": np.linspace(100, 130, 31),
            }
        )

        labeled = atr_triple_barrier_labels(
            df,
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
            max_holding_bars=20,
        )

        assert len(labeled) == 31
        assert "label" in labeled.columns

    def test_short_position_example(self):
        """Test short position example from docstring."""
        timestamps = pl.datetime_range(
            eager=True,
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 20),
            interval="1d",
            time_zone="UTC",
        )

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "high": np.linspace(101, 120, 20),
                "low": np.linspace(99, 118, 20),
                "close": np.linspace(100, 119, 20),
            }
        )

        labeled = atr_triple_barrier_labels(
            df,
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
            side=-1,
            max_holding_bars=10,
        )

        assert len(labeled) == 20
        assert "label" in labeled.columns
