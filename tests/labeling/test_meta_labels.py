"""Tests for meta-labeling utilities.

Tests cover:
- meta_labels: Binary classification of signal profitability
- compute_bet_size: Probability-to-size transformation methods
- apply_meta_model: End-to-end signal sizing
- Edge cases: zero signals, extreme probabilities, thresholds
"""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.labeling import (
    apply_meta_model,
    compute_bet_size,
    meta_labels,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def signal_data():
    """Sample data with signals and returns."""
    return pl.DataFrame(
        {
            "signal": [1, -1, 1, -1, 1, -1, 0, 1, -1],
            "fwd_return": [0.02, -0.01, -0.01, 0.01, 0.0, 0.0, 0.02, 0.005, -0.002],
        }
    )


@pytest.fixture
def probability_data():
    """Sample data with probabilities."""
    return pl.DataFrame({"prob": [0.1, 0.3, 0.5, 0.7, 0.9, 0.0, 1.0]})


@pytest.fixture
def full_pipeline_data():
    """Data for testing full meta-model pipeline."""
    return pl.DataFrame(
        {
            "signal": [1, -1, 1, -1, 1, 0],
            "meta_prob": [0.8, 0.3, 0.5, 0.9, 0.1, 0.7],
        }
    )


# =============================================================================
# Tests for meta_labels
# =============================================================================


class TestMetaLabels:
    """Tests for meta_labels function."""

    def test_long_profitable(self, signal_data):
        """Test that long signal with positive return is labeled 1."""
        result = meta_labels(signal_data, "signal", "fwd_return")
        # Row 0: signal=1, return=0.02 -> profitable (1)
        assert result["meta_label"][0] == 1

    def test_short_profitable(self, signal_data):
        """Test that short signal with negative return is labeled 1."""
        result = meta_labels(signal_data, "signal", "fwd_return")
        # Row 1: signal=-1, return=-0.01 -> signal*return = 0.01 > 0 -> profitable
        assert result["meta_label"][1] == 1

    def test_long_unprofitable(self, signal_data):
        """Test that long signal with negative return is labeled 0."""
        result = meta_labels(signal_data, "signal", "fwd_return")
        # Row 2: signal=1, return=-0.01 -> signal*return = -0.01 < 0 -> unprofitable
        assert result["meta_label"][2] == 0

    def test_short_unprofitable(self, signal_data):
        """Test that short signal with positive return is labeled 0."""
        result = meta_labels(signal_data, "signal", "fwd_return")
        # Row 3: signal=-1, return=0.01 -> signal*return = -0.01 < 0 -> unprofitable
        assert result["meta_label"][3] == 0

    def test_zero_return_is_unprofitable(self, signal_data):
        """Test that zero return is labeled as unprofitable (0)."""
        result = meta_labels(signal_data, "signal", "fwd_return")
        # Row 4: signal=1, return=0.0 -> 0 not > 0 -> unprofitable
        assert result["meta_label"][4] == 0
        # Row 5: signal=-1, return=0.0 -> 0 not > 0 -> unprofitable
        assert result["meta_label"][5] == 0

    def test_zero_signal_is_null(self, signal_data):
        """Test that zero signal produces null meta_label."""
        result = meta_labels(signal_data, "signal", "fwd_return")
        # Row 6: signal=0 -> null
        assert result["meta_label"][6] is None

    def test_threshold_filters_small_returns(self, signal_data):
        """Test that threshold filters small profitable returns."""
        # With threshold=0.01, only returns > 0.01 are profitable
        result = meta_labels(signal_data, "signal", "fwd_return", threshold=0.01)
        # Row 0: signal=1, return=0.02 -> 0.02 > 0.01 -> profitable
        assert result["meta_label"][0] == 1
        # Row 7: signal=1, return=0.005 -> 0.005 not > 0.01 -> unprofitable
        assert result["meta_label"][7] == 0

    def test_output_has_meta_label_column(self, signal_data):
        """Test that output contains meta_label column."""
        result = meta_labels(signal_data, "signal", "fwd_return")
        assert "meta_label" in result.columns

    def test_preserves_original_columns(self, signal_data):
        """Test that original columns are preserved."""
        result = meta_labels(signal_data, "signal", "fwd_return")
        assert "signal" in result.columns
        assert "fwd_return" in result.columns
        assert len(result.columns) == 3  # original 2 + meta_label


# =============================================================================
# Tests for compute_bet_size
# =============================================================================


class TestComputeBetSize:
    """Tests for compute_bet_size function."""

    def test_linear_at_half(self, probability_data):
        """Test that linear method returns 0 at probability 0.5."""
        result = probability_data.with_columns(
            compute_bet_size("prob", method="linear").alias("bet_size")
        )
        # prob=0.5 -> (0.5-0.5)*2 = 0
        assert np.isclose(result.filter(pl.col("prob") == 0.5)["bet_size"][0], 0.0)

    def test_linear_at_extremes(self, probability_data):
        """Test that linear method returns +/-1 at extremes."""
        result = probability_data.with_columns(
            compute_bet_size("prob", method="linear").alias("bet_size")
        )
        # prob=0.0 -> (0-0.5)*2 = -1
        assert np.isclose(result.filter(pl.col("prob") == 0.0)["bet_size"][0], -1.0)
        # prob=1.0 -> (1-0.5)*2 = 1
        assert np.isclose(result.filter(pl.col("prob") == 1.0)["bet_size"][0], 1.0)

    def test_linear_symmetry(self, probability_data):
        """Test that linear method is symmetric around 0.5."""
        result = probability_data.with_columns(
            compute_bet_size("prob", method="linear").alias("bet_size")
        )
        # 0.3 and 0.7 should have opposite signs, same magnitude
        bet_03 = result.filter(pl.col("prob") == 0.3)["bet_size"][0]
        bet_07 = result.filter(pl.col("prob") == 0.7)["bet_size"][0]
        assert np.isclose(bet_03, -bet_07)

    def test_sigmoid_at_half(self, probability_data):
        """Test that sigmoid method returns 0 at probability 0.5."""
        result = probability_data.with_columns(
            compute_bet_size("prob", method="sigmoid", scale=5.0).alias("bet_size")
        )
        bet_05 = result.filter(pl.col("prob") == 0.5)["bet_size"][0]
        assert np.isclose(bet_05, 0.0, atol=0.01)

    def test_sigmoid_monotonic(self, probability_data):
        """Test that sigmoid method is monotonically increasing."""
        result = probability_data.with_columns(
            compute_bet_size("prob", method="sigmoid", scale=5.0).alias("bet_size")
        ).sort("prob")
        bet_sizes = result["bet_size"].to_numpy()
        # Should be strictly increasing
        assert all(bet_sizes[i] < bet_sizes[i + 1] for i in range(len(bet_sizes) - 1))

    def test_sigmoid_scale_effect(self):
        """Test that higher scale creates sharper sigmoid."""
        df = pl.DataFrame({"prob": [0.4, 0.6]})
        result_low = df.with_columns(
            compute_bet_size("prob", method="sigmoid", scale=1.0).alias("low")
        )
        result_high = df.with_columns(
            compute_bet_size("prob", method="sigmoid", scale=10.0).alias("high")
        )
        # Higher scale should push values closer to +/-1
        assert abs(result_high["high"][1]) > abs(result_low["low"][1])

    def test_discrete_above_threshold(self, probability_data):
        """Test discrete method returns 1 above threshold."""
        result = probability_data.with_columns(
            compute_bet_size("prob", method="discrete", threshold=0.5).alias("bet_size")
        )
        # prob > 0.5 should give 1.0
        bet_07 = result.filter(pl.col("prob") == 0.7)["bet_size"][0]
        bet_09 = result.filter(pl.col("prob") == 0.9)["bet_size"][0]
        assert bet_07 == 1.0
        assert bet_09 == 1.0

    def test_discrete_at_or_below_threshold(self, probability_data):
        """Test discrete method returns 0 at or below threshold."""
        result = probability_data.with_columns(
            compute_bet_size("prob", method="discrete", threshold=0.5).alias("bet_size")
        )
        # prob <= 0.5 should give 0.0
        bet_05 = result.filter(pl.col("prob") == 0.5)["bet_size"][0]
        bet_03 = result.filter(pl.col("prob") == 0.3)["bet_size"][0]
        assert bet_05 == 0.0
        assert bet_03 == 0.0

    def test_unknown_method_raises(self, probability_data):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            probability_data.with_columns(
                compute_bet_size("prob", method="unknown").alias("bet_size")  # type: ignore[arg-type]
            )


# =============================================================================
# Tests for apply_meta_model
# =============================================================================


class TestApplyMetaModel:
    """Tests for apply_meta_model function."""

    def test_high_prob_long_signal(self, full_pipeline_data):
        """Test high probability amplifies long signal."""
        result = apply_meta_model(
            full_pipeline_data, "signal", "meta_prob", bet_size_method="sigmoid"
        )
        # Row 0: signal=1, prob=0.8 -> positive sized signal
        sized = result["sized_signal"][0]
        assert sized > 0.5  # High probability should give strong signal

    def test_low_prob_short_signal(self, full_pipeline_data):
        """Test low probability dampens short signal."""
        result = apply_meta_model(
            full_pipeline_data, "signal", "meta_prob", bet_size_method="sigmoid"
        )
        # Row 1: signal=-1, prob=0.3 -> weak negative (low confidence)
        sized = result["sized_signal"][1]
        # Direction is negative (short), but magnitude is reduced
        assert sized < 0  # Still negative (short direction)
        assert abs(sized) < 0.5  # But weak (low probability)

    def test_half_prob_gives_near_zero(self, full_pipeline_data):
        """Test that 0.5 probability gives near-zero sized signal."""
        result = apply_meta_model(
            full_pipeline_data, "signal", "meta_prob", bet_size_method="sigmoid"
        )
        # Row 2: signal=1, prob=0.5 -> near zero
        sized = result["sized_signal"][2]
        assert abs(sized) < 0.1

    def test_zero_signal_remains_zero(self, full_pipeline_data):
        """Test that zero signal remains zero regardless of probability."""
        result = apply_meta_model(
            full_pipeline_data, "signal", "meta_prob", bet_size_method="sigmoid"
        )
        # Row 5: signal=0, prob=0.7 -> 0 (no trade regardless of confidence)
        sized = result["sized_signal"][5]
        assert sized == 0.0

    def test_output_column_name(self, full_pipeline_data):
        """Test that output column name is configurable."""
        result = apply_meta_model(
            full_pipeline_data,
            "signal",
            "meta_prob",
            output_col="custom_signal",
        )
        assert "custom_signal" in result.columns
        assert "sized_signal" not in result.columns

    def test_linear_method(self, full_pipeline_data):
        """Test apply_meta_model with linear bet sizing."""
        result = apply_meta_model(
            full_pipeline_data, "signal", "meta_prob", bet_size_method="linear"
        )
        # Row 3: signal=-1, prob=0.9 -> linear: (0.9-0.5)*2 = 0.8, sized = -1 * 0.8 = -0.8
        sized = result["sized_signal"][3]
        assert np.isclose(sized, -0.8)

    def test_discrete_method(self, full_pipeline_data):
        """Test apply_meta_model with discrete bet sizing."""
        result = apply_meta_model(
            full_pipeline_data,
            "signal",
            "meta_prob",
            bet_size_method="discrete",
            threshold=0.5,
        )
        # Row 0: signal=1, prob=0.8 > 0.5 -> sized = 1 * 1 = 1
        assert result["sized_signal"][0] == 1.0
        # Row 1: signal=-1, prob=0.3 <= 0.5 -> sized = -1 * 0 = 0
        assert result["sized_signal"][1] == 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetaLabelingPipeline:
    """Integration tests for full meta-labeling workflow."""

    def test_full_pipeline_workflow(self):
        """Test complete meta-labeling workflow from signals to sized positions."""
        # Step 1: Generate signals and returns
        df = pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "signal": [1, -1, 1, -1],
                "fwd_return": [0.02, -0.015, -0.01, 0.01],
            }
        )

        # Step 2: Create meta-labels (training targets)
        df_labeled = meta_labels(df, "signal", "fwd_return")

        # Check meta-labels are correct
        # Row 0: long + positive = profitable (1)
        # Row 1: short + negative = profitable (1)
        # Row 2: long + negative = unprofitable (0)
        # Row 3: short + positive = unprofitable (0)
        assert df_labeled["meta_label"].to_list() == [1, 1, 0, 0]

        # Step 3: Simulate meta-model predictions
        df_with_probs = df_labeled.with_columns(pl.Series("meta_prob", [0.85, 0.75, 0.4, 0.3]))

        # Step 4: Apply meta-model to size positions
        df_final = apply_meta_model(
            df_with_probs, "signal", "meta_prob", bet_size_method="sigmoid", scale=5.0
        )

        # Verify sized signals
        sized = df_final["sized_signal"].to_list()

        # High confidence correct predictions -> large magnitude
        assert sized[0] > 0.5  # Long, high prob
        assert sized[1] < -0.5  # Short, high prob

        # Low confidence wrong predictions -> small magnitude
        assert 0 < sized[2] < 0.5  # Long, low prob (still positive direction)
        assert -0.5 < sized[3] < 0  # Short, low prob

    def test_transaction_cost_threshold(self):
        """Test using threshold to filter trades below transaction cost."""
        df = pl.DataFrame(
            {
                "signal": [1, 1, 1, 1],
                "fwd_return": [0.001, 0.005, 0.01, 0.02],  # Small to large returns
            }
        )

        # 50 bps transaction cost threshold
        result = meta_labels(df, "signal", "fwd_return", threshold=0.005)

        # Only returns > 0.5% are considered profitable
        assert result["meta_label"][0] == 0  # 0.1% < 0.5%
        assert result["meta_label"][1] == 0  # 0.5% not > 0.5%
        assert result["meta_label"][2] == 1  # 1.0% > 0.5%
        assert result["meta_label"][3] == 1  # 2.0% > 0.5%
