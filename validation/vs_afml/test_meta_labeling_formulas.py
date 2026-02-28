# mypy: disable-error-code="no-any-return,arg-type"
"""Direct verification of meta-labeling against AFML Chapter 3 formulas.

Meta-labeling creates a binary classification target indicating whether
the primary model's directional prediction was correct.

AFML Meta-Label Formula:
    meta_label = 1 if sign(signal) * return > threshold else 0

This means:
    - Signal=+1 (long), return>0: meta_label=1 (correct prediction)
    - Signal=+1 (long), return<0: meta_label=0 (wrong prediction)
    - Signal=-1 (short), return<0: meta_label=1 (correct, short profits from decline)
    - Signal=-1 (short), return>0: meta_label=0 (wrong, short loses on gain)

Reference:
    López de Prado, M. (2018). "Advances in Financial Machine Learning".
    Wiley. Chapter 3: Meta-Labeling.
"""

from __future__ import annotations

import polars as pl

from ml4t.engineer.labeling import apply_meta_model, compute_bet_size, meta_labels


class TestMetaLabelFormula:
    """Tests for core meta-labeling formula."""

    def test_long_correct_positive_return(self):
        """Long signal + positive return = meta_label 1 (correct)."""
        df = pl.DataFrame(
            {
                "signal": [1],
                "fwd_return": [0.05],  # 5% gain
            }
        )

        result = meta_labels(df, "signal", "fwd_return")

        assert result["meta_label"][0] == 1

    def test_long_wrong_negative_return(self):
        """Long signal + negative return = meta_label 0 (wrong)."""
        df = pl.DataFrame(
            {
                "signal": [1],
                "fwd_return": [-0.03],  # 3% loss
            }
        )

        result = meta_labels(df, "signal", "fwd_return")

        assert result["meta_label"][0] == 0

    def test_short_correct_negative_return(self):
        """Short signal + negative return = meta_label 1 (correct).

        Shorts profit when price goes down.
        """
        df = pl.DataFrame(
            {
                "signal": [-1],
                "fwd_return": [-0.04],  # 4% decline = profit for short
            }
        )

        result = meta_labels(df, "signal", "fwd_return")

        assert result["meta_label"][0] == 1

    def test_short_wrong_positive_return(self):
        """Short signal + positive return = meta_label 0 (wrong).

        Shorts lose when price goes up.
        """
        df = pl.DataFrame(
            {
                "signal": [-1],
                "fwd_return": [0.02],  # 2% gain = loss for short
            }
        )

        result = meta_labels(df, "signal", "fwd_return")

        assert result["meta_label"][0] == 0

    def test_no_signal_returns_null(self):
        """Zero signal = null meta_label (no prediction to evaluate)."""
        df = pl.DataFrame(
            {
                "signal": [0],
                "fwd_return": [0.01],
            }
        )

        result = meta_labels(df, "signal", "fwd_return")

        assert result["meta_label"][0] is None

    def test_multiple_signals(self):
        """Test batch of signals with various outcomes."""
        df = pl.DataFrame(
            {
                "signal": [1, 1, -1, -1, 0, 1, -1],
                "fwd_return": [0.02, -0.01, -0.03, 0.02, 0.01, 0.00, 0.00],
            }
        )

        result = meta_labels(df, "signal", "fwd_return")

        expected = [
            1,  # Long + positive = correct
            0,  # Long + negative = wrong
            1,  # Short + negative = correct
            0,  # Short + positive = wrong
            None,  # No signal
            0,  # Long + zero return (not > 0)
            0,  # Short + zero return (not > 0)
        ]

        for i, exp in enumerate(expected):
            assert result["meta_label"][i] == exp, f"Row {i}: expected {exp}"


class TestMetaLabelThreshold:
    """Tests for meta-labeling with threshold."""

    def test_threshold_filters_small_profits(self):
        """Returns below threshold are considered wrong.

        This accounts for transaction costs - a 0.1% gain isn't profitable
        if transaction costs are 0.2%.
        """
        df = pl.DataFrame(
            {
                "signal": [1, 1, 1],
                "fwd_return": [0.001, 0.002, 0.003],  # 0.1%, 0.2%, 0.3%
            }
        )

        # Threshold of 0.2% (0.002)
        result = meta_labels(df, "signal", "fwd_return", threshold=0.002)

        assert result["meta_label"][0] == 0, "0.1% < threshold"
        assert result["meta_label"][1] == 0, "0.2% == threshold (not >)"
        assert result["meta_label"][2] == 1, "0.3% > threshold"

    def test_threshold_with_short(self):
        """Threshold works for short positions too."""
        df = pl.DataFrame(
            {
                "signal": [-1, -1, -1],
                "fwd_return": [-0.001, -0.002, -0.003],  # Short profits
            }
        )

        # For shorts: signed_return = -1 * fwd_return = 0.001, 0.002, 0.003
        result = meta_labels(df, "signal", "fwd_return", threshold=0.002)

        assert result["meta_label"][0] == 0, "Signed return 0.1% < threshold"
        assert result["meta_label"][1] == 0, "Signed return 0.2% == threshold"
        assert result["meta_label"][2] == 1, "Signed return 0.3% > threshold"


class TestBetSizeFormulas:
    """Tests for bet sizing formulas from AFML."""

    def test_linear_bet_size(self):
        """Linear bet sizing: bet = 2 * (prob - 0.5).

        Range: [0,1] → [-1, 1]
        """
        df = pl.DataFrame(
            {
                "prob": [0.0, 0.25, 0.5, 0.75, 1.0],
            }
        )

        result = df.with_columns(compute_bet_size("prob", method="linear").alias("bet"))

        expected = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for i, exp in enumerate(expected):
            assert abs(result["bet"][i] - exp) < 1e-10, f"prob={df['prob'][i]}"

    def test_discrete_bet_size(self):
        """Discrete bet sizing: bet = 1 if prob > threshold else 0."""
        df = pl.DataFrame(
            {
                "prob": [0.4, 0.5, 0.6, 0.7],
            }
        )

        result = df.with_columns(
            compute_bet_size("prob", method="discrete", threshold=0.5).alias("bet")
        )

        expected = [0.0, 0.0, 1.0, 1.0]

        for i, exp in enumerate(expected):
            assert abs(result["bet"][i] - exp) < 1e-10

    def test_sigmoid_bet_size_centered(self):
        """Sigmoid bet sizing: centered at 0.5, S-curve shape."""
        df = pl.DataFrame(
            {
                "prob": [0.5],  # Center point
            }
        )

        result = df.with_columns(compute_bet_size("prob", method="sigmoid", scale=1.0).alias("bet"))

        # At prob=0.5, sigmoid should output 0 (centered)
        assert abs(result["bet"][0]) < 1e-10

    def test_sigmoid_bet_size_extremes(self):
        """Sigmoid approaches ±1 at probability extremes."""
        df = pl.DataFrame(
            {
                "prob": [0.0, 1.0],
            }
        )

        result = df.with_columns(
            compute_bet_size("prob", method="sigmoid", scale=10.0).alias("bet")
        )

        # High scale makes sigmoid approach (but not quite reach) ±1
        # At prob=0: x = 10*(0-0.5) = -5, tanh(-5) ≈ -0.9999
        # At prob=1: x = 10*(1-0.5) = 5, tanh(5) ≈ 0.9999
        # With scale=10, we get ~0.9866 not 0.99+
        assert result["bet"][0] < -0.98, "prob=0 should give bet≈-1"
        assert result["bet"][1] > 0.98, "prob=1 should give bet≈+1"


class TestApplyMetaModel:
    """Tests for combining primary signal with meta-model probability."""

    def test_high_prob_long_signal(self):
        """High probability + long signal → strong positive sized signal."""
        df = pl.DataFrame(
            {
                "signal": [1],
                "meta_prob": [0.9],  # High confidence
            }
        )

        result = apply_meta_model(df, "signal", "meta_prob", bet_size_method="linear")

        # signal=1, bet_size = 2*(0.9-0.5) = 0.8
        # sized_signal = sign(1) * abs(0.8) = 0.8
        assert abs(result["sized_signal"][0] - 0.8) < 1e-10

    def test_low_prob_long_signal(self):
        """Low probability + long signal → weak positive sized signal."""
        df = pl.DataFrame(
            {
                "signal": [1],
                "meta_prob": [0.3],  # Low confidence
            }
        )

        result = apply_meta_model(df, "signal", "meta_prob", bet_size_method="linear")

        # signal=1, bet_size = 2*(0.3-0.5) = -0.4
        # sized_signal = sign(1) * abs(-0.4) = 0.4
        assert abs(result["sized_signal"][0] - 0.4) < 1e-10

    def test_high_prob_short_signal(self):
        """High probability + short signal → strong negative sized signal."""
        df = pl.DataFrame(
            {
                "signal": [-1],
                "meta_prob": [0.9],
            }
        )

        result = apply_meta_model(df, "signal", "meta_prob", bet_size_method="linear")

        # signal=-1, bet_size = 0.8
        # sized_signal = sign(-1) * abs(0.8) = -0.8
        assert abs(result["sized_signal"][0] - (-0.8)) < 1e-10

    def test_uncertain_probability(self):
        """50% probability → near-zero sized signal."""
        df = pl.DataFrame(
            {
                "signal": [1],
                "meta_prob": [0.5],
            }
        )

        result = apply_meta_model(df, "signal", "meta_prob", bet_size_method="linear")

        # bet_size = 2*(0.5-0.5) = 0
        assert abs(result["sized_signal"][0]) < 1e-10

    def test_discrete_filtering(self):
        """Discrete method can be used to filter low-confidence trades."""
        df = pl.DataFrame(
            {
                "signal": [1, 1, -1, -1],
                "meta_prob": [0.4, 0.7, 0.3, 0.8],
            }
        )

        result = apply_meta_model(
            df,
            "signal",
            "meta_prob",
            bet_size_method="discrete",
            threshold=0.5,
        )

        expected = [0.0, 1.0, 0.0, -1.0]

        for i, exp in enumerate(expected):
            assert abs(result["sized_signal"][i] - exp) < 1e-10


class TestEdgeCases:
    """Edge cases for meta-labeling."""

    def test_zero_return_long(self):
        """Zero return is not profitable (not > threshold)."""
        df = pl.DataFrame(
            {
                "signal": [1],
                "fwd_return": [0.0],
            }
        )

        result = meta_labels(df, "signal", "fwd_return")

        assert result["meta_label"][0] == 0

    def test_very_small_return(self):
        """Very small positive return is still correct (> 0)."""
        df = pl.DataFrame(
            {
                "signal": [1],
                "fwd_return": [1e-10],
            }
        )

        result = meta_labels(df, "signal", "fwd_return")

        assert result["meta_label"][0] == 1

    def test_large_signal_values(self):
        """Signal values other than ±1 should work (use sign)."""
        df = pl.DataFrame(
            {
                "signal": [10.0, -5.0, 0.5, -0.1],  # Explicit floats for Polars type inference
                "fwd_return": [0.01, -0.01, 0.01, -0.01],
            }
        )

        result = meta_labels(df, "signal", "fwd_return")

        # sign(10)=1, sign(-5)=-1, sign(0.5)=1, sign(-0.1)=-1
        expected = [1, 1, 1, 1]  # All predictions correct

        for i, exp in enumerate(expected):
            assert result["meta_label"][i] == exp
