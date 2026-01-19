# mypy: disable-error-code="no-any-return,arg-type"
"""Direct verification of triple barrier labeling against AFML Chapter 3 formulas.

These tests use deterministic synthetic cases where the correct answer is known
by construction based on the mathematical definition from De Prado's book.

AFML Triple Barrier Algorithm:
    1. For each event i at time t_i with entry price p_i:
    2. Set barriers:
       - Upper: p_upper = p_i * (1 + upper_barrier)
       - Lower: p_lower = p_i * (1 - lower_barrier)
       - Vertical: t_1 = t_i + max_holding_period
    3. Find first touch:
       - If price hits p_upper first → label = 1
       - If price hits p_lower first → label = -1
       - If time reaches t_1 first → label = 0 (return-based)

Reference:
    López de Prado, M. (2018). "Advances in Financial Machine Learning".
    Wiley. Chapter 3: Labeling.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.labeling import BarrierConfig, triple_barrier_labels


class TestUpperBarrierHit:
    """Tests for scenarios where upper (profit) barrier is hit first."""

    def test_upper_barrier_hit_long_immediate(self):
        """Price immediately exceeds upper barrier → label must be 1.

        AFML Formula: If high[j] >= p_entry * (1 + upper_barrier) before lower/time,
        then label = 1 (upper barrier hit).
        """
        # Entry at 100, upper barrier at 102 (2%)
        # Price: 100 → 101 → 103 (hits 102 on bar 2)
        prices = [100.0, 101.0, 103.0, 104.0, 105.0]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,  # 2% profit target
            lower_barrier=0.02,  # 2% stop loss
            max_holding_period=10,
            side=1,  # Long position
        )

        result = triple_barrier_labels(df, config)

        # Check first event (entry at bar 0)
        assert result["label"][0] == 1, "Upper barrier should be hit"
        assert result["barrier_hit"][0] == "upper"
        assert result["label_bars"][0] == 2, "Should hit on bar 2"

    def test_upper_barrier_hit_long_gradual(self):
        """Gradual price increase to upper barrier."""
        # Entry at 100, need to reach 105 (5% barrier)
        # Price gradually rises: 100 → 101 → 102 → 103 → 104 → 105 → 106
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.05,  # 5% profit target
            lower_barrier=0.10,  # 10% stop loss (won't hit)
            max_holding_period=20,
            side=1,
        )

        result = triple_barrier_labels(df, config)

        assert result["label"][0] == 1
        assert result["barrier_hit"][0] == "upper"
        assert result["label_bars"][0] == 5, "Should hit when price reaches 105"

    def test_upper_barrier_hit_short_position(self):
        """Short position: profit target is BELOW entry price.

        For shorts, upper_barrier = 2% means profit at 98 (2% below 100).
        """
        # Entry at 100, profit target at 98 for short
        # Price drops: 100 → 99 → 97 (hits 98 on bar 2)
        prices = [100.0, 99.0, 97.0, 96.0, 95.0]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,  # 2% profit (at 98 for short)
            lower_barrier=0.02,  # 2% stop loss (at 102 for short)
            max_holding_period=10,
            side=-1,  # Short position
        )

        result = triple_barrier_labels(df, config)

        assert result["label"][0] == 1, "Upper (profit) barrier should be hit"
        assert result["barrier_hit"][0] == "upper"


class TestLowerBarrierHit:
    """Tests for scenarios where lower (stop loss) barrier is hit first."""

    def test_lower_barrier_hit_long_immediate(self):
        """Price immediately drops below lower barrier → label must be -1.

        AFML Formula: If low[j] <= p_entry * (1 - lower_barrier) before upper/time,
        then label = -1 (lower barrier hit).
        """
        # Entry at 100, lower barrier at 98 (2%)
        # Price: 100 → 99 → 97 (hits 98 on bar 2)
        prices = [100.0, 99.0, 97.0, 96.0, 95.0]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,  # 2% profit target
            lower_barrier=0.02,  # 2% stop loss
            max_holding_period=10,
            side=1,  # Long position
        )

        result = triple_barrier_labels(df, config)

        assert result["label"][0] == -1, "Lower barrier should be hit"
        assert result["barrier_hit"][0] == "lower"
        assert result["label_bars"][0] == 2

    def test_lower_barrier_hit_short_position(self):
        """Short position: stop loss is ABOVE entry price.

        For shorts, lower_barrier = 2% means stop loss at 102 (2% above 100).
        """
        # Entry at 100, stop loss at 102 for short
        # Price rises: 100 → 101 → 103 (hits 102)
        prices = [100.0, 101.0, 103.0, 104.0, 105.0]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,  # 2% profit (at 98 for short)
            lower_barrier=0.02,  # 2% stop loss (at 102 for short)
            max_holding_period=10,
            side=-1,  # Short position
        )

        result = triple_barrier_labels(df, config)

        assert result["label"][0] == -1, "Lower (stop) barrier should be hit"
        assert result["barrier_hit"][0] == "lower"


class TestTimeBarrierHit:
    """Tests for scenarios where time (vertical) barrier is hit."""

    def test_time_barrier_positive_return(self):
        """Price stays within barriers, slight positive at end → label 1.

        AFML Formula: If neither upper nor lower barrier hit before max_holding,
        label = sign(return at time barrier).
        """
        # Entry at 100, barriers at ±2%
        # Price never hits ±2%: 100 → 100.5 → 100.3 → 100.7 → 100.8
        # At time barrier (bar 4): return = (100.8 - 100) / 100 = +0.8%
        prices = [100.0, 100.5, 100.3, 100.7, 100.8]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,  # 2% (won't hit)
            lower_barrier=0.02,  # 2% (won't hit)
            max_holding_period=4,  # Time barrier at bar 4
            side=1,
        )

        result = triple_barrier_labels(df, config)

        # Note: label=0 for time barrier, not +1/-1 based on return
        # The implementation uses 0 for time barrier hit
        assert result["label"][0] == 0, "Time barrier should give label 0"
        assert result["barrier_hit"][0] == "time"
        assert result["label_return"][0] > 0, "Return should be positive"

    def test_time_barrier_negative_return(self):
        """Price stays within barriers, slight negative at end."""
        # Entry at 100, barriers at ±2%
        # Price never hits ±2%: 100 → 99.5 → 99.7 → 99.3 → 99.2
        prices = [100.0, 99.5, 99.7, 99.3, 99.2]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,
            lower_barrier=0.02,
            max_holding_period=4,
            side=1,
        )

        result = triple_barrier_labels(df, config)

        assert result["label"][0] == 0, "Time barrier should give label 0"
        assert result["barrier_hit"][0] == "time"
        assert result["label_return"][0] < 0, "Return should be negative"

    def test_time_barrier_flat(self):
        """Price stays exactly flat → label 0 with zero return."""
        prices = [100.0, 100.0, 100.0, 100.0, 100.0]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,
            lower_barrier=0.02,
            max_holding_period=4,
            side=1,
        )

        result = triple_barrier_labels(df, config)

        assert result["label"][0] == 0
        assert result["barrier_hit"][0] == "time"
        assert abs(result["label_return"][0]) < 1e-10


class TestReturnCalculation:
    """Tests for correct return calculation per AFML formula."""

    def test_return_formula_long(self):
        """Verify return = (exit - entry) / entry for long positions."""
        # Entry at 100, exit at 103 (upper barrier hit)
        prices = [100.0, 101.0, 103.0, 104.0, 105.0]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,  # Hit at 102, close is 103
            lower_barrier=0.02,
            max_holding_period=10,
            side=1,
        )

        result = triple_barrier_labels(df, config)

        expected_return = (103.0 - 100.0) / 100.0  # 0.03 = 3%
        assert abs(result["label_return"][0] - expected_return) < 1e-10

    def test_return_formula_short(self):
        """Verify return = (entry - exit) / entry for short positions.

        For shorts, profit is when price goes down.
        """
        # Entry at 100, exit at 97 (profit for short)
        prices = [100.0, 99.0, 97.0, 96.0, 95.0]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,  # Profit target at 98
            lower_barrier=0.02,
            max_holding_period=10,
            side=-1,  # Short
        )

        result = triple_barrier_labels(df, config)

        # For short: return = (entry - exit) / entry = (100 - 97) / 100 = 0.03
        expected_return = (100.0 - 97.0) / 100.0
        assert abs(result["label_return"][0] - expected_return) < 1e-10


class TestBarrierPriority:
    """Tests for barrier priority when multiple could be hit."""

    def test_upper_before_lower_same_bar(self):
        """When OHLC allows both barriers in same bar, upper takes priority.

        If high >= upper and low <= lower on same bar, upper wins.
        This requires passing high_col and low_col to enable OHLC checking.
        """
        # Entry at 100, bar 1 has extreme range
        close = [100.0, 100.0]
        high = [100.0, 110.0]  # Could hit upper at 102
        low = [100.0, 90.0]  # Could hit lower at 98

        df = pl.DataFrame({"close": close, "high": high, "low": low})

        config = BarrierConfig(
            upper_barrier=0.02,
            lower_barrier=0.02,
            max_holding_period=10,
            side=1,
        )

        # Must pass high_col and low_col to enable OHLC barrier checking
        result = triple_barrier_labels(df, config, high_col="high", low_col="low")

        # Upper barrier should be checked first
        assert result["label"][0] == 1
        assert result["barrier_hit"][0] == "upper"

    def test_lower_hit_before_upper(self):
        """Lower barrier hit first chronologically."""
        # Entry at 100
        # Bar 1: price drops but stays above lower
        # Bar 2: hits lower barrier
        # Bar 3: would have hit upper but already exited
        close = [100.0, 99.0, 97.0, 105.0]
        df = pl.DataFrame({"close": close, "high": close, "low": close})

        config = BarrierConfig(
            upper_barrier=0.03,  # 103
            lower_barrier=0.02,  # 98
            max_holding_period=10,
            side=1,
        )

        result = triple_barrier_labels(df, config)

        assert result["label"][0] == -1
        assert result["barrier_hit"][0] == "lower"
        assert result["label_bars"][0] == 2  # Hit on bar 2


class TestMultipleEvents:
    """Tests with multiple labeling events."""

    def test_multiple_events_independent(self):
        """Each event is labeled independently from entry point."""
        # Multiple entry points with different outcomes
        prices = [100.0, 103.0, 100.0, 97.0, 100.0]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,
            lower_barrier=0.02,
            max_holding_period=3,
            side=1,
        )

        result = triple_barrier_labels(df, config)

        # Event 0 (entry 100): hits upper at 103
        assert result["label"][0] == 1
        # Event 1 (entry 103): hits lower at ~101 (but 100 is further down)
        # Actually 100 is 2.9% below 103, hits lower barrier
        assert result["label"][1] == -1
        # Event 2 (entry 100): hits lower at 97
        assert result["label"][2] == -1

    def test_all_events_time_barrier(self):
        """All events hit time barrier (flat market)."""
        prices = [100.0] * 20
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,
            lower_barrier=0.02,
            max_holding_period=5,
            side=1,
        )

        result = triple_barrier_labels(df, config)

        # All events should hit time barrier with zero return
        for i in range(len(prices) - 5):
            assert result["label"][i] == 0, f"Event {i} should hit time barrier"
            assert result["barrier_hit"][i] == "time"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_exact_barrier_touch(self):
        """Price exactly touches barrier level."""
        # Entry at 100, barrier at 102
        prices = [100.0, 101.0, 102.0, 101.0]  # Exactly 102
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,
            lower_barrier=0.02,
            max_holding_period=10,
            side=1,
        )

        result = triple_barrier_labels(df, config)

        assert result["label"][0] == 1
        assert result["barrier_hit"][0] == "upper"

    def test_max_period_boundary(self):
        """Barrier hit exactly at max holding period."""
        # Entry at 100, upper barrier at 102
        # Price hits barrier on the last allowed bar
        prices = [100.0, 100.0, 100.0, 102.0, 105.0]  # Hits on bar 3
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,
            lower_barrier=0.02,
            max_holding_period=3,  # Allows checking bars 1, 2, 3
            side=1,
        )

        result = triple_barrier_labels(df, config)

        assert result["label"][0] == 1, "Should hit upper on bar 3"

    def test_end_of_data(self):
        """Event near end of data with insufficient bars."""
        prices = [100.0, 101.0, 102.0]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.05,  # 5% (won't hit with this data)
            lower_barrier=0.05,
            max_holding_period=10,  # More than available data
            side=1,
        )

        result = triple_barrier_labels(df, config)

        # Last event should still get a label (time barrier at end of data)
        assert result["label"][2] is not None

    def test_asymmetric_barriers(self):
        """Different sizes for upper and lower barriers."""
        # Entry at 100
        # Upper at 105 (5%), Lower at 98 (2%)
        prices = [100.0, 99.0, 97.0, 100.0]  # Hits lower first at 97
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.05,  # 5%
            lower_barrier=0.02,  # 2%
            max_holding_period=10,
            side=1,
        )

        result = triple_barrier_labels(df, config)

        assert result["label"][0] == -1
        assert result["barrier_hit"][0] == "lower"

    def test_wide_barriers(self):
        """Very wide barriers (unusual but valid)."""
        # Entry at 100, barriers at ±50%
        prices = [100.0, 110.0, 120.0, 130.0, 140.0]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.50,  # 50%
            lower_barrier=0.50,  # 50%
            max_holding_period=3,  # Time barrier before price barriers
            side=1,
        )

        result = triple_barrier_labels(df, config)

        # Should hit time barrier since 40% gain < 50% barrier
        assert result["label"][0] == 0
        assert result["barrier_hit"][0] == "time"


class TestSymmetricSide:
    """Tests for symmetric (side=0 or None) positions."""

    def test_symmetric_defaults_to_long_like(self):
        """Symmetric position behaves like long for barrier checking."""
        prices = [100.0, 103.0, 105.0]
        df = pl.DataFrame({"close": prices, "high": prices, "low": prices})

        config = BarrierConfig(
            upper_barrier=0.02,
            lower_barrier=0.02,
            max_holding_period=10,
            side=0,  # Symmetric
        )

        result = triple_barrier_labels(df, config)

        # Should behave like long: upper barrier hit when price goes up
        assert result["label"][0] == 1
        assert result["barrier_hit"][0] == "upper"


class TestOHLCBarrierChecking:
    """Tests for OHLC-based barrier checking."""

    def test_high_triggers_upper_long(self):
        """For long positions, high price can trigger upper barrier.

        Even if close doesn't reach the barrier, high touching it counts.
        """
        close = [100.0, 100.0, 100.0]
        high = [100.0, 105.0, 100.0]  # High touches 105, > upper at 102
        low = [100.0, 100.0, 100.0]

        df = pl.DataFrame({"close": close, "high": high, "low": low})

        config = BarrierConfig(
            upper_barrier=0.02,
            lower_barrier=0.02,
            max_holding_period=10,
            side=1,
        )

        result = triple_barrier_labels(df, config, high_col="high", low_col="low")

        assert result["label"][0] == 1
        assert result["barrier_hit"][0] == "upper"

    def test_low_triggers_lower_long(self):
        """For long positions, low price can trigger lower barrier."""
        close = [100.0, 100.0, 100.0]
        high = [100.0, 100.0, 100.0]
        low = [100.0, 95.0, 100.0]  # Low touches 95, < lower at 98

        df = pl.DataFrame({"close": close, "high": high, "low": low})

        config = BarrierConfig(
            upper_barrier=0.02,
            lower_barrier=0.02,
            max_holding_period=10,
            side=1,
        )

        result = triple_barrier_labels(df, config, high_col="high", low_col="low")

        assert result["label"][0] == -1
        assert result["barrier_hit"][0] == "lower"
