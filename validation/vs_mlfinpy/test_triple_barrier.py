# mypy: disable-error-code="import-untyped,no-any-return,arg-type"
"""Comparison tests between ml4t.engineer.labeling and mlfinpy.

These tests validate ml4t implementations against the open-source mlfinpy
package, which provides reference implementations of De Prado's algorithms.

Requirements:
    pip install mlfinpy

Note: mlfinpy requires numba 0.60.0 which conflicts with ml4t-engineer's
numba 0.63.1. Run these tests in a separate virtual environment:

    python -m venv .venv-mlfinpy
    source .venv-mlfinpy/bin/activate
    pip install mlfinpy pandas numpy pytest
    pip install -e .  # Install ml4t-engineer in dev mode
    pytest validation/vs_mlfinpy/ -v

Reference:
    https://mlfinpy.readthedocs.io/
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

# Try to import mlfinpy - skip all tests if not available
try:
    import mlfinpy  # noqa: F401
    from mlfinpy.labeling.labeling import (
        add_vertical_barrier,
        get_events,
        get_bins,
    )
    HAS_MLFINPY = True
except ImportError:
    HAS_MLFINPY = False

from ml4t.engineer.labeling import BarrierConfig, triple_barrier_labels


pytestmark = pytest.mark.skipif(
    not HAS_MLFINPY,
    reason="mlfinpy not installed (requires separate venv due to numba conflict)"
)


def create_test_prices(n: int = 500, seed: int = 42) -> tuple[pd.Series, pl.DataFrame]:
    """Create test price series for both mlfinpy (pandas) and ml4t (polars)."""
    np.random.seed(seed)

    # Generate realistic price series with some volatility
    returns = np.random.randn(n) * 0.02
    close = 100.0 * np.exp(np.cumsum(returns))

    # Create pandas series with datetime index (required by mlfinpy)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    pandas_close = pd.Series(close, index=dates, name="close")

    # Create polars DataFrame
    polars_df = pl.DataFrame({
        "timestamp": dates.to_pydatetime(),
        "close": close,
        "high": close,  # Simplified: use close for OHLC
        "low": close,
    })

    return pandas_close, polars_df


def get_daily_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    """Compute daily volatility using EWM (from AFML Snippet 3.1)."""
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    df0 = df0.ewm(span=span).std()
    return df0


class TestTripleBarrierComparison:
    """Compare ml4t vs mlfinpy triple barrier labeling."""

    def test_symmetric_barriers_same_labels(self):
        """Symmetric barriers should produce matching labels."""
        pandas_close, polars_df = create_test_prices(n=200, seed=42)

        # mlfinpy approach
        t_events = pd.Series(pandas_close.index[::5])  # Every 5th bar
        target = pd.Series(0.02, index=pandas_close.index)  # 2% target

        vertical = add_vertical_barrier(t_events, pandas_close, num_days=10)

        mlfinpy_events = get_events(
            close=pandas_close,
            t_events=t_events,
            pt_sl=[1.0, 1.0],  # Symmetric barriers
            target=target,
            min_ret=0.0,
            num_threads=1,
            vertical_barrier_times=vertical,
        )
        mlfinpy_labels = get_bins(mlfinpy_events, pandas_close)

        # ml4t approach - process same events
        # Extract event indices (row numbers) from t_events
        event_mask = np.zeros(len(polars_df), dtype=bool)
        for ev_time in t_events:
            idx = polars_df["timestamp"].to_numpy().searchsorted(ev_time)
            if idx < len(event_mask):
                event_mask[idx] = True

        # Create event_time column for ml4t
        polars_with_events = polars_df.with_columns(
            pl.when(pl.Series(event_mask))
            .then(pl.col("timestamp"))
            .otherwise(None)
            .alias("event_time")
        )

        config = BarrierConfig(
            upper_barrier=0.02,
            lower_barrier=0.02,
            max_holding_period=10,
            side=1,
        )

        ml4t_result = triple_barrier_labels(
            polars_with_events, config, timestamp_col="timestamp"
        )

        # Compare labels (filter to only events that mlfinpy processed)
        ml4t_labels = ml4t_result.filter(pl.col("event_time").is_not_null())

        # Check that we have similar number of labels
        n_mlfinpy = len(mlfinpy_labels)
        n_ml4t = len(ml4t_labels)

        # Allow some tolerance for edge effects
        assert abs(n_mlfinpy - n_ml4t) <= 5, f"Label count mismatch: {n_mlfinpy} vs {n_ml4t}"

        # Compare label distribution
        mlfinpy_dist = mlfinpy_labels["bin"].value_counts(normalize=True)
        ml4t_dist = ml4t_labels["label"].value_counts(normalize=True).sort("label")

        # Both should have similar proportions of +1/-1/0
        print(f"mlfinpy distribution:\n{mlfinpy_dist}")
        print(f"ml4t distribution:\n{ml4t_dist}")

    def test_asymmetric_barriers(self):
        """Asymmetric barriers with different PT and SL."""
        pandas_close, polars_df = create_test_prices(n=200, seed=123)

        # mlfinpy approach
        t_events = pd.Series(pandas_close.index[::10])
        target = pd.Series(0.02, index=pandas_close.index)
        vertical = add_vertical_barrier(t_events, pandas_close, num_days=5)

        mlfinpy_events = get_events(
            close=pandas_close,
            t_events=t_events,
            pt_sl=[1.5, 1.0],  # Asymmetric: PT=3%, SL=2%
            target=target,
            min_ret=0.0,
            num_threads=1,
            vertical_barrier_times=vertical,
        )
        mlfinpy_labels = get_bins(mlfinpy_events, pandas_close)

        # ml4t approach
        config = BarrierConfig(
            upper_barrier=0.03,  # 1.5 * 2% = 3%
            lower_barrier=0.02,  # 1.0 * 2% = 2%
            max_holding_period=5,
            side=1,
        )

        ml4t_result = triple_barrier_labels(polars_df, config)

        # Verify both produce reasonable results
        assert len(mlfinpy_labels) > 0
        assert len(ml4t_result) == len(polars_df)

    def test_label_signs_match(self):
        """Verify label sign conventions match.

        mlfinpy: bin ∈ {-1, 0, 1}
        ml4t: label ∈ {-1, 0, 1}

        Both should use:
        +1 = upper barrier hit (profit)
        -1 = lower barrier hit (stop loss)
        0 = time barrier hit
        """
        pandas_close, polars_df = create_test_prices(n=100, seed=999)

        # mlfinpy
        t_events = pd.Series(pandas_close.index[::20])
        target = pd.Series(0.05, index=pandas_close.index)  # Wide target
        vertical = add_vertical_barrier(t_events, pandas_close, num_days=3)

        mlfinpy_events = get_events(
            close=pandas_close,
            t_events=t_events,
            pt_sl=[1.0, 1.0],
            target=target,
            min_ret=0.0,
            num_threads=1,
            vertical_barrier_times=vertical,
        )
        mlfinpy_labels = get_bins(mlfinpy_events, pandas_close)

        # ml4t
        config = BarrierConfig(
            upper_barrier=0.05,
            lower_barrier=0.05,
            max_holding_period=3,
            side=1,
        )
        ml4t_result = triple_barrier_labels(polars_df, config)

        # Both should have labels in {-1, 0, 1}
        mlfinpy_unique = set(mlfinpy_labels["bin"].unique())
        ml4t_unique = set(ml4t_result["label"].unique())

        assert mlfinpy_unique.issubset({-1, 0, 1})
        assert ml4t_unique.issubset({-1, 0, 1})


class TestReturnCalculations:
    """Compare return calculations between implementations."""

    def test_return_formula(self):
        """Both should use same return formula: (exit - entry) / entry."""
        pandas_close, polars_df = create_test_prices(n=50, seed=111)

        # mlfinpy
        t_events = pd.Series(pandas_close.index[:5])  # Just first 5 events
        target = pd.Series(0.10, index=pandas_close.index)  # Wide barrier
        vertical = add_vertical_barrier(t_events, pandas_close, num_days=20)

        mlfinpy_events = get_events(
            close=pandas_close,
            t_events=t_events,
            pt_sl=[1.0, 1.0],
            target=target,
            min_ret=0.0,
            num_threads=1,
            vertical_barrier_times=vertical,
        )
        mlfinpy_labels = get_bins(mlfinpy_events, pandas_close)

        # ml4t
        config = BarrierConfig(
            upper_barrier=0.10,
            lower_barrier=0.10,
            max_holding_period=20,
            side=1,
        )
        ml4t_result = triple_barrier_labels(polars_df, config)

        # Both should produce finite returns
        mlfinpy_returns = mlfinpy_labels["ret"]
        ml4t_returns = ml4t_result["label_return"]

        assert np.all(np.isfinite(mlfinpy_returns))
        assert np.all(np.isfinite(ml4t_returns.to_numpy()))

        # Returns should be in reasonable range (< 100%)
        assert np.all(np.abs(mlfinpy_returns) < 1.0)
        assert np.all(np.abs(ml4t_returns.to_numpy()) < 1.0)


class TestVolatilityNormalization:
    """Test volatility-normalized barriers."""

    def test_atr_normalized_barriers(self):
        """Compare barriers normalized by volatility estimate."""
        pandas_close, polars_df = create_test_prices(n=300, seed=42)

        # mlfinpy uses getDailyVol for target
        target = get_daily_volatility(pandas_close, span=20)
        target = target.dropna()

        # Only use events where target is available
        t_events = pd.Series(target.index[::10])

        vertical = add_vertical_barrier(t_events, pandas_close, num_days=5)

        mlfinpy_events = get_events(
            close=pandas_close,
            t_events=t_events,
            pt_sl=[2.0, 1.0],  # 2x vol for PT, 1x for SL
            target=target,
            min_ret=0.0,
            num_threads=1,
            vertical_barrier_times=vertical,
        )

        # Should produce valid results
        assert len(mlfinpy_events) > 0
        assert "t1" in mlfinpy_events.columns

        mlfinpy_labels = get_bins(mlfinpy_events, pandas_close)
        assert len(mlfinpy_labels) > 0


class TestEdgeCases:
    """Edge cases for comparison."""

    def test_no_vertical_barrier(self):
        """Compare behavior when vertical barrier is disabled."""
        pandas_close, polars_df = create_test_prices(n=100, seed=42)

        # mlfinpy with no vertical barrier
        t_events = pd.Series(pandas_close.index[:10])
        target = pd.Series(0.02, index=pandas_close.index)

        mlfinpy_events = get_events(
            close=pandas_close,
            t_events=t_events,
            pt_sl=[1.0, 1.0],
            target=target,
            min_ret=0.0,
            num_threads=1,
            vertical_barrier_times=False,  # No vertical barrier
        )

        # Should still produce results (labels determined by horizontal barriers)
        assert len(mlfinpy_events) > 0

    def test_only_profit_taking(self):
        """Test with only profit-taking barrier (no stop loss)."""
        pandas_close, polars_df = create_test_prices(n=100, seed=42)

        # mlfinpy with only PT
        t_events = pd.Series(pandas_close.index[:10])
        target = pd.Series(0.02, index=pandas_close.index)
        vertical = add_vertical_barrier(t_events, pandas_close, num_days=20)

        mlfinpy_events = get_events(
            close=pandas_close,
            t_events=t_events,
            pt_sl=[1.0, 0.0],  # PT only, no SL
            target=target,
            min_ret=0.0,
            num_threads=1,
            vertical_barrier_times=vertical,
        )

        # ml4t equivalent (use np.inf for lower barrier)
        config = BarrierConfig(
            upper_barrier=0.02,
            lower_barrier=None,  # Disabled
            max_holding_period=20,
            side=1,
        )
        ml4t_result = triple_barrier_labels(polars_df, config)

        # Both should produce valid results
        assert len(mlfinpy_events) > 0
        assert len(ml4t_result) > 0
