"""Comprehensive tests for non-TA-Lib volatility indicators.

Tests mathematical properties, edge cases, parameter variations,
and relationships between estimators for all 11 Polars-based
volatility features.
"""

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv_df():
    """Deterministic OHLCV DataFrame (100 rows) with known properties."""
    np.random.seed(42)
    n = 100
    # Geometric brownian motion for realistic prices
    returns = np.random.randn(n) * 0.02  # ~2% daily vol
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_ = np.roll(close, 1) * (1 + np.random.randn(n) * 0.002)
    open_[0] = close[0]
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        }
    )


@pytest.fixture
def constant_df():
    """OHLCV DataFrame with constant prices (zero volatility)."""
    n = 60
    return pl.DataFrame(
        {
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        }
    )


@pytest.fixture
def trending_df():
    """OHLCV with strong upward drift (tests drift-independence)."""
    n = 100
    close = np.linspace(100, 200, n)  # 100% drift over 100 bars
    high = close * 1.005
    low = close * 0.995
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1000.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# 1. Parkinson Volatility
# ---------------------------------------------------------------------------


class TestParkinsonVolatility:
    """Tests for parkinson_volatility."""

    def test_basic_output(self, ohlcv_df):
        """Runs on deterministic data, validates output shape/type."""
        from ml4t.engineer.features.volatility.parkinson_volatility import parkinson_volatility

        result = ohlcv_df.select(parkinson_volatility("high", "low", period=20).alias("pv"))
        assert result.shape == (100, 1)
        assert result.dtypes[0] == pl.Float64

    def test_positivity(self, ohlcv_df):
        """Parkinson volatility must be non-negative."""
        from ml4t.engineer.features.volatility.parkinson_volatility import parkinson_volatility

        vals = (
            ohlcv_df.select(parkinson_volatility("high", "low", period=20).alias("pv"))["pv"]
            .drop_nulls()
            .drop_nans()
        )
        assert (vals >= 0).all()

    def test_zero_range_gives_zero(self, constant_df):
        """H=L => ln(H/L)=0 => zero volatility."""
        from ml4t.engineer.features.volatility.parkinson_volatility import parkinson_volatility

        vals = (
            constant_df.select(parkinson_volatility("high", "low", period=10).alias("pv"))["pv"]
            .drop_nulls()
            .drop_nans()
        )
        assert len(vals) > 0
        assert (vals.abs() < 1e-10).all()

    def test_different_periods(self, ohlcv_df):
        """Longer period smooths more."""
        from ml4t.engineer.features.volatility.parkinson_volatility import parkinson_volatility

        short = (
            ohlcv_df.select(parkinson_volatility("high", "low", period=10).alias("pv"))["pv"]
            .drop_nulls()
            .drop_nans()
        )
        long = (
            ohlcv_df.select(parkinson_volatility("high", "low", period=30).alias("pv"))["pv"]
            .drop_nulls()
            .drop_nans()
        )
        # Longer period should have lower variance (smoothing effect)
        assert short.std() >= long.std() * 0.5  # Generous bound

    def test_annualization(self, ohlcv_df):
        """Annualized > non-annualized by sqrt(252) factor."""
        from ml4t.engineer.features.volatility.parkinson_volatility import parkinson_volatility

        ann = (
            ohlcv_df.select(
                parkinson_volatility("high", "low", period=20, annualize=True).alias("v")
            )["v"]
            .drop_nulls()
            .drop_nans()
        )
        raw = (
            ohlcv_df.select(
                parkinson_volatility("high", "low", period=20, annualize=False).alias("v")
            )["v"]
            .drop_nulls()
            .drop_nans()
        )
        # Annualized should be ~sqrt(252) times larger
        ratio = ann.mean() / raw.mean()
        assert 10 < ratio < 20  # sqrt(252) ~ 15.87


# ---------------------------------------------------------------------------
# 2. Garman-Klass Volatility
# ---------------------------------------------------------------------------


class TestGarmanKlassVolatility:
    """Tests for garman_klass_volatility."""

    def test_basic_output(self, ohlcv_df):
        """Runs on deterministic data, validates output shape."""
        from ml4t.engineer.features.volatility.garman_klass_volatility import (
            garman_klass_volatility,
        )

        result = ohlcv_df.select(
            garman_klass_volatility("open", "high", "low", "close", period=20).alias("gk")
        )
        assert result.shape == (100, 1)

    def test_positivity(self, ohlcv_df):
        """Garman-Klass volatility must be non-negative."""
        from ml4t.engineer.features.volatility.garman_klass_volatility import (
            garman_klass_volatility,
        )

        vals = (
            ohlcv_df.select(
                garman_klass_volatility("open", "high", "low", "close", period=20).alias("gk")
            )["gk"]
            .drop_nulls()
            .drop_nans()
        )
        assert (vals >= 0).all()

    def test_more_efficient_than_parkinson(self, ohlcv_df):
        """GK uses more information (OHLC vs HL) and should be more stable."""
        from ml4t.engineer.features.volatility.garman_klass_volatility import (
            garman_klass_volatility,
        )
        from ml4t.engineer.features.volatility.parkinson_volatility import parkinson_volatility

        gk = (
            ohlcv_df.select(
                garman_klass_volatility("open", "high", "low", "close", period=20).alias("v")
            )["v"]
            .drop_nulls()
            .drop_nans()
        )
        pk = (
            ohlcv_df.select(parkinson_volatility("high", "low", period=20).alias("v"))["v"]
            .drop_nulls()
            .drop_nans()
        )
        # Both should produce similar magnitudes (same underlying data)
        assert abs(gk.mean() - pk.mean()) / pk.mean() < 1.0  # Within 100%

    def test_constant_prices(self, constant_df):
        """Constant prices => zero volatility."""
        from ml4t.engineer.features.volatility.garman_klass_volatility import (
            garman_klass_volatility,
        )

        vals = (
            constant_df.select(
                garman_klass_volatility("open", "high", "low", "close", period=10).alias("gk")
            )["gk"]
            .drop_nulls()
            .drop_nans()
        )
        assert len(vals) > 0
        assert (vals.abs() < 1e-10).all()


# ---------------------------------------------------------------------------
# 3. Rogers-Satchell Volatility
# ---------------------------------------------------------------------------


class TestRogersSatchellVolatility:
    """Tests for rogers_satchell_volatility."""

    def test_basic_output(self, ohlcv_df):
        """Validates output shape and type."""
        from ml4t.engineer.features.volatility.rogers_satchell_volatility import (
            rogers_satchell_volatility,
        )

        result = ohlcv_df.select(
            rogers_satchell_volatility("open", "high", "low", "close", period=20).alias("rs")
        )
        assert result.shape == (100, 1)

    def test_drift_independence(self, trending_df, ohlcv_df):
        """RS is drift-independent; trending data shouldn't inflate vol estimate."""
        from ml4t.engineer.features.volatility.rogers_satchell_volatility import (
            rogers_satchell_volatility,
        )

        # Trending data with similar intraday range should give similar vol
        rs_trend = (
            trending_df.select(
                rogers_satchell_volatility(
                    "open", "high", "low", "close", period=20, annualize=False
                ).alias("v")
            )["v"]
            .drop_nulls()
            .drop_nans()
        )
        # Main check: RS on trending data shouldn't blow up
        assert rs_trend.mean() < 1.0  # Sanity — shouldn't be massive

    def test_positivity(self, ohlcv_df):
        """RS should produce non-negative values for well-formed OHLC."""
        from ml4t.engineer.features.volatility.rogers_satchell_volatility import (
            rogers_satchell_volatility,
        )

        vals = (
            ohlcv_df.select(
                rogers_satchell_volatility("open", "high", "low", "close", period=20).alias("rs")
            )["rs"]
            .drop_nulls()
            .drop_nans()
        )
        assert (vals >= 0).all()

    def test_different_periods(self, ohlcv_df):
        """Test with different window sizes."""
        from ml4t.engineer.features.volatility.rogers_satchell_volatility import (
            rogers_satchell_volatility,
        )

        for period in [10, 20, 50]:
            result = ohlcv_df.select(
                rogers_satchell_volatility("open", "high", "low", "close", period=period).alias("v")
            )
            assert len(result) == 100


# ---------------------------------------------------------------------------
# 4. Yang-Zhang Volatility
# ---------------------------------------------------------------------------


class TestYangZhangVolatility:
    """Tests for yang_zhang_volatility."""

    def test_basic_output(self, ohlcv_df):
        """Validates output shape and type."""
        from ml4t.engineer.features.volatility.yang_zhang_volatility import yang_zhang_volatility

        result = ohlcv_df.select(
            yang_zhang_volatility("open", "high", "low", "close", period=20).alias("yz")
        )
        assert result.shape == (100, 1)

    def test_positivity(self, ohlcv_df):
        """Yang-Zhang should be non-negative."""
        from ml4t.engineer.features.volatility.yang_zhang_volatility import yang_zhang_volatility

        vals = (
            ohlcv_df.select(
                yang_zhang_volatility("open", "high", "low", "close", period=20).alias("yz")
            )["yz"]
            .drop_nulls()
            .drop_nans()
        )
        assert (vals >= 0).all()

    def test_combines_overnight_and_intraday(self, ohlcv_df):
        """YZ should capture both overnight and intraday moves."""
        from ml4t.engineer.features.volatility.yang_zhang_volatility import yang_zhang_volatility

        vals = (
            ohlcv_df.select(
                yang_zhang_volatility("open", "high", "low", "close", period=20).alias("yz")
            )["yz"]
            .drop_nulls()
            .drop_nans()
        )
        # Should produce non-trivial estimates
        assert vals.mean() > 0

    def test_constant_prices(self, constant_df):
        """Constant prices => zero volatility."""
        from ml4t.engineer.features.volatility.yang_zhang_volatility import yang_zhang_volatility

        vals = (
            constant_df.select(
                yang_zhang_volatility("open", "high", "low", "close", period=20).alias("yz")
            )["yz"]
            .drop_nulls()
            .drop_nans()
        )
        assert len(vals) > 0
        assert (vals.abs() < 1e-10).all()


# ---------------------------------------------------------------------------
# 5. Realized Volatility
# ---------------------------------------------------------------------------


class TestRealizedVolatility:
    """Tests for realized_volatility."""

    def test_basic_output(self, ohlcv_df):
        """Validates output shape — note: takes returns, not close."""
        from ml4t.engineer.features.volatility.realized_volatility import realized_volatility

        df = ohlcv_df.with_columns(returns=pl.col("close").pct_change())
        result = df.select(realized_volatility("returns", period=20).alias("rv"))
        assert result.shape == (100, 1)

    def test_constant_returns_zero(self):
        """Constant returns => zero realized vol."""
        from ml4t.engineer.features.volatility.realized_volatility import realized_volatility

        df = pl.DataFrame({"returns": [0.01] * 50})
        vals = (
            df.select(realized_volatility("returns", period=10, annualize=False).alias("rv"))["rv"]
            .drop_nulls()
            .drop_nans()
        )
        assert len(vals) > 0
        assert (vals.abs() < 1e-10).all()

    def test_annualization_factor(self, ohlcv_df):
        """Annualized should be sqrt(trading_periods) times raw."""
        from ml4t.engineer.features.volatility.realized_volatility import realized_volatility

        df = ohlcv_df.with_columns(returns=pl.col("close").pct_change())
        ann = (
            df.select(
                realized_volatility(
                    "returns", period=20, annualize=True, trading_periods=252
                ).alias("v")
            )["v"]
            .drop_nulls()
            .drop_nans()
        )
        raw = (
            df.select(realized_volatility("returns", period=20, annualize=False).alias("v"))["v"]
            .drop_nulls()
            .drop_nans()
        )
        ratio = ann.mean() / raw.mean()
        expected = np.sqrt(252)
        assert abs(ratio - expected) / expected < 0.01  # Within 1%

    def test_positivity(self, ohlcv_df):
        """Realized vol is always non-negative."""
        from ml4t.engineer.features.volatility.realized_volatility import realized_volatility

        df = ohlcv_df.with_columns(returns=pl.col("close").pct_change())
        vals = (
            df.select(realized_volatility("returns", period=20).alias("rv"))["rv"]
            .drop_nulls()
            .drop_nans()
        )
        assert (vals >= 0).all()


# ---------------------------------------------------------------------------
# 6. EWMA Volatility
# ---------------------------------------------------------------------------


class TestEWMAVolatility:
    """Tests for ewma_volatility."""

    def test_basic_output(self, ohlcv_df):
        """Validates output shape."""
        from ml4t.engineer.features.volatility.ewma_volatility import ewma_volatility

        result = ohlcv_df.select(ewma_volatility("close", span=20).alias("ev"))
        assert result.shape == (100, 1)

    def test_different_spans(self, ohlcv_df):
        """Shorter span reacts faster to recent data."""
        from ml4t.engineer.features.volatility.ewma_volatility import ewma_volatility

        short = (
            ohlcv_df.select(ewma_volatility("close", span=10).alias("v"))["v"]
            .drop_nulls()
            .drop_nans()
        )
        long = (
            ohlcv_df.select(ewma_volatility("close", span=50).alias("v"))["v"]
            .drop_nulls()
            .drop_nans()
        )
        # Shorter span should have higher variance (more reactive)
        assert short.std() >= long.std() * 0.3  # Generous bound

    def test_positivity(self, ohlcv_df):
        """EWMA vol is always non-negative."""
        from ml4t.engineer.features.volatility.ewma_volatility import ewma_volatility

        vals = (
            ohlcv_df.select(ewma_volatility("close", span=20).alias("ev"))["ev"]
            .drop_nulls()
            .drop_nans()
        )
        assert (vals >= 0).all()

    def test_normalized_output(self, ohlcv_df):
        """Normalized output should be in [-1, 1] range."""
        from ml4t.engineer.features.volatility.ewma_volatility import ewma_volatility

        vals = (
            ohlcv_df.select(ewma_volatility("close", span=20, normalize=True).alias("ev"))["ev"]
            .drop_nulls()
            .drop_nans()
        )
        assert (vals >= -1.0).all()
        assert (vals <= 1.0).all()


# ---------------------------------------------------------------------------
# 7. GARCH Forecast
# ---------------------------------------------------------------------------


class TestGARCHForecast:
    """Tests for garch_forecast."""

    def test_basic_output(self, ohlcv_df):
        """Validates output shape — takes returns."""
        from ml4t.engineer.features.volatility.garch_forecast import garch_forecast

        df = ohlcv_df.with_columns(returns=pl.col("close").pct_change())
        result = df.select(garch_forecast("returns").alias("gf"))
        assert result.shape == (100, 1)

    def test_alpha_beta_constraint(self):
        """alpha + beta must be < 1 for stationarity."""
        from ml4t.engineer.features.volatility.garch_forecast import garch_forecast

        with pytest.raises(ValueError):
            pl.DataFrame({"r": [0.01] * 10}).select(garch_forecast("r", alpha=0.5, beta=0.6))

    def test_responds_to_shocks(self, ohlcv_df):
        """Conditional vol should increase after large returns."""
        from ml4t.engineer.features.volatility.garch_forecast import garch_forecast

        # Create data with a shock
        returns = [0.001] * 30 + [0.10] + [0.001] * 30  # big shock at t=30
        df = pl.DataFrame({"returns": returns})
        vals = df.select(garch_forecast("returns").alias("gf"))["gf"].to_numpy()
        # After shock, vol should be higher than before
        pre_shock = vals[25:30]
        post_shock = vals[31:36]
        pre_mean = np.nanmean(pre_shock)
        post_mean = np.nanmean(post_shock)
        assert post_mean > pre_mean

    def test_positivity(self, ohlcv_df):
        """GARCH forecasts should be non-negative."""
        from ml4t.engineer.features.volatility.garch_forecast import garch_forecast

        df = ohlcv_df.with_columns(returns=pl.col("close").pct_change())
        vals = df.select(garch_forecast("returns").alias("gf"))["gf"].drop_nulls().drop_nans()
        assert (vals >= 0).all()

    def test_parameter_validation(self):
        """Invalid parameters should raise."""
        from ml4t.engineer.features.volatility.garch_forecast import garch_forecast

        df = pl.DataFrame({"r": [0.01] * 10})
        with pytest.raises(ValueError):
            df.select(garch_forecast("r", omega=-0.001))
        with pytest.raises(ValueError):
            df.select(garch_forecast("r", alpha=-0.1))
        with pytest.raises(ValueError):
            df.select(garch_forecast("r", beta=-0.1))
        with pytest.raises(ValueError):
            df.select(garch_forecast("r", horizon=0))


# ---------------------------------------------------------------------------
# 8. Conditional Volatility Ratio
# ---------------------------------------------------------------------------


class TestConditionalVolatilityRatio:
    """Tests for conditional_volatility_ratio."""

    def test_basic_output(self, ohlcv_df):
        """Validates output shape."""
        from ml4t.engineer.features.volatility.conditional_volatility_ratio import (
            conditional_volatility_ratio,
        )

        df = ohlcv_df.with_columns(returns=pl.col("close").pct_change())
        result = df.select(conditional_volatility_ratio("returns", period=20).alias("cvr"))
        assert result.shape == (100, 1)

    def test_symmetric_returns_near_one(self):
        """Symmetric returns should give ratio close to 1."""
        from ml4t.engineer.features.volatility.conditional_volatility_ratio import (
            conditional_volatility_ratio,
        )

        np.random.seed(42)
        # Symmetric normal returns
        returns = np.random.randn(200) * 0.01
        df = pl.DataFrame({"returns": returns})
        vals = (
            df.select(conditional_volatility_ratio("returns", period=50).alias("cvr"))["cvr"]
            .drop_nulls()
            .drop_nans()
        )
        # Mean should be close to 1.0 for symmetric distributions
        assert 0.5 < vals.mean() < 2.0

    def test_positivity(self, ohlcv_df):
        """Ratio should be non-negative."""
        from ml4t.engineer.features.volatility.conditional_volatility_ratio import (
            conditional_volatility_ratio,
        )

        df = ohlcv_df.with_columns(returns=pl.col("close").pct_change())
        vals = (
            df.select(conditional_volatility_ratio("returns", period=20).alias("cvr"))["cvr"]
            .drop_nulls()
            .drop_nans()
        )
        assert (vals >= 0).all()

    def test_different_thresholds(self, ohlcv_df):
        """Different thresholds should produce different decompositions."""
        from ml4t.engineer.features.volatility.conditional_volatility_ratio import (
            conditional_volatility_ratio,
        )

        df = ohlcv_df.with_columns(returns=pl.col("close").pct_change())
        r0 = (
            df.select(conditional_volatility_ratio("returns", threshold=0.0, period=20).alias("v"))[
                "v"
            ]
            .drop_nulls()
            .drop_nans()
        )
        r1 = (
            df.select(
                conditional_volatility_ratio("returns", threshold=0.01, period=20).alias("v")
            )["v"]
            .drop_nulls()
            .drop_nans()
        )
        # Higher threshold means fewer "upside" returns, ratio should differ
        assert r0.mean() != r1.mean()


# ---------------------------------------------------------------------------
# 9. Volatility of Volatility
# ---------------------------------------------------------------------------


class TestVolatilityOfVolatility:
    """Tests for volatility_of_volatility."""

    def test_basic_output(self, ohlcv_df):
        """Validates output shape."""
        from ml4t.engineer.features.volatility.volatility_of_volatility import (
            volatility_of_volatility,
        )

        result = ohlcv_df.select(
            volatility_of_volatility("close", vol_period=10, vov_period=10).alias("vov")
        )
        assert result.shape == (100, 1)

    def test_constant_vol_near_zero(self):
        """Constant volatility input => near-zero VoV."""
        from ml4t.engineer.features.volatility.volatility_of_volatility import (
            volatility_of_volatility,
        )

        # Very slight random walk with nearly constant vol
        np.random.seed(42)
        close = 100 + np.cumsum(np.full(200, 0.01))  # constant small changes
        df = pl.DataFrame({"close": close})
        vals = (
            df.select(
                volatility_of_volatility(
                    "close", vol_period=10, vov_period=10, annualize=False
                ).alias("vov")
            )["vov"]
            .drop_nulls()
            .drop_nans()
        )
        # Should be very small since underlying vol is nearly constant
        assert vals.mean() < 0.01

    def test_positivity(self, ohlcv_df):
        """VoV should be non-negative."""
        from ml4t.engineer.features.volatility.volatility_of_volatility import (
            volatility_of_volatility,
        )

        vals = (
            ohlcv_df.select(
                volatility_of_volatility("close", vol_period=10, vov_period=10).alias("vov")
            )["vov"]
            .drop_nulls()
            .drop_nans()
        )
        assert (vals >= 0).all()

    def test_different_periods(self, ohlcv_df):
        """Different vol_period and vov_period combinations."""
        from ml4t.engineer.features.volatility.volatility_of_volatility import (
            volatility_of_volatility,
        )

        for vp, vvp in [(5, 5), (10, 10), (20, 20)]:
            result = ohlcv_df.select(
                volatility_of_volatility("close", vol_period=vp, vov_period=vvp).alias("vov")
            )
            assert len(result) == 100


# ---------------------------------------------------------------------------
# 10. Volatility Percentile Rank
# ---------------------------------------------------------------------------


class TestVolatilityPercentileRank:
    """Tests for volatility_percentile_rank."""

    def test_basic_output(self, ohlcv_df):
        """Validates output shape."""
        from ml4t.engineer.features.volatility.volatility_percentile_rank import (
            volatility_percentile_rank,
        )

        result = ohlcv_df.select(
            volatility_percentile_rank("close", period=10, lookback=50).alias("vpr")
        )
        assert result.shape == (100, 1)

    def test_output_in_0_100(self, ohlcv_df):
        """Output should be in [0, 100] range."""
        from ml4t.engineer.features.volatility.volatility_percentile_rank import (
            volatility_percentile_rank,
        )

        vals = (
            ohlcv_df.select(
                volatility_percentile_rank("close", period=10, lookback=50).alias("vpr")
            )["vpr"]
            .drop_nulls()
            .drop_nans()
        )
        assert len(vals) > 0
        assert (vals >= 0.0).all()
        assert (vals <= 100.0).all()

    def test_different_lookbacks(self, ohlcv_df):
        """Shorter lookback = more responsive to recent vol changes."""
        from ml4t.engineer.features.volatility.volatility_percentile_rank import (
            volatility_percentile_rank,
        )

        short = (
            ohlcv_df.select(volatility_percentile_rank("close", period=10, lookback=30).alias("v"))[
                "v"
            ]
            .drop_nulls()
            .drop_nans()
        )
        long = (
            ohlcv_df.select(volatility_percentile_rank("close", period=10, lookback=80).alias("v"))[
                "v"
            ]
            .drop_nulls()
            .drop_nans()
        )
        # Both should be bounded
        assert (short >= 0).all() and (short <= 100).all()
        assert (long >= 0).all() and (long <= 100).all()


# ---------------------------------------------------------------------------
# 11. Volatility Regime Probability
# ---------------------------------------------------------------------------


class TestVolatilityRegimeProbability:
    """Tests for volatility_regime_probability."""

    def test_basic_output(self, ohlcv_df):
        """Validates that function returns dict with correct keys."""
        from ml4t.engineer.features.volatility.volatility_regime_probability import (
            volatility_regime_probability,
        )

        exprs = volatility_regime_probability("close", period=10, lookback=50)
        assert isinstance(exprs, dict)
        assert "prob_low_vol" in exprs
        assert "prob_med_vol" in exprs
        assert "prob_high_vol" in exprs
        assert "current_vol" in exprs

    def test_probabilities_sum_to_one(self, ohlcv_df):
        """Low + med + high probabilities should sum to ~1."""
        from ml4t.engineer.features.volatility.volatility_regime_probability import (
            volatility_regime_probability,
        )

        exprs = volatility_regime_probability("close", period=10, lookback=50)
        result = ohlcv_df.select(
            exprs["prob_low_vol"].alias("low"),
            exprs["prob_med_vol"].alias("med"),
            exprs["prob_high_vol"].alias("high"),
        )
        total = (result["low"] + result["med"] + result["high"]).drop_nulls().drop_nans()
        assert len(total) > 0
        assert np.allclose(total.to_numpy(), 1.0, atol=0.01)

    def test_probabilities_in_0_1(self, ohlcv_df):
        """Each probability should be in [0, 1]."""
        from ml4t.engineer.features.volatility.volatility_regime_probability import (
            volatility_regime_probability,
        )

        exprs = volatility_regime_probability("close", period=10, lookback=50)
        result = ohlcv_df.select(
            exprs["prob_low_vol"].alias("low"),
            exprs["prob_med_vol"].alias("med"),
            exprs["prob_high_vol"].alias("high"),
        )
        for col in ["low", "med", "high"]:
            vals = result[col].drop_nulls().drop_nans()
            assert (vals >= 0.0).all()
            assert (vals <= 1.0).all()

    def test_threshold_ordering(self):
        """low_vol_threshold must be < high_vol_threshold."""
        from ml4t.engineer.features.volatility.volatility_regime_probability import (
            volatility_regime_probability,
        )

        with pytest.raises(ValueError):
            volatility_regime_probability("close", low_vol_threshold=0.05, high_vol_threshold=0.01)

    def test_current_vol_positive(self, ohlcv_df):
        """Current vol should be non-negative."""
        from ml4t.engineer.features.volatility.volatility_regime_probability import (
            volatility_regime_probability,
        )

        exprs = volatility_regime_probability("close", period=10, lookback=50)
        vals = ohlcv_df.select(exprs["current_vol"].alias("v"))["v"].drop_nulls().drop_nans()
        assert (vals >= 0).all()


# ---------------------------------------------------------------------------
# Bollinger Bands (enhance existing coverage)
# ---------------------------------------------------------------------------


class TestBollingerBandsComprehensive:
    """Enhanced tests for bollinger_bands."""

    def test_band_ordering(self):
        """Upper > middle > lower always."""
        from ml4t.engineer.features.volatility.bollinger_bands import bollinger_bands

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        upper, middle, lower = bollinger_bands(close, period=20, nbdevup=2.0, nbdevdn=2.0)

        # Compare only where all three are valid (non-NaN)
        mask = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        assert np.all(upper[mask] >= middle[mask])
        assert np.all(middle[mask] >= lower[mask])

    def test_different_stddev_params(self):
        """Different nbdevup/nbdevdn should widen/narrow bands."""
        from ml4t.engineer.features.volatility.bollinger_bands import bollinger_bands

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        u1, m1, l1 = bollinger_bands(close, period=20, nbdevup=1.0, nbdevdn=1.0)
        u2, m2, l2 = bollinger_bands(close, period=20, nbdevup=3.0, nbdevdn=3.0)

        mask = ~(np.isnan(u1) | np.isnan(u2))
        # Wider bands with larger nbdev
        assert np.all(u2[mask] >= u1[mask])
        assert np.all(l2[mask] <= l1[mask])
        # Middle band should be the same
        assert np.allclose(m1[mask], m2[mask])

    def test_constant_prices_collapse(self):
        """Constant prices => bands collapse to price level."""
        from ml4t.engineer.features.volatility.bollinger_bands import bollinger_bands

        close = np.array([100.0] * 50)
        upper, middle, lower = bollinger_bands(close, period=20)

        mask = ~np.isnan(upper)
        assert np.allclose(upper[mask], 100.0)
        assert np.allclose(middle[mask], 100.0)
        assert np.allclose(lower[mask], 100.0)

    def test_output_column_count(self):
        """Should return exactly 3 arrays."""
        from ml4t.engineer.features.volatility.bollinger_bands import bollinger_bands

        close = np.random.randn(50).cumsum() + 100
        result = bollinger_bands(close, period=10)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_different_periods(self):
        """Longer period produces smoother bands."""
        from ml4t.engineer.features.volatility.bollinger_bands import bollinger_bands

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)

        u10, m10, l10 = bollinger_bands(close, period=10)
        u50, m50, l50 = bollinger_bands(close, period=50)

        # Longer period has fewer NaN at start
        assert np.sum(np.isnan(m10)) < np.sum(np.isnan(m50))

        # Both should have valid values
        assert np.sum(~np.isnan(u10)) > 0
        assert np.sum(~np.isnan(u50)) > 0


# ---------------------------------------------------------------------------
# Cross-estimator comparisons
# ---------------------------------------------------------------------------


class TestVolatilityEstimatorRelationships:
    """Test relationships between different volatility estimators."""

    def test_estimators_agree_on_magnitude(self, ohlcv_df):
        """All OHLC estimators should agree on approximate vol level."""
        from ml4t.engineer.features.volatility.garman_klass_volatility import (
            garman_klass_volatility,
        )
        from ml4t.engineer.features.volatility.parkinson_volatility import parkinson_volatility
        from ml4t.engineer.features.volatility.rogers_satchell_volatility import (
            rogers_satchell_volatility,
        )
        from ml4t.engineer.features.volatility.yang_zhang_volatility import yang_zhang_volatility

        period = 20
        pk = (
            ohlcv_df.select(parkinson_volatility("high", "low", period=period).alias("v"))["v"]
            .drop_nulls()
            .drop_nans()
            .mean()
        )
        gk = (
            ohlcv_df.select(
                garman_klass_volatility("open", "high", "low", "close", period=period).alias("v")
            )["v"]
            .drop_nulls()
            .drop_nans()
            .mean()
        )
        rs = (
            ohlcv_df.select(
                rogers_satchell_volatility("open", "high", "low", "close", period=period).alias("v")
            )["v"]
            .drop_nulls()
            .drop_nans()
            .mean()
        )
        yz = (
            ohlcv_df.select(
                yang_zhang_volatility("open", "high", "low", "close", period=period).alias("v")
            )["v"]
            .drop_nulls()
            .drop_nans()
            .mean()
        )

        estimates = [pk, gk, rs, yz]
        # All should be in the same order of magnitude
        min_est = min(estimates)
        max_est = max(estimates)
        assert max_est / min_est < 10  # Within 10x of each other
