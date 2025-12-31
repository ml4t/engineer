"""Tests for risk management features."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features import risk


class TestRiskFeatures:
    """Test suite for risk management features."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 500

        # Generate realistic return data with different characteristics
        # Normal returns
        normal_returns = np.random.normal(0.001, 0.02, n)

        # Fat-tailed returns (t-distribution)
        from scipy import stats

        fat_tail_returns = stats.t.rvs(df=4, loc=0.001, scale=0.02, size=n)

        # Skewed returns
        skewed_returns = stats.skewnorm.rvs(a=-2, loc=0.001, scale=0.02, size=n)

        # Create price series from returns
        prices = 100 * np.cumprod(1 + normal_returns)

        # Create test DataFrame
        self.df = pl.DataFrame(
            {
                "datetime": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
                "normal_returns": normal_returns,
                "fat_tail_returns": fat_tail_returns,
                "skewed_returns": skewed_returns,
                "prices": prices,
            },
        )

        # Create drawdown scenario
        dd_prices = np.ones(100) * 100
        dd_prices[20:40] = 100 * np.exp(np.linspace(0, -0.3, 20))  # 30% drawdown
        dd_prices[40:60] = dd_prices[39]  # Flat bottom
        dd_prices[60:80] = dd_prices[59] * np.exp(np.linspace(0, 0.3, 20))  # Recovery
        dd_prices[80:] = 100  # Back to peak

        self.dd_df = pl.DataFrame(
            {
                "prices": dd_prices,
                "returns": np.concatenate([[0], np.diff(dd_prices) / dd_prices[:-1]]),
            },
        )

    def test_value_at_risk_historical(self):
        """Test historical VaR calculation."""
        result = self.df.with_columns(
            [
                risk.value_at_risk(
                    "normal_returns",
                    confidence_level=0.95,
                    window=100,
                    method="historical",
                ).alias("var_95"),
                risk.value_at_risk(
                    "normal_returns",
                    confidence_level=0.99,
                    window=100,
                    method="historical",
                ).alias("var_99"),
            ],
        )

        # Check that values exist and are negative (losses)
        var_95 = result["var_95"].drop_nulls()
        var_99 = result["var_99"].drop_nulls()

        assert len(var_95) > 0
        assert len(var_99) > 0
        assert var_95.mean() < 0  # VaR should be negative
        assert var_99.mean() < var_95.mean()  # 99% VaR should be more extreme

    def test_value_at_risk_parametric(self):
        """Test parametric VaR calculation."""
        result = self.df.with_columns(
            [
                risk.value_at_risk(
                    "normal_returns",
                    confidence_level=0.95,
                    window=100,
                    method="parametric",
                ).alias("var_param"),
            ],
        )

        var_param = result["var_param"].drop_nulls()
        assert len(var_param) > 0

        # For normal returns, parametric should be close to -1.645 * std
        expected_var = -1.645 * self.df["normal_returns"].std()
        assert abs(var_param.mean() - expected_var) < 0.01

    def test_value_at_risk_cornish_fisher(self):
        """Test Cornish-Fisher VaR for non-normal distributions."""
        result = self.df.with_columns(
            [
                risk.value_at_risk(
                    "skewed_returns",
                    confidence_level=0.95,
                    window=100,
                    method="cornish_fisher",
                ).alias("var_cf"),
                risk.value_at_risk(
                    "skewed_returns",
                    confidence_level=0.95,
                    window=100,
                    method="parametric",
                ).alias("var_param"),
            ],
        )

        var_cf = result["var_cf"].drop_nulls()
        var_param = result["var_param"].drop_nulls()

        assert len(var_cf) > 0
        # CF adjustment should differ from parametric for skewed data
        assert abs(var_cf.mean() - var_param.mean()) > 0.001

    def test_conditional_value_at_risk(self):
        """Test CVaR calculation."""
        result = self.df.with_columns(
            [
                risk.value_at_risk(
                    "normal_returns",
                    confidence_level=0.95,
                    window=100,
                ).alias("var"),
                risk.conditional_value_at_risk(
                    "normal_returns",
                    confidence_level=0.95,
                    window=100,
                ).alias("cvar"),
            ],
        )

        var = result["var"].drop_nulls()
        cvar = result["cvar"].drop_nulls()

        assert len(cvar) > 0
        # CVaR should be more extreme than VaR
        assert cvar.mean() < var.mean()

    def test_maximum_drawdown(self):
        """Test maximum drawdown calculation."""
        dd_stats = risk.maximum_drawdown("prices")

        # Apply to drawdown scenario
        result = self.dd_df.with_columns(
            [
                dd_stats["max_drawdown"].alias("max_dd"),
                dd_stats["max_duration"].alias("max_duration"),
                dd_stats["current_drawdown"].alias("current_dd"),
                dd_stats["time_underwater"].alias("time_underwater"),
            ],
        )

        # Check maximum drawdown is approximately -26% (1 - exp(-0.3))
        max_dd = result["max_dd"].min()
        assert -0.27 < max_dd < -0.25

        # Check duration (only for rolling window, not expanding)
        if result["max_duration"].dtype != pl.Null:
            max_duration = result["max_duration"].max()
            assert max_duration > 20  # Should capture the drawdown period

        # Check recovery
        final_dd = result["current_dd"][-1]
        assert abs(final_dd) < 0.01  # Should be back to peak

    def test_maximum_drawdown_rolling(self):
        """Test rolling maximum drawdown."""
        dd_stats = risk.maximum_drawdown("prices", window=50)

        result = self.df.with_columns(
            [
                dd_stats["max_drawdown"].alias("rolling_dd"),
            ],
        )

        rolling_dd = result["rolling_dd"].drop_nulls()
        assert len(rolling_dd) > 0
        assert rolling_dd.max() <= 0  # Drawdowns are negative

    def test_downside_deviation(self):
        """Test downside deviation calculation."""
        result = self.df.with_columns(
            [
                risk.downside_deviation(
                    "normal_returns",
                    target_return=0.0,
                    window=100,
                ).alias("downside_dev"),
                pl.col("normal_returns").rolling_std(100).alias("total_std"),
            ],
        )

        downside_dev = result["downside_dev"].drop_nulls()
        total_std = result["total_std"].drop_nulls()

        assert len(downside_dev) > 0
        # Downside deviation should be less than total std
        assert downside_dev.mean() < total_std.mean()

    def test_tail_ratio(self):
        """Test tail ratio calculation."""
        result = self.df.with_columns(
            [
                risk.tail_ratio(
                    "normal_returns",
                    confidence_level=0.95,
                    window=100,
                ).alias("tail_ratio"),
                risk.tail_ratio(
                    "fat_tail_returns",
                    confidence_level=0.95,
                    window=100,
                ).alias("tail_ratio_fat"),
            ],
        )

        tail_ratio = result["tail_ratio"].drop_nulls()
        tail_ratio_fat = result["tail_ratio_fat"].drop_nulls()

        assert len(tail_ratio) > 0
        # For symmetric distribution, should be close to 1
        assert 0.8 < tail_ratio.mean() < 1.4  # Allow some variance in finite samples

        # Fat tails might have different ratio
        assert len(tail_ratio_fat) > 0

    def test_higher_moments(self):
        """Test higher moment calculations."""
        moments = risk.higher_moments("skewed_returns", window=100)

        result = self.df.with_columns(
            [
                moments["skewness"].alias("skew"),
                moments["kurtosis"].alias("kurt"),
                moments["hyperskewness"].alias("hyper_skew"),
                moments["hyperkurtosis"].alias("hyper_kurt"),
            ],
        )

        skew = result["skew"].drop_nulls()
        kurt = result["kurt"].drop_nulls()

        assert len(skew) > 0
        assert len(kurt) > 0

        # Skewed returns should have negative skewness
        assert skew.mean() < -0.1

        # Check kurtosis exists
        assert kurt.mean() != 0

    def test_risk_adjusted_returns(self):
        """Test risk-adjusted return metrics."""
        ratios = risk.risk_adjusted_returns(
            "normal_returns",
            risk_free_rate=0.02,
            window=100,
        )

        result = self.df.with_columns(
            [
                ratios["sharpe_ratio"].alias("sharpe"),
                ratios["sortino_ratio"].alias("sortino"),
                ratios["calmar_ratio"].alias("calmar"),
                ratios["omega_ratio"].alias("omega"),
            ],
        )

        sharpe = result["sharpe"].drop_nulls()
        sortino = result["sortino"].drop_nulls()
        calmar = result["calmar"].drop_nulls()
        omega = result["omega"].drop_nulls()

        # All ratios should exist
        assert len(sharpe) > 0
        assert len(sortino) > 0
        assert len(calmar) > 0
        assert len(omega) > 0

        # Sortino should typically be higher than Sharpe
        assert sortino.mean() > sharpe.mean()

        # Omega should be positive for positive returns
        assert omega.mean() > 0

    def test_ulcer_index(self):
        """Test Ulcer Index calculation."""
        result = self.dd_df.with_columns(
            [
                risk.ulcer_index("prices", window=50).alias("ulcer"),
            ],
        )

        ulcer = result["ulcer"].drop_nulls()
        assert len(ulcer) > 0

        # Should capture drawdown volatility
        assert ulcer.max() > 5  # Significant during drawdown

    def test_information_ratio(self):
        """Test Information Ratio calculation."""
        # Create benchmark returns
        benchmark_returns = self.df["normal_returns"] * 0.8 + np.random.normal(
            0,
            0.005,
            len(self.df),
        )

        result = self.df.with_columns(
            [
                pl.Series("benchmark_returns", benchmark_returns),
            ],
        ).with_columns(
            [
                risk.information_ratio(
                    "normal_returns",
                    "benchmark_returns",
                    window=100,
                ).alias("info_ratio"),
            ],
        )

        info_ratio = result["info_ratio"].drop_nulls()
        assert len(info_ratio) > 0

        # Should be positive if outperforming
        assert info_ratio.mean() > 0

    def test_input_validation(self):
        """Test input validation."""
        # Invalid confidence level
        with pytest.raises(ValueError, match="confidence_level"):
            self.df.with_columns(
                [risk.value_at_risk("normal_returns", confidence_level=1.5)],
            )

        # Invalid window
        with pytest.raises(ValueError, match="window"):
            self.df.with_columns([risk.value_at_risk("normal_returns", window=0)])

        # Invalid method
        with pytest.raises(ValueError, match="method"):
            self.df.with_columns(
                [risk.value_at_risk("normal_returns", method="invalid")],
            )

    def test_edge_cases(self):
        """Test edge cases."""
        # Small dataset
        small_df = pl.DataFrame({"returns": [0.01] * 20})  # Minimum window size
        result = small_df.with_columns(
            [
                risk.value_at_risk("returns", window=20).alias("var"),
            ],
        )
        assert len(result) == 20

        # All positive returns
        pos_df = pl.DataFrame({"returns": np.abs(np.random.normal(0.01, 0.02, 100))})
        result = pos_df.with_columns(
            [
                risk.value_at_risk("returns", window=20).alias("var"),
            ],
        )
        var = result["var"].drop_nulls()
        assert var.min() >= 0  # No losses

        # Constant returns
        const_df = pl.DataFrame({"returns": [0.01] * 100})
        result = const_df.with_columns(
            [
                risk.downside_deviation("returns", window=20).alias("dd"),
            ],
        )
        dd = result["dd"].drop_nulls()
        assert dd.max() < 1e-10  # Should be near zero

    def test_performance(self):
        """Test performance with larger dataset."""
        import time

        # Create larger dataset
        n = 10000
        large_df = pl.DataFrame(
            {
                "returns": np.random.normal(0.001, 0.02, n),
                "prices": 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n)),
            },
        )

        # Time VaR calculation
        start = time.perf_counter()
        _ = large_df.with_columns(
            [
                risk.value_at_risk("returns", window=252).alias("var"),
                risk.conditional_value_at_risk("returns", window=252).alias("cvar"),
            ],
        )
        end = time.perf_counter()

        processing_time = end - start
        rows_per_second = n / processing_time

        print(f"\nRisk calculations performance: {rows_per_second:,.0f} rows/second")
        assert (
            rows_per_second > 5000
        )  # Should process at least 5K rows/second (conservative for CI)

    def test_integration_with_other_features(self):
        """Test integration with other feature modules."""
        # This would test combining risk features with other modules
        # For now, just ensure we can chain multiple risk calculations
        result = self.df.with_columns(
            [
                risk.value_at_risk("normal_returns", window=50).alias("var_50"),
                risk.value_at_risk("normal_returns", window=100).alias("var_100"),
                risk.downside_deviation("normal_returns", window=50).alias("dd_50"),
            ],
        )

        assert "var_50" in result.columns
        assert "var_100" in result.columns
        assert "dd_50" in result.columns

        # All should have some non-null values
        assert result["var_50"].drop_nulls().len() > 0
        assert result["var_100"].drop_nulls().len() > 0
        assert result["dd_50"].drop_nulls().len() > 0
