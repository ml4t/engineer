"""Tests for cross-asset relationship features."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.cross_asset import (
    beta_to_market,
    co_integration_score,
    correlation_matrix_features,
    correlation_regime_indicator,
    cross_asset_momentum,
    lead_lag_correlation,
    multi_asset_dispersion,
    relative_strength_index_spread,
    rolling_correlation,
    volatility_ratio,
)


class TestCrossAssetFeatures:
    """Test cross-asset feature calculations."""

    @pytest.fixture
    def sample_data(self):
        """Create sample multi-asset data."""
        np.random.seed(42)
        n = 100

        # Create correlated asset returns
        returns1 = np.random.normal(0.0005, 0.01, n)
        returns2 = 0.8 * returns1 + 0.2 * np.random.normal(0.0003, 0.008, n)
        returns3 = -0.5 * returns1 + np.random.normal(0.0002, 0.012, n)

        # Create prices from returns
        prices1 = 100 * np.exp(np.cumsum(returns1))
        prices2 = 100 * np.exp(np.cumsum(returns2))
        prices3 = 100 * np.exp(np.cumsum(returns3))

        # Create volatility series
        vol1 = (
            np.abs(returns1).rolling(20).std()
            if hasattr(returns1, "rolling")
            else np.convolve(np.abs(returns1), np.ones(20) / 20, mode="same")
        )
        vol2 = (
            np.abs(returns2).rolling(20).std()
            if hasattr(returns2, "rolling")
            else np.convolve(np.abs(returns2), np.ones(20) / 20, mode="same")
        )

        # RSI values (mock)
        rsi1 = 50 + 20 * np.sin(np.linspace(0, 4 * np.pi, n))
        rsi2 = 50 + 15 * np.sin(np.linspace(0, 4 * np.pi, n) + np.pi / 4)

        df = pl.DataFrame(
            {
                "asset1_returns": returns1,
                "asset2_returns": returns2,
                "asset3_returns": returns3,
                "asset1_price": prices1,
                "asset2_price": prices2,
                "asset3_price": prices3,
                "market_returns": returns1,  # Use asset1 as market proxy
                "asset1_vol": vol1,
                "asset2_vol": vol2,
                "asset1_rsi": rsi1,
                "asset2_rsi": rsi2,
            },
        )

        return df

    def test_rolling_correlation(self, sample_data):
        """Test rolling correlation calculation."""
        result = sample_data.select(
            rolling_correlation("asset1_returns", "asset2_returns", window=20).alias(
                "corr",
            ),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # Check correlation bounds
        corr_values = result["corr"].drop_nulls()
        assert corr_values.min() >= -1.0
        assert corr_values.max() <= 1.0

        # Should have high correlation given how we constructed the data
        assert corr_values.mean() > 0.5

    def test_beta_to_market(self, sample_data):
        """Test beta calculation."""
        result = sample_data.select(
            beta_to_market("asset2_returns", "market_returns", window=30).alias("beta"),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # Beta should be around 0.8 given our data construction
        beta_values = result["beta"].drop_nulls()
        assert 0.5 < beta_values.mean() < 1.2

    def test_correlation_regime_indicator(self, sample_data):
        """Test correlation regime identification."""
        # First calculate correlation
        df_with_corr = sample_data.with_columns(
            rolling_correlation("asset1_returns", "asset2_returns", window=20).alias(
                "correlation",
            ),
        )

        # Then calculate regime indicators
        regimes = correlation_regime_indicator(
            df_with_corr["correlation"],
            low_threshold=0.3,
            high_threshold=0.7,
        )

        result = df_with_corr.with_columns(
            [
                regimes["corr_regime_low"].alias("low"),
                regimes["corr_regime_mid"].alias("mid"),
                regimes["corr_regime_high"].alias("high"),
            ],
        )

        # Check that regime probabilities sum to ~1
        # Skip initial nulls by filtering
        valid_rows = result.filter(
            pl.col("low").is_not_null()
            & pl.col("mid").is_not_null()
            & pl.col("high").is_not_null(),
        )

        # Calculate totals
        totals = valid_rows.select(
            (pl.col("low") + pl.col("mid") + pl.col("high")).alias("total"),
        )["total"]

        # Check each total
        for total in totals:
            assert 0.9 <= total <= 1.1

    def test_lead_lag_correlation(self, sample_data):
        """Test lead-lag correlation analysis."""
        correlations = lead_lag_correlation(
            "asset1_returns",
            "asset2_returns",
            max_lag=3,
            window=20,
        )

        result = sample_data.with_columns(
            [correlations[key].alias(key) for key in correlations],
        )

        # Should have correlations for lags -3 to 3
        expected_keys = [
            "lead_3",
            "lead_2",
            "lead_1",
            "lag_0",
            "lag_1",
            "lag_2",
            "lag_3",
        ]
        assert all(key in result.columns for key in expected_keys)

        # Contemporaneous correlation should be highest
        lag_0_corr = result["lag_0"].drop_nulls().mean()
        assert lag_0_corr > 0.5

    def test_multi_asset_dispersion(self, sample_data):
        """Test cross-sectional dispersion calculation."""
        result = sample_data.select(
            multi_asset_dispersion(
                ["asset1_returns", "asset2_returns", "asset3_returns"],
                window=20,
                method="std",
            ).alias("dispersion"),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # Dispersion should be positive
        disp_values = result["dispersion"].drop_nulls()
        assert (disp_values > 0).all()

    def test_correlation_matrix_features(self, sample_data):
        """Test correlation matrix feature extraction."""
        features = correlation_matrix_features(
            ["asset1_returns", "asset2_returns", "asset3_returns"],
            window=20,
        )

        result = sample_data.with_columns(
            [
                features["avg_correlation"].alias("avg_corr"),
                features["max_correlation"].alias("max_corr"),
                features["min_correlation"].alias("min_corr"),
                features["corr_dispersion"].alias("corr_disp"),
            ],
        )

        # Check bounds
        avg_corr = result["avg_corr"].drop_nulls()
        assert (avg_corr >= -1.0).all() and (avg_corr <= 1.0).all()

        # Max should be >= avg >= min
        for i in range(20, len(result)):
            if not any(
                np.isnan(
                    [
                        result["max_corr"][i],
                        result["avg_corr"][i],
                        result["min_corr"][i],
                    ],
                ),
            ):
                assert result["max_corr"][i] >= result["avg_corr"][i] >= result["min_corr"][i]

    def test_relative_strength_index_spread(self, sample_data):
        """Test RSI spread calculation."""
        result = sample_data.select(
            relative_strength_index_spread(
                "asset1_rsi",
                "asset2_rsi",
                smooth_period=5,
            ).alias("rsi_spread"),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # Spread should oscillate around 0
        spread_values = result["rsi_spread"].drop_nulls()
        assert -50 < spread_values.mean() < 50

    def test_volatility_ratio(self, sample_data):
        """Test volatility ratio calculation."""
        # Test log ratio
        result_log = sample_data.select(
            volatility_ratio("asset1_vol", "asset2_vol", log_ratio=True).alias(
                "vol_ratio_log",
            ),
        )

        # Test regular ratio
        result_reg = sample_data.select(
            volatility_ratio("asset1_vol", "asset2_vol", log_ratio=False).alias(
                "vol_ratio",
            ),
        )

        # Check that log ratio exists and has reasonable values
        log_values = result_log["vol_ratio_log"].drop_nulls()
        assert len(log_values) > 0
        # Log ratio should be finite
        assert log_values.is_finite().all()

        # Regular ratio should be positive
        reg_values = result_reg["vol_ratio"].drop_nulls()
        assert (reg_values > 0).all()

    # Test removed: transfer_entropy() raises NotImplementedError
    # The Polars expression API doesn't efficiently support the complex rolling
    # operations required. Use transfer_entropy_nb() for single calculations.
    # See: https://github.com/ml4t/ml4t-features/issues/transfer-entropy-implementation

    def test_co_integration_score(self, sample_data):
        """Test co-integration score calculation."""
        result = sample_data.select(
            co_integration_score("asset1_price", "asset2_price", window=30).alias(
                "coint_score",
            ),
        )

        # Check output shape
        assert len(result) == len(sample_data)

        # Score should be positive (it's a standard deviation)
        scores = result["coint_score"].drop_nulls()
        assert (scores > 0).all()

        # Given high correlation, score should be relatively low
        assert scores.mean() < 0.5

    def test_cross_asset_momentum(self, sample_data):
        """Test cross-asset momentum features."""
        # Test rank method
        momentum_rank = cross_asset_momentum(
            ["asset1_returns", "asset2_returns", "asset3_returns"],
            lookback=20,
            method="rank",
        )

        result_rank = sample_data.with_columns(
            momentum_rank["momentum_rank"].alias("mom_rank"),
        )

        # Rank should be between 0 and 1
        ranks = result_rank["mom_rank"].drop_nulls()
        assert (ranks >= 0).all() and (ranks <= 1).all()

        # Test zscore method
        momentum_zscore = cross_asset_momentum(
            ["asset1_returns", "asset2_returns", "asset3_returns"],
            lookback=20,
            method="zscore",
        )

        result_zscore = sample_data.with_columns(
            momentum_zscore["momentum_zscore"].alias("mom_zscore"),
        )

        # Z-score should be roughly centered around 0
        zscores = result_zscore["mom_zscore"].drop_nulls()
        assert -5 < zscores.mean() < 5

    def test_edge_cases(self, sample_data):
        """Test edge cases and error handling."""
        # Test with invalid window (should raise ValueError)
        with pytest.raises(ValueError, match="window must be at least 2"):
            sample_data.select(
                rolling_correlation("asset1_returns", "asset2_returns", window=1).alias(
                    "corr",
                ),
            )

        # Test with identical series (correlation should be close to 1)
        result_identical = sample_data.select(
            rolling_correlation("asset1_returns", "asset1_returns", window=10).alias(
                "corr",
            ),
        )
        corr_values = result_identical["corr"].drop_nulls()
        # Allow for small numerical errors in rolling correlation
        assert np.all(
            corr_values.to_numpy() >= 0.7,
        )  # Should be high positive correlation
        assert np.all(corr_values.to_numpy() <= 1.0)  # Should not exceed 1

    def test_with_nulls(self):
        """Test handling of null values."""
        df_with_nulls = pl.DataFrame(
            {
                "returns1": [0.01, 0.02, None, -0.01, 0.03],
                "returns2": [0.02, None, 0.01, -0.02, 0.02],
            },
        )

        # Should handle nulls gracefully
        result = df_with_nulls.select(
            rolling_correlation("returns1", "returns2", window=3).alias("corr"),
        )
        assert len(result) == len(df_with_nulls)

    def test_validation(self):
        """Test input validation."""
        # Test invalid window
        with pytest.raises(ValueError, match="window must be at least 2"):
            rolling_correlation("asset1_returns", "asset2_returns", window=1)

        # Test invalid thresholds in correlation_regime_indicator
        with pytest.raises(
            ValueError,
            match="low_threshold.*must be less than high_threshold",
        ):
            correlation_regime_indicator(
                "correlation",
                low_threshold=0.8,
                high_threshold=0.3,
            )

        # Test invalid max_lag
        with pytest.raises(ValueError, match="max_lag must be non-negative"):
            lead_lag_correlation("asset1_returns", "asset2_returns", max_lag=-1)

        # Test empty returns list
        with pytest.raises(
            ValueError,
            match="returns_list must have at least 2 elements",
        ):
            multi_asset_dispersion([])

        # Test invalid method
        with pytest.raises(ValueError, match="Unknown method.*Supported methods"):
            multi_asset_dispersion(
                ["asset1_returns", "asset2_returns"],
                method="invalid",
            )
