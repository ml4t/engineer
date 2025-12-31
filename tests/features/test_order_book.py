"""
Tests for order book microstructure features.

Tests cover:
- Basic functionality for bid_ask_imbalance, book_depth_ratio, weighted_mid_price
- Edge cases (zero depth, NaN handling)
- Parameter validation
- Output ranges and structure
- Formula correctness
"""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.microstructure import (
    bid_ask_imbalance,
    book_depth_ratio,
    weighted_mid_price,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_quote_data():
    """Sample quote data with bid/ask prices and sizes."""
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 9, 30),
                end=pl.datetime(2024, 1, 1, 16, 0),
                interval="5m",
                eager=True,
            )[:20],
            "bid_price": [
                100.00,
                100.01,
                100.02,
                100.01,
                100.03,
                100.02,
                100.04,
                100.03,
                100.05,
                100.04,
                100.06,
                100.05,
                100.07,
                100.06,
                100.08,
                100.07,
                100.09,
                100.08,
                100.10,
                100.09,
            ],
            "ask_price": [
                100.02,
                100.03,
                100.04,
                100.03,
                100.05,
                100.04,
                100.06,
                100.05,
                100.07,
                100.06,
                100.08,
                100.07,
                100.09,
                100.08,
                100.10,
                100.09,
                100.11,
                100.10,
                100.12,
                100.11,
            ],
            "bid_size": [
                1000,
                1200,
                800,
                1500,
                1100,
                900,
                1400,
                1000,
                1600,
                1200,
                1000,
                1300,
                900,
                1500,
                1100,
                800,
                1400,
                1000,
                1700,
                1200,
            ],
            "ask_size": [
                1000,
                1000,
                1200,
                500,
                1100,
                1100,
                600,
                1000,
                400,
                1200,
                1000,
                700,
                1100,
                500,
                900,
                1200,
                600,
                1000,
                300,
                800,
            ],
        }
    )


@pytest.fixture
def balanced_book_data():
    """Data with perfectly balanced order book."""
    return pl.DataFrame(
        {
            "bid_price": [100.0, 100.0, 100.0],
            "ask_price": [100.02, 100.02, 100.02],
            "bid_size": [1000, 1000, 1000],
            "ask_size": [1000, 1000, 1000],
        }
    )


@pytest.fixture
def imbalanced_book_data():
    """Data with heavily imbalanced order book."""
    return pl.DataFrame(
        {
            "bid_price": [100.0, 100.0, 100.0],
            "ask_price": [100.02, 100.02, 100.02],
            "bid_size": [2000, 0, 1000],  # bid-heavy, zero bid, normal
            "ask_size": [500, 1000, 1000],  # ask-light, normal, normal
        }
    )


@pytest.fixture
def edge_case_data():
    """Data with edge cases: zeros, very small values."""
    return pl.DataFrame(
        {
            "bid_price": [100.0, 100.0, 100.0, 100.0],
            "ask_price": [100.02, 100.02, 100.02, 100.02],
            "bid_size": [0, 0, 1000, 1],
            "ask_size": [0, 1000, 0, 1000000],
        }
    )


# =============================================================================
# Tests for bid_ask_imbalance
# =============================================================================


class TestBidAskImbalance:
    """Tests for bid_ask_imbalance function."""

    def test_basic_imbalance_calculation(self, sample_quote_data):
        """Test basic imbalance calculation returns correct values."""
        result = sample_quote_data.with_columns(
            bid_ask_imbalance("bid_size", "ask_size").alias("imbalance")
        )

        # Check output column exists
        assert "imbalance" in result.columns

        # Check no nulls (all inputs are valid)
        assert result["imbalance"].null_count() == 0

        # Check range is [-1, 1]
        assert result["imbalance"].min() >= -1.0
        assert result["imbalance"].max() <= 1.0

    def test_balanced_book_returns_zero(self, balanced_book_data):
        """Test that equal bid/ask sizes return zero imbalance."""
        result = balanced_book_data.with_columns(
            bid_ask_imbalance("bid_size", "ask_size").alias("imbalance")
        )

        # All values should be 0 (balanced)
        assert np.allclose(result["imbalance"].to_numpy(), 0.0)

    def test_bid_heavy_returns_positive(self, imbalanced_book_data):
        """Test that bid-heavy book returns positive imbalance."""
        result = imbalanced_book_data.with_columns(
            bid_ask_imbalance("bid_size", "ask_size").alias("imbalance")
        )

        # First row: bid_size=2000, ask_size=500 -> positive imbalance
        first_imbalance = result["imbalance"][0]
        expected = (2000 - 500) / (2000 + 500)  # 0.6
        assert np.isclose(first_imbalance, expected)

    def test_ask_heavy_returns_negative(self):
        """Test that ask-heavy book returns negative imbalance."""
        df = pl.DataFrame({"bid_size": [500], "ask_size": [2000]})
        result = df.with_columns(bid_ask_imbalance("bid_size", "ask_size").alias("imbalance"))

        expected = (500 - 2000) / (500 + 2000)  # -0.6
        assert np.isclose(result["imbalance"][0], expected)

    def test_zero_total_depth_returns_zero(self, edge_case_data):
        """Test that zero total depth returns 0 (not NaN/inf)."""
        result = edge_case_data.with_columns(
            bid_ask_imbalance("bid_size", "ask_size").alias("imbalance")
        )

        # First row has bid_size=0, ask_size=0 -> should return 0
        assert result["imbalance"][0] == 0.0

    def test_with_rolling_period(self, sample_quote_data):
        """Test imbalance with rolling average smoothing."""
        result = sample_quote_data.with_columns(
            bid_ask_imbalance("bid_size", "ask_size", period=5).alias("imbalance_5")
        )

        # First 4 values should be null (rolling window not filled)
        assert result["imbalance_5"][:4].null_count() == 4

        # Remaining values should be valid
        assert result["imbalance_5"][4:].null_count() == 0

    def test_formula_correctness(self):
        """Test formula: (bid - ask) / (bid + ask)."""
        df = pl.DataFrame({"bid_size": [1500], "ask_size": [500]})
        result = df.with_columns(bid_ask_imbalance("bid_size", "ask_size").alias("imbalance"))

        # Manual calculation: (1500 - 500) / (1500 + 500) = 1000 / 2000 = 0.5
        assert np.isclose(result["imbalance"][0], 0.5)


# =============================================================================
# Tests for book_depth_ratio
# =============================================================================


class TestBookDepthRatio:
    """Tests for book_depth_ratio function."""

    def test_basic_depth_ratio(self, sample_quote_data):
        """Test basic depth ratio calculation."""
        result = sample_quote_data.with_columns(
            book_depth_ratio("bid_size", "ask_size").alias("depth_ratio")
        )

        # Check output exists
        assert "depth_ratio" in result.columns

        # Check range is [0, 1]
        assert result["depth_ratio"].min() >= 0.0
        assert result["depth_ratio"].max() <= 1.0

    def test_balanced_returns_half(self, balanced_book_data):
        """Test that equal depth returns 0.5."""
        result = balanced_book_data.with_columns(
            book_depth_ratio("bid_size", "ask_size").alias("depth_ratio")
        )

        assert np.allclose(result["depth_ratio"].to_numpy(), 0.5)

    def test_all_bid_returns_one(self):
        """Test that all-bid depth returns 1.0."""
        df = pl.DataFrame({"bid_size": [1000], "ask_size": [0]})
        result = df.with_columns(book_depth_ratio("bid_size", "ask_size").alias("depth_ratio"))

        assert result["depth_ratio"][0] == 1.0

    def test_all_ask_returns_zero(self):
        """Test that all-ask depth returns 0.0."""
        df = pl.DataFrame({"bid_size": [0], "ask_size": [1000]})
        result = df.with_columns(book_depth_ratio("bid_size", "ask_size").alias("depth_ratio"))

        assert result["depth_ratio"][0] == 0.0

    def test_zero_total_depth_returns_half(self, edge_case_data):
        """Test that zero total depth returns 0.5 (balanced default)."""
        result = edge_case_data.with_columns(
            book_depth_ratio("bid_size", "ask_size").alias("depth_ratio")
        )

        # First row has both sizes = 0, should return 0.5
        assert result["depth_ratio"][0] == 0.5

    def test_formula_correctness(self):
        """Test formula: bid / (bid + ask)."""
        df = pl.DataFrame({"bid_size": [3000], "ask_size": [1000]})
        result = df.with_columns(book_depth_ratio("bid_size", "ask_size").alias("depth_ratio"))

        # 3000 / (3000 + 1000) = 0.75
        assert np.isclose(result["depth_ratio"][0], 0.75)


# =============================================================================
# Tests for weighted_mid_price
# =============================================================================


class TestWeightedMidPrice:
    """Tests for weighted_mid_price function."""

    def test_basic_weighted_mid(self, sample_quote_data):
        """Test basic weighted mid calculation."""
        result = sample_quote_data.with_columns(
            weighted_mid_price("bid_price", "ask_price", "bid_size", "ask_size").alias("wmid")
        )

        # Check output exists
        assert "wmid" in result.columns

        # Check weighted mid is between bid and ask
        assert (result["wmid"] >= result["bid_price"]).all()
        assert (result["wmid"] <= result["ask_price"]).all()

    def test_balanced_equals_simple_mid(self, balanced_book_data):
        """Test that balanced book gives simple midpoint."""
        result = balanced_book_data.with_columns(
            weighted_mid_price("bid_price", "ask_price", "bid_size", "ask_size").alias("wmid")
        )

        simple_mid = (balanced_book_data["bid_price"] + balanced_book_data["ask_price"]) / 2
        assert np.allclose(result["wmid"].to_numpy(), simple_mid.to_numpy())

    def test_bid_heavy_closer_to_ask(self):
        """Test that bid-heavy book gives weighted mid closer to ask."""
        df = pl.DataFrame(
            {
                "bid_price": [100.0],
                "ask_price": [100.02],
                "bid_size": [2000],  # bid-heavy
                "ask_size": [1000],
            }
        )
        result = df.with_columns(
            weighted_mid_price("bid_price", "ask_price", "bid_size", "ask_size").alias("wmid")
        )

        simple_mid = 100.01
        # With more bid depth, weighted mid should be > simple mid (closer to ask)
        assert result["wmid"][0] > simple_mid

    def test_ask_heavy_closer_to_bid(self):
        """Test that ask-heavy book gives weighted mid closer to bid."""
        df = pl.DataFrame(
            {
                "bid_price": [100.0],
                "ask_price": [100.02],
                "bid_size": [1000],
                "ask_size": [2000],  # ask-heavy
            }
        )
        result = df.with_columns(
            weighted_mid_price("bid_price", "ask_price", "bid_size", "ask_size").alias("wmid")
        )

        simple_mid = 100.01
        # With more ask depth, weighted mid should be < simple mid (closer to bid)
        assert result["wmid"][0] < simple_mid

    def test_zero_depth_returns_simple_mid(self):
        """Test that zero total depth falls back to simple mid."""
        df = pl.DataFrame(
            {
                "bid_price": [100.0],
                "ask_price": [100.02],
                "bid_size": [0],
                "ask_size": [0],
            }
        )
        result = df.with_columns(
            weighted_mid_price("bid_price", "ask_price", "bid_size", "ask_size").alias("wmid")
        )

        expected = 100.01  # simple mid
        assert np.isclose(result["wmid"][0], expected)

    def test_formula_correctness(self):
        """Test formula: (bid*ask_size + ask*bid_size) / (bid_size + ask_size)."""
        df = pl.DataFrame(
            {
                "bid_price": [100.0],
                "ask_price": [100.02],
                "bid_size": [1500],
                "ask_size": [500],
            }
        )
        result = df.with_columns(
            weighted_mid_price("bid_price", "ask_price", "bid_size", "ask_size").alias("wmid")
        )

        # Manual: (100.0 * 500 + 100.02 * 1500) / (1500 + 500)
        # = (50000 + 150030) / 2000 = 200030 / 2000 = 100.015
        expected = (100.0 * 500 + 100.02 * 1500) / (1500 + 500)
        assert np.isclose(result["wmid"][0], expected)


# =============================================================================
# Tests for Expression API
# =============================================================================


class TestExpressionAPI:
    """Test that functions work with both string column names and pl.Expr."""

    def test_bid_ask_imbalance_with_expr(self, sample_quote_data):
        """Test bid_ask_imbalance accepts pl.Expr inputs."""
        result = sample_quote_data.with_columns(
            bid_ask_imbalance(pl.col("bid_size"), pl.col("ask_size")).alias("imbalance")
        )
        assert "imbalance" in result.columns
        assert result["imbalance"].null_count() == 0

    def test_book_depth_ratio_with_expr(self, sample_quote_data):
        """Test book_depth_ratio accepts pl.Expr inputs."""
        result = sample_quote_data.with_columns(
            book_depth_ratio(pl.col("bid_size"), pl.col("ask_size")).alias("depth_ratio")
        )
        assert "depth_ratio" in result.columns

    def test_weighted_mid_with_expr(self, sample_quote_data):
        """Test weighted_mid_price accepts pl.Expr inputs."""
        result = sample_quote_data.with_columns(
            weighted_mid_price(
                pl.col("bid_price"),
                pl.col("ask_price"),
                pl.col("bid_size"),
                pl.col("ask_size"),
            ).alias("wmid")
        )
        assert "wmid" in result.columns


# =============================================================================
# Tests for Registry
# =============================================================================


class TestRegistry:
    """Test that features are properly registered."""

    def test_features_registered(self):
        """Test that order book features are registered in the registry."""
        from ml4t.engineer.core.registry import get_registry

        registry = get_registry()

        # Check features are registered
        assert registry.get("bid_ask_imbalance") is not None
        assert registry.get("book_depth_ratio") is not None
        assert registry.get("weighted_mid_price") is not None

    def test_feature_metadata(self):
        """Test that feature metadata is correct."""
        from ml4t.engineer.core.registry import get_registry

        registry = get_registry()

        # Check bid_ask_imbalance metadata
        meta = registry.get("bid_ask_imbalance")
        assert meta.category == "microstructure"
        assert meta.normalized is True
        assert "order-book" in meta.tags

        # Check weighted_mid_price metadata
        meta = registry.get("weighted_mid_price")
        assert meta.category == "microstructure"
        assert "mid-price" in meta.tags
