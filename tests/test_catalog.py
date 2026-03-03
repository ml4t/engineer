"""Tests for discovery/catalog.py - FeatureCatalog API."""

import pytest

from ml4t.engineer.core.registry import FeatureMetadata, FeatureRegistry
from ml4t.engineer.discovery.catalog import FeatureCatalog


@pytest.fixture
def registry():
    """Create a test registry with known features."""
    reg = FeatureRegistry()

    reg.register(
        FeatureMetadata(
            name="rsi",
            func=lambda _close, _period=14: None,
            category="momentum",
            description="Relative Strength Index",
            formula="100 - 100 / (1 + RS)",
            normalized=True,
            ta_lib_compatible=True,
            input_type="close",
            output_type="indicator",
            parameters={"period": 14},
            tags=["oscillator", "momentum"],
            value_range=(0.0, 100.0),
            lookback=lambda period=14, **_: period,
        )
    )

    reg.register(
        FeatureMetadata(
            name="sma",
            func=lambda _close, _period=20: None,
            category="trend",
            description="Simple Moving Average",
            formula="sum(close, period) / period",
            normalized=False,
            ta_lib_compatible=True,
            input_type="close",
            output_type="indicator",
            parameters={"period": 20},
            tags=["trend", "average"],
            lookback=lambda period=20, **_: period,
        )
    )

    reg.register(
        FeatureMetadata(
            name="atr",
            func=lambda _high, _low, _close, _period=14: None,
            category="volatility",
            description="Average True Range",
            formula="EMA(TR, period)",
            normalized=False,
            ta_lib_compatible=True,
            input_type="OHLCV",
            output_type="indicator",
            parameters={"period": 14},
            tags=["volatility", "range"],
            lookback=lambda period=14, **_: period,
        )
    )

    reg.register(
        FeatureMetadata(
            name="macd",
            func=lambda _close, _fast=12, _slow=26, _signal=9: None,
            category="momentum",
            description="Moving Average Convergence Divergence",
            formula="EMA(fast) - EMA(slow)",
            normalized=False,
            ta_lib_compatible=True,
            input_type="close",
            output_type="indicator",
            parameters={"fast": 12, "slow": 26, "signal": 9},
            dependencies=["sma"],
            tags=["momentum", "trend"],
            lookback=lambda slow=26, signal=9, **_: slow + signal,
        )
    )

    return reg


@pytest.fixture
def catalog(registry):
    """Create a FeatureCatalog wrapping test registry."""
    return FeatureCatalog(registry)


class TestFeatureCatalogList:
    """Tests for FeatureCatalog.list()."""

    def test_list_all(self, catalog):
        result = catalog.list()
        assert len(result) == 4
        assert result == sorted(result), "Results should be alphabetically sorted"

    def test_list_by_category(self, catalog):
        result = catalog.list(category="momentum")
        assert set(result) == {"rsi", "macd"}

    def test_list_by_category_no_match(self, catalog):
        result = catalog.list(category="nonexistent")
        assert result == []

    def test_list_normalized(self, catalog):
        result = catalog.list(normalized=True)
        assert result == ["rsi"]

    def test_list_not_normalized(self, catalog):
        result = catalog.list(normalized=False)
        assert set(result) == {"sma", "atr", "macd"}

    def test_list_ta_lib_compatible(self, catalog):
        result = catalog.list(ta_lib_compatible=True)
        assert len(result) == 4

    def test_list_by_input_type(self, catalog):
        result = catalog.list(input_type="close")
        assert set(result) == {"rsi", "sma", "macd"}

    def test_list_by_input_type_ohlcv(self, catalog):
        result = catalog.list(input_type="OHLCV")
        assert result == ["atr"]

    def test_list_with_tags(self, catalog):
        result = catalog.list(tags=["momentum"])
        assert set(result) == {"rsi", "macd"}

    def test_list_with_multiple_tags(self, catalog):
        result = catalog.list(tags=["momentum", "trend"])
        assert result == ["macd"]

    def test_list_has_dependencies(self, catalog):
        result = catalog.list(has_dependencies=True)
        assert result == ["macd"]

    def test_list_no_dependencies(self, catalog):
        result = catalog.list(has_dependencies=False)
        assert set(result) == {"rsi", "sma", "atr"}

    def test_list_with_limit(self, catalog):
        result = catalog.list(limit=2)
        assert len(result) == 2

    def test_list_combined_filters(self, catalog):
        result = catalog.list(category="momentum", normalized=True)
        assert result == ["rsi"]


class TestFeatureCatalogDescribe:
    """Tests for FeatureCatalog.describe()."""

    def test_describe_existing(self, catalog):
        info = catalog.describe("rsi")
        assert info["name"] == "rsi"
        assert info["category"] == "momentum"
        assert info["normalized"] is True
        assert info["ta_lib_compatible"] is True
        assert info["formula"] == "100 - 100 / (1 + RS)"
        assert info["parameters"] == {"period": 14}
        assert info["value_range"] == (0.0, 100.0)
        assert info["lookback_period"] == 14

    def test_describe_with_dependencies(self, catalog):
        info = catalog.describe("macd")
        assert info["dependencies"] == ["sma"]
        assert info["lookback_period"] == 35  # 26 + 9

    def test_describe_not_found(self, catalog):
        with pytest.raises(KeyError, match="not found"):
            catalog.describe("nonexistent")


class TestFeatureCatalogSearch:
    """Tests for FeatureCatalog.search()."""

    def test_search_by_name(self, catalog):
        results = catalog.search("rsi")
        assert len(results) > 0
        names = [name for name, _ in results]
        assert "rsi" in names

    def test_search_exact_name_highest_score(self, catalog):
        results = catalog.search("rsi")
        assert results[0][0] == "rsi"
        assert results[0][1] == 1.0  # Exact match

    def test_search_by_description(self, catalog):
        results = catalog.search("Average True Range")
        names = [name for name, _ in results]
        assert "atr" in names

    def test_search_partial_match(self, catalog):
        results = catalog.search("average")
        names = [name for name, _ in results]
        assert "sma" in names  # "Simple Moving Average"
        assert "atr" in names  # "Average True Range"

    def test_search_empty_query(self, catalog):
        results = catalog.search("")
        assert results == []

    def test_search_whitespace_query(self, catalog):
        results = catalog.search("   ")
        assert results == []

    def test_search_max_results(self, catalog):
        results = catalog.search("a", max_results=2)
        assert len(results) <= 2

    def test_search_specific_fields(self, catalog):
        results = catalog.search("momentum", search_fields=["tags"])
        names = [name for name, _ in results]
        assert "rsi" in names
        assert "macd" in names


class TestFeatureCatalogConvenience:
    """Tests for convenience methods."""

    def test_by_input_type(self, catalog):
        result = catalog.by_input_type("close")
        assert set(result) == {"rsi", "sma", "macd"}

    def test_by_lookback(self, catalog):
        result = catalog.by_lookback(14)
        assert "rsi" in result
        assert "atr" in result
        assert "macd" not in result  # lookback = 35

    def test_categories(self, catalog):
        result = catalog.categories()
        assert result == ["momentum", "trend", "volatility"]

    def test_input_types(self, catalog):
        result = catalog.input_types()
        assert set(result) == {"close", "OHLCV"}

    def test_stats(self, catalog):
        stats = catalog.stats()
        assert stats["total"] == 4
        assert stats["by_category"]["momentum"] == 2
        assert stats["by_category"]["volatility"] == 1
        assert stats["normalized"] == 1
        assert stats["ta_lib_compatible"] == 4

    def test_len(self, catalog):
        assert len(catalog) == 4

    def test_repr(self, catalog):
        assert "FeatureCatalog" in repr(catalog)
        assert "4" in repr(catalog)
