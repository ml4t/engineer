"""Tests for DataContractConfig."""

import sys
from types import ModuleType

import pytest

from ml4t.engineer.config import DataContractConfig


class TestDataContractConfig:
    """Shared dataframe contract tests."""

    def test_defaults(self):
        """Default contract should match canonical ML4T column names."""
        contract = DataContractConfig()
        assert contract.timestamp_col == "timestamp"
        assert contract.symbol_col == "symbol"
        assert contract.price_col == "close"
        assert contract.open_col == "open"
        assert contract.high_col == "high"
        assert contract.low_col == "low"
        assert contract.close_col == "close"
        assert contract.volume_col == "volume"

    def test_group_col_alias_maps_to_symbol_col(self):
        """group_col alias should map to symbol_col for panel configs."""
        contract = DataContractConfig.from_dict(
            {
                "timestamp_col": "ts",
                "group_col": "ticker",
                "price_col": "px",
            }
        )
        assert contract.symbol_col == "ticker"
        assert contract.timestamp_col == "ts"
        assert contract.price_col == "px"

    def test_from_mapping(self):
        """Contract should load from generic dict-like mappings."""
        contract = DataContractConfig.from_mapping(
            {
                "timestamp_col": "ts",
                "symbol_col": "asset_id",
                "price_col": "mid_price",
            }
        )
        assert contract.timestamp_col == "ts"
        assert contract.symbol_col == "asset_id"
        assert contract.price_col == "mid_price"

    def test_from_ml4t_data_or_actionable_error(self):
        """Bridge should either work or fail with an explicit import message."""
        try:
            contract = DataContractConfig.from_ml4t_data()
            assert contract.timestamp_col == "timestamp"
            assert contract.symbol_col == "symbol"
            assert contract.price_col == "close"
        except ImportError as exc:
            assert "ml4t-data is required" in str(exc)

    def test_from_ml4t_data_uses_available_schema_aliases(self, monkeypatch: pytest.MonkeyPatch):
        """Bridge should map to available schema aliases when canonical names differ."""
        class _SchemaWithAliases:
            SCHEMA = {
                "ts_event": object(),
                "ticker": object(),
                "open_price": object(),
                "high_price": object(),
                "low_price": object(),
                "last_price": object(),
                "volume_base": object(),
            }

        data_module = ModuleType("ml4t.data")
        data_module.__path__ = []  # type: ignore[attr-defined]
        core_module = ModuleType("ml4t.data.core")
        core_module.MultiAssetSchema = _SchemaWithAliases
        data_module.core = core_module

        monkeypatch.setitem(sys.modules, "ml4t.data", data_module)
        monkeypatch.setitem(sys.modules, "ml4t.data.core", core_module)

        contract = DataContractConfig.from_ml4t_data()
        assert contract.timestamp_col == "ts_event"
        assert contract.symbol_col == "ticker"
        assert contract.open_col == "open_price"
        assert contract.high_col == "high_price"
        assert contract.low_col == "low_price"
        assert contract.close_col == "last_price"
        assert contract.price_col == "last_price"
        assert contract.volume_col == "volume_base"
