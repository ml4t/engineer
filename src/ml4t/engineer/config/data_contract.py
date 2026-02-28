"""Shared dataframe column contract for cross-library interoperability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import AliasChoices, Field

from ml4t.engineer.config.base import BaseConfig


class DataContractConfig(BaseConfig):
    """Canonical dataframe column mapping shared across ML4T libraries."""

    timestamp_col: str = Field("timestamp", description="Timestamp column name")
    symbol_col: str | list[str] | None = Field(
        "symbol",
        validation_alias=AliasChoices("symbol_col", "group_col", "ticker_col", "asset_col"),
        description="Asset identifier column(s) used for panel grouping",
    )
    price_col: str = Field("close", description="Primary price column")
    open_col: str = Field("open", description="Open price column")
    high_col: str = Field("high", description="High price column")
    low_col: str = Field("low", description="Low price column")
    close_col: str = Field("close", description="Close price column")
    volume_col: str = Field("volume", description="Volume column")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> DataContractConfig:
        """Create contract from a generic mapping source."""
        return cls(**dict(mapping))

    @classmethod
    def from_ml4t_data(cls) -> DataContractConfig:
        """Create a contract from ml4t-data's canonical multi-asset schema."""
        try:
            from ml4t.data.core import MultiAssetSchema
        except ImportError as exc:  # pragma: no cover - environment-dependent
            msg = (
                "ml4t-data is required to build DataContractConfig.from_ml4t_data(). "
                "Install/enable ml4t-data or provide contract fields explicitly."
            )
            raise ImportError(msg) from exc

        schema = getattr(MultiAssetSchema, "SCHEMA", {})
        schema_cols = {str(col) for col in schema}

        def pick(candidates: tuple[str, ...], default: str | None) -> str | None:
            for candidate in candidates:
                if candidate in schema_cols:
                    return candidate
            return default

        close_col = pick(("close", "close_price", "last", "last_price"), "close")
        return cls(
            timestamp_col=pick(("timestamp", "ts", "ts_event", "datetime", "date"), "timestamp"),
            symbol_col=pick(("symbol", "ticker", "asset", "asset_id"), None),
            price_col=pick(
                ("close", "close_price", "price", "mid_price", "last", "last_price"),
                close_col,
            ),
            open_col=pick(("open", "open_price"), "open"),
            high_col=pick(("high", "high_price"), "high"),
            low_col=pick(("low", "low_price"), "low"),
            close_col=close_col,
            volume_col=pick(("volume", "volume_base", "size"), "volume"),
        )


__all__ = ["DataContractConfig"]
