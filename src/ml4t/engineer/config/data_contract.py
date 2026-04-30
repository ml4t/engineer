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


__all__ = ["DataContractConfig"]
