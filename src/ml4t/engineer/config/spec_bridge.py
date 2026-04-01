"""Compatibility bridge from shared market-data specs to engineer contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ml4t.specs import FeedSpec

from ml4t.engineer.config.data_contract import DataContractConfig


def data_contract_from_market_data_spec(
    spec: Mapping[str, Any] | object,
) -> DataContractConfig:
    """Build a dataframe contract from a shared market-data spec shape.

    The bridge accepts either a nested mapping produced by ``MarketDataSpec.to_dict()``
    or an object exposing a ``schema`` attribute with the same field names.
    """
    feed_spec = FeedSpec.from_any(spec)

    return DataContractConfig(
        timestamp_col=feed_spec.timestamp_col,
        symbol_col=feed_spec.entity_col or "asset",
        price_col=feed_spec.price_col,
        open_col=feed_spec.open_col,
        high_col=feed_spec.high_col,
        low_col=feed_spec.low_col,
        close_col=feed_spec.close_col,
        volume_col=feed_spec.volume_col,
    )


__all__ = ["data_contract_from_market_data_spec"]
