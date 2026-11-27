"""Shared feature-function dispatch metadata."""

import inspect
from collections.abc import Callable
from typing import Any

# Function argument names that resolve to input DataFrame columns.
COLUMN_ARG_MAP: dict[str, str | list[str]] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "returns": "returns",
    "price": "close",
    "value": "close",
    "feature": "close",
    "features": ["close"],
    "volatility": "close",
    "regime": "close",
}

# Parameters that are always dispatched as keyword arguments.
KEYWORD_ONLY_PARAMS: frozenset[str] = frozenset({"implementation"})

# Columns each ``input_type`` label promises. A feature's label must cover every column its
# signature requires, so a frame holding exactly these columns can compute it.
INPUT_TYPE_COLUMNS: dict[str, list[str]] = {
    "close": ["close"],
    "returns": ["returns"],
    "volume": ["volume"],
    "HL": ["high", "low"],
    "HLC": ["high", "low", "close"],
    "OHLC": ["open", "high", "low", "close"],
    "OHLCV": ["open", "high", "low", "close", "volume"],
    "returns_volume": ["returns", "volume"],
    "returns_volume_price": ["returns", "volume", "close"],
    "quote": ["bid_price", "ask_price", "bid_size", "ask_size"],
}


def required_columns(func: Callable[..., Any]) -> list[str]:
    """Return the input columns dispatch passes to ``func``, in signature order.

    These are the column arguments without a default; ``compute_features`` passes each as the
    column name ``COLUMN_ARG_MAP`` assigns to it.
    """
    columns: list[str] = []
    for name, parameter in inspect.signature(func).parameters.items():
        if name in COLUMN_ARG_MAP and parameter.default is inspect.Parameter.empty:
            mapped = COLUMN_ARG_MAP[name]
            for column in mapped if isinstance(mapped, list) else [mapped]:
                if column not in columns:
                    columns.append(column)
    return columns


def infer_input_type(columns: list[str]) -> str:
    """Return the first ``INPUT_TYPE_COLUMNS`` label, in declaration order, covering ``columns``."""
    needed = set(columns)
    for label, provided in INPUT_TYPE_COLUMNS.items():
        if needed <= set(provided):
            return label
    raise TypeError(f"No input_type covers the columns {sorted(needed)}")


__all__ = [
    "COLUMN_ARG_MAP",
    "INPUT_TYPE_COLUMNS",
    "KEYWORD_ONLY_PARAMS",
    "infer_input_type",
    "required_columns",
]
