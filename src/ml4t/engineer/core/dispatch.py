"""Shared feature-function dispatch metadata."""

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

__all__ = ["COLUMN_ARG_MAP", "KEYWORD_ONLY_PARAMS"]
