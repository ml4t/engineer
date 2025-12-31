"""Core type definitions for ml4t-engineer.

Defines the fundamental types used throughout the library.
"""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl

# Type aliases for clarity
Symbol = str
AssetId = str
OrderId = str
Price = float
Quantity = float
Timestamp = int  # nanoseconds since epoch

# Feature computation types
FeatureValue = float | int | bool
FeatureArray = npt.NDArray[np.float64] | pl.Series | pl.Expr

# Implementation modes
Implementation = Literal["auto", "numba", "polars"]

# Time-related types
TimeUnit = Literal["ns", "us", "ms", "s"]
Frequency = str  # e.g., "1D", "5T", "1H"

# Pipeline types
StepName = str
StepConfig = dict[str, Any]

__all__ = [
    "AssetId",
    "FeatureArray",
    "FeatureValue",
    "Frequency",
    "Implementation",
    "OrderId",
    "Price",
    "Quantity",
    "StepConfig",
    "StepName",
    "Symbol",
    "TimeUnit",
    "Timestamp",
]
