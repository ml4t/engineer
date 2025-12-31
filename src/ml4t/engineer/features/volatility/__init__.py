"""Volatility indicators for ml4t-engineer.

This module provides volatility-based technical analysis indicators including:
- ATR (Average True Range)
- NATR (Normalized ATR)
- TRANGE (True Range)
- Bollinger Bands

All indicators are validated against TA-Lib reference implementation.
"""

from ml4t.engineer.features.volatility.atr import atr
from ml4t.engineer.features.volatility.bollinger_bands import bollinger_bands
from ml4t.engineer.features.volatility.natr import natr
from ml4t.engineer.features.volatility.trange import trange

__all__ = [
    "atr",
    "bollinger_bands",
    "natr",
    "trange",
]
