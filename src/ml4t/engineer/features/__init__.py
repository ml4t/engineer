"""Features module for ml4t-engineer.

Contains all feature engineering functionality organized intuitively:

Core Feature Categories:
    - momentum: Oscillators and momentum indicators (RSI, MACD, Stochastic, etc.)
    - trend: Moving averages and trend-following indicators (SMA, EMA, etc.)
    - volatility: Volatility measures and bands (ATR, NATR, True Range, Bollinger)
    - volume: Volume-based indicators (OBV, AD, ADOSC)
    - statistics: Statistical indicators (variance, regression, etc.)
    - math: Mathematical transformations (max, min, sum)
    - price_transform: Price transformations (typical price, weighted close, etc.)

Note: All features are organized as individual modules within their category directories.
      Use the registry system to discover and access features programmatically.
"""

# Import all feature modules to trigger @feature decorator registration
# This ensures all features are available in the global registry
from ml4t.engineer.features import (
    math,
    momentum,
    price_transform,
    statistics,
    trend,
    volatility,
    volume,
)

__all__ = [
    "math",
    "momentum",
    "price_transform",
    "statistics",
    "trend",
    "volatility",
    "volume",
]
