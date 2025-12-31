"""Features module for ml4t-engineer.

Contains all feature engineering functionality organized intuitively:

Core Feature Categories:
    - momentum: Oscillators and momentum indicators (RSI, MACD, Stochastic, etc.)
    - trend: Moving averages and trend-following indicators (SMA, EMA, etc.)
    - volatility: Volatility measures and bands (ATR, NATR, Yang-Zhang, etc.)
    - volume: Volume-based indicators (OBV, AD, ADOSC)
    - statistics: Statistical indicators (variance, regression, etc.)
    - math: Mathematical transformations (max, min, sum)
    - price_transform: Price transformations (typical price, weighted close, etc.)

Advanced Features:
    - microstructure: Market microstructure (Kyle lambda, VPIN, Amihud, etc.)
    - ml: ML-specific features (fractional differencing, entropy, lags, etc.)
    - cross_asset: Cross-sectional and multi-asset features
    - risk: Risk metrics and measures
    - regime: Market regime detection
    - fdiff: Fractional differencing utilities

Note: All features are organized as individual modules within their category directories.
      Use the registry system to discover and access features programmatically.
"""

# Import all feature modules to trigger @feature decorator registration
from ml4t.engineer.features import (
    composite,
    cross_asset,
    fdiff,
    math,
    microstructure,
    ml,
    momentum,
    price_transform,
    regime,
    risk,
    statistics,
    trend,
    volatility,
    volume,
)

__all__ = [
    # Core feature categories
    "math",
    "momentum",
    "price_transform",
    "statistics",
    "trend",
    "volatility",
    "volume",
    # Advanced features
    "microstructure",
    "ml",
    # Standalone modules
    "composite",
    "cross_asset",
    "fdiff",
    "regime",
    "risk",
]
