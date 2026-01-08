"""Features module for ml4t-engineer.

Contains all feature engineering functionality organized intuitively:

Core Feature Categories:
    - momentum: Oscillators and momentum indicators (RSI, MACD, Stochastic, etc.)
    - trend: Moving averages and trend-following indicators (SMA, EMA, Bollinger Bands, etc.)
    - volatility: Volatility measures and bands (ATR, NATR, True Range, Bollinger Bands)
    - volume: Volume-based indicators (OBV, AD, ADOSC)
    - statistics: Statistical indicators (variance, correlation, regression, etc.)
    - math: Mathematical transformations (max, min, sum, etc.)
    - price_transform: Price transformations (typical price, weighted close, etc.)

Advanced Features:
    - microstructure: Market microstructure (Kyle lambda, VPIN, Amihud, etc.)
    - ml: ML-specific features (fractional differencing, entropy, lags, etc.)
    - cross_asset: Cross-sectional and multi-asset features
    - risk: Risk metrics and measures
    - regime: Market regime detection
    - fdiff: Fractional differencing utilities

Note: All features are now organized as individual modules within their category directories.
      Use the registry system to discover and access features programmatically.
"""

# Import all feature modules to trigger @feature decorator registration
# This ensures all features are available in the global registry
# Import all feature categories to register features
from ml4t.engineer.features import (
    composite,
    cross_asset,
    fdiff,
    math,  # noqa: F401
    microstructure,  # noqa: F401
    ml,  # noqa: F401
    momentum,  # noqa: F401
    price_transform,  # noqa: F401
    regime,
    risk,
    statistics,  # noqa: F401
    trend,  # noqa: F401
    volatility,  # noqa: F401
    volume,  # noqa: F401
)

__all__ = [
    # Advanced feature modules (single files)
    "composite",
    "cross_asset",
    "fdiff",
    "regime",
    "risk",
]
