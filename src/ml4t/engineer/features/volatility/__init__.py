"""Volatility indicators for ml4t-engineer.

This module provides volatility-based technical analysis indicators including:

TA-Lib Compatible:
- ATR (Average True Range)
- NATR (Normalized ATR)
- TRANGE (True Range)
- Bollinger Bands

Advanced Volatility Models (Academic):
- Yang-Zhang volatility (most efficient estimator)
- Parkinson volatility (high-low range-based)
- Garman-Klass volatility (OHLC-based)
- Rogers-Satchell volatility (drift-adjusted)
- Realized volatility (squared returns)
- EWMA volatility (exponentially weighted)
- GARCH forecast (conditional volatility)

Volatility Analysis:
- Conditional volatility ratio
- Volatility of volatility
- Volatility percentile rank
- Volatility regime probability
"""

# Import all features to trigger registration
from ml4t.engineer.features.volatility.atr import *  # noqa: F403
from ml4t.engineer.features.volatility.bollinger_bands import *  # noqa: F403
from ml4t.engineer.features.volatility.conditional_volatility_ratio import *  # noqa: F403
from ml4t.engineer.features.volatility.ewma_volatility import *  # noqa: F403
from ml4t.engineer.features.volatility.garch_forecast import *  # noqa: F403
from ml4t.engineer.features.volatility.garman_klass_volatility import *  # noqa: F403
from ml4t.engineer.features.volatility.natr import *  # noqa: F403
from ml4t.engineer.features.volatility.parkinson_volatility import *  # noqa: F403
from ml4t.engineer.features.volatility.realized_volatility import *  # noqa: F403
from ml4t.engineer.features.volatility.rogers_satchell_volatility import *  # noqa: F403
from ml4t.engineer.features.volatility.trange import *  # noqa: F403
from ml4t.engineer.features.volatility.volatility_of_volatility import *  # noqa: F403
from ml4t.engineer.features.volatility.volatility_percentile_rank import *  # noqa: F403
from ml4t.engineer.features.volatility.volatility_regime_probability import *  # noqa: F403
from ml4t.engineer.features.volatility.yang_zhang_volatility import *  # noqa: F403
