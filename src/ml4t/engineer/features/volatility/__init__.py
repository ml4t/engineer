"""
Volatility features.

All features in this category are auto-registered via @feature decorators.
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
