"""
Trend features.

All features in this category are auto-registered via @feature decorators.
"""

# Import all features to trigger registration
from ml4t.engineer.features.trend.dema import *  # noqa: F403
from ml4t.engineer.features.trend.donchian import *  # noqa: F403
from ml4t.engineer.features.trend.ema import *  # noqa: F403
from ml4t.engineer.features.trend.kama import *  # noqa: F403
from ml4t.engineer.features.trend.midpoint import *  # noqa: F403
from ml4t.engineer.features.trend.sma import *  # noqa: F403
from ml4t.engineer.features.trend.t3 import *  # noqa: F403
from ml4t.engineer.features.trend.tema import *  # noqa: F403
from ml4t.engineer.features.trend.trima import *  # noqa: F403
from ml4t.engineer.features.trend.wma import *  # noqa: F403
