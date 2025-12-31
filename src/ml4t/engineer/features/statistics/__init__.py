"""
Statistics features.

All features in this category are auto-registered via @feature decorators.
"""

# Import all features to trigger registration
from ml4t.engineer.features.statistics.avgdev import *  # noqa: F403
from ml4t.engineer.features.statistics.linearreg import *  # noqa: F403
from ml4t.engineer.features.statistics.linearreg_angle import *  # noqa: F403
from ml4t.engineer.features.statistics.linearreg_intercept import *  # noqa: F403
from ml4t.engineer.features.statistics.linearreg_slope import *  # noqa: F403
from ml4t.engineer.features.statistics.stddev import *  # noqa: F403
from ml4t.engineer.features.statistics.structural_break import *  # noqa: F403
from ml4t.engineer.features.statistics.tsf import *  # noqa: F403
from ml4t.engineer.features.statistics.var import *  # noqa: F403
