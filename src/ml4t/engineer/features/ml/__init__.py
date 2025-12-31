"""
ML features.

All features in this category are auto-registered via @feature decorators.
"""

# Import all features to trigger registration
from ml4t.engineer.features.ml.create_lag_features import *  # noqa: F403
from ml4t.engineer.features.ml.cyclical_encode import *  # noqa: F403
from ml4t.engineer.features.ml.directional_targets import *  # noqa: F403
from ml4t.engineer.features.ml.fourier_features import *  # noqa: F403
from ml4t.engineer.features.ml.interaction_features import *  # noqa: F403
from ml4t.engineer.features.ml.multi_horizon_returns import *  # noqa: F403
from ml4t.engineer.features.ml.percentile_rank_features import *  # noqa: F403
from ml4t.engineer.features.ml.regime_conditional_features import *  # noqa: F403
from ml4t.engineer.features.ml.rolling_entropy import *  # noqa: F403
from ml4t.engineer.features.ml.time_decay_weights import *  # noqa: F403
from ml4t.engineer.features.ml.volatility_adjusted_returns import *  # noqa: F403
