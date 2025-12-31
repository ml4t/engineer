"""
Microstructure features.

All features in this category are auto-registered via @feature decorators.
"""

# Import all features to trigger registration
from ml4t.engineer.features.microstructure.amihud_illiquidity import *  # noqa: F403
from ml4t.engineer.features.microstructure.effective_tick_rule import *  # noqa: F403
from ml4t.engineer.features.microstructure.kyle_lambda import *  # noqa: F403
from ml4t.engineer.features.microstructure.order_book import *  # noqa: F403
from ml4t.engineer.features.microstructure.order_flow_imbalance import *  # noqa: F403
from ml4t.engineer.features.microstructure.price_impact_ratio import *  # noqa: F403
from ml4t.engineer.features.microstructure.quote_stuffing_indicator import *  # noqa: F403
from ml4t.engineer.features.microstructure.realized_spread import *  # noqa: F403
from ml4t.engineer.features.microstructure.roll_spread_estimator import *  # noqa: F403
from ml4t.engineer.features.microstructure.trade_intensity import *  # noqa: F403
from ml4t.engineer.features.microstructure.volume_at_price_ratio import *  # noqa: F403
from ml4t.engineer.features.microstructure.volume_synchronicity import *  # noqa: F403
from ml4t.engineer.features.microstructure.volume_weighted_price_momentum import *  # noqa: F403
