"""Information-driven bars for financial data sampling.

This module implements various bar types that sample data based on
information content rather than fixed time intervals:

**Standard Event-Driven Bars:**
- Tick bars: Sample every N ticks
- Volume bars: Sample when volume reaches threshold
- Dollar bars: Sample when dollar value traded reaches threshold

**Advanced Information-Driven Bars:**
- Imbalance bars: Sample based on order flow imbalance (tick, volume, dollar)
- Run bars: Sample based on consecutive buy/sell runs (tick, volume, dollar)

The vectorized implementations are used by default for improved performance.
Original implementations are available with the 'Original' suffix if needed.

Based on "Advances in Financial Machine Learning" by Marcos López de Prado.
"""

from ml4t.engineer.bars.base import BarSampler
from ml4t.engineer.bars.imbalance import (
    FixedTickImbalanceBarSampler,
    FixedVolumeImbalanceBarSampler,
    TickImbalanceBarSampler,
    WindowTickImbalanceBarSampler,
    WindowVolumeImbalanceBarSampler,
)

# Import original implementations with renamed identifiers
from ml4t.engineer.bars.imbalance import ImbalanceBarSampler as ImbalanceBarSamplerOriginal

# Import run bars (new implementation)
from ml4t.engineer.bars.run import DollarRunBarSampler, TickRunBarSampler, VolumeRunBarSampler
from ml4t.engineer.bars.tick import TickBarSampler as TickBarSamplerOriginal

# Import vectorized implementations as default
from ml4t.engineer.bars.vectorized import (
    DollarBarSamplerVectorized as DollarBarSampler,
)
from ml4t.engineer.bars.vectorized import (
    ImbalanceBarSamplerVectorized as ImbalanceBarSampler,
)
from ml4t.engineer.bars.vectorized import (
    TickBarSamplerVectorized as TickBarSampler,
)
from ml4t.engineer.bars.vectorized import (
    VolumeBarSamplerVectorized as VolumeBarSampler,
)
from ml4t.engineer.bars.volume import (
    DollarBarSampler as DollarBarSamplerOriginal,
)
from ml4t.engineer.bars.volume import (
    VolumeBarSampler as VolumeBarSamplerOriginal,
)

__all__ = [
    # Base class
    "BarSampler",
    # Standard event-driven bars (vectorized)
    "TickBarSampler",
    "VolumeBarSampler",
    "DollarBarSampler",
    # Advanced information-driven bars (adaptive)
    "ImbalanceBarSampler",  # Volume imbalance bars (VIBs)
    "TickImbalanceBarSampler",  # Tick imbalance bars (TIBs)
    # Fixed threshold bars (recommended for production)
    "FixedTickImbalanceBarSampler",
    "FixedVolumeImbalanceBarSampler",
    # Window-based bars (bounded adaptation via rolling windows)
    "WindowTickImbalanceBarSampler",
    "WindowVolumeImbalanceBarSampler",
    # Run bars
    "TickRunBarSampler",
    "VolumeRunBarSampler",
    "DollarRunBarSampler",
    # Original implementations (if needed)
    "TickBarSamplerOriginal",
    "VolumeBarSamplerOriginal",
    "DollarBarSamplerOriginal",
    "ImbalanceBarSamplerOriginal",
]
