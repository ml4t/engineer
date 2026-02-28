"""Compatibility shim for removed legacy barrier config module."""

_REMOVAL_MESSAGE = (
    "ml4t.engineer.labeling.barriers has been removed. "
    "Use ml4t.engineer.config.LabelingConfig.triple_barrier(...) for triple-barrier settings "
    "or LabelingConfig.atr_barrier(...) for ATR-adjusted settings."
)

raise ImportError(_REMOVAL_MESSAGE)
