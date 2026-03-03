"""Feature selection has moved to ml4t-diagnostic.

Use ``ml4t.diagnostic.selection`` instead::

    from ml4t.diagnostic.selection import FeatureSelector

Install with: ``pip install ml4t-diagnostic``
"""

_MOVED_EXPORTS = {
    "FeatureSelector",
    "SelectionReport",
    "SelectionStep",
}


def __getattr__(name: str) -> object:
    if name in _MOVED_EXPORTS:
        raise ImportError(
            f"ml4t.engineer.selection.{name} has moved to ml4t-diagnostic. "
            f"Use: from ml4t.diagnostic.selection import {name}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
