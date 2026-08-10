"""Numba decorators with a pure-Python fallback for unsupported runtimes."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, TypeVar

_Function = TypeVar("_Function", bound=Callable[..., Any])


def _identity_jit(*args: Any, **kwargs: Any) -> Any:
    """Return the undecorated function when Numba is unavailable."""
    del kwargs
    if len(args) == 1 and callable(args[0]):
        return args[0]

    def decorate(function: _Function) -> _Function:
        return function

    return decorate


def _load_numba_decorators() -> tuple[Any, Any, bool]:
    try:
        numba = import_module("numba")
    except ImportError:
        return _identity_jit, _identity_jit, False
    return numba.jit, numba.njit, True


jit, njit, NUMBA_AVAILABLE = _load_numba_decorators()


def warm_rolling_callback(callback: Callable[[Any], Any], window: int) -> None:
    """Run ``callback`` once so its kernel compiles outside a rolling window.

    ``polars.Expr.rolling_map`` does not advance its window while the callback is
    being compiled. The first time a process evaluates a feature whose callback
    calls a lazily-jitted kernel, every window therefore receives the first
    window's value: the column comes out near-constant, and nothing raises.

    Measured on a 600-row series, each feature returning two distinct values on its
    first use in a process and the full set on every later one:
    ``regime.hurst_exponent``, ``statistics.coefficient_of_variation``,
    ``statistics.rolling_cv_zscore``, ``statistics.rolling_drift`` and
    ``ml.rolling_entropy_lz``. It is what made ``etfs/03_financial_features`` fail
    its withheld-rows check in CI and pass on a re-run, since the second run found
    the kernel already compiled on disk.

    Call this with the same ``window`` the ``rolling_map`` will use, so the probe
    has the shape the kernel will be typed against.
    """
    if not NUMBA_AVAILABLE:
        return

    import numpy as np
    import polars as pl

    probe = pl.Series("warmup", np.linspace(1.0, 2.0, max(int(window), 2), dtype=np.float64))
    try:
        callback(probe)
    except Exception:  # noqa: BLE001
        # A probe the kernel rejects leaves the feature exactly as it was: the
        # warm-up is an optimisation of when compilation happens, never a result.
        return
