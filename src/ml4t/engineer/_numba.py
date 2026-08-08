"""Numba decorators with a pure-Python fallback for unsupported runtimes."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, TypeVar, cast

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
    return cast(Any, numba.jit), cast(Any, numba.njit), True


jit, njit, NUMBA_AVAILABLE = _load_numba_decorators()
