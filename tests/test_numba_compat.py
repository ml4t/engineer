"""Tests for runtime behavior when Numba is unavailable."""

from __future__ import annotations

from typing import NoReturn

from ml4t.engineer import _numba


def test_missing_numba_uses_working_identity_decorators(monkeypatch) -> None:
    def missing_numba(_name: str) -> NoReturn:
        raise ImportError("Numba is unavailable")

    monkeypatch.setattr(_numba, "import_module", missing_numba)
    jit, njit, available = _numba._load_numba_decorators()

    @jit(nopython=True, cache=True)
    def add_one(value: int) -> int:
        return value + 1

    @njit
    def double(value: int) -> int:
        return value * 2

    assert available is False
    assert add_one(2) == 3
    assert double(3) == 6


def test_loader_reports_current_numba_availability() -> None:
    jit, njit, available = _numba._load_numba_decorators()

    assert available is _numba.NUMBA_AVAILABLE
    assert (jit is _numba._identity_jit) is not available
    assert (njit is _numba._identity_jit) is not available
