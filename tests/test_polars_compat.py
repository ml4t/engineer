"""Tests for the CPython 3.15 Polars dispatch compatibility repair."""

from types import FunctionType

from ml4t.engineer._polars_compat import _has_py315_docstring_layout


def test_detects_python_315_docstring_constant_layout() -> None:
    def empty_method() -> None:
        """Method implemented through expression dispatch."""

    code = empty_method.__code__.replace(co_consts=(empty_method.__doc__,))
    python_315_method = FunctionType(code, globals())

    assert _has_py315_docstring_layout(python_315_method, {code.co_code})


def test_rejects_nonempty_method() -> None:
    def implemented_method() -> int:
        """Method with a concrete implementation."""
        return 1

    code = implemented_method.__code__.replace(co_consts=(implemented_method.__doc__, 1))
    method = FunctionType(code, globals())

    assert not _has_py315_docstring_layout(method, {code.co_code})
