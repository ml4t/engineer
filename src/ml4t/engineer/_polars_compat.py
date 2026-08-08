"""Compatibility repair for Polars Series dispatch on CPython 3.15."""

from __future__ import annotations

import sys
from collections.abc import Container
from datetime import datetime
from importlib import import_module
from types import FunctionType

_SERIES_CLASSES = (
    ("polars.series.series", "Series"),
    ("polars.series.array", "ArrayNameSpace"),
    ("polars.series.binary", "BinaryNameSpace"),
    ("polars.series.categorical", "CatNameSpace"),
    ("polars.series.datetime", "DateTimeNameSpace"),
    ("polars.series.ext", "ExtensionNameSpace"),
    ("polars.series.list", "ListNameSpace"),
    ("polars.series.string", "StringNameSpace"),
    ("polars.series.struct", "StructNameSpace"),
)
_BROKEN_SERIES_ERROR = "'NoneType' object has no attribute '_s'"


def _has_py315_docstring_layout(function: FunctionType, empty_bytecode: Container[bytes]) -> bool:
    code = function.__code__
    return (
        code.co_code in empty_bytecode
        and isinstance(function.__doc__, str)
        and code.co_consts == (function.__doc__,)
    )


def ensure_polars_series_dispatch() -> None:
    """Repair pola-rs/polars#28347 only when its Python 3.15 failure is present."""
    if sys.version_info < (3, 15):
        return

    import polars as pl

    expected = datetime(2000, 1, 1)
    try:
        pl.Series("_ml4t_polars_probe", [expected])
        return
    except AttributeError as error:
        if _BROKEN_SERIES_ERROR not in str(error):
            raise

    utils = import_module("polars.series.utils")
    original_is_empty = utils._is_empty_method
    empty_bytecode = utils._EMPTY_BYTECODE

    def is_empty_method(function: FunctionType) -> bool:
        return original_is_empty(function) or _has_py315_docstring_layout(function, empty_bytecode)

    utils._is_empty_method = is_empty_method
    for module_name, class_name in _SERIES_CLASSES:
        utils.expr_dispatch(getattr(import_module(module_name), class_name))

    try:
        probe = pl.Series("_ml4t_polars_probe", [expected])
        if probe.to_list() != [expected]:
            raise RuntimeError("Polars Series dispatch probe returned an invalid result")
    except Exception as error:
        utils._is_empty_method = original_is_empty
        raise RuntimeError("Unable to repair Polars Series dispatch on Python 3.15") from error


ensure_polars_series_dispatch()
