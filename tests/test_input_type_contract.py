"""A feature's input_type must name the columns it needs.

`feature_catalog.by_input_type(...)` is how a caller finds which features a given set of columns
supports. Before this contract, 48 registered features carried the decorator's default label
"close" while reading high, low, volume or returns, and 14 returns-based features failed on every
OHLCV frame with a raw polars ColumnNotFoundError (issue #135).
"""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer import compute_features
from ml4t.engineer.api import INPUT_TYPE_COLUMNS, _execute_feature
from ml4t.engineer.core.registry import get_registry

_N = 300


def _frame_for(columns: list[str]) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    frames = []
    for symbol in ("AAA", "BBB"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, _N)))
        open_ = close * np.exp(rng.normal(0, 0.003, _N))
        spread = np.abs(rng.normal(0, 0.005, _N)) * close
        values = {
            "open": open_,
            "close": close,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "volume": rng.integers(1_000, 10_000, _N).astype(float),
            "returns": np.concatenate([[0.0], np.diff(np.log(close))]),
            "bid_price": close - 0.01,
            "ask_price": close + 0.01,
            "bid_size": rng.integers(100, 1_000, _N).astype(float),
            "ask_size": rng.integers(100, 1_000, _N).astype(float),
        }
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": pl.datetime_range(
                        pl.datetime(2020, 1, 1),
                        pl.datetime(2020, 1, 1) + pl.duration(days=_N - 1),
                        "1d",
                        eager=True,
                    ),
                    "symbol": [symbol] * _N,
                    **{c: values[c] for c in columns},
                }
            )
        )
    return pl.concat(frames)


_REQUIRED_PARAMS = {"time_decay_weights": {"lookback": 20}, "fourier_features": {"period": 21}}


@pytest.mark.parametrize("name", sorted(get_registry().list_all()))
def test_feature_computes_on_the_columns_its_input_type_promises(name: str) -> None:
    """Error prevented: a catalog label that promises columns the feature cannot run on.

    Runs the dispatch step compute_features uses on a frame holding exactly the label's columns,
    so a column read through a defaulted parameter (donchian's high and low) is caught too.
    """
    metadata = get_registry().get(name)
    assert metadata.input_type in INPUT_TYPE_COLUMNS
    frame = _frame_for(INPUT_TYPE_COLUMNS[metadata.input_type])
    try:
        _execute_feature(frame, name, _REQUIRED_PARAMS.get(name, {}), name, ["symbol"])
    except pl.exceptions.ColumnNotFoundError as exc:
        pytest.fail(f"{name} (input_type={metadata.input_type!r}) read a missing column: {exc}")
    except ValueError as exc:
        if "needs column" in str(exc):
            pytest.fail(f"{name} (input_type={metadata.input_type!r}): {exc}")
        raise


def test_missing_returns_column_names_the_column_and_the_feature() -> None:
    """Error prevented: a raw polars ColumnNotFoundError deep inside the feature expression."""
    ohlcv = _frame_for(INPUT_TYPE_COLUMNS["OHLCV"])
    with pytest.raises(ValueError, match=r"realized_volatility.*\['returns'\].*returns"):
        compute_features(ohlcv, ["realized_volatility"], group_col="symbol", calendar=None)


def test_missing_column_is_reported_for_lazy_input() -> None:
    """Error prevented: the check skipped when the caller passes a LazyFrame."""
    lazy = _frame_for(INPUT_TYPE_COLUMNS["OHLCV"]).lazy()
    with pytest.raises(ValueError, match=r"garch_forecast.*\['returns'\]"):
        compute_features(lazy, ["garch_forecast"], group_col="symbol", calendar=None)


def test_returns_features_compute_once_a_returns_column_is_added() -> None:
    """Error prevented: the new check rejecting a frame that does carry the declared column."""
    frame = _frame_for(INPUT_TYPE_COLUMNS["OHLCV"] + ["returns"])
    out = compute_features(frame, ["realized_volatility"], group_col="symbol", calendar=None)
    assert out["realized_volatility"].drop_nulls().len() > 0


def test_close_only_catalog_entries_exclude_range_and_volume_features() -> None:
    """Error prevented: by_input_type('close') listing features that need high, low or volume."""
    from ml4t.engineer import feature_catalog

    close_only = set(feature_catalog.by_input_type("close"))
    assert {"atr", "adx", "obv", "realized_volatility", "donchian_channels"}.isdisjoint(close_only)
    assert {"rsi", "ema"} <= close_only
