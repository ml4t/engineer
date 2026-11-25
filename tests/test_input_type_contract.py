"""A feature's input_type must name the columns it needs.

`feature_catalog.by_input_type(...)` is how a caller finds which features a given set of columns
supports. Before this contract, 48 registered features carried the decorator's default label
"close" while reading high, low, volume or returns, and 14 returns-based features failed on every
OHLCV frame with a raw polars ColumnNotFoundError (issue #135).
"""

import numpy as np
import polars as pl
import pytest
from ml4t.specs import FeedSpec

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


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("unit,period", [("session", 2), ("elapsed", "2h")])
def test_missing_returns_are_named_for_valid_calendar_window_inputs(unit, period, lazy) -> None:
    from datetime import UTC, datetime, timedelta

    frame = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=i + 1) for i in range(4)
            ],
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.0] * 4,
            "volume": [1000.0] * 4,
        }
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        timezone="UTC",
        data_frequency="1h",
        timestamp_semantics="bar_close",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="right",
        partial_bar_policy="exclude",
    )
    with pytest.raises(ValueError, match=r"realized_volatility.*\['returns'\]"):
        compute_features(
            frame.lazy() if lazy else frame,
            [{"name": "realized_volatility", "params": {"period": period}}],
            feed_spec=feed,
            window_unit=unit,
        )
    supplied = frame.with_columns(pl.lit(0.0).alias("returns"))
    valid = compute_features(
        supplied.lazy() if lazy else supplied,
        [{"name": "realized_volatility", "params": {"period": period}}],
        feed_spec=feed,
        window_unit=unit,
    )
    if isinstance(valid, pl.LazyFrame):
        valid = valid.collect()
    assert valid.height == frame.height
    assert "realized_volatility" in valid.columns


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


_QUOTE_PARAMS = {
    "bid_ask_imbalance": {"bid_size": "buy_size", "ask_size": "sell_size"},
    "book_depth_ratio": {"bid_depth": "buy_size", "ask_depth": "sell_size"},
    "weighted_mid_price": {
        "bid_price": "buy_price",
        "ask_price": "sell_price",
        "bid_size": "buy_size",
        "ask_size": "sell_size",
    },
}


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("name", sorted(_QUOTE_PARAMS))
def test_missing_default_quote_columns_name_the_feature(name: str, lazy: bool) -> None:
    frame = _frame_for(INPUT_TYPE_COLUMNS["OHLCV"])
    data = frame.lazy() if lazy else frame
    with pytest.raises(ValueError, match=rf"{name}.*needs column") as error:
        compute_features(data, [name], calendar=None)
    assert "'bid_size'" in str(error.value)
    assert "'ask_size'" in str(error.value)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("expression", [False, True])
@pytest.mark.parametrize("name", sorted(_QUOTE_PARAMS))
def test_custom_quote_columns_do_not_require_unused_defaults(
    name: str, expression: bool, lazy: bool
) -> None:
    from polars.testing import assert_frame_equal

    frame = _frame_for(INPUT_TYPE_COLUMNS["OHLCV"]).with_columns(
        buy_size=pl.lit(300.0),
        sell_size=pl.lit(100.0),
        buy_price=pl.lit(99.0),
        sell_price=pl.lit(101.0),
    )
    params = {
        key: pl.col(value) if expression else value for key, value in _QUOTE_PARAMS[name].items()
    }
    data = frame.lazy() if lazy else frame
    out = compute_features(data, [{"name": name, "params": params}], calendar=None)
    if isinstance(out, pl.LazyFrame):
        out = out.collect()
    expected = {"bid_ask_imbalance": 0.5, "book_depth_ratio": 0.75, "weighted_mid_price": 100.5}
    assert out[name].to_list() == [expected[name]] * frame.height
    assert_frame_equal(out.select(frame.columns), frame)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("expression", [False, True])
@pytest.mark.parametrize("name", sorted(_QUOTE_PARAMS))
def test_missing_custom_quote_column_names_the_actual_input(
    name: str, expression: bool, lazy: bool
) -> None:
    frame = _frame_for(INPUT_TYPE_COLUMNS["OHLCV"] + INPUT_TYPE_COLUMNS["quote"])
    params = {
        key: pl.col(value) if expression else value for key, value in _QUOTE_PARAMS[name].items()
    }
    data = frame.lazy() if lazy else frame
    with pytest.raises(ValueError, match=rf"{name}.*needs column") as error:
        compute_features(data, [{"name": name, "params": params}], calendar=None)
    assert "'buy_size'" in str(error.value)
    assert "'sell_size'" in str(error.value)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("name", ["bid_ask_imbalance", "book_depth_ratio"])
def test_size_only_quotes_do_not_require_unused_prices(name: str, lazy: bool) -> None:
    frame = _frame_for(INPUT_TYPE_COLUMNS["OHLCV"] + ["bid_size", "ask_size"])
    out = compute_features(frame.lazy() if lazy else frame, [name], calendar=None)
    if isinstance(out, pl.LazyFrame):
        out = out.collect()
    assert out[name].null_count() == 0


@pytest.mark.parametrize("lazy", [False, True])
def test_multi_output_feature_names_an_overridden_missing_column(lazy: bool) -> None:
    frame = _frame_for(INPUT_TYPE_COLUMNS["OHLCV"])
    request = [{"name": "donchian_channels", "params": {"period": 2, "low_col": "floor"}}]
    with pytest.raises(ValueError, match=r"donchian_channels.*needs column.*floor"):
        compute_features(frame.lazy() if lazy else frame, request, calendar=None)
    supplied = frame.with_columns(pl.lit(3.0).alias("floor"))
    out = compute_features(supplied.lazy() if lazy else supplied, request, calendar=None)
    if isinstance(out, pl.LazyFrame):
        out = out.collect()
    assert out.height == frame.height
    assert set(out["donchian_channels_1"].drop_nulls().to_list()) == {3.0}


@pytest.mark.parametrize("input_type", ["close", "unknown"])
def test_registration_rejects_a_label_that_cannot_supply_required_returns(input_type: str) -> None:
    from ml4t.engineer.core.decorators import feature

    with pytest.raises(TypeError, match="input_type"):

        @feature(
            name="invalid_returns_contract",
            category="volatility",
            description="Invalid label",
            lookback=0,
            input_type=input_type,
        )
        def invalid_returns_contract(returns: pl.Expr | str) -> pl.Expr:
            return pl.col(returns) if isinstance(returns, str) else returns


def test_third_party_returns_feature_is_registered_and_computes_from_inferred_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ml4t.engineer.core import decorators
    from ml4t.engineer.core.registry import FeatureRegistry

    registry = FeatureRegistry()
    monkeypatch.setattr(decorators, "get_registry", lambda: registry)

    @decorators.feature(
        name="custom_returns_contract",
        category="volatility",
        description="Returns variance",
        lookback=0,
    )
    def custom_returns_contract(returns: pl.Expr | str) -> pl.Expr:
        expr = pl.col(returns) if isinstance(returns, str) else returns
        return expr * expr

    metadata = registry.get("custom_returns_contract")
    assert metadata is not None
    assert metadata.input_type == "returns"
    result = pl.DataFrame({"returns": [0.0, 0.5, -0.25]}).select(metadata.func("returns"))
    assert result.to_series().to_list() == [0.0, 0.25, 0.0625]
